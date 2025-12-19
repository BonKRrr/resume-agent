#include "commands/analyze.hpp"
#include "jobs/JobCorpus.hpp"
#include "jobs/RequirementExtractor.hpp"
#include "jobs/TextUtil.hpp"
#include "jobs/EmbeddingIndex.hpp"
#include "emb/MiniLmEmbedder.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
    }
    return def;
}

struct Printer {
    std::ostream* a = nullptr; // stdout
    std::ostream* b = nullptr; // optional file
    template <typename T>
    Printer& operator<<(const T& v) {
        if (a) (*a) << v;
        if (b) (*b) << v;
        return *this;
    }
    // support std::endl
    Printer& operator<<(std::ostream& (*manip)(std::ostream&)) {
        if (a) manip(*a);
        if (b) manip(*b);
        return *this;
    }
};

static bool open_out(std::ofstream& out, const std::string& out_path) {
    if (out_path.empty()) return false;
    try {
        fs::path p(out_path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path());
        }
        out.open(p, std::ios::out | std::ios::trunc);
        return (bool)out;
    } catch (...) {
        return false;
    }
}

static std::unordered_set<std::string> tokenize_post(const std::string& raw) {
    std::string norm = textutil::normalize(raw);
    auto toks = textutil::tokenize(norm);
    auto ntoks = textutil::normalize_tokens(toks);

    std::unordered_set<std::string> s;
    s.reserve(ntoks.size());
    for (auto& t : ntoks) {
        if (!t.empty()) s.insert(t);
    }
    return s;
}

static void print_reqs(Printer& pr, const std::string& id, const ExtractedReqs& r) {
    pr << "\nPOST " << id << "\n";
    for (const auto& [cat, items] : r.by_category) {
        if (items.empty()) continue;
        pr << "- " << cat << ": ";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) pr << ", ";
            pr << items[i];
        }
        pr << "\n";
    }
}

int cmd_analyze(int argc, char** argv) {
    std::string role        = get_arg(argc, argv, "--role", "");
    std::string jobs_dir    = get_arg(argc, argv, "--jobs", "data/jobs/raw");
    std::string topk_s      = get_arg(argc, argv, "--topk", "25");

    std::string emb_path    = get_arg(argc, argv, "--emb", "data/embeddings/jobs.bin");
    std::string model       = get_arg(argc, argv, "--model", "models/emb/model.onnx");
    std::string vocab       = get_arg(argc, argv, "--vocab", "models/emb/vocab.txt");

    std::string min_score_s = get_arg(argc, argv, "--min_score", "0.30");
    std::string out_path    = get_arg(argc, argv, "--out", "");

    // rerank knobs (Day 2 constants)
    const double alpha       = 0.70;
    const size_t topn_seed   = 10;
    const size_t topx_tokens = 30;
    const size_t bigk_floor  = 50;

    if (role.empty()) {
        std::cerr << "error: missing --role\n";
        return 1;
    }

    double min_score = 0.0;
    try {
        min_score = std::stod(min_score_s);
    } catch (...) {
        std::cerr << "error: invalid --min_score\n";
        return 1;
    }

    size_t topk = 0;
    try {
        topk = (size_t)std::stoul(topk_s);
    } catch (...) {
        std::cerr << "error: invalid --topk\n";
        return 1;
    }
    if (topk == 0) topk = 1;

    // Optional file output
    std::ofstream out;
    bool write_out = false;
    if (!out_path.empty()) {
        write_out = open_out(out, out_path);
        if (!write_out) {
            std::cerr << "error: failed to open --out path: " << out_path << "\n";
            return 1;
        }
    }

    Printer pr;
    pr.a = &std::cout;
    pr.b = write_out ? (std::ostream*)&out : nullptr;

    if (write_out) {
        try {
            pr << "OUT: " << fs::absolute(fs::path(out_path)).string() << "\n";
        } catch (...) {
            pr << "OUT: " << out_path << "\n";
        }
    }

    JobCorpus corpus = JobCorpus::load_from_dir(jobs_dir);

    pr << "ROLE: " << role << "\n";
    pr << "JOBS_DIR: " << jobs_dir << "\n";
    pr << "POSTINGS: " << corpus.postings().size() << "\n";

    // Map id -> posting pointer
    std::unordered_map<std::string, const JobPosting*> by_id;
    by_id.reserve(corpus.postings().size());
    for (const auto& p : corpus.postings()) by_id[p.id] = &p;

    // Token sets + DF
    std::unordered_map<std::string, std::unordered_set<std::string>> post_tokens;
    post_tokens.reserve(corpus.postings().size());

    std::unordered_map<std::string, int> df;
    df.reserve(4096);

    for (const auto& p : corpus.postings()) {
        auto s = tokenize_post(p.raw_text);
        for (const auto& tok : s) df[tok] += 1;
        post_tokens.emplace(p.id, std::move(s));
    }

    // Load embedding index
    EmbeddingIndex idx;
    if (!idx.load(emb_path)) {
        std::cerr << "error: failed to load embeddings cache: " << emb_path << "\n";
        std::cerr << "hint: run `resume-agent embed` first\n";
        return 1;
    }

    // Embed query
    MiniLmEmbedder emb;
    if (!emb.init(model, vocab)) {
        std::cerr << "error: failed to init embedder for query\n";
        return 1;
    }

    auto q = emb.embed(role, 64);
    if (q.empty() || q.size() != idx.dim()) {
        std::cerr << "error: query embedding dim mismatch\n";
        return 1;
    }

    // Retrieve more than needed, then filter + rerank
    size_t bigk = std::max(topk, bigk_floor);
    auto hits = idx.topk(q, bigk);

    pr << "RAW_HITS: " << hits.size() << "\n";

    if (hits.empty()) {
        pr << "KEPT: 0 (min_score=" << min_score << ")\n";
        if (write_out) { out.flush(); out.close(); }
        return 0;
    }

    // Filter by min_score
    std::vector<decltype(hits)::value_type> kept;
    kept.reserve(hits.size());
    for (const auto& h : hits) {
        if (h.score >= min_score) kept.push_back(h);
    }
    pr << "KEPT: " << kept.size() << " (min_score=" << min_score << ")\n";

    if (kept.empty()) {
        if (write_out) { out.flush(); out.close(); }
        return 0;
    }

    const size_t M = corpus.postings().size();

    auto idf = [&](const std::string& tok) -> double {
        auto it = df.find(tok);
        int d = (it == df.end()) ? 0 : it->second;
        return std::log((1.0 + (double)M) / (1.0 + (double)d));
    };

    // Build "role vocab" from top seeds
    const size_t seedN = std::min(topn_seed, kept.size());
    std::unordered_map<std::string, int> tf_top;
    tf_top.reserve(1024);

    for (size_t i = 0; i < seedN; ++i) {
        const auto& h = kept[i];
        auto pt_it = post_tokens.find(h.job_id);
        if (pt_it == post_tokens.end()) continue;
        for (const auto& tok : pt_it->second) tf_top[tok] += 1;
    }

    struct TokScore { std::string tok; double score; };
    std::vector<TokScore> scored;
    scored.reserve(tf_top.size());
    for (const auto& kv : tf_top) {
        double s = (double)kv.second * idf(kv.first);
        if (s > 0.0) scored.push_back({kv.first, s});
    }

    std::sort(scored.begin(), scored.end(),
              [](const TokScore& a, const TokScore& b){ return a.score > b.score; });

    if (scored.size() > topx_tokens) scored.resize(topx_tokens);

    std::unordered_set<std::string> top_tokens;
    top_tokens.reserve(scored.size() * 2);
    for (const auto& ts : scored) top_tokens.insert(ts.tok);

    // Rerank
    struct RankedHit {
        std::string job_id;
        double emb_score;
        double lex_score;
        double combined;
    };

    std::vector<RankedHit> ranked;
    ranked.reserve(kept.size());

    for (const auto& h : kept) {
        auto pt_it = post_tokens.find(h.job_id);
        if (pt_it == post_tokens.end()) continue;

        double lex = 0.0;
        for (const auto& tok : pt_it->second) {
            if (top_tokens.find(tok) != top_tokens.end()) lex += idf(tok);
        }

        double emb_s = (double)h.score;
        double comb = alpha * emb_s + (1.0 - alpha) * lex;
        ranked.push_back({h.job_id, emb_s, lex, comb});
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const RankedHit& a, const RankedHit& b){ return a.combined > b.combined; });

    if (ranked.size() > topk) ranked.resize(topk);

    pr << "TOPK: " << ranked.size() << "\n";

    RequirementExtractor ex;

    for (size_t i = 0; i < ranked.size(); ++i) {
        const auto& rh = ranked[i];
        auto it = by_id.find(rh.job_id);
        if (it == by_id.end()) continue;

        pr << "\n# hit " << rh.job_id
           << " combined=" << rh.combined
           << " emb=" << rh.emb_score
           << " lex=" << rh.lex_score
           << "\n";

        auto reqs = ex.extract(it->second->raw_text);
        print_reqs(pr, it->second->id, reqs);
    }

    if (write_out) {
        out.flush();
        out.close();
        std::cout << "\nWROTE: " << out_path << "\n";
    }

    return 0;
}
