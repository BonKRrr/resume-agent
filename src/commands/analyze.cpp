#include "commands/analyze.hpp"
#include "jobs/JobCorpus.hpp"
#include "jobs/RequirementExtractor.hpp"
#include "jobs/TextUtil.hpp"
#include "jobs/EmbeddingIndex.hpp"
#include "emb/MiniLmEmbedder.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
    }
    return def;
}

static void print_reqs(const std::string& id, const ExtractedReqs& r) {
    std::cout << "\nPOST " << id << "\n";
    for (const auto& [cat, items] : r.by_category) {
        if (items.empty()) continue;
        std::cout << "- " << cat << ": ";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << items[i];
        }
        std::cout << "\n";
    }
}

static std::unordered_set<std::string> tokenize_post(const std::string& raw) {
    // normalize -> tokenize -> normalize_tokens
    std::string norm = textutil::normalize(raw);
    auto toks = textutil::tokenize(norm);
    auto ntoks = textutil::normalize_tokens(toks);

    // dedupe into a set for DF/overlap counting
    std::unordered_set<std::string> s;
    s.reserve(ntoks.size());
    for (auto& t : ntoks) {
        if (t.empty()) continue;
        s.insert(t);
    }
    return s;
}

int cmd_analyze(int argc, char** argv) {
    std::string role     = get_arg(argc, argv, "--role", "");
    std::string jobs_dir = get_arg(argc, argv, "--jobs", "data/jobs/raw");
    std::string topk_s   = get_arg(argc, argv, "--topk", "25");

    std::string emb_path = get_arg(argc, argv, "--emb", "data/embeddings/jobs.bin");
    std::string model    = get_arg(argc, argv, "--model", "models/emb/model.onnx");
    std::string vocab    = get_arg(argc, argv, "--vocab", "models/emb/vocab.txt");

    // rerank knobs (all optional)
    double alpha = 0.70; // weight on embedding score
    size_t topn_seed = 10; // how many top hits to learn "role vocab" from
    size_t topx_tokens = 30; // how many bootstrapped tokens to use
    size_t bigk_floor = 50; // how many candidates to retrieve before reranking

    if (role.empty()) {
        std::cerr << "error: missing --role\n";
        return 1;
    }

    JobCorpus corpus = JobCorpus::load_from_dir(jobs_dir);

    std::cout << "ROLE: " << role << "\n";
    std::cout << "JOBS_DIR: " << jobs_dir << "\n";
    std::cout << "POSTINGS: " << corpus.postings().size() << "\n";

    // Map id -> posting pointer
    std::unordered_map<std::string, const JobPosting*> by_id;
    by_id.reserve(corpus.postings().size());
    for (const auto& p : corpus.postings()) by_id[p.id] = &p;

    // Precompute token sets + document frequencies
    std::unordered_map<std::string, std::unordered_set<std::string>> post_tokens;
    post_tokens.reserve(corpus.postings().size());

    std::unordered_map<std::string, int> df;
    df.reserve(4096);

    for (const auto& p : corpus.postings()) {
        auto s = tokenize_post(p.raw_text);
        for (const auto& tok : s) {
            df[tok] += 1;
        }
        post_tokens.emplace(p.id, std::move(s));
    }

    // Load embeddings
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

    size_t topk = (size_t)std::stoul(topk_s);

    // Step 1: get a bigger candidate set than requested so reranking has room
    size_t bigk = std::max(topk, bigk_floor);
    auto hits = idx.topk(q, bigk);

    if (hits.empty()) {
        std::cout << "TOPK: 0\n";
        return 0;
    }

    // Step 2: learn "role vocabulary" from top seed hits, weighted by IDF
    const size_t M = corpus.postings().size();
    topn_seed = std::min(topn_seed, hits.size());

    std::unordered_map<std::string, int> tf_top;
    tf_top.reserve(1024);

    auto idf = [&](const std::string& tok) -> double {
        auto it = df.find(tok);
        int d = (it == df.end()) ? 0 : it->second;
        // smoothed IDF
        return std::log((1.0 + (double)M) / (1.0 + (double)d));
    };

    for (size_t i = 0; i < topn_seed; ++i) {
        const auto& h = hits[i];
        auto pt_it = post_tokens.find(h.job_id);
        if (pt_it == post_tokens.end()) continue;
        for (const auto& tok : pt_it->second) {
            tf_top[tok] += 1;
        }
    }

    // Rank candidate tokens by (tf_top * idf) and pick top X
    struct TokScore { std::string tok; double score; };
    std::vector<TokScore> scored;
    scored.reserve(tf_top.size());

    for (const auto& kv : tf_top) {
        // ignore ultra-common junk and extremely rare singletons if you want,
        // but we keep it simple and let IDF handle it.
        double s = (double)kv.second * idf(kv.first);
        if (s > 0.0) scored.push_back({kv.first, s});
    }

    std::sort(scored.begin(), scored.end(),
              [](const TokScore& a, const TokScore& b){ return a.score > b.score; });

    if (scored.size() > topx_tokens) scored.resize(topx_tokens);

    std::unordered_set<std::string> top_tokens;
    top_tokens.reserve(scored.size() * 2);
    for (const auto& ts : scored) top_tokens.insert(ts.tok);

    // Step 3: rerank candidates by embedding_score + overlap_IDF(top_tokens)
    struct RankedHit {
        std::string job_id;
        double emb_score;
        double lex_score;
        double combined;
    };

    std::vector<RankedHit> ranked;
    ranked.reserve(hits.size());

    for (const auto& h : hits) {
        auto pt_it = post_tokens.find(h.job_id);
        if (pt_it == post_tokens.end()) continue;

        double lex = 0.0;
        for (const auto& tok : pt_it->second) {
            if (top_tokens.find(tok) != top_tokens.end()) {
                lex += idf(tok);
            }
        }

        double emb_s = (double)h.score;
        double comb = alpha * emb_s + (1.0 - alpha) * lex;
        ranked.push_back({h.job_id, emb_s, lex, comb});
    }

    std::sort(ranked.begin(), ranked.end(),
              [](const RankedHit& a, const RankedHit& b){ return a.combined > b.combined; });

    if (ranked.size() > topk) ranked.resize(topk);

    std::cout << "TOPK: " << ranked.size() << "\n";

    RequirementExtractor ex;

    for (size_t i = 0; i < ranked.size(); ++i) {
        const auto& rh = ranked[i];
        auto it = by_id.find(rh.job_id);
        if (it == by_id.end()) continue;

        std::cout << "\n# hit " << rh.job_id
                  << " combined=" << rh.combined
                  << " emb=" << rh.emb_score
                  << " lex=" << rh.lex_score
                  << "\n";

        auto reqs = ex.extract(it->second->raw_text);
        print_reqs(it->second->id, reqs);
    }

    return 0;
}
