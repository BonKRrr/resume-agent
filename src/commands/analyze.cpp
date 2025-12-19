#include "commands/analyze.hpp"
#include "jobs/JobCorpus.hpp"
#include "jobs/RequirementExtractor.hpp"
#include "jobs/TextUtil.hpp"
#include "jobs/EmbeddingIndex.hpp"
#include "emb/MiniLmEmbedder.hpp"
#include "llm/LLMClient.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

static bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == key) return true;
    }
    return false;
}

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

// ---------- Day 3: RoleProfile scaffolding ----------

static std::string to_lower_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    return s;
}

static std::string trim_ascii(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    size_t j = s.size();
    while (j > i && (s[j - 1] == ' ' || s[j - 1] == '\t' || s[j - 1] == '\n' || s[j - 1] == '\r')) --j;
    return s.substr(i, j - i);
}

static std::string canonicalize_skill(const std::string& raw) {
    // Minimal canonicalization for Day 3; expand later.
    std::string s = trim_ascii(raw);
    s = to_lower_ascii(s);

    // common quick aliases
    if (s == "c++17" || s == "c++20" || s == "c++14" || s == "c++11") return "c++";
    if (s == "cpp") return "c++";
    if (s == "js") return "javascript";
    if (s == "ts") return "typescript";
    if (s == "py") return "python";

    return s;
}

static double span_weight_from_category(const std::string& cat) {
    // For now, categories from RequirementExtractor are treated as "requirement-ish".
    // You can refine later once you add real span typing from LLM segmentation.
    (void)cat;
    return 1.0;
}

static std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '\\': oss << "\\\\"; break;
            case '"':  oss << "\\\""; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if ((unsigned char)c < 0x20) {
                    oss << "\\u";
                    const char* hex = "0123456789abcdef";
                    oss << "00" << hex[(c >> 4) & 0xF] << hex[c & 0xF];
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

struct Mention {
    std::string posting_id;
    std::string category;
    std::string raw;
    std::string canonical;
    std::string strength;   // "must"|"should"|"nice"|"unknown" (Day 3 placeholder)
    std::string polarity;   // "positive"|"negated" (Day 3 placeholder)
    double confidence;      // 0..1
    double contrib;         // for weighting
};

struct SkillAgg {
    int raw_count = 0;      // number of postings where present
    double sum_contrib = 0.0;
    std::vector<std::string> evidence; // small sample of raw mentions
};

static bool ensure_dir(const fs::path& p) {
    try {
        if (p.empty()) return true;
        fs::create_directories(p);
        return true;
    } catch (...) {
        return false;
    }
}

static void write_mentions_jsonl(const fs::path& path, const std::vector<Mention>& mentions) {
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f) return;

    for (const auto& m : mentions) {
        f << "{"
          << "\"posting_id\":\"" << json_escape(m.posting_id) << "\","
          << "\"category\":\""   << json_escape(m.category) << "\","
          << "\"raw\":\""        << json_escape(m.raw) << "\","
          << "\"canonical\":\""  << json_escape(m.canonical) << "\","
          << "\"strength\":\""   << json_escape(m.strength) << "\","
          << "\"polarity\":\""   << json_escape(m.polarity) << "\","
          << "\"confidence\":"   << m.confidence << ","
          << "\"contrib\":"      << m.contrib
          << "}\n";
    }
}

static void write_profile_json(
    const fs::path& path,
    const std::string& role,
    int num_postings,
    const std::vector<std::pair<std::string, double>>& weights_sorted,
    const std::vector<std::string>& core,
    const std::vector<std::string>& secondary,
    const std::vector<std::string>& nice,
    const std::unordered_map<std::string, SkillAgg>& agg
) {
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    if (!f) return;

    f << "{\n";
    f << "  \"role\": \"" << json_escape(role) << "\",\n";
    f << "  \"num_postings\": " << num_postings << ",\n";

    auto write_str_array = [&](const char* key, const std::vector<std::string>& arr) {
        f << "  \"" << key << "\": [";
        for (size_t i = 0; i < arr.size(); ++i) {
            if (i) f << ", ";
            f << "\"" << json_escape(arr[i]) << "\"";
        }
        f << "],\n";
    };

    write_str_array("core_skills", core);
    write_str_array("secondary_skills", secondary);
    write_str_array("nice_to_have", nice);

    f << "  \"skill_weights\": {\n";
    for (size_t i = 0; i < weights_sorted.size(); ++i) {
        const auto& kv = weights_sorted[i];
        f << "    \"" << json_escape(kv.first) << "\": " << kv.second;
        f << (i + 1 < weights_sorted.size() ? ",\n" : "\n");
    }
    f << "  },\n";

    // Small evidence samples (optional but nice for explainability)
    f << "  \"evidence\": {\n";
    size_t written = 0;
    for (const auto& kv : weights_sorted) {
        auto it = agg.find(kv.first);
        if (it == agg.end()) continue;
        const auto& ev = it->second.evidence;
        f << "    \"" << json_escape(kv.first) << "\": [";
        for (size_t j = 0; j < ev.size(); ++j) {
            if (j) f << ", ";
            f << "\"" << json_escape(ev[j]) << "\"";
        }
        f << "]";
        written++;
        f << (written < weights_sorted.size() ? ",\n" : "\n");
        // keep file smaller; you can remove this cap later
        if (written >= 50) break;
    }
    f << "  }\n";

    f << "}\n";
}

// ---------------------------------------------------

int cmd_analyze(int argc, char** argv) {
    std::string role        = get_arg(argc, argv, "--role", "");
    std::string jobs_dir    = get_arg(argc, argv, "--jobs", "data/jobs/raw");
    std::string topk_s      = get_arg(argc, argv, "--topk", "25");

    std::string emb_path    = get_arg(argc, argv, "--emb", "data/embeddings/jobs.bin");
    std::string model       = get_arg(argc, argv, "--model", "models/emb/model.onnx");
    std::string vocab       = get_arg(argc, argv, "--vocab", "models/emb/vocab.txt");

    std::string min_score_s = get_arg(argc, argv, "--min_score", "0.30");
    std::string out_path    = get_arg(argc, argv, "--out", "");

    // Day 3 flags
    bool use_llm   = has_flag(argc, argv, "--llm");
    bool do_profile = has_flag(argc, argv, "--profile");
    std::string outdir_s = get_arg(argc, argv, "--outdir", "out");

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

    // Optional file output (existing Day 2 behavior)
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

    // Prepare outdir
    fs::path outdir(outdir_s);
    if (do_profile) {
        if (!ensure_dir(outdir)) {
            std::cerr << "error: failed to create --outdir: " << outdir_s << "\n";
            return 1;
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

    // LLM slot (Day 3: wired, Null by default)
    llm::NullLLMClient null_llm;
    llm::LLMClient* llm_client = use_llm ? (llm::LLMClient*)&null_llm : nullptr;
    (void)llm_client; // will be used once you implement a real adapter

    RequirementExtractor ex;

    // Day 3 aggregation buffers
    std::vector<Mention> all_mentions;
    all_mentions.reserve(ranked.size() * 32);

    // skill -> aggregate
    std::unordered_map<std::string, SkillAgg> agg;

    // presence tracking to ensure "once per posting per skill"
    std::unordered_map<std::string, std::unordered_map<std::string, Mention>> best_by_posting;
    best_by_posting.reserve(ranked.size());

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

        if (!do_profile) continue;

        // Convert extracted requirements into mentions
        for (const auto& [cat, items] : reqs.by_category) {
            if (items.empty()) continue;

            const double sw = span_weight_from_category(cat);
            for (const auto& raw_skill : items) {
                std::string canon = canonicalize_skill(raw_skill);
                if (canon.empty()) continue;

                Mention m;
                m.posting_id = it->second->id;
                m.category   = cat;
                m.raw        = raw_skill;
                m.canonical  = canon;

                // Day 3 placeholders until LLM span parsing exists
                m.strength   = "must";
                m.polarity   = "positive";
                m.confidence = 1.0;

                const double strength_w = 1.0; // must
                m.contrib = sw * strength_w * m.confidence;

                // choose best mention per posting per canonical skill
                auto& mp = best_by_posting[m.posting_id];
                auto it2 = mp.find(m.canonical);
                if (it2 == mp.end() || m.contrib > it2->second.contrib) {
                    mp[m.canonical] = m;
                }
            }
        }
    }

    if (do_profile) {
        // flatten best_by_posting into all_mentions
        for (auto& pkv : best_by_posting) {
            for (auto& skv : pkv.second) {
                all_mentions.push_back(skv.second);
            }
        }

        // aggregate
        for (const auto& m : all_mentions) {
            if (m.polarity == "negated") continue;

            auto& a = agg[m.canonical];
            a.raw_count += 1;
            a.sum_contrib += m.contrib;

            if (a.evidence.size() < 3) {
                a.evidence.push_back(m.raw);
            }
        }

        const int N = (int)ranked.size();
        // compute weights
        std::vector<std::pair<std::string, double>> weights;
        weights.reserve(agg.size());

        std::vector<std::string> core, secondary, nice;

        for (const auto& kv : agg) {
            const std::string& skill = kv.first;
            const SkillAgg& a = kv.second;

            double freq = (N > 0) ? ((double)a.raw_count / (double)N) : 0.0;
            double avg_contrib = (a.raw_count > 0) ? (a.sum_contrib / (double)a.raw_count) : 0.0;

            // simple combined weight (tunable)
            double w = 0.7 * freq + 0.3 * avg_contrib;

            weights.push_back({skill, w});

            // buckets (tweak later)
            if (freq >= 0.55 && w >= 0.75) core.push_back(skill);
            else if (freq >= 0.25 && w >= 0.45) secondary.push_back(skill);
            else nice.push_back(skill);
        }

        std::sort(weights.begin(), weights.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });

        auto sort_alpha = [](std::vector<std::string>& v) {
            std::sort(v.begin(), v.end());
        };
        sort_alpha(core);
        sort_alpha(secondary);
        sort_alpha(nice);

        // write artifacts
        fs::path mentions_path = outdir / "mentions.jsonl";
        fs::path profile_path  = outdir / "profile.json";

        write_mentions_jsonl(mentions_path, all_mentions);
        write_profile_json(profile_path, role, (int)ranked.size(), weights, core, secondary, nice, agg);

        pr << "\nDAY3: wrote " << mentions_path.string() << "\n";
        pr << "DAY3: wrote " << profile_path.string() << "\n";
    }

    if (write_out) {
        out.flush();
        out.close();
        std::cout << "\nWROTE: " << out_path << "\n";
    }

    return 0;
}
