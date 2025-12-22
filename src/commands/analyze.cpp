#include "commands/analyze.hpp"
#include "llm/MockLLMClient.hpp"
#include "llm/OllamaLLMClient.hpp"
#include "llm/LLMClient.hpp"

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
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <functional>

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
    std::ostream* a = nullptr;
    std::ostream* b = nullptr;
    template <typename T>
    Printer& operator<<(const T& v) {
        if (a) (*a) << v;
        if (b) (*b) << v;
        return *this;
    }
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
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
        out.open(p, std::ios::out | std::ios::trunc);
        return (bool)out;
    } catch (...) {
        return false;
    }
}

// ---------- tokenize helpers ----------

static std::string to_lower_ascii(std::string s) {
    for (char& c : s) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return s;
}

static std::string trim_ascii(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
    size_t j = s.size();
    while (j > i && (s[j - 1] == ' ' || s[j - 1] == '\t' || s[j - 1] == '\n' || s[j - 1] == '\r')) --j;
    return s.substr(i, j - i);
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

static std::unordered_set<std::string> tokenize_text(const std::string& raw) {
    std::string norm = textutil::normalize(raw);
    auto toks = textutil::tokenize(norm);
    auto ntoks = textutil::normalize_tokens(toks);
    std::unordered_set<std::string> s;
    s.reserve(ntoks.size());
    for (auto& t : ntoks) if (!t.empty()) s.insert(t);
    if (s.find("cpp") != s.end()) {
        s.erase("cpp");
        s.insert("c++");
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

// ---------- Day 3 helpers ----------

static std::string canonicalize_skill(const std::string& raw) {
    std::string s = trim_ascii(raw);
    s = to_lower_ascii(s);

    if (s == "c++17" || s == "c++20" || s == "c++14" || s == "c++11") return "c++";
    if (s == "cpp") return "c++";
    if (s == "js") return "javascript";
    if (s == "ts") return "typescript";
    if (s == "py") return "python";

    return s;
}

static double span_weight_from_category(const std::string& cat) {
    (void)cat;
    return 1.0;
}

static double span_weight_from_span_type(const std::string& t) {
    if (t == "requirement") return 1.0;
    if (t == "preferred") return 0.6;
    if (t == "responsibility") return 0.4;
    return 0.2;
}

static double strength_weight(const std::string& s) {
    if (s == "must") return 1.0;
    if (s == "should") return 0.7;
    if (s == "nice") return 0.4;
    return 0.6;
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
                } else oss << c;
        }
    }
    return oss.str();
}

struct Mention {
    std::string posting_id;
    std::string category;   // for non-LLM mode
    std::string raw;
    std::string canonical;
    std::string strength;
    std::string polarity;
    double confidence = 0.0;
    double contrib = 0.0;
};

struct SkillAgg {
    int raw_count = 0;
    double sum_contrib = 0.0;
    std::vector<std::string> evidence;
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

    // skill_weights
    f << "  \"skill_weights\": {\n";
    for (size_t i = 0; i < weights_sorted.size(); ++i) {
        const auto& kv = weights_sorted[i];
        f << "    \"" << json_escape(kv.first) << "\": " << kv.second;
        f << (i + 1 < weights_sorted.size() ? ",\n" : "\n");
    }
    f << "  },\n";

    // evidence (cap to 50 keys, no trailing comma ever)
    f << "  \"evidence\": {\n";

    std::vector<std::string> keys;
    keys.reserve(std::min<size_t>(50, weights_sorted.size()));

    for (const auto& kv : weights_sorted) {
        if (keys.size() >= 50) break;
        if (agg.find(kv.first) == agg.end()) continue;
        keys.push_back(kv.first);
    }

    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string& k = keys[i];
        auto it = agg.find(k);
        const auto& ev = it->second.evidence;

        f << "    \"" << json_escape(k) << "\": [";
        for (size_t j = 0; j < ev.size(); ++j) {
            if (j) f << ", ";
            f << "\"" << json_escape(ev[j]) << "\"";
        }
        f << "]";
        f << (i + 1 < keys.size() ? ",\n" : "\n");
    }

    f << "  }\n";
    f << "}\n";
}


// --- shrink what you send to the LLM ---
static std::string shrink_posting_for_llm(const std::string& raw) {
    std::string lower = to_lower_ascii(raw);

    struct Key { const char* k; };
    const Key starts[] = {
        {"requirements"}, {"requirement"}, {"qualifications"}, {"qualification"},
        {"responsibilities"}, {"responsibility"},
        {"what you will do"}, {"what you'll do"}, {"what you bring"},
        {"preferred"}, {"nice to have"}, {"nice-to-have"}, {"optional"}, {"bonus"}, {"plus"}
    };

    const Key stops[] = {
        {"benefits"}, {"perks"}, {"about us"}, {"about the company"},
        {"equal opportunity"}, {"eeo"}, {"privacy"}, {"legal"}, {"who we are"}
    };

    auto find_any = [&](const Key* arr, size_t n, size_t from) -> size_t {
        size_t best = std::string::npos;
        for (size_t i = 0; i < n; ++i) {
            size_t pos = lower.find(arr[i].k, from);
            if (pos != std::string::npos) {
                if (best == std::string::npos || pos < best) best = pos;
            }
        }
        return best;
    };

    std::string out;
    out.reserve(9000);

    size_t from = 0;
    int blocks = 0;
    while (blocks < 3) {
        size_t s = find_any(starts, sizeof(starts)/sizeof(starts[0]), from);
        if (s == std::string::npos) break;

        size_t e = find_any(stops, sizeof(stops)/sizeof(stops[0]), s + 1);
        if (e == std::string::npos) e = std::min(raw.size(), s + (size_t)4500);
        else e = std::min(e, s + (size_t)4500);

        if (!out.empty()) out += "\n\n";
        out += raw.substr(s, e - s);

        from = e;
        blocks++;
    }

    if (out.empty()) {
        size_t cap = std::min(raw.size(), (size_t)8000);
        out = raw.substr(0, cap);
    } else {
        if (out.size() > 9000) out.resize(9000);
    }

    return out;
}

// ---------- NEW: zone extraction for rerank ----------

struct Zones {
    std::string title;
    std::string lead;
    std::string req;
};

static std::string extract_title_from_kv_blob(const std::string& raw) {
    std::string lower = to_lower_ascii(raw);
    size_t tpos = lower.find(":title");
    if (tpos == std::string::npos) return "";

    size_t start = raw.find_first_not_of(" \t\r\n", tpos + 6);
    if (start == std::string::npos) return "";

    while (start < raw.size() && (raw[start] == ' ' || raw[start] == '\t' || raw[start] == ':')) start++;
    while (start < raw.size() && (raw[start] == ' ' || raw[start] == '\t')) start++;

    size_t end = raw.find(", :description", start);
    if (end == std::string::npos) end = raw.find(", :location", start);
    if (end == std::string::npos) end = raw.find(", :employer", start);
    if (end == std::string::npos) end = raw.find(", :skills", start);
    if (end == std::string::npos) end = raw.find('\n', start);
    if (end == std::string::npos) end = std::min(raw.size(), start + (size_t)160);

    if (end <= start) return "";
    std::string t = trim_ascii(raw.substr(start, end - start));
    if (t.size() > 200) t.resize(200);
    return t;
}

static std::string extract_title_fallback_line(const std::string& raw) {
    size_t i = 0;
    while (i < raw.size() && i < 2000) {
        size_t j = raw.find('\n', i);
        if (j == std::string::npos) j = raw.size();
        std::string line = trim_ascii(raw.substr(i, j - i));
        if (!line.empty()) {
            if (line.size() <= 90) return line;
            break;
        }
        i = (j == raw.size()) ? j : j + 1;
    }
    return "";
}

static std::string extract_requirements_block(const std::string& raw) {
    std::string lower = to_lower_ascii(raw);

    struct Key { const char* k; };
    const Key starts[] = {
        {"requirements"}, {"requirement"}, {"qualifications"}, {"qualification"},
        {"what you bring"}, {"what you'll bring"}, {"skills"}, {"you have"}, {"must have"}
    };

    const Key stops[] = {
        {"responsibilities"}, {"responsibility"},
        {"benefits"}, {"perks"}, {"about us"}, {"about the company"},
        {"equal opportunity"}, {"eeo"}, {"privacy"}, {"legal"}
    };

    auto find_any = [&](const Key* arr, size_t n, size_t from) -> size_t {
        size_t best = std::string::npos;
        for (size_t i = 0; i < n; ++i) {
            size_t pos = lower.find(arr[i].k, from);
            if (pos != std::string::npos) {
                if (best == std::string::npos || pos < best) best = pos;
            }
        }
        return best;
    };

    std::string out;
    out.reserve(5000);

    size_t from = 0;
    int blocks = 0;
    while (blocks < 2) {
        size_t s = find_any(starts, sizeof(starts)/sizeof(starts[0]), from);
        if (s == std::string::npos) break;

        size_t e = find_any(stops, sizeof(stops)/sizeof(stops[0]), s + 1);
        if (e == std::string::npos) e = std::min(raw.size(), s + (size_t)3500);
        else e = std::min(e, s + (size_t)3500);

        if (!out.empty()) out += "\n\n";
        out += raw.substr(s, e - s);

        from = e;
        blocks++;
    }

    if (out.size() > 6000) out.resize(6000);
    return out;
}

static Zones extract_zones(const std::string& raw) {
    Zones z;

    z.title = extract_title_from_kv_blob(raw);
    if (z.title.empty()) z.title = extract_title_fallback_line(raw);

    // lead: very top of posting, domain-agnostic
    const size_t LEAD_CAP = 1400;
    if (!raw.empty()) {
        size_t cap = std::min(raw.size(), LEAD_CAP);
        z.lead = raw.substr(0, cap);
    }

    z.req = extract_requirements_block(raw);
    return z;
}

// --- tokenize the query/role so lex rerank is anchored to what user asked ---
static std::unordered_set<std::string> tokenize_query(const std::string& role) {
    std::string norm = textutil::normalize(role);
    auto toks = textutil::tokenize(norm);
    auto ntoks = textutil::normalize_tokens(toks);
    std::unordered_set<std::string> s;
    s.reserve(ntoks.size());
    for (auto& t : ntoks) if (!t.empty()) s.insert(t);

    if (s.find("cpp") != s.end()) {
        s.erase("cpp");
        s.insert("c++");
    }
    return s;
}

static bool role_mentions_cpp(const std::unordered_set<std::string>& q) {
    return (q.find("c++") != q.end() || q.find("cpp") != q.end());
}

static bool post_has_cpp(const std::unordered_set<std::string>& toks) {
    return (toks.find("c++") != toks.end() || toks.find("cpp") != toks.end());
}

static bool tokens_has_any(const std::unordered_set<std::string>& toks, const std::unordered_set<std::string>& need) {
    for (const auto& x : need) {
        if (toks.find(x) != toks.end()) return true;
    }
    return false;
}

static double zone_query_score(const std::unordered_set<std::string>& zone_toks,
                               const std::unordered_set<std::string>& q_tokens,
                               const std::function<double(const std::string&)>& idf) {
    double s = 0.0;
    for (const auto& qt : q_tokens) {
        if (zone_toks.find(qt) != zone_toks.end()) s += idf(qt);
    }
    return s;
}

static bool title_has_conflicting_lang(const std::unordered_set<std::string>& title_toks,
                                       const std::unordered_set<std::string>& q_tokens) {
    const bool q_has_cpp = (q_tokens.find("c++") != q_tokens.end() || q_tokens.find("cpp") != q_tokens.end());
    if (!q_has_cpp) return false;

    const bool title_has_cpp = (title_toks.find("c++") != title_toks.end() || title_toks.find("cpp") != title_toks.end());
    if (title_has_cpp) return false;

    static const char* langs[] = {
        "java","python","ruby","c#","csharp","javascript","typescript","php","scala","kotlin","golang","go"
    };

    for (const char* l : langs) {
        if (title_toks.find(l) != title_toks.end()) return true;
    }
    return false;
}

// ---------------------------------------------------

int cmd_analyze(int argc, char** argv) {
    std::string role         = get_arg(argc, argv, "--role", "");
    std::string jobs_dir     = get_arg(argc, argv, "--jobs", "data/jobs/sample500");
    std::string topk_s       = get_arg(argc, argv, "--topk", "10"); // default changed to 15

    // LLM args
    std::string llm_mock_dir = get_arg(argc, argv, "--llm_mock", "");
    std::string llm_model    = get_arg(argc, argv, "--llm_model", "llama3.2:3b");
    std::string llm_cache    = get_arg(argc, argv, "--llm_cache", "out/llm_cache");

    std::string emb_path     = get_arg(argc, argv, "--emb", "data/embeddings/jobs.bin");
    std::string model        = get_arg(argc, argv, "--model", "models/emb/model.onnx");
    std::string vocab        = get_arg(argc, argv, "--vocab", "models/emb/vocab.txt");

    std::string min_score_s  = get_arg(argc, argv, "--min_score", "0.30");
    std::string out_path     = get_arg(argc, argv, "--out", "");

    bool use_llm    = has_flag(argc, argv, "--llm");
    bool do_profile = has_flag(argc, argv, "--profile");
    std::string outdir_s = get_arg(argc, argv, "--outdir", "out");

    // IMPORTANT CHANGE:
    // Title/top-part is FIRST PRIORITY now.
    // We still keep embedding in the mix, but it’s a *tie-breaker*.
    const size_t topn_seed   = 10;
    const size_t topx_tokens = 30;
    const size_t bigk_floor  = 80; // increased so title-only postings have more chance to show up

    if (role.empty()) {
        std::cerr << "error: missing --role\n";
        return 1;
    }

    double min_score = 0.0;
    try { min_score = std::stod(min_score_s); }
    catch (...) {
        std::cerr << "error: invalid --min_score\n";
        return 1;
    }

    size_t topk = 0;
    try { topk = (size_t)std::stoul(topk_s); }
    catch (...) {
        std::cerr << "error: invalid --topk\n";
        return 1;
    }
    if (topk == 0) topk = 1;

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
        try { pr << "OUT: " << fs::absolute(fs::path(out_path)).string() << "\n"; }
        catch (...) { pr << "OUT: " << out_path << "\n"; }
    }

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

    std::unordered_map<std::string, const JobPosting*> by_id;
    by_id.reserve(corpus.postings().size());
    for (const auto& p : corpus.postings()) by_id[p.id] = &p;

    std::unordered_map<std::string, std::unordered_set<std::string>> post_tokens;
    post_tokens.reserve(corpus.postings().size());

    std::unordered_map<std::string, int> df;
    df.reserve(4096);

    for (const auto& p : corpus.postings()) {
        auto s = tokenize_post(p.raw_text);
        for (const auto& tok : s) df[tok] += 1;
        post_tokens.emplace(p.id, std::move(s));
    }

    EmbeddingIndex idx;
    if (!idx.load(emb_path)) {
        std::cerr << "error: failed to load embeddings cache: " << emb_path << "\n";
        std::cerr << "hint: run `resume-agent embed` first\n";
        return 1;
    }

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

    size_t bigk = std::max(topk, bigk_floor);
    auto hits = idx.topk(q, bigk);

    pr << "RAW_HITS: " << hits.size() << "\n";

    if (hits.empty()) {
        pr << "KEPT: 0 (min_score=" << min_score << ")\n";
        if (write_out) { out.flush(); out.close(); }
        return 0;
    }

    // query tokens early (so we can "rescue" strong-title hits even if emb score is low)
    auto q_tokens = tokenize_query(role);
    const bool wants_cpp = role_mentions_cpp(q_tokens);

    // IMPORTANT CHANGE:
    // Do NOT throw away postings just because embedding is below min_score
    // if the TITLE / TOP PART matches the query tokens.
    std::vector<decltype(hits)::value_type> kept;
    kept.reserve(hits.size());
    for (const auto& h : hits) {
        bool keep_by_emb = (h.score >= min_score);

        bool keep_by_title_or_lead = false;
        auto itp = by_id.find(h.job_id);
        if (itp != by_id.end()) {
            Zones z = extract_zones(itp->second->raw_text);
            auto title_toks = tokenize_text(z.title);
            auto lead_toks  = tokenize_text(z.lead);

            // "title/top is first priority": if any query token appears in title OR lead, keep it.
            // This is what makes your one-line "C++ Backend Engineer" reliably survive filtering.
            keep_by_title_or_lead = tokens_has_any(title_toks, q_tokens) || tokens_has_any(lead_toks, q_tokens);
        }

        if (keep_by_emb || keep_by_title_or_lead) kept.push_back(h);
    }

    pr << "KEPT: " << kept.size() << " (min_score=" << min_score << ", title/lead rescue enabled)\n";

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

    // seed top tokens from seed hits (keeps your existing "top_tokens" flavor)
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
        double s2 = (double)kv.second * idf(kv.first);
        if (s2 > 0.0) scored.push_back({kv.first, s2});
    }

    std::sort(scored.begin(), scored.end(),
              [](const TokScore& a, const TokScore& b){ return a.score > b.score; });

    if (scored.size() > topx_tokens) scored.resize(topx_tokens);

    std::unordered_set<std::string> top_tokens;
    top_tokens.reserve(scored.size() * 2);
    for (const auto& ts : scored) top_tokens.insert(ts.tok);

    // always include query tokens in lex scoring set
    for (const auto& qt : q_tokens) {
        if (!qt.empty()) top_tokens.insert(qt);
    }

    struct RankedHit {
        std::string job_id;

        double emb_score = 0.0;
        double lex_score = 0.0;      // still printed (now mostly header-driven)
        double combined  = 0.0;      // now effectively "header-first"

        // debug flags
        bool has_cpp = false;
        bool has_title = false;
        bool title_conflict = false;
        bool identity_match = false;

        // debug scores
        double s_title = 0.0;
        double s_lead  = 0.0;
        double s_req   = 0.0;
    };

    std::vector<RankedHit> ranked;
    ranked.reserve(kept.size());

    // IMPORTANT CHANGE: SUPER HEAVY title weighting.
    // If you want “title/top part is first priority”, you do NOT want alpha=0.7 anymore.
    // Here we make title dominate; embedding becomes a weak tie-breaker.
    const double WT_TITLE = 200.0;
    const double WT_LEAD  = 80.0;
    const double WT_REQ   = 20.0;
    const double WT_BODYQ = 4.0;
    const double WT_BODYLEX = 1.0;

    // penalties/bonuses for identity mismatch (C++ example, but only kicks in when query has C++)
    const double PENALTY_TITLE_CONFLICT = 500.0; // "Java ..." title when query asks for C++
    const double PENALTY_MISSING_IDENTITY = 200.0;
    const double BONUS_IDENTITY_IN_TITLE  = 120.0;

    for (const auto& h : kept) {
        auto itp = by_id.find(h.job_id);
        if (itp == by_id.end()) continue;

        const auto& post = *itp->second;

        auto pt_it = post_tokens.find(h.job_id);
        if (pt_it == post_tokens.end()) continue;

        const auto& body_toks = pt_it->second;

        Zones z = extract_zones(post.raw_text);

        auto title_toks = tokenize_text(z.title);
        auto lead_toks  = tokenize_text(z.lead);
        auto req_toks   = tokenize_text(z.req);

        const bool has_title = !trim_ascii(z.title).empty();

        // Title/lead identity match: query tokens appear in title or lead
        const bool title_match = has_title ? tokens_has_any(title_toks, q_tokens) : false;
        const bool lead_match  = tokens_has_any(lead_toks,  q_tokens);
        const bool identity_match = (has_title ? (title_match || lead_match) : lead_match);

        // base body lex (seeded tokens) — now small, since you want header first
        double base_lex = 0.0;
        for (const auto& tok : body_toks) {
            if (top_tokens.find(tok) != top_tokens.end()) base_lex += idf(tok);
        }

        // query-token presence scores per zone
        double s_title = zone_query_score(title_toks, q_tokens, idf);
        double s_lead  = zone_query_score(lead_toks,  q_tokens, idf);
        double s_req   = zone_query_score(req_toks,   q_tokens, idf);
        double s_bodyq = zone_query_score(body_toks,  q_tokens, idf);

        // C++ conflict logic retained (helps avoid "Java Software Engineer" when role says C++)
        const bool title_conflict = title_has_conflicting_lang(title_toks, q_tokens);
        const bool title_has_cpp = (title_toks.find("c++") != title_toks.end() || title_toks.find("cpp") != title_toks.end());
        const bool lead_has_cpp  = (lead_toks.find("c++")  != lead_toks.end()  || lead_toks.find("cpp")  != lead_toks.end());

        double identity_adj = 0.0;
        if (wants_cpp) {
            if (title_conflict) identity_adj -= PENALTY_TITLE_CONFLICT;

            const bool in_title_or_lead = (title_has_cpp || lead_has_cpp);
            if (!in_title_or_lead) identity_adj -= PENALTY_MISSING_IDENTITY;
            else if (title_has_cpp) identity_adj += BONUS_IDENTITY_IN_TITLE;
        }

        // FINAL: Header-first score.
        // This is what forces a one-line "C++ Backend Engineer" posting to rise to the top.
        double header_first_score =
            WT_TITLE * s_title +
            WT_LEAD  * s_lead  +
            WT_REQ   * s_req   +
            WT_BODYQ * s_bodyq +
            WT_BODYLEX * base_lex +
            identity_adj;

        // Embedding is now ONLY a tiny tie-breaker.
        // (We keep it so the system still works when postings have no usable titles.)
        const double emb_tiebreak = 5.0 * (double)h.score;

        double combined = header_first_score + emb_tiebreak;

        RankedHit rh;
        rh.job_id = h.job_id;
        rh.emb_score = (double)h.score;
        rh.lex_score = header_first_score; // keep printing as "lex" for continuity
        rh.combined  = combined;

        rh.has_cpp = post_has_cpp(body_toks);
        rh.has_title = has_title;
        rh.title_conflict = title_conflict;
        rh.identity_match = identity_match;

        rh.s_title = s_title;
        rh.s_lead  = s_lead;
        rh.s_req   = s_req;

        ranked.push_back(std::move(rh));
    }

    // HARD RULE: title/lead identity matches are always first.
    // Then sort by the header-first score.
    std::stable_partition(ranked.begin(), ranked.end(),
                          [](const RankedHit& r){ return r.identity_match; });

    std::sort(ranked.begin(), ranked.end(),
              [](const RankedHit& a, const RankedHit& b){ return a.combined > b.combined; });

    if (ranked.size() > topk) ranked.resize(topk);

    pr << "TOPK: " << ranked.size() << "\n";

    // LLM clients
    llm::NullLLMClient null_llm;
    llm::MockLLMClient mock_llm(llm_mock_dir.empty() ? "llm_mock" : llm_mock_dir);
    llm::OllamaLLMClient ollama_llm(llm_model, llm_cache);

    llm::LLMClient* llm_client = nullptr;

    if (use_llm) {
        if (!llm_mock_dir.empty()) {
            llm_client = (llm::LLMClient*)&mock_llm;
        } else {
            llm_client = (llm::LLMClient*)&ollama_llm;
        }
    }

    RequirementExtractor ex;

    std::vector<Mention> all_mentions;
    all_mentions.reserve(ranked.size() * 32);

    std::unordered_map<std::string, SkillAgg> agg;

    // posting_id -> canonical skill -> best mention
    std::unordered_map<std::string, std::unordered_map<std::string, Mention>> best_by_posting;
    best_by_posting.reserve(ranked.size());

    for (size_t i = 0; i < ranked.size(); ++i) {
        const auto& rh = ranked[i];
        auto it = by_id.find(rh.job_id);
        if (it == by_id.end()) continue;

        pr << "\n# hit " << rh.job_id
           << " combined=" << rh.combined
           << " emb=" << rh.emb_score
           << " header=" << rh.lex_score
           << " TITLE=" << rh.s_title
           << " LEAD=" << rh.s_lead
           << " REQ=" << rh.s_req
           << " ID_MATCH=" << (rh.identity_match ? "yes" : "no");

        if (wants_cpp) pr << " TITLE_CONFLICT=" << (rh.title_conflict ? "yes" : "no");
        pr << "\n";

        const std::string& post_id = it->second->id;
        const std::string& text    = it->second->raw_text;

        if (!use_llm) {
            // ------- Non-LLM path -------
            auto reqs = ex.extract(text);
            print_reqs(pr, post_id, reqs);

            if (!do_profile) continue;

            for (const auto& [cat, items] : reqs.by_category) {
                if (items.empty()) continue;

                const double sw = span_weight_from_category(cat);
                for (const auto& raw_skill : items) {
                    std::string canon = canonicalize_skill(raw_skill);
                    if (canon.empty()) continue;

                    Mention m;
                    m.posting_id = post_id;
                    m.category   = cat;
                    m.raw        = raw_skill;
                    m.canonical  = canon;

                    m.strength   = "must";
                    m.polarity   = "positive";
                    m.confidence = 1.0;
                    m.contrib    = sw * 1.0 * m.confidence;

                    auto& mp = best_by_posting[m.posting_id];
                    auto it2 = mp.find(m.canonical);
                    if (it2 == mp.end() || m.contrib > it2->second.contrib) mp[m.canonical] = m;
                }
            }
        } else {
            // ------- LLM path -------
            if (!do_profile) continue;
            if (!llm_client) continue;

            std::string shrunk = shrink_posting_for_llm(text);
            auto evidences = llm_client->analyze_posting(post_id, shrunk);

            for (const auto& ev0 : evidences) {
                std::string pol = ev0.polarity;
                if (pol.empty()) pol = "positive";
                if (pol == "negated") continue;

                std::string st = ev0.strength;
                if (st.empty()) st = "unknown";

                std::string stype = ev0.span_type;
                if (stype.empty()) stype = "other";

                const double sw  = span_weight_from_span_type(stype);
                const double stw = strength_weight(st);

                for (const auto& sh : ev0.skills) {
                    std::string canon = canonicalize_skill(!sh.canonical.empty() ? sh.canonical : sh.raw);
                    if (canon.empty()) continue;

                    Mention m;
                    m.posting_id = post_id;
                    m.category   = "";
                    m.raw        = sh.raw;
                    m.canonical  = canon;

                    m.strength   = st;
                    m.polarity   = pol;
                    m.confidence = sh.confidence;
                    m.contrib    = sw * stw * m.confidence;

                    auto& mp = best_by_posting[m.posting_id];
                    auto it2 = mp.find(m.canonical);
                    if (it2 == mp.end() || m.contrib > it2->second.contrib) mp[m.canonical] = m;
                }
            }
        }
    }

    if (do_profile) {
        for (auto& pkv : best_by_posting) {
            for (auto& skv : pkv.second) all_mentions.push_back(skv.second);
        }

        for (const auto& m : all_mentions) {
            if (m.polarity == "negated") continue;
            auto& a = agg[m.canonical];
            a.raw_count += 1;
            a.sum_contrib += m.contrib;
            if (a.evidence.size() < 3) a.evidence.push_back(m.raw);
        }

        const int N = (int)ranked.size();

        std::vector<std::pair<std::string, double>> weights;
        weights.reserve(agg.size());

        std::vector<std::string> core, secondary, nice;

        for (const auto& kv : agg) {
            const std::string& skill = kv.first;
            const SkillAgg& a = kv.second;

            double freq = (N > 0) ? ((double)a.raw_count / (double)N) : 0.0;
            double avg_contrib = (a.raw_count > 0) ? (a.sum_contrib / (double)a.raw_count) : 0.0;

            double w = 0.7 * freq + 0.3 * avg_contrib;
            weights.push_back({skill, w});

            // --- adaptive thresholds based on sample size ---
            double core_freq_cutoff =
                (N >= 20) ? 0.55 :
                (N >= 10) ? 0.50 :
                            0.40;

            double core_weight_cutoff =
                (N >= 20) ? 0.75 :
                (N >= 10) ? 0.65 :
                            0.55;

            double secondary_freq_cutoff =
                (N >= 20) ? 0.25 :
                (N >= 10) ? 0.20 :
                            0.15;

            double secondary_weight_cutoff =
                (N >= 20) ? 0.45 :
                (N >= 10) ? 0.40 :
                            0.35;

            // --- classification ---
            if (freq >= core_freq_cutoff && w >= core_weight_cutoff) {
                core.push_back(skill);
            }
            else if (freq >= secondary_freq_cutoff && w >= secondary_weight_cutoff) {
                secondary.push_back(skill);
            }
            else {
                nice.push_back(skill);
            }


        }

        std::sort(weights.begin(), weights.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });

        auto sort_alpha = [](std::vector<std::string>& v) { std::sort(v.begin(), v.end()); };
        sort_alpha(core);
        sort_alpha(secondary);
        sort_alpha(nice);

        fs::path mentions_path = outdir / "mentions.jsonl";
        fs::path profile_path  = outdir / "profile.json";

        write_mentions_jsonl(mentions_path, all_mentions);
        write_profile_json(profile_path, role, (int)ranked.size(), weights, core, secondary, nice, agg);

        pr << "\nwrote " << mentions_path.string() << "\n";
        pr << "wrote " << profile_path.string() << "\n";
    }

    if (write_out) {
        out.flush();
        out.close();
        std::cout << "\nWROTE: " << out_path << "\n";
    }

    return 0;
}
