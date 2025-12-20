#include "resume/SemanticMatcher.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace resume {

static std::string trim_copy(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;

    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;

    return s.substr(a, b - a);
}

static std::string to_lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Minimal normalization shared with scorer/build: trim + lowercase
static std::string normalize_key(const std::string& s) {
    return to_lower_copy(trim_copy(s));
}

static std::string canonicalize_skill(const std::string& s) {
    static const std::unordered_map<std::string, std::string> alias = {
        {"c++ programming language", "c++"},
        {"ruby on rails expertise", "ruby on rails"},
        {"server-side framework expertise", "server-side framework"},
        {"server-side framework experience", "server-side framework"},
        {"client-side framework experience", "client-side framework"},
        {"testing framework expertise", "testing framework"},
        {"open source contribution experience", "open source contribution"},
        {"stakeholder management experience", "stakeholder management"},
        {"technical debt management experience", "technical debt management"},
        {"refactoring expertise", "refactoring"},
        {"no sql database", "nosql database"},
    };

    auto it = alias.find(s);
    if (it != alias.end()) return it->second;
    return s;
}

static std::string norm_and_canon(const std::string& s) {
    return canonicalize_skill(normalize_key(s));
}

// ---------- NEW: filter out junk semantic targets ----------

static bool looks_like_real_skill_target(const std::string& s) {
    // exact allowlist for common short true-skills
    static const std::unordered_set<std::string> allow = {
        "c", "c++", "c#", "java", "python", "rust", "go",
        "sql", "linux", "git", "docker", "kubernetes",
        "aws", "gcp", "azure",
        "grpc", "http", "rest",
        "mongodb", "postgres", "mysql"
    };

    // exact banlist of generic nouns that embeddings love to over-match
    static const std::unordered_set<std::string> ban = {
        "engineer", "engineers", "developer", "developers",
        "development", "software", "coding",
        "experience", "best practices", "practices",
        "talent", "team", "teams",
        "framework", "frameworks" // too generic as a semantic target
    };

    if (s.empty()) return false;

    // keep allowlisted items
    if (allow.find(s) != allow.end()) return true;

    // kill obvious junk
    if (ban.find(s) != ban.end()) return false;

    // require at least one reasonably long token unless allowlisted
    // (prevents "dev", "eng", etc.)
    bool has_long_token = false;
    int token_count = 0;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && !std::isspace(static_cast<unsigned char>(s[j]))) ++j;
        token_count++;
        if ((j - i) >= 4) has_long_token = true;
        i = j;
    }

    // single-token generic skills are the most error-prone semantically
    // (e.g. "development", "design", "engineers"). We already banned some,
    // but also require multi-token unless it’s allowlisted.
    if (token_count <= 1) return false;

    return has_long_token;
}

class SemanticMatcherImpl final : public SemanticMatcher {
public:
    SemanticMatcherImpl(EmbeddingIndex idx, const MiniLmEmbedder* emb, SemanticMatcherConfig cfg)
        : m_idx(std::move(idx)), m_emb(emb), m_cfg(std::move(cfg)) {}

    SemanticHit best_match(const std::string& text) const override {
        if (!m_emb) return SemanticHit{};
        if (m_idx.size() == 0 || m_idx.dim() == 0) return SemanticHit{};

        const std::string q = norm_and_canon(text);
        if (q.empty()) return SemanticHit{};

        std::vector<float> qv = m_emb->embed(q);
        if (qv.empty()) return SemanticHit{};

        const size_t k = (m_cfg.topk == 0) ? 1 : m_cfg.topk;
        auto hits = m_idx.topk(qv, k);
        if (hits.empty()) return SemanticHit{};

        const auto& h = hits[0];

        SemanticHit out;
        out.similarity = h.score;

        if (h.score < m_cfg.threshold) {
            out.ok = false;
            return out;
        }

        out.ok = true;
        out.skill = h.job_id;   // we store skill string in job_id
        return out;
    }

private:
    EmbeddingIndex m_idx;
    const MiniLmEmbedder* m_emb = nullptr;
    SemanticMatcherConfig m_cfg;
};

static EmbeddingIndex build_index_from_profile(
    const std::map<std::string, double>& profile_skill_weights,
    const MiniLmEmbedder& embedder
) {
    std::vector<std::string> skills;
    skills.reserve(profile_skill_weights.size());

    for (const auto& kv : profile_skill_weights) {
        const std::string s = norm_and_canon(kv.first);
        if (s.empty()) continue;

        // Only include high-quality targets to prevent junk matches like "engineers"
        if (!looks_like_real_skill_target(s)) continue;

        skills.push_back(s);
    }

    // Deduplicate deterministically
    std::sort(skills.begin(), skills.end());
    skills.erase(std::unique(skills.begin(), skills.end()), skills.end());

    std::vector<float> packed;
    size_t dim = 0;

    for (const auto& s : skills) {
        std::vector<float> v = embedder.embed(s);
        if (v.empty()) continue;

        if (dim == 0) dim = v.size();
        if (v.size() != dim) {
            throw std::runtime_error("SemanticMatcher: inconsistent embedding dim");
        }

        packed.insert(packed.end(), v.begin(), v.end());
    }

    EmbeddingIndex idx;
    if (dim == 0 || skills.empty()) {
        return idx;
    }

    idx.set(std::move(skills), std::move(packed), dim);
    return idx;
}

std::unique_ptr<SemanticMatcher> build_profile_semantic_matcher(
    const std::map<std::string, double>& profile_skill_weights,
    const MiniLmEmbedder& embedder,
    const SemanticMatcherConfig& cfg
) {
    // Try loading cached index if requested and file exists
    if (!cfg.cache_path.empty()) {
        EmbeddingIndex cached;
        if (cached.load(cfg.cache_path)) {
            return std::make_unique<SemanticMatcherImpl>(std::move(cached), &embedder, cfg);
        }
    }

    EmbeddingIndex idx = build_index_from_profile(profile_skill_weights, embedder);

    // Save cache if requested
    if (!cfg.cache_path.empty()) {
        fs::create_directories(fs::path(cfg.cache_path).parent_path());
        (void)idx.save(cfg.cache_path);
    }

    return std::make_unique<SemanticMatcherImpl>(std::move(idx), &embedder, cfg);
}

}  // namespace resume
