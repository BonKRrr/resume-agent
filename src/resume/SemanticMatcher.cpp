#include "resume/SemanticMatcher.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unordered_map>
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
        out.ok = (h.score >= m_cfg.threshold);
        out.skill = h.job_id;   // we store skill string in job_id
        out.similarity = h.score;
        if (!out.ok) {
            out.skill.clear();
            out.similarity = h.score;
        }
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
        if (!s.empty()) skills.push_back(s);
    }

    // Deduplicate skills while keeping deterministic order:
    // since profile_skill_weights is a std::map, iteration is sorted by key.
    // After normalization, duplicates can appear; remove them deterministically.
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
        // empty index
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
            // NOTE: We don't validate "same profile" here; deterministic, but user-controlled cache.
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
