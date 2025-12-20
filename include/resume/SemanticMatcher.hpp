#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "emb/MiniLmEmbedder.hpp"
#include "jobs/EmbeddingIndex.hpp"

namespace resume {

struct SemanticHit {
    bool ok = false;
    std::string skill;     // matched profile skill (normalized key)
    float similarity = 0;  // cosine similarity (embedding vectors are L2-normalized)
};

struct SemanticMatcherConfig {
    float threshold = 0.66f; // accept match if similarity >= threshold
    size_t topk = 1;         // query topk; we use best hit (index 0)
    std::string cache_path;  // optional: load/save profile skill index
};

class SemanticMatcher {
public:
    virtual ~SemanticMatcher() = default;
    virtual SemanticHit best_match(const std::string& text) const = 0;
};

std::unique_ptr<SemanticMatcher> build_profile_semantic_matcher(
    const std::map<std::string, double>& profile_skill_weights,
    const MiniLmEmbedder& embedder,
    const SemanticMatcherConfig& cfg
);

}  // namespace resume
