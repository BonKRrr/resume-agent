#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "resume/Models.hpp"

namespace resume {

struct RoleProfileLite {
    std::string role;
    std::vector<std::string> core_skills;
    std::map<std::string, double> skill_weights;
};

enum class MatchType {
    Exact,
    Semantic
};

struct MatchEvidence {
    MatchType type = MatchType::Exact;

    // What we saw on the resume side (already normalized/canonicalized)
    std::string source;

    // Which profile skill we credited (normalized key)
    std::string matched_skill;

    // For semantic matches (cosine). For exact matches this is 1.0
    double similarity = 1.0;

    // Profile weight (base) and actual contribution credited to this bullet
    double profile_weight = 0.0;
    double contribution = 0.0;
};

struct ScoreConfig {
    double core_bonus = 0.15;

    // semantic matching (embedding fallback)
    bool semantic_enabled = false;
    double semantic_threshold = 0.66;     // accept match if cosine >= threshold

    // IMPORTANT: semantic matches should help, but never dominate exact.
    // contribution = profile_weight * semantic_weight_scale * similarity
    double semantic_weight_scale = 0.25;

    // skip extremely tiny semantic contributions (noise guard)
    double semantic_min_contribution = 0.01;
};

struct MatchedSkill {
    std::string skill;     // profile skill key
    double weight = 0.0;   // contribution credited (not the raw profile weight)
};

struct BulletScoreBreakdown {
    double raw_skill_sum = 0.0;  // sum of contributions
    int tag_count = 0;           // number of tags on bullet after normalization/canonicalization (before de-dupe)
    double normalized_skill = 0.0;
    double core_bonus = 0.0;
    double total = 0.0;
};

struct ScoredBullet {
    std::string bullet_id;
    std::string section;       // "Experience" or "Project"
    std::string parent_id;     // experience.id or project.id
    std::string parent_title;  // experience.title or project.name
    std::string text;

    std::vector<std::string> tags;  // normalized + canonicalized tags from bullet.tags
    std::vector<MatchedSkill> matched_skills;
    std::vector<std::string> core_hits;

    // Explainability: one entry per credited match (exact or semantic)
    std::vector<MatchEvidence> match_evidence;

    BulletScoreBreakdown score;
};

// forward declaration to avoid including matcher header here
class SemanticMatcher;

std::vector<ScoredBullet> score_bullets(
    const AbstractResume& resume,
    const RoleProfileLite& profile,
    const ScoreConfig& cfg = {},
    const SemanticMatcher* semantic = nullptr
);

}  // namespace resume
