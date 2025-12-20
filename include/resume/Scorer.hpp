// include/resume/Scorer.hpp
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

struct ScoreConfig {
    double core_bonus = 0.15;
};

struct MatchedSkill {
    std::string skill;
    double weight = 0.0;
};

struct BulletScoreBreakdown {
    double raw_skill_sum = 0.0;
    int tag_count = 0;              // original tag count on bullet (after canonicalization, before de-dupe)
    double normalized_skill = 0.0;
    double core_bonus = 0.0;
    double total = 0.0;
};

struct ScoredBullet {
    std::string bullet_id;
    std::string section;            // "Experience" or "Project"
    std::string parent_id;          // experience.id or project.id
    std::string parent_title;       // experience.title or project.name
    std::string text;

    std::vector<std::string> tags;  // canonicalized tags from bullet.tags
    std::vector<MatchedSkill> matched_skills;
    std::vector<std::string> core_hits;

    BulletScoreBreakdown score;
};

std::vector<ScoredBullet> score_bullets(
    const AbstractResume& resume,
    const RoleProfileLite& profile,
    const ScoreConfig& cfg = {}
);

}  // namespace resume
