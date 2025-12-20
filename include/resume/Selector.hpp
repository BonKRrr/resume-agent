#pragma once

#include <string>
#include <vector>

#include "resume/Scorer.hpp"

namespace resume {

struct SelectorConfig {
    int max_total_bullets = 10;
    int max_bullets_per_parent = 3;
    int max_experience_bullets = 6;
    int max_project_bullets = 4;
    int min_unique_parents = 2;
};

struct SelectionDecision {
    std::string bullet_id;
    bool accepted = false;
    std::string reason;
};

struct SelectorResult {
    SelectorConfig cfg;
    std::vector<ScoredBullet> selected;          // selected bullets (scored objects)
    std::vector<SelectionDecision> decisions;    // one per candidate in scored list order
};

SelectorResult select_bullets(const std::vector<ScoredBullet>& scored, const SelectorConfig& cfg);

}  // namespace resume
