#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "resume/Scorer.hpp"
#include "resume/Selector.hpp"

namespace resume {

struct ExplainabilityArtifact {
    std::string role;
    std::string resume_path;
    std::string profile_path;

    ScoreConfig score_cfg;
    SelectorConfig selector_cfg;

    std::vector<ScoredBullet> selected;
    std::vector<SelectionDecision> decisions;

    nlohmann::json to_json() const;
    void write_to(const std::filesystem::path& out_path) const;
};

}  // namespace resume
