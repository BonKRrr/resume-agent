// include/resume/BulletScoresArtifact.hpp
#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "resume/Scorer.hpp"

namespace resume {

struct BulletScoresArtifact {
    std::string role;
    int num_bullets = 0;

    std::string resume_path;
    std::string profile_path;

    std::vector<ScoredBullet> bullets;

    nlohmann::json to_json() const;
    void write_to(const std::filesystem::path& out_path) const;
};

}  // namespace resume
