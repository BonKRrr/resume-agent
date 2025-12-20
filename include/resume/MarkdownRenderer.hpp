#pragma once

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "resume/ConcreteResume.hpp"
#include "resume/Models.hpp"
#include "resume/Scorer.hpp"

namespace resume {

ConcreteResume build_concrete_resume(const AbstractResume& resume, const std::vector<ScoredBullet>& selected);

std::string render_markdown(const ConcreteResume& cr);

void write_markdown(const std::filesystem::path& out_path, const std::string& md);

}  // namespace resume
