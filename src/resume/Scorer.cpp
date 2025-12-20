// src/resume/Scorer.cpp
#include "resume/Scorer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace resume {

static std::string canonicalize_skill(const std::string& s) {
    // Keep this small + explicit. Add more aliases as you discover them.
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

static std::unordered_set<std::string> build_core_set(const RoleProfileLite& profile) {
    std::unordered_set<std::string> out;
    out.reserve(profile.core_skills.size() * 2 + 8);
    for (const auto& s : profile.core_skills) out.insert(canonicalize_skill(s));
    return out;
}

static double safe_norm(int tag_count) {
    return std::sqrt(1.0 + static_cast<double>(tag_count));
}

static void finalize_and_push(
    std::vector<ScoredBullet>& out,
    const Bullet& b,
    const std::string& section,
    const std::string& parent_id,
    const std::string& parent_title,
    const RoleProfileLite& profile,
    const std::unordered_set<std::string>& core,
    const ScoreConfig& cfg
) {
    ScoredBullet sb;
    sb.bullet_id = b.id;
    sb.section = section;
    sb.parent_id = parent_id;
    sb.parent_title = parent_title;
    sb.text = b.text;

    sb.tags.reserve(b.tags.size());
    for (const auto& t : b.tags) sb.tags.push_back(canonicalize_skill(t));

    sb.score.tag_count = static_cast<int>(sb.tags.size());

    std::unordered_set<std::string> seen;
    seen.reserve(sb.tags.size() * 2 + 4);

    double raw = 0.0;
    bool has_core = false;

    for (const auto& tag : sb.tags) {
        if (!seen.insert(tag).second) continue; // de-dupe for scoring

        auto wit = profile.skill_weights.find(tag);
        if (wit == profile.skill_weights.end()) continue;

        raw += wit->second;
        sb.matched_skills.push_back(MatchedSkill{tag, wit->second});

        if (core.find(tag) != core.end()) {
            has_core = true;
            sb.core_hits.push_back(tag);
        }
    }

    std::sort(sb.matched_skills.begin(), sb.matched_skills.end(),
              [](const MatchedSkill& a, const MatchedSkill& b) {
                  if (a.weight != b.weight) return a.weight > b.weight;
                  return a.skill < b.skill;
              });

    std::sort(sb.core_hits.begin(), sb.core_hits.end());
    sb.core_hits.erase(std::unique(sb.core_hits.begin(), sb.core_hits.end()), sb.core_hits.end());

    sb.score.raw_skill_sum = raw;
    sb.score.normalized_skill = raw / safe_norm(sb.score.tag_count);
    sb.score.core_bonus = has_core ? cfg.core_bonus : 0.0;
    sb.score.total = sb.score.normalized_skill + sb.score.core_bonus;

    out.push_back(std::move(sb));
}

std::vector<ScoredBullet> score_bullets(
    const AbstractResume& resume,
    const RoleProfileLite& profile,
    const ScoreConfig& cfg
) {
    const auto core = build_core_set(profile);

    std::vector<ScoredBullet> scored;
    // Reserve roughly: sum of all bullets in experiences + projects
    size_t approx = 0;
    for (const auto& e : resume.experiences) approx += e.bullets.size();
    for (const auto& p : resume.projects) approx += p.bullets.size();
    scored.reserve(approx);

    for (const auto& e : resume.experiences) {
        for (const auto& b : e.bullets) {
            finalize_and_push(scored, b, "Experience", e.id, e.title, profile, core, cfg);
        }
    }

    for (const auto& p : resume.projects) {
        for (const auto& b : p.bullets) {
            finalize_and_push(scored, b, "Project", p.id, p.name, profile, core, cfg);
        }
    }

    std::sort(scored.begin(), scored.end(),
              [](const ScoredBullet& a, const ScoredBullet& b) {
                  if (a.score.total != b.score.total) return a.score.total > b.score.total;
                  if (a.score.raw_skill_sum != b.score.raw_skill_sum) return a.score.raw_skill_sum > b.score.raw_skill_sum;
                  if (a.core_hits.size() != b.core_hits.size()) return a.core_hits.size() > b.core_hits.size();
                  if (a.section != b.section) return a.section < b.section;
                  return a.bullet_id < b.bullet_id;
              });

    return scored;
}

}  // namespace resume
