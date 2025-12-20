#include "resume/ExplainabilityArtifact.hpp"

#include <fstream>
#include <stdexcept>

namespace resume {

static const char* match_type_str(MatchType t) {
    switch (t) {
        case MatchType::Exact: return "exact";
        case MatchType::Semantic: return "semantic";
        default: return "unknown";
    }
}

static nlohmann::json scored_bullet_to_json(const ScoredBullet& b) {
    nlohmann::json j;

    j["bullet_id"] = b.bullet_id;
    j["section"] = b.section;
    j["parent_id"] = b.parent_id;
    j["parent_title"] = b.parent_title;
    j["text"] = b.text;

    j["tags"] = b.tags;

    nlohmann::json matched = nlohmann::json::array();
    for (const auto& ms : b.matched_skills) {
        matched.push_back({{"skill", ms.skill}, {"weight", ms.weight}});
    }
    j["matched_skills"] = matched;

    j["core_hits"] = b.core_hits;

    nlohmann::json evidence = nlohmann::json::array();
    for (const auto& ev : b.match_evidence) {
        evidence.push_back({
            {"type", match_type_str(ev.type)},
            {"source", ev.source},
            {"matched_skill", ev.matched_skill},
            {"similarity", ev.similarity},
            {"profile_weight", ev.profile_weight},
            {"contribution", ev.contribution}
        });
    }
    j["match_evidence"] = evidence;

    j["score"] = {
        {"raw_skill_sum", b.score.raw_skill_sum},
        {"tag_count", b.score.tag_count},
        {"normalized_skill", b.score.normalized_skill},
        {"core_bonus", b.score.core_bonus},
        {"total", b.score.total},
    };

    return j;
}

nlohmann::json ExplainabilityArtifact::to_json() const {
    nlohmann::json j;

    j["role"] = role;
    j["resume_path"] = resume_path;
    j["profile_path"] = profile_path;

    j["score_config"] = {
        {"core_bonus", score_cfg.core_bonus},
        {"semantic_enabled", score_cfg.semantic_enabled},
        {"semantic_threshold", score_cfg.semantic_threshold}
    };

    j["selector_config"] = {
        {"max_total_bullets", selector_cfg.max_total_bullets},
        {"max_bullets_per_parent", selector_cfg.max_bullets_per_parent},
        {"max_experience_bullets", selector_cfg.max_experience_bullets},
        {"max_project_bullets", selector_cfg.max_project_bullets},
        {"min_unique_parents", selector_cfg.min_unique_parents}
    };

    nlohmann::json sel = nlohmann::json::array();
    for (const auto& b : selected) sel.push_back(scored_bullet_to_json(b));
    j["selected_bullets"] = sel;

    nlohmann::json dec = nlohmann::json::array();
    for (const auto& d : decisions) {
        dec.push_back({
            {"bullet_id", d.bullet_id},
            {"accepted", d.accepted},
            {"reason", d.reason}
        });
    }
    j["selection_decisions"] = dec;

    return j;
}

void ExplainabilityArtifact::write_to(const std::filesystem::path& out_path) const {
    std::filesystem::create_directories(out_path.parent_path());

    std::ofstream out(out_path);
    if (!out) throw std::runtime_error("Failed to open output file: " + out_path.string());

    out << to_json().dump(2) << "\n";
}

}  // namespace resume
