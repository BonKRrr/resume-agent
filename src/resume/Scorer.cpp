#include "resume/Scorer.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <unordered_map>

#include "resume/SemanticMatcher.hpp"

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

static std::string normalize_tag(const std::string& s) {
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
    return canonicalize_skill(normalize_tag(s));
}

static std::unordered_set<std::string> build_core_set(const RoleProfileLite& profile) {
    std::unordered_set<std::string> out;
    out.reserve(profile.core_skills.size() * 2 + 8);
    for (const auto& s : profile.core_skills) out.insert(norm_and_canon(s));
    return out;
}

static double safe_norm(int tag_count) {
    return std::sqrt(1.0 + static_cast<double>(tag_count));
}

static double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

// Scale semantic similarity into [0,1] so borderline matches contribute less.
// Exact matches effectively have scale=1.
static double semantic_scale(double sim, double thr) {
    if (sim <= thr) return 0.0;
    double x = (sim - thr) / (1.0 - thr);
    return clamp01(x);
}

static void finalize_and_push(
    std::vector<ScoredBullet>& out,
    const Bullet& b,
    const std::string& section,
    const std::string& parent_id,
    const std::string& parent_title,
    const RoleProfileLite& profile,
    const std::unordered_set<std::string>& core,
    const ScoreConfig& cfg,
    const SemanticMatcher* semantic
) {
    ScoredBullet sb;
    sb.bullet_id = b.id;
    sb.section = section;
    sb.parent_id = parent_id;
    sb.parent_title = parent_title;
    sb.text = b.text;

    sb.tags.reserve(b.tags.size());
    for (const auto& t : b.tags) {
        sb.tags.push_back(norm_and_canon(t));
    }

    sb.score.tag_count = static_cast<int>(sb.tags.size());

    // De-dupe on "credited profile skill" so multiple tags mapping to same skill don't double count.
    std::unordered_set<std::string> credited_skills;
    credited_skills.reserve(sb.tags.size() * 2 + 8);

    double raw = 0.0;
    bool has_core = false;

    for (const auto& tag : sb.tags) {
        if (tag.empty()) continue;

        // 1) Exact match
        auto wit = profile.skill_weights.find(tag);
        if (wit != profile.skill_weights.end()) {
            const std::string matched_skill = tag;

            if (!credited_skills.insert(matched_skill).second) continue;

            const double profile_w = wit->second;
            const double contrib = profile_w;

            raw += contrib;
            sb.matched_skills.push_back(MatchedSkill{matched_skill, contrib});

            MatchEvidence ev;
            ev.type = MatchType::Exact;
            ev.source = tag;
            ev.matched_skill = matched_skill;
            ev.similarity = 1.0;
            ev.profile_weight = profile_w;
            ev.contribution = contrib;
            sb.match_evidence.push_back(std::move(ev));

            if (core.find(matched_skill) != core.end()) {
                has_core = true;
                sb.core_hits.push_back(matched_skill);
            }
            continue;
        }

        // 2) Semantic match (embedding fallback)
        if (cfg.semantic_enabled && semantic) {
            SemanticHit hit = semantic->best_match(tag);
            if (!hit.ok) continue;

            const std::string matched_skill = hit.skill;
            if (matched_skill.empty()) continue;

            auto pwit = profile.skill_weights.find(matched_skill);
            if (pwit == profile.skill_weights.end()) continue;

            if (!credited_skills.insert(matched_skill).second) continue;

            const double profile_w = pwit->second;
            const double scale = semantic_scale(hit.similarity, cfg.semantic_threshold);
            const double contrib = profile_w * scale;

            // If scale is extremely small, skip entirely to reduce noise.
            if (contrib <= 0.0) continue;

            raw += contrib;
            sb.matched_skills.push_back(MatchedSkill{matched_skill, contrib});

            MatchEvidence ev;
            ev.type = MatchType::Semantic;
            ev.source = tag;
            ev.matched_skill = matched_skill;
            ev.similarity = hit.similarity;
            ev.profile_weight = profile_w;
            ev.contribution = contrib;
            sb.match_evidence.push_back(std::move(ev));

            if (core.find(matched_skill) != core.end()) {
                has_core = true;
                sb.core_hits.push_back(matched_skill);
            }
        }
    }

    std::sort(sb.matched_skills.begin(), sb.matched_skills.end(),
              [](const MatchedSkill& a, const MatchedSkill& b) {
                  if (a.weight != b.weight) return a.weight > b.weight;
                  return a.skill < b.skill;
              });

    std::sort(sb.core_hits.begin(), sb.core_hits.end());
    sb.core_hits.erase(std::unique(sb.core_hits.begin(), sb.core_hits.end()), sb.core_hits.end());

    // Sort evidence by contribution desc for readability
    std::sort(sb.match_evidence.begin(), sb.match_evidence.end(),
              [](const MatchEvidence& a, const MatchEvidence& b) {
                  if (a.contribution != b.contribution) return a.contribution > b.contribution;
                  if (a.matched_skill != b.matched_skill) return a.matched_skill < b.matched_skill;
                  return a.source < b.source;
              });

    sb.score.raw_skill_sum = raw;
    sb.score.normalized_skill = raw / safe_norm(sb.score.tag_count);
    sb.score.core_bonus = has_core ? cfg.core_bonus : 0.0;
    sb.score.total = sb.score.normalized_skill + sb.score.core_bonus;

    out.push_back(std::move(sb));
}

std::vector<ScoredBullet> score_bullets(
    const AbstractResume& resume,
    const RoleProfileLite& profile,
    const ScoreConfig& cfg,
    const SemanticMatcher* semantic
) {
    const auto core = build_core_set(profile);

    std::vector<ScoredBullet> scored;
    size_t approx = 0;
    for (const auto& e : resume.experiences) approx += e.bullets.size();
    for (const auto& p : resume.projects) approx += p.bullets.size();
    scored.reserve(approx);

    for (const auto& e : resume.experiences) {
        for (const auto& b : e.bullets) {
            finalize_and_push(scored, b, "Experience", e.id, e.title, profile, core, cfg, semantic);
        }
    }

    for (const auto& p : resume.projects) {
        for (const auto& b : p.bullets) {
            finalize_and_push(scored, b, "Project", p.id, p.name, profile, core, cfg, semantic);
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
