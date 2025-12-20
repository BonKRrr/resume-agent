#include "resume/Selector.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace resume {

static bool is_experience(const ScoredBullet& b) { return b.section == "Experience"; }
static bool is_project(const ScoredBullet& b) { return b.section == "Project"; }

static std::string parent_key(const ScoredBullet& b) {
    return b.section + "::" + b.parent_id;
}

static size_t unique_parent_count(const std::vector<ScoredBullet>& v) {
    std::unordered_set<std::string> s;
    s.reserve(v.size() * 2 + 8);
    for (const auto& b : v) s.insert(parent_key(b));
    return s.size();
}

static int count_section(const std::vector<ScoredBullet>& v, const std::string& section) {
    int c = 0;
    for (const auto& b : v) if (b.section == section) ++c;
    return c;
}

static int count_parent(const std::vector<ScoredBullet>& v, const std::string& pkey) {
    int c = 0;
    for (const auto& b : v) if (parent_key(b) == pkey) ++c;
    return c;
}

static int find_lowest_replaceable_index(
    const std::vector<ScoredBullet>& selected,
    const std::string& section,
    const std::string& new_parent,
    const SelectorConfig& cfg
) {
    // Replace the lowest-scoring bullet from a parent that currently has >1 bullet
    // (to increase diversity), and only within the same section to preserve section caps.
    int best_i = -1;
    double best_score = 1e100;

    std::unordered_map<std::string, int> parent_counts;
    parent_counts.reserve(selected.size() * 2 + 8);
    for (const auto& b : selected) parent_counts[parent_key(b)]++;

    for (int i = 0; i < (int)selected.size(); ++i) {
        const auto& b = selected[i];
        if (b.section != section) continue;

        const std::string pk = parent_key(b);
        if (pk == new_parent) continue;

        auto it = parent_counts.find(pk);
        if (it == parent_counts.end()) continue;
        if (it->second <= 1) continue;

        const double s = b.score.total;
        if (s < best_score) {
            best_score = s;
            best_i = i;
        }
    }

    (void)cfg;
    return best_i;
}

SelectorResult select_bullets(const std::vector<ScoredBullet>& scored, const SelectorConfig& cfg) {
    SelectorResult res;
    res.cfg = cfg;
    res.decisions.reserve(scored.size());

    std::vector<ScoredBullet> selected;
    selected.reserve(std::min((int)scored.size(), cfg.max_total_bullets));

    std::unordered_map<std::string, int> parent_counts;
    parent_counts.reserve(64);

    int exp_count = 0;
    int proj_count = 0;

    auto can_take = [&](const ScoredBullet& b) -> std::pair<bool, std::string> {
        if ((int)selected.size() >= cfg.max_total_bullets) return {false, "total_cap"};

        const std::string pk = parent_key(b);
        auto it = parent_counts.find(pk);
        const int pc = (it == parent_counts.end()) ? 0 : it->second;
        if (pc >= cfg.max_bullets_per_parent) return {false, "parent_cap"};

        if (is_experience(b)) {
            if (exp_count >= cfg.max_experience_bullets) return {false, "experience_cap"};
        } else if (is_project(b)) {
            if (proj_count >= cfg.max_project_bullets) return {false, "project_cap"};
        }
        return {true, ""};
    };

    // Greedy selection pass
    for (const auto& b : scored) {
        auto ok = can_take(b);
        if (!ok.first) {
            res.decisions.push_back(SelectionDecision{b.bullet_id, false, ok.second});
            continue;
        }

        selected.push_back(b);
        parent_counts[parent_key(b)]++;
        if (is_experience(b)) ++exp_count;
        else if (is_project(b)) ++proj_count;

        res.decisions.push_back(SelectionDecision{b.bullet_id, true, "selected"});
        if ((int)selected.size() >= cfg.max_total_bullets) break;
    }

    // Diversity fix-up: try to reach min_unique_parents by swapping in candidates
    // that introduce a new parent (deterministic order: scored list order).
    if (cfg.min_unique_parents > 0) {
        size_t up = unique_parent_count(selected);
        if ((int)up < cfg.min_unique_parents && !selected.empty()) {
            std::unordered_set<std::string> sel_parents;
            sel_parents.reserve(selected.size() * 2 + 8);
            for (const auto& b : selected) sel_parents.insert(parent_key(b));

            for (const auto& cand : scored) {
                if ((int)unique_parent_count(selected) >= cfg.min_unique_parents) break;

                const std::string cpk = parent_key(cand);
                if (sel_parents.find(cpk) != sel_parents.end()) continue;

                const std::string sec = cand.section;

                // Can we add without breaking caps? If total not full, just add.
                if ((int)selected.size() < cfg.max_total_bullets) {
                    auto ok = can_take(cand);
                    if (!ok.first) continue;

                    selected.push_back(cand);
                    sel_parents.insert(cpk);

                    parent_counts[parent_key(cand)]++;
                    if (is_experience(cand)) ++exp_count;
                    else if (is_project(cand)) ++proj_count;

                    continue;
                }

                // Otherwise swap with a replaceable low-score bullet in same section
                int replace_i = find_lowest_replaceable_index(selected, sec, cpk, cfg);
                if (replace_i < 0) continue;

                const ScoredBullet old = selected[replace_i];

                // Check parent cap for cand (with old removed)
                {
                    const std::string oldpk = parent_key(old);
                    parent_counts[oldpk]--;
                    if (parent_counts[oldpk] <= 0) parent_counts.erase(oldpk);

                    auto ok = can_take(cand);
                    if (!ok.first) {
                        // revert
                        parent_counts[oldpk]++;
                        continue;
                    }

                    // accept swap
                    selected[replace_i] = cand;
                    parent_counts[parent_key(cand)]++;
                    sel_parents.erase(oldpk);
                    sel_parents.insert(cpk);
                }
            }
        }
    }

    // Final deterministic ordering: keep selection in descending score order.
    std::sort(selected.begin(), selected.end(),
              [](const ScoredBullet& a, const ScoredBullet& b) {
                  if (a.score.total != b.score.total) return a.score.total > b.score.total;
                  if (a.score.raw_skill_sum != b.score.raw_skill_sum) return a.score.raw_skill_sum > b.score.raw_skill_sum;
                  if (a.core_hits.size() != b.core_hits.size()) return a.core_hits.size() > b.core_hits.size();
                  if (a.section != b.section) return a.section < b.section;
                  return a.bullet_id < b.bullet_id;
              });

    res.selected = std::move(selected);
    return res;
}

}  // namespace resume
