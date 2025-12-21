#include "resume/Validator.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace resume {

static nlohmann::json read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open JSON file: " + path.string());
    nlohmann::json j;
    in >> j;
    return j;
}

static void add_error(ValidationReport& rep, const std::string& code, const std::string& msg, const std::string& bullet_id = "") {
    rep.pass = false;
    ValidationError e;
    e.code = code;
    e.message = msg;
    e.bullet_id = bullet_id;
    rep.errors.push_back(std::move(e));
}

static std::unordered_set<std::string> collect_resume_bullet_ids(const nlohmann::json& resume_j) {
    std::unordered_set<std::string> ids;

    auto collect_from_parent_array = [&](const char* key) {
        if (!resume_j.contains(key) || !resume_j[key].is_array()) return;
        for (const auto& parent : resume_j[key]) {
            if (!parent.is_object()) continue;
            if (!parent.contains("bullets") || !parent["bullets"].is_array()) continue;
            for (const auto& b : parent["bullets"]) {
                if (!b.is_object()) continue;
                const std::string id = b.value("id", "");
                if (!id.empty()) ids.insert(id);
            }
        }
    };

    collect_from_parent_array("experiences");
    collect_from_parent_array("projects");
    return ids;
}

static int get_int_or(const nlohmann::json& j, const char* key, int def) {
    if (!j.contains(key)) return def;
    if (!j[key].is_number_integer()) return def;
    return j[key].get<int>();
}

ValidationReport validate_run(const ValidationInputs& in) {
    ValidationReport rep;

    const fs::path outdir_p(in.outdir);
    const fs::path profile_path = outdir_p / "profile.json";
    const fs::path scores_path  = outdir_p / "bullet_scores.json";
    const fs::path resume_md    = outdir_p / "resume.md";

    if (!fs::exists(in.resume_path)) add_error(rep, "missing_file", "resume file does not exist: " + in.resume_path);
    if (!fs::exists(in.explainability_path)) add_error(rep, "missing_file", "explainability file does not exist: " + in.explainability_path);

    if (!fs::exists(profile_path)) add_error(rep, "missing_file", "profile.json missing in outdir: " + profile_path.string());
    if (!fs::exists(scores_path))  add_error(rep, "missing_file", "bullet_scores.json missing in outdir: " + scores_path.string());
    if (!fs::exists(resume_md))    add_error(rep, "missing_file", "resume.md missing in outdir: " + resume_md.string());

    if (!rep.pass) return rep;

    nlohmann::json resume_j;
    nlohmann::json explain_j;
    try {
        resume_j = read_json_file(fs::path(in.resume_path));
        explain_j = read_json_file(fs::path(in.explainability_path));
    } catch (const std::exception& e) {
        add_error(rep, "json_parse_error", e.what());
        return rep;
    }

    const auto resume_ids = collect_resume_bullet_ids(resume_j);

    if (!explain_j.contains("selected_bullets") || !explain_j["selected_bullets"].is_array()) {
        add_error(rep, "bad_explainability", "missing or invalid selected_bullets array");
        return rep;
    }

    if (!explain_j.contains("selector_config") || !explain_j["selector_config"].is_object()) {
        add_error(rep, "bad_explainability", "missing or invalid selector_config object");
        return rep;
    }

    const nlohmann::json& cfg = explain_j["selector_config"];
    const int max_total = get_int_or(cfg, "max_total_bullets", 10);
    const int max_per_parent = get_int_or(cfg, "max_bullets_per_parent", 3);
    const int max_exp = get_int_or(cfg, "max_experience_bullets", 6);
    const int max_proj = get_int_or(cfg, "max_project_bullets", 4);
    const int min_unique_parents = get_int_or(cfg, "min_unique_parents", 2);

    const auto& selected = explain_j["selected_bullets"];

    if ((int)selected.size() > max_total) {
        add_error(rep, "constraint_violation", "selected_bullets exceeds max_total_bullets");
    }

    std::unordered_set<std::string> seen;
    std::unordered_map<std::string, int> count_by_parent;
    int exp_count = 0;
    int proj_count = 0;

    std::unordered_set<std::string> unique_parents;

    for (const auto& sb : selected) {
        if (!sb.is_object()) continue;

        const std::string bullet_id = sb.value("bullet_id", "");
        const std::string section = sb.value("section", "");
        const std::string parent_id = sb.value("parent_id", "");

        if (bullet_id.empty()) {
            add_error(rep, "bad_explainability", "selected_bullets contains item with empty bullet_id");
            continue;
        }

        if (seen.find(bullet_id) != seen.end()) {
            add_error(rep, "duplicate_bullet", "duplicate bullet_id selected", bullet_id);
        } else {
            seen.insert(bullet_id);
        }

        if (resume_ids.find(bullet_id) == resume_ids.end()) {
            add_error(rep, "unknown_bullet", "selected bullet_id not found in resume input", bullet_id);
        }

        if (!parent_id.empty()) {
            count_by_parent[parent_id] += 1;
            unique_parents.insert(parent_id);
            if (count_by_parent[parent_id] > max_per_parent) {
                add_error(rep, "constraint_violation", "max_bullets_per_parent exceeded for parent_id=" + parent_id, bullet_id);
            }
        }

        if (section == "Experience") exp_count += 1;
        if (section == "Project") proj_count += 1;
    }

    if (exp_count > max_exp) add_error(rep, "constraint_violation", "max_experience_bullets exceeded");
    if (proj_count > max_proj) add_error(rep, "constraint_violation", "max_project_bullets exceeded");

    if ((int)unique_parents.size() < min_unique_parents) {
        add_error(rep, "constraint_violation", "min_unique_parents not satisfied");
    }

    return rep;
}

void write_validation_report(const fs::path& path, const ValidationReport& rep) {
    try {
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
    } catch (...) {
    }

    nlohmann::json j;
    j["pass"] = rep.pass;
    j["errors"] = nlohmann::json::array();

    for (const auto& e : rep.errors) {
        nlohmann::json ej;
        ej["code"] = e.code;
        ej["message"] = e.message;
        if (!e.bullet_id.empty()) ej["bullet_id"] = e.bullet_id;
        j["errors"].push_back(ej);
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) return;
    out << j.dump(2) << "\n";
}

}
