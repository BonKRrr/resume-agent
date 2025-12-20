// src/commands/build.cpp
#include "commands/build.hpp"

#include "nlohmann/json.hpp"
#include "resume/BulletScoresArtifact.hpp"
#include "resume/Scorer.hpp"
#include "resume/Models.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static bool has_flag(int argc, char** argv, const std::string& key) {
    for (int i = 0; i < argc; ++i) {
        if (std::string(argv[i]) == key) return true;
    }
    return false;
}

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
    }
    return def;
}

static nlohmann::json read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open JSON file: " + path.string());
    nlohmann::json j;
    in >> j;
    return j;
}

// Manual parsing: matches exactly your structs (no ADL / no hidden converters).
static Bullet parse_bullet(const nlohmann::json& j) {
    Bullet b;
    b.id = j.value("id", "");
    b.text = j.value("text", "");
    if (j.contains("tags") && j["tags"].is_array()) {
        for (const auto& t : j["tags"]) {
            if (t.is_string()) b.tags.push_back(t.get<std::string>());
        }
    }
    return b;
}

static Experience parse_experience(const nlohmann::json& j) {
    Experience e;
    e.id = j.value("id", "");
    e.title = j.value("title", "");
    e.organization = j.value("organization", "");
    e.dates = j.value("dates", "");
    if (j.contains("bullets") && j["bullets"].is_array()) {
        for (const auto& bj : j["bullets"]) {
            e.bullets.push_back(parse_bullet(bj));
        }
    }
    return e;
}

static Project parse_project(const nlohmann::json& j) {
    Project p;
    p.id = j.value("id", "");
    p.name = j.value("name", "");
    p.context = j.value("context", "");
    if (j.contains("bullets") && j["bullets"].is_array()) {
        for (const auto& bj : j["bullets"]) {
            p.bullets.push_back(parse_bullet(bj));
        }
    }
    return p;
}

static AbstractResume parse_resume(const nlohmann::json& j) {
    AbstractResume r;

    if (j.contains("experiences") && j["experiences"].is_array()) {
        for (const auto& ej : j["experiences"]) {
            r.experiences.push_back(parse_experience(ej));
        }
    }

    if (j.contains("projects") && j["projects"].is_array()) {
        for (const auto& pj : j["projects"]) {
            r.projects.push_back(parse_project(pj));
        }
    }

    return r;
}

static resume::RoleProfileLite parse_profile(const nlohmann::json& j) {
    resume::RoleProfileLite p;
    p.role = j.value("role", "");

    if (j.contains("core_skills") && j["core_skills"].is_array()) {
        for (const auto& s : j["core_skills"]) {
            if (s.is_string()) p.core_skills.push_back(s.get<std::string>());
        }
    }

    if (j.contains("skill_weights") && j["skill_weights"].is_object()) {
        for (auto it = j["skill_weights"].begin(); it != j["skill_weights"].end(); ++it) {
            const std::string key = it.key();
            if (it.value().is_number()) {
                p.skill_weights[key] = it.value().get<double>();
            }
        }
    }

    return p;
}

int cmd_build(int argc, char** argv) {
    try {
        const std::string role_arg = get_arg(argc, argv, "--role", "");
        const fs::path resume_path = get_arg(argc, argv, "--resume", "data/abstract_resume.json");
        const fs::path profile_path = get_arg(argc, argv, "--profile", "out/profile.json");
        const fs::path outdir = get_arg(argc, argv, "--outdir", "out");

        // present for compatibility, not used yet
        (void)has_flag(argc, argv, "--scores_only");

        const nlohmann::json resume_j = read_json_file(resume_path);
        const nlohmann::json profile_j = read_json_file(profile_path);

        AbstractResume resume = parse_resume(resume_j);
        resume::RoleProfileLite profile = parse_profile(profile_j);

        const std::string effective_role = !role_arg.empty() ? role_arg : profile.role;

        const auto scored = resume::score_bullets(resume, profile);

        int bullet_count = 0;
        for (const auto& e : resume.experiences) bullet_count += static_cast<int>(e.bullets.size());
        for (const auto& p : resume.projects) bullet_count += static_cast<int>(p.bullets.size());

        resume::BulletScoresArtifact artifact;
        artifact.role = effective_role;
        artifact.num_bullets = bullet_count;
        artifact.resume_path = resume_path.string();
        artifact.profile_path = profile_path.string();
        artifact.bullets = scored;

        const fs::path out_path = outdir / "bullet_scores.json";
        artifact.write_to(out_path);

        std::cout << "ROLE: " << effective_role << "\n";
        std::cout << "RESUME: " << resume_path.string() << "\n";
        std::cout << "PROFILE: " << profile_path.string() << "\n";
        std::cout << "OUT: " << out_path.string() << "\n";
        std::cout << "BULLETS: " << artifact.num_bullets << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "build failed: " << e.what() << "\n";
        return 1;
    }
}
