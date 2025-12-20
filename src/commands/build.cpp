#include "commands/build.hpp"

#include "nlohmann/json.hpp"
#include "resume/BulletScoresArtifact.hpp"
#include "resume/Scorer.hpp"
#include "resume/SemanticMatcher.hpp"
#include "resume/Selector.hpp"
#include "resume/MarkdownRenderer.hpp"
#include "resume/ExplainabilityArtifact.hpp"
#include "resume/Models.hpp"

#include "emb/MiniLmEmbedder.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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

static int get_arg_int(int argc, char** argv, const std::string& key, int def) {
    const std::string s = get_arg(argc, argv, key, "");
    if (s.empty()) return def;
    try { return std::stoi(s); } catch (...) { return def; }
}

static double get_arg_double(int argc, char** argv, const std::string& key, double def) {
    const std::string s = get_arg(argc, argv, key, "");
    if (s.empty()) return def;
    try { return std::stod(s); } catch (...) { return def; }
}

static nlohmann::json read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Failed to open JSON file: " + path.string());
    nlohmann::json j;
    in >> j;
    return j;
}

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

static std::string normalize_key(const std::string& s) {
    return to_lower_copy(trim_copy(s));
}

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
        for (const auto& bj : j["bullets"]) e.bullets.push_back(parse_bullet(bj));
    }
    return e;
}

static Project parse_project(const nlohmann::json& j) {
    Project p;
    p.id = j.value("id", "");
    p.name = j.value("name", "");
    p.context = j.value("context", "");
    if (j.contains("bullets") && j["bullets"].is_array()) {
        for (const auto& bj : j["bullets"]) p.bullets.push_back(parse_bullet(bj));
    }
    return p;
}

static AbstractResume parse_resume(const nlohmann::json& j) {
    AbstractResume r;

    if (j.contains("experiences") && j["experiences"].is_array()) {
        for (const auto& ej : j["experiences"]) r.experiences.push_back(parse_experience(ej));
    }

    if (j.contains("projects") && j["projects"].is_array()) {
        for (const auto& pj : j["projects"]) r.projects.push_back(parse_project(pj));
    }

    return r;
}

static resume::RoleProfileLite parse_profile(const nlohmann::json& j) {
    resume::RoleProfileLite p;
    p.role = j.value("role", "");

    if (j.contains("core_skills") && j["core_skills"].is_array()) {
        for (const auto& s : j["core_skills"]) {
            if (s.is_string()) p.core_skills.push_back(normalize_key(s.get<std::string>()));
        }
    }

    if (j.contains("skill_weights") && j["skill_weights"].is_object()) {
        for (auto it = j["skill_weights"].begin(); it != j["skill_weights"].end(); ++it) {
            const std::string key = normalize_key(it.key());
            if (it.value().is_number()) p.skill_weights[key] = it.value().get<double>();
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

        const bool scores_only = has_flag(argc, argv, "--scores_only");

        const bool semantic = has_flag(argc, argv, "--semantic");
        const std::string emb_model = get_arg(argc, argv, "--emb_model", "models/emb/model.onnx");
        const std::string emb_vocab = get_arg(argc, argv, "--emb_vocab", "models/emb/vocab.txt");
        const double semantic_threshold = get_arg_double(argc, argv, "--semantic_threshold", 0.66);
        const int semantic_topk_i = get_arg_int(argc, argv, "--semantic_topk", 1);
        const std::string semantic_cache = get_arg(argc, argv, "--semantic_cache", "");

        resume::SelectorConfig sel_cfg;
        sel_cfg.max_total_bullets = get_arg_int(argc, argv, "--max_total_bullets", sel_cfg.max_total_bullets);
        sel_cfg.max_bullets_per_parent = get_arg_int(argc, argv, "--max_bullets_per_parent", sel_cfg.max_bullets_per_parent);
        sel_cfg.max_experience_bullets = get_arg_int(argc, argv, "--max_experience_bullets", sel_cfg.max_experience_bullets);
        sel_cfg.max_project_bullets = get_arg_int(argc, argv, "--max_project_bullets", sel_cfg.max_project_bullets);
        sel_cfg.min_unique_parents = get_arg_int(argc, argv, "--min_unique_parents", sel_cfg.min_unique_parents);

        const nlohmann::json resume_j = read_json_file(resume_path);
        const nlohmann::json profile_j = read_json_file(profile_path);

        AbstractResume resume = parse_resume(resume_j);
        resume::RoleProfileLite profile = parse_profile(profile_j);

        const std::string effective_role = !role_arg.empty() ? role_arg : profile.role;

        resume::ScoreConfig score_cfg;
        score_cfg.semantic_enabled = semantic;
        score_cfg.semantic_threshold = semantic_threshold;

        std::unique_ptr<resume::SemanticMatcher> matcher;
        MiniLmEmbedder embedder;

        if (semantic) {
            if (emb_model.empty() || emb_vocab.empty()) {
                throw std::runtime_error("Semantic matching enabled but missing --emb_model and/or --emb_vocab");
            }
            if (!embedder.init(emb_model, emb_vocab)) {
                throw std::runtime_error("Failed to init MiniLmEmbedder (check model/vocab paths)");
            }

            resume::SemanticMatcherConfig mcfg;
            mcfg.threshold = static_cast<float>(score_cfg.semantic_threshold);
            mcfg.topk = (semantic_topk_i <= 0) ? 1u : static_cast<size_t>(semantic_topk_i);
            mcfg.cache_path = semantic_cache;

            matcher = resume::build_profile_semantic_matcher(profile.skill_weights, embedder, mcfg);
        }

        const auto scored = resume::score_bullets(resume, profile, score_cfg, matcher.get());

        int bullet_count = 0;
        for (const auto& e : resume.experiences) bullet_count += static_cast<int>(e.bullets.size());
        for (const auto& p : resume.projects) bullet_count += static_cast<int>(p.bullets.size());

        resume::BulletScoresArtifact artifact;
        artifact.role = effective_role;
        artifact.num_bullets = bullet_count;
        artifact.resume_path = resume_path.string();
        artifact.profile_path = profile_path.string();
        artifact.bullets = scored;

        const fs::path scores_path = outdir / "bullet_scores.json";
        artifact.write_to(scores_path);

        std::cout << "ROLE: " << effective_role << "\n";
        std::cout << "RESUME: " << resume_path.string() << "\n";
        std::cout << "PROFILE: " << profile_path.string() << "\n";
        std::cout << "OUT_SCORES: " << scores_path.string() << "\n";
        std::cout << "BULLETS: " << artifact.num_bullets << "\n";
        std::cout << "SEMANTIC: " << (semantic ? "on" : "off") << "\n";

        if (semantic) {
            std::cout << "EMB_MODEL: " << emb_model << "\n";
            std::cout << "EMB_VOCAB: " << emb_vocab << "\n";
            std::cout << "SEM_THRESHOLD: " << score_cfg.semantic_threshold << "\n";
            std::cout << "SEM_TOPK: " << semantic_topk_i << "\n";
            if (!semantic_cache.empty()) std::cout << "SEM_CACHE: " << semantic_cache << "\n";
        }

        if (scores_only) {
            return 0;
        }

        const resume::SelectorResult sel = resume::select_bullets(scored, sel_cfg);

        const resume::ConcreteResume cr = resume::build_concrete_resume(resume, sel.selected);
        const std::string md = resume::render_markdown(cr);

        const fs::path resume_md_path = outdir / "resume.md";
        resume::write_markdown(resume_md_path, md);

        resume::ExplainabilityArtifact ex;
        ex.role = effective_role;
        ex.resume_path = resume_path.string();
        ex.profile_path = profile_path.string();
        ex.score_cfg = score_cfg;
        ex.selector_cfg = sel_cfg;
        ex.selected = sel.selected;
        ex.decisions = sel.decisions;

        const fs::path explain_path = outdir / "explainability.json";
        ex.write_to(explain_path);

        std::cout << "OUT_RESUME_MD: " << resume_md_path.string() << "\n";
        std::cout << "OUT_EXPLAIN: " << explain_path.string() << "\n";
        std::cout << "SELECTED: " << sel.selected.size() << "\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "build failed: " << e.what() << "\n";
        return 1;
    }
}
