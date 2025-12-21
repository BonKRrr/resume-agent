#include "commands/run.hpp"

#include "commands/analyze.hpp"
#include "commands/build.hpp"
#include "resume/Validator.hpp"

#include "nlohmann/json.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int run_usage() {
    std::cerr
        << "usage:\n"
        << "  resume-agent run --role \"<job title>\" --resume <path> [--outdir <dir>]\n";
    return 1;
}

static std::vector<char*> to_argv(std::vector<std::string>& storage) {
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& s : storage) argv.push_back(s.data());
    return argv;
}

static void append_jsonl(const fs::path& path, const nlohmann::json& j) {
    try {
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::out | std::ios::app);
        if (!out) return;
        out << j.dump() << "\n";
    } catch (...) {
    }
}

static void write_json(const fs::path& path, const nlohmann::json& j) {
    try {
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out) return;
        out << j.dump(2) << "\n";
    } catch (...) {
    }
}

static nlohmann::json args_to_json_array(const std::vector<std::string>& args) {
    nlohmann::json a = nlohmann::json::array();
    for (const auto& s : args) a.push_back(s);
    return a;
}

int cmd_run(int argc, char** argv) {
    std::string role;
    std::string resume_path;
    std::string outdir = "out";

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        if (a == "--help") return run_usage();

        if (a == "--role") {
            if (i + 1 >= argc) {
                std::cerr << "error: --role requires a value\n";
                return 2;
            }
            role = argv[++i];
            continue;
        }

        if (a == "--resume") {
            if (i + 1 >= argc) {
                std::cerr << "error: --resume requires a value\n";
                return 2;
            }
            resume_path = argv[++i];
            continue;
        }

        if (a == "--outdir") {
            if (i + 1 >= argc) {
                std::cerr << "error: --outdir requires a value\n";
                return 2;
            }
            outdir = argv[++i];
            continue;
        }

        std::cerr << "error: unknown arg: " << a << "\n";
        return run_usage();
    }

    if (role.empty()) {
        std::cerr << "error: missing --role\n";
        return run_usage();
    }
    if (resume_path.empty()) {
        std::cerr << "error: missing --resume\n";
        return run_usage();
    }

    try {
        fs::create_directories(outdir);
    } catch (const std::exception& e) {
        std::cerr << "error: failed to create outdir '" << outdir << "': " << e.what() << "\n";
        return 2;
    }

    const fs::path outdir_p(outdir);

    const std::string profile_path = (outdir_p / "profile.json").string();
    const std::string llm_cache_dir = (outdir_p / "llm_cache").string();
    const std::string semantic_cache_path = (outdir_p / "profile_skill_index.bin").string();

    const fs::path explain_path = outdir_p / "explainability.json";
    const fs::path report_path  = outdir_p / "validation_report.json";
    const fs::path attempts_path = outdir_p / "run_attempts.jsonl";
    const fs::path manifest_path = outdir_p / "run_manifest.json";

    // -------------------------
    // 1) ANALYZE (always)
    // -------------------------
    std::vector<std::string> analyze_args;
    analyze_args.push_back("analyze");
    analyze_args.push_back("--role");
    analyze_args.push_back(role);
    analyze_args.push_back("--profile");
    analyze_args.push_back("--llm");
    analyze_args.push_back("--outdir");
    analyze_args.push_back(outdir);
    analyze_args.push_back("--llm_cache");
    analyze_args.push_back(llm_cache_dir);

    {
        auto cargv = to_argv(analyze_args);
        const int rc = cmd_analyze((int)cargv.size(), cargv.data());
        if (rc != 0) return rc;
    }

    if (!fs::exists(profile_path)) {
        std::cerr << "error: analyze did not produce expected profile: " << profile_path << "\n";
        return 2;
    }

    // -------------------------
    // 2) BUILD+VALIDATE (agent loop)
    // -------------------------
    // Deterministic retry schedule.
    // Attempt 1: base
    // Attempt 2: relax bullet caps slightly
    // Attempt 3: relax semantic threshold slightly
    // Attempt 4: relax per-parent slightly
    struct BuildTweak {
        int max_total_bullets = -1;        // -1 means don't pass flag
        int max_experience_bullets = -1;
        int max_project_bullets = -1;
        int max_bullets_per_parent = -1;
        double semantic_threshold = -1.0;  // -1 means don't pass flag
    };

    std::vector<BuildTweak> plan;
    {
        BuildTweak t1; // baseline: no extra flags
        plan.push_back(t1);

        BuildTweak t2;
        t2.max_total_bullets = 12;
        t2.max_experience_bullets = 7;
        t2.max_project_bullets = 5;
        plan.push_back(t2);

        BuildTweak t3 = t2;
        t3.semantic_threshold = 0.62;
        plan.push_back(t3);

        BuildTweak t4 = t3;
        t4.max_bullets_per_parent = 4;
        plan.push_back(t4);
    }

    resume::ValidationReport last_rep;
    bool success = false;

    for (size_t attempt = 0; attempt < plan.size(); ++attempt) {
        const BuildTweak& tw = plan[attempt];

        std::vector<std::string> build_args;
        build_args.push_back("build");
        build_args.push_back("--semantic");
        build_args.push_back("--resume");
        build_args.push_back(resume_path);
        build_args.push_back("--profile");
        build_args.push_back(profile_path);
        build_args.push_back("--outdir");
        build_args.push_back(outdir);
        build_args.push_back("--semantic_cache");
        build_args.push_back(semantic_cache_path);

        if (tw.semantic_threshold >= 0.0) {
            build_args.push_back("--semantic_threshold");
            build_args.push_back(std::to_string(tw.semantic_threshold));
        }
        if (tw.max_total_bullets > 0) {
            build_args.push_back("--max_total_bullets");
            build_args.push_back(std::to_string(tw.max_total_bullets));
        }
        if (tw.max_experience_bullets > 0) {
            build_args.push_back("--max_experience_bullets");
            build_args.push_back(std::to_string(tw.max_experience_bullets));
        }
        if (tw.max_project_bullets > 0) {
            build_args.push_back("--max_project_bullets");
            build_args.push_back(std::to_string(tw.max_project_bullets));
        }
        if (tw.max_bullets_per_parent > 0) {
            build_args.push_back("--max_bullets_per_parent");
            build_args.push_back(std::to_string(tw.max_bullets_per_parent));
        }

        int build_rc = 0;
        {
            auto cargv = to_argv(build_args);
            build_rc = cmd_build((int)cargv.size(), cargv.data());
        }

        resume::ValidationInputs vin;
        vin.resume_path = resume_path;
        vin.explainability_path = explain_path.string();
        vin.outdir = outdir;

        last_rep = resume::validate_run(vin);
        resume::write_validation_report(report_path, last_rep);

        nlohmann::json attempt_j;
        attempt_j["attempt"] = (int)(attempt + 1);
        attempt_j["build_rc"] = build_rc;
        attempt_j["pass"] = last_rep.pass;
        attempt_j["analyze_args"] = args_to_json_array(analyze_args);
        attempt_j["build_args"] = args_to_json_array(build_args);

        nlohmann::json errs = nlohmann::json::array();
        for (const auto& e : last_rep.errors) {
            nlohmann::json ej;
            ej["code"] = e.code;
            ej["message"] = e.message;
            if (!e.bullet_id.empty()) ej["bullet_id"] = e.bullet_id;
            errs.push_back(ej);
        }
        attempt_j["errors"] = errs;

        append_jsonl(attempts_path, attempt_j);

        if (build_rc == 0 && last_rep.pass) {
            success = true;
            break;
        }
    }

    // -------------------------
    // 3) MANIFEST (always write)
    // -------------------------
    nlohmann::json manifest;
    manifest["role"] = role;
    manifest["resume_path"] = resume_path;
    manifest["outdir"] = outdir;
    manifest["artifacts"] = {
        {"profile_json", (outdir_p / "profile.json").string()},
        {"mentions_jsonl", (outdir_p / "mentions.jsonl").string()},
        {"bullet_scores_json", (outdir_p / "bullet_scores.json").string()},
        {"resume_md", (outdir_p / "resume.md").string()},
        {"explainability_json", explain_path.string()},
        {"validation_report_json", report_path.string()},
        {"run_attempts_jsonl", attempts_path.string()},
        {"run_manifest_json", manifest_path.string()}
    };
    manifest["defaults"] = {
        {"llm_cache_dir", llm_cache_dir},
        {"semantic_cache_path", semantic_cache_path}
    };
    manifest["analyze_args"] = args_to_json_array(analyze_args);

    write_json(manifest_path, manifest);

    if (!success) {
        std::cerr << "validation failed after retries: wrote " << report_path.string() << "\n";
        std::cerr << "attempt log: " << attempts_path.string() << "\n";
        std::cerr << "manifest: " << manifest_path.string() << "\n";
        for (const auto& e : last_rep.errors) {
            std::cerr << "- " << e.code << ": " << e.message;
            if (!e.bullet_id.empty()) std::cerr << " (bullet_id=" << e.bullet_id << ")";
            std::cerr << "\n";
        }
        return 1;
    }

    std::cout << "VALIDATION: pass\n";
    std::cout << "OUT_VALIDATE: " << report_path.string() << "\n";
    std::cout << "OUT_ATTEMPTS: " << attempts_path.string() << "\n";
    std::cout << "OUT_MANIFEST: " << manifest_path.string() << "\n";
    return 0;
}
