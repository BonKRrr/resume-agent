#include "commands/validate.hpp"

#include "resume/Validator.hpp"

#include <filesystem>
#include <iostream>
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

static int validate_usage() {
    std::cerr
        << "usage:\n"
        << "  resume-agent validate --resume <path> [--outdir <dir>] [--explain <path>] [--out <path>]\n";
    return 1;
}

int cmd_validate(int argc, char** argv) {
    (void)has_flag;

    const std::string resume_path = get_arg(argc, argv, "--resume", "");
    const std::string outdir      = get_arg(argc, argv, "--outdir", "out");

    if (resume_path.empty()) {
        std::cerr << "error: missing --resume\n";
        return validate_usage();
    }

    const fs::path outdir_p(outdir);
    const std::string explain_path = get_arg(argc, argv, "--explain", (outdir_p / "explainability.json").string());
    const std::string out_path     = get_arg(argc, argv, "--out", (outdir_p / "validation_report.json").string());

    resume::ValidationInputs vin;
    vin.resume_path = resume_path;
    vin.explainability_path = explain_path;
    vin.outdir = outdir;

    const resume::ValidationReport rep = resume::validate_run(vin);
    resume::write_validation_report(fs::path(out_path), rep);

    if (!rep.pass) {
        std::cerr << "validation failed: wrote " << out_path << "\n";
        for (const auto& e : rep.errors) {
            std::cerr << "- " << e.code << ": " << e.message;
            if (!e.bullet_id.empty()) std::cerr << " (bullet_id=" << e.bullet_id << ")";
            std::cerr << "\n";
        }
        return 1;
    }

    std::cout << "VALIDATION: pass\n";
    std::cout << "OUT_VALIDATE: " << out_path << "\n";
    return 0;
}
