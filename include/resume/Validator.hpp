#pragma once

#include <string>
#include <vector>
#include <filesystem>

namespace resume {

struct ValidationError {
    std::string code;
    std::string message;
    std::string bullet_id;
};

struct ValidationReport {
    bool pass = true;
    std::vector<ValidationError> errors;
};

struct ValidationInputs {
    std::string resume_path;
    std::string explainability_path;
    std::string outdir;
};

ValidationReport validate_run(const ValidationInputs& in);
void write_validation_report(const std::filesystem::path& path, const ValidationReport& rep);

}
