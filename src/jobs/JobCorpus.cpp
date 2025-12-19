#include "jobs/JobCorpus.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

static std::string read_all(const fs::path& p) {
    std::ifstream in(p);
    if (!in) throw std::runtime_error("failed to open: " + p.string());
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

JobCorpus JobCorpus::load_from_dir(const std::string& dir) {
    JobCorpus c;

    fs::path root(dir);
    if (!fs::exists(root)) throw std::runtime_error("dir not found: " + dir);

    for (auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        auto p = entry.path();
        if (p.extension() != ".txt") continue;

        JobPosting jp;
        jp.id = p.stem().string();
        jp.raw_text = read_all(p);
        // title can be empty for now; we’ll add meta.json later
        c.m_posts.push_back(std::move(jp));
    }

    return c;
}
