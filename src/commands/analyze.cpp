#include "commands/analyze.hpp"
#include "jobs/JobCorpus.hpp"
#include "jobs/TextUtil.hpp"
#include "jobs/RequirementExtractor.hpp"
#include <iostream>
#include <string>

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return std::string(argv[i + 1]);
    }
    return def;
}

static void print_reqs(const std::string& id, const ExtractedReqs& r) {
    std::cout << "\nPOST " << id << "\n";
    for (const auto& [cat, items] : r.by_category) {
        if (items.empty()) continue;
        std::cout << "- " << cat << ": ";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i) std::cout << ", ";
            std::cout << items[i];
        }
        std::cout << "\n";
    }
}

int cmd_analyze(int argc, char** argv) {
    std::string role = get_arg(argc, argv, "--role", "");
    std::string jobs_dir = get_arg(argc, argv, "--jobs", "data/jobs/raw");

    if (role.empty()) {
        std::cerr << "error: missing --role\n";
        return 1;
    }

    JobCorpus corpus = JobCorpus::load_from_dir(jobs_dir);

    std::cout << "ROLE: " << role << "\n";
    std::cout << "JOBS_DIR: " << jobs_dir << "\n";
    std::cout << "POSTINGS: " << corpus.postings().size() << "\n";

    RequirementExtractor ex;

    for (const auto& post : corpus.postings()) {
        auto reqs = ex.extract(post.raw_text);
        print_reqs(post.id, reqs);
    }

    return 0;
}
