#include "commands/analyze.hpp"
#include "jobs/JobCorpus.hpp"
#include "jobs/RequirementExtractor.hpp"
#include "jobs/TextUtil.hpp"
#include "jobs/EmbeddingIndex.hpp"
#include "emb/MiniLmEmbedder.hpp"
#include <iostream>
#include <string>
#include <unordered_map>

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
    std::string role     = get_arg(argc, argv, "--role", "");
    std::string jobs_dir = get_arg(argc, argv, "--jobs", "data/jobs/raw");
    std::string topk_s   = get_arg(argc, argv, "--topk", "25");

    std::string emb_path = get_arg(argc, argv, "--emb", "data/embeddings/jobs.bin");
    std::string model    = get_arg(argc, argv, "--model", "models/emb/model.onnx");
    std::string vocab    = get_arg(argc, argv, "--vocab", "models/emb/vocab.txt");

    if (role.empty()) {
        std::cerr << "error: missing --role\n";
        return 1;
    }

    JobCorpus corpus = JobCorpus::load_from_dir(jobs_dir);

    std::cout << "ROLE: " << role << "\n";
    std::cout << "JOBS_DIR: " << jobs_dir << "\n";
    std::cout << "POSTINGS: " << corpus.postings().size() << "\n";

    // Map id -> posting pointer for fast lookup
    std::unordered_map<std::string, const JobPosting*> by_id;
    by_id.reserve(corpus.postings().size());
    for (const auto& p : corpus.postings()) by_id[p.id] = &p;

    // Load embedding index
    EmbeddingIndex idx;
    if (!idx.load(emb_path)) {
        std::cerr << "error: failed to load embeddings cache: " << emb_path << "\n";
        std::cerr << "hint: run `resume-agent embed` first\n";
        return 1;
    }

    // Embed the role query
    MiniLmEmbedder emb;
    if (!emb.init(model, vocab)) {
        std::cerr << "error: failed to init embedder for query\n";
        return 1;
    }

    auto q = emb.embed(role, 64);
    if (q.empty() || q.size() != idx.dim()) {
        std::cerr << "error: query embedding dim mismatch\n";
        return 1;
    }

    size_t topk = (size_t)std::stoul(topk_s);
    auto hits = idx.topk(q, topk);

    std::cout << "TOPK: " << hits.size() << "\n";

    RequirementExtractor ex;

    for (const auto& h : hits) {
        auto it = by_id.find(h.job_id);
        if (it == by_id.end()) continue; // embedding exists but posting missing

        std::cout << "\n# hit " << h.job_id << " score=" << h.score << "\n";
        auto reqs = ex.extract(it->second->raw_text);
        print_reqs(it->second->id, reqs);
    }

    return 0;
}
