#include "commands/embed.hpp"
#include "jobs/JobCorpus.hpp"
#include "jobs/EmbeddingIndex.hpp"
#include "emb/MiniLmEmbedder.hpp"
#include <iostream>
#include <string>

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def) {
    for (int i = 0; i + 1 < argc; ++i) {
        if (argv[i] == key) return argv[i + 1];
    }
    return def;
}

int cmd_embed(int argc, char** argv) {
    std::string jobs_dir = get_arg(argc, argv, "--jobs", "data/jobs/raw");
    std::string model    = get_arg(argc, argv, "--model", "models/emb/model.onnx");
    std::string vocab    = get_arg(argc, argv, "--vocab", "models/emb/vocab.txt");
    std::string outp     = get_arg(argc, argv, "--out", "data/embeddings/jobs.bin");

    JobCorpus corpus = JobCorpus::load_from_dir(jobs_dir);

    MiniLmEmbedder emb;
    if (!emb.init(model, vocab)) {
        std::cerr << "error: failed to init MiniLmEmbedder\n";
        return 1;
    }

    std::vector<std::string> ids;
    std::vector<float> vecs;

    size_t dim = 0;
    for (const auto& p : corpus.postings()) {
        auto v = emb.embed(p.raw_text, 256);
        if (v.empty()) continue;

        if (dim == 0) dim = v.size();
        if (v.size() != dim) continue;

        ids.push_back(p.id);
        vecs.insert(vecs.end(), v.begin(), v.end());

        std::cout << "embedded " << p.id << "\n";
    }

    EmbeddingIndex idx;
    idx.set(std::move(ids), std::move(vecs), dim);

    if (!idx.save(outp)) {
        std::cerr << "error: failed to save embeddings to " << outp << "\n";
        return 1;
    }

    std::cout << "saved: " << outp << " (n=" << idx.size() << ", dim=" << idx.dim() << ")\n";
    return 0;
}
