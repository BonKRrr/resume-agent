#pragma once
#include <string>
#include <vector>

struct EmbHit {
    std::string job_id;
    float score;
};

class EmbeddingIndex {
public:
    // vectors[i] corresponds to job_ids[i], each vector has dim floats
    void set(std::vector<std::string> job_ids, std::vector<float> vectors, size_t dim);

    std::vector<EmbHit> topk(const std::vector<float>& query_vec, size_t k) const;

    // cache I/O (binary)
    bool save(const std::string& path) const;
    bool load(const std::string& path);

    size_t dim() const { return m_dim; }
    size_t size() const { return m_job_ids.size(); }

private:
    size_t m_dim = 0;
    std::vector<std::string> m_job_ids;
    std::vector<float> m_vecs; // packed: size = size()*dim()

    static float cosine(const float* a, const float* b, size_t dim);
};
