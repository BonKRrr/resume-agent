#include "jobs/EmbeddingIndex.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>

void EmbeddingIndex::set(std::vector<std::string> job_ids, std::vector<float> vectors, size_t dim) {
    m_job_ids = std::move(job_ids);
    m_vecs = std::move(vectors);
    m_dim = dim;
}

float EmbeddingIndex::cosine(const float* a, const float* b, size_t dim) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double x = a[i], y = b[i];
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if (na == 0.0 || nb == 0.0) return 0.0f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

std::vector<EmbHit> EmbeddingIndex::topk(const std::vector<float>& query_vec, size_t k) const {
    std::vector<EmbHit> hits;
    if (m_dim == 0 || query_vec.size() != m_dim) return hits;

    hits.reserve(m_job_ids.size());
    for (size_t i = 0; i < m_job_ids.size(); ++i) {
        const float* v = &m_vecs[i * m_dim];
        float s = cosine(query_vec.data(), v, m_dim);
        hits.push_back({m_job_ids[i], s});
    }

    std::partial_sort(hits.begin(), hits.begin() + std::min(k, hits.size()), hits.end(),
                      [](const auto& a, const auto& b){ return a.score > b.score; });

    if (hits.size() > k) hits.resize(k);
    return hits;
}

bool EmbeddingIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    uint32_t dim = (uint32_t)m_dim;
    uint32_t n = (uint32_t)m_job_ids.size();
    out.write((char*)&dim, sizeof(dim));
    out.write((char*)&n, sizeof(n));

    for (const auto& id : m_job_ids) {
        uint32_t len = (uint32_t)id.size();
        out.write((char*)&len, sizeof(len));
        out.write(id.data(), len);
    }

    uint64_t vec_count = (uint64_t)m_vecs.size();
    out.write((char*)&vec_count, sizeof(vec_count));
    out.write((char*)m_vecs.data(), (std::streamsize)(sizeof(float) * m_vecs.size()));
    return true;
}

bool EmbeddingIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    uint32_t dim = 0, n = 0;
    in.read((char*)&dim, sizeof(dim));
    in.read((char*)&n, sizeof(n));
    if (!in || dim == 0) return false;

    std::vector<std::string> ids;
    ids.reserve(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = 0;
        in.read((char*)&len, sizeof(len));
        if (!in) return false;
        std::string s(len, '\0');
        in.read(s.data(), len);
        if (!in) return false;
        ids.push_back(std::move(s));
    }

    uint64_t vec_count = 0;
    in.read((char*)&vec_count, sizeof(vec_count));
    if (!in) return false;

    std::vector<float> vecs((size_t)vec_count);
    in.read((char*)vecs.data(), (std::streamsize)(sizeof(float) * vecs.size()));
    if (!in) return false;

    m_dim = dim;
    m_job_ids = std::move(ids);
    m_vecs = std::move(vecs);
    return true;
}
