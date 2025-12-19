#pragma once
#include "jobs/JobCorpus.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>



struct SearchHit {
    std::string job_id;
    double score;
    size_t token_count;
};

class TfidfSearch {
public:
    explicit TfidfSearch(const JobCorpus& corpus);

    std::vector<SearchHit> topk(const std::string& query, size_t k) const;

private:
    struct PostingVec {
        std::string job_id;
        size_t token_count = 0;
        std::vector<std::pair<uint32_t, float>> weights; // (term_id, tf-idf weight)
        double norm = 0.0;
    };

    // vocab
    std::vector<std::string> m_terms;             // term_id -> term
    std::vector<uint32_t> m_df;                   // term_id -> document frequency
    std::vector<double> m_idf;                    // term_id -> idf
    std::unordered_map<std::string, uint32_t> m_term_to_id;

    std::vector<PostingVec> m_postings;

    static double dot_sparse(
        const std::vector<std::pair<uint32_t, float>>& a,
        const std::vector<std::pair<uint32_t, float>>& b
    );
};
