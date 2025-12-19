#include "jobs/TfidfSearch.hpp"
#include "jobs/TextUtil.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>

static double safe_log(double x) {
    return std::log(x);
}

static void sort_and_merge(std::vector<std::pair<uint32_t, float>>& v) {
    std::sort(v.begin(), v.end(), [](auto& x, auto& y){ return x.first < y.first; });
    size_t w = 0;
    for (size_t i = 0; i < v.size(); ) {
        uint32_t id = v[i].first;
        float sum = 0.0f;
        size_t j = i;
        while (j < v.size() && v[j].first == id) {
            sum += v[j].second;
            ++j;
        }
        v[w++] = {id, sum};
        i = j;
    }
    v.resize(w);
}

double TfidfSearch::dot_sparse(
    const std::vector<std::pair<uint32_t, float>>& a,
    const std::vector<std::pair<uint32_t, float>>& b
) {
    size_t i = 0, j = 0;
    double s = 0.0;
    while (i < a.size() && j < b.size()) {
        if (a[i].first == b[j].first) {
            s += (double)a[i].second * (double)b[j].second;
            ++i; ++j;
        } else if (a[i].first < b[j].first) {
            ++i;
        } else {
            ++j;
        }
    }
    return s;
}

TfidfSearch::TfidfSearch(const JobCorpus& corpus) {
    const auto& posts = corpus.postings();
    const uint32_t N = (uint32_t)posts.size();

    // Pass 1: build DF + vocab from normalized tokens
    std::unordered_map<std::string, uint32_t> df_map;

    std::vector<std::vector<std::string>> posting_tokens;
    posting_tokens.reserve(posts.size());

    for (const auto& p : posts) {
        auto norm = textutil::normalize(p.raw_text);
        auto toks = textutil::tokenize(norm);
        posting_tokens.push_back(toks);

        // unique terms in this doc for DF
        std::unordered_map<std::string, bool> seen;
        seen.reserve(toks.size());
        for (const auto& t : toks) seen.emplace(t, true);

        for (const auto& kv : seen) {
            df_map[kv.first] += 1;
        }
    }

    // Freeze vocab: assign term_ids
    m_terms.reserve(df_map.size());
    m_df.reserve(df_map.size());

    for (const auto& kv : df_map) {
        m_term_to_id.emplace(kv.first, (uint32_t)m_terms.size());
        m_terms.push_back(kv.first);
        m_df.push_back(kv.second);
    }

    // Compute IDF
    m_idf.resize(m_terms.size());
    for (size_t term_id = 0; term_id < m_terms.size(); ++term_id) {
        // smooth: idf = log((N + 1)/(df + 1)) + 1
        double df = (double)m_df[term_id];
        m_idf[term_id] = safe_log(((double)N + 1.0) / (df + 1.0)) + 1.0;
    }

    // Pass 2: build posting vectors (TF-IDF)
    m_postings.reserve(posts.size());

    for (size_t idx = 0; idx < posts.size(); ++idx) {
        const auto& p = posts[idx];
        const auto& toks = posting_tokens[idx];

        std::unordered_map<uint32_t, uint32_t> tf;
        tf.reserve(toks.size());

        for (const auto& t : toks) {
            auto it = m_term_to_id.find(t);
            if (it == m_term_to_id.end()) continue;
            tf[it->second] += 1;
        }

        PostingVec pv;
        pv.job_id = p.id;
        pv.token_count = toks.size();

        pv.weights.reserve(tf.size());
        double norm2 = 0.0;

        for (const auto& kv : tf) {
            uint32_t term_id = kv.first;
            uint32_t freq = kv.second;

            // log TF
            double w = (1.0 + safe_log((double)freq)) * m_idf[term_id];
            pv.weights.push_back({term_id, (float)w});
            norm2 += w * w;
        }

        sort_and_merge(pv.weights);
        pv.norm = std::sqrt(norm2);

        m_postings.push_back(std::move(pv));
    }
}

std::vector<SearchHit> TfidfSearch::topk(const std::string& query, size_t k) const {
    auto qnorm = textutil::normalize(query);
    auto qtoks = textutil::tokenize(qnorm);

    // TF for query
    std::unordered_map<uint32_t, uint32_t> qtf;
    qtf.reserve(qtoks.size());

    for (const auto& t : qtoks) {
        auto it = m_term_to_id.find(t);
        if (it == m_term_to_id.end()) continue;
        qtf[it->second] += 1;
    }

    std::vector<std::pair<uint32_t, float>> qvec;
    qvec.reserve(qtf.size());

    double qnorm2 = 0.0;
    for (const auto& kv : qtf) {
        uint32_t term_id = kv.first;
        uint32_t freq = kv.second;
        double w = (1.0 + safe_log((double)freq)) * m_idf[term_id];
        qvec.push_back({term_id, (float)w});
        qnorm2 += w * w;
    }

    sort_and_merge(qvec);
    double qn = std::sqrt(qnorm2);
    if (qn == 0.0) return {}; // no known terms

    std::vector<SearchHit> hits;
    hits.reserve(m_postings.size());

    for (const auto& p : m_postings) {
        if (p.norm == 0.0) continue;
        double d = dot_sparse(qvec, p.weights);
        double score = d / (qn * p.norm);
        if (score > 0.0) {
            hits.push_back({p.job_id, score, p.token_count});
        }
    }

    std::sort(hits.begin(), hits.end(), [](const auto& a, const auto& b){
        return a.score > b.score;
    });

    if (hits.size() > k) hits.resize(k);
    return hits;
}
