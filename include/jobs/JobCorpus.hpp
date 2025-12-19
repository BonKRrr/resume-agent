#pragma once
#include "jobs/JobPosting.hpp"
#include <string>
#include <vector>

class JobCorpus {
public:
    static JobCorpus load_from_dir(const std::string& dir); // loads *.txt
    const std::vector<JobPosting>& postings() const { return m_posts; }

private:
    std::vector<JobPosting> m_posts;
};
