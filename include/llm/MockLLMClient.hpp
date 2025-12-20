#pragma once

#include "llm/LLMClient.hpp"

#include <filesystem>
#include <string>

namespace llm {

class MockLLMClient final : public LLMClient {
    std::filesystem::path root_;

public:
    explicit MockLLMClient(const std::string& root_dir);

    // Day 3 (B)
    std::vector<EvidenceSpan> analyze_posting(const std::string& posting_id,
                                             const std::string& posting_text) override;

    // Legacy (kept)
    std::vector<Span> segment(const std::string& posting_text) override;
    EvidenceSpan extract(const Span& span) override;

    std::vector<Span> segment_for_posting_id(const std::string& posting_id);
    std::vector<EvidenceSpan> evidence_for_posting_id(const std::string& posting_id);

private:
    std::vector<EvidenceSpan> load_file_for_posting_id(const std::string& posting_id);
};

} // namespace llm
