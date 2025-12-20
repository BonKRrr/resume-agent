#pragma once

#include "llm/LLMClient.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace llm {

class OllamaLLMClient final : public LLMClient {
    std::string model_;
    std::filesystem::path cache_dir_;

public:
    OllamaLLMClient(const std::string& model, const std::string& cache_dir);

    std::vector<EvidenceSpan> analyze_posting(const std::string& posting_id,
                                             const std::string& posting_text) override;

    std::vector<Span> segment(const std::string& posting_text) override;
    EvidenceSpan extract(const Span& span) override;

private:
    std::string prompt_analyzer_onecall(const std::string& posting_text) const;

    std::string prompt_segmenter(const std::string& posting_text) const;
    std::string prompt_extractor(const Span& span) const;

    std::string run_ollama_json(const std::string& prompt) const;

    std::string cache_key(const std::string& task, const std::string& input) const;
    bool load_cache(const std::string& key, std::string& out) const;
    void save_cache(const std::string& key, const std::string& content) const;

    std::vector<Span> parse_spans_json(const std::string& s) const;
    EvidenceSpan parse_evidence_json(const std::string& s) const;
    std::vector<EvidenceSpan> parse_evidence_list_json(const std::string& s) const;
};

} // namespace llm
