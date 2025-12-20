#pragma once
#include <string>
#include <vector>

namespace llm {

struct Span {
    std::string type; // "requirement" | "preferred" | "responsibility" | "other"
    std::string text;
};

struct SkillHit {
    std::string raw;
    std::string canonical;
    double confidence = 0.0; // 0..1
};

struct EvidenceSpan {
    std::string span_type;
    std::string span_text;
    std::string polarity;  // "positive" | "negated"
    std::string strength;  // "must" | "should" | "nice" | "unknown"
    std::vector<SkillHit> skills;
};

class LLMClient {
public:
    virtual ~LLMClient() = default;

    // Day 3 (B): ONE CALL per posting => evidence spans with skills
    virtual std::vector<EvidenceSpan> analyze_posting(const std::string& posting_id,
                                                     const std::string& posting_text) = 0;

    // Legacy (kept for compatibility / mock tooling; analyze.cpp won't use these anymore)
    virtual std::vector<Span> segment(const std::string& posting_text) = 0;
    virtual EvidenceSpan extract(const Span& span) = 0;
};

class NullLLMClient final : public LLMClient {
public:
    std::vector<EvidenceSpan> analyze_posting(const std::string&, const std::string&) override { return {}; }
    std::vector<Span> segment(const std::string&) override { return {}; }
    EvidenceSpan extract(const Span&) override { return EvidenceSpan{}; }
};

} // namespace llm
