#include "llm/MockLLMClient.hpp"
#include "nlohmann/json.hpp"

#include <fstream>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace llm {

MockLLMClient::MockLLMClient(const std::string& root_dir) : root_(root_dir) {}

std::vector<EvidenceSpan> MockLLMClient::load_file_for_posting_id(const std::string& posting_id) {
    std::vector<EvidenceSpan> out;

    fs::path p = root_ / (posting_id + ".json");
    std::ifstream f(p);
    if (!f) return out;

    json j;
    try {
        f >> j;
    } catch (...) {
        return out;
    }

    if (!j.is_object()) return out;
    if (!j.contains("evidence") || !j["evidence"].is_array()) return out;

    for (const auto& e : j["evidence"]) {
        if (!e.is_object()) continue;

        EvidenceSpan ev;

        if (e.contains("span_type") && e["span_type"].is_string()) ev.span_type = e["span_type"].get<std::string>();
        if (e.contains("span_text") && e["span_text"].is_string()) ev.span_text = e["span_text"].get<std::string>();
        if (e.contains("polarity") && e["polarity"].is_string()) ev.polarity = e["polarity"].get<std::string>();
        if (e.contains("strength") && e["strength"].is_string()) ev.strength = e["strength"].get<std::string>();

        if (e.contains("skills") && e["skills"].is_array()) {
            for (const auto& s : e["skills"]) {
                if (!s.is_object()) continue;
                SkillHit sh;

                if (s.contains("raw") && s["raw"].is_string()) sh.raw = s["raw"].get<std::string>();
                if (s.contains("canonical") && s["canonical"].is_string()) sh.canonical = s["canonical"].get<std::string>();
                if (s.contains("confidence") && (s["confidence"].is_number_float() || s["confidence"].is_number_integer()))
                    sh.confidence = s["confidence"].get<double>();

                if (!sh.raw.empty() || !sh.canonical.empty()) ev.skills.push_back(sh);
            }
        }

        if (!ev.span_type.empty() || !ev.span_text.empty() || !ev.skills.empty()) out.push_back(std::move(ev));
    }

    return out;
}

std::vector<EvidenceSpan> MockLLMClient::evidence_for_posting_id(const std::string& posting_id) {
    return load_file_for_posting_id(posting_id);
}

std::vector<Span> MockLLMClient::segment_for_posting_id(const std::string& posting_id) {
    std::vector<Span> spans;
    auto evs = load_file_for_posting_id(posting_id);
    spans.reserve(evs.size());
    for (const auto& ev : evs) {
        Span sp;
        sp.type = ev.span_type;
        sp.text = ev.span_text;
        spans.push_back(std::move(sp));
    }
    return spans;
}

// Day 3 (B): ignore text, use posting_id to fetch mock evidence
std::vector<EvidenceSpan> MockLLMClient::analyze_posting(const std::string& posting_id,
                                                        const std::string&) {
    return load_file_for_posting_id(posting_id);
}

std::vector<Span> MockLLMClient::segment(const std::string&) {
    return {};
}

EvidenceSpan MockLLMClient::extract(const Span& span) {
    EvidenceSpan ev;
    ev.span_type = span.type;
    ev.span_text = span.text;
    ev.polarity = "positive";
    ev.strength = "unknown";
    return ev;
}

} // namespace llm
