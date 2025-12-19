#pragma once
#include <string>
#include <utility>
#include <vector>

struct ExtractedReqs {
    // ordered categories for printing
    std::vector<std::pair<std::string, std::vector<std::string>>> by_category;
};

class RequirementExtractor {
public:
    ExtractedReqs extract(const std::string& raw_text) const;

private:
    static std::string to_lower_ascii(const std::string& s);
    static std::string trim(const std::string& s);
    static std::vector<std::string> split_lines(const std::string& s);

    struct SectionSlices {
        std::string must;
        std::string preferred;
    };

    static SectionSlices slice_requirement_sections(const std::string& raw_text);

    static bool contains_phrase(const std::string& normalized_haystack, const std::string& normalized_phrase);
};
