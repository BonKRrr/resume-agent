#pragma once

#include <string>
#include <vector>

namespace resume {

struct ConcreteEntry {
    std::string header;              // e.g. "Software Engineer — Company"
    std::vector<std::string> bullets;
};

struct ConcreteSection {
    std::string title;               // "Experience", "Projects"
    std::vector<ConcreteEntry> entries;
};

struct ConcreteResume {
    std::vector<ConcreteSection> sections;
};

}  // namespace resume
