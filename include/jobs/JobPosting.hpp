#pragma once
#include <string>
#include <vector>

struct JobPosting {
    std::string id;        // filename stem, e.g., "001"
    std::string title;     // optional for now; can leave empty
    std::string raw_text;  // full posting text
};
