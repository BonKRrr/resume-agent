#include "commands/ResumeDump.hpp"
#include "io/JsonIO.hpp"
#include "resume/Models.hpp"

#include <iostream>

static void printTags(const std::vector<std::string>& tags) {
    std::cout << "    tags: ";
    for (size_t i = 0; i < tags.size(); ++i) {
        std::cout << tags[i];
        if (i + 1 < tags.size()) std::cout << ", ";
    }
    std::cout << "\n";
}

int resumeDump(const std::string& resumePath) {
    AbstractResume ar;
    try {
        ar = loadAbstractResume(resumePath);
    } catch (const std::exception& e) {
        std::cerr << "[error] failed to load resume: " << e.what() << "\n";
        return 1;
    }

    for (const auto& exp : ar.experiences) {
        std::cout << "[Experience] " << exp.title << " - " << exp.organization
                  << " (" << exp.dates << ")\n";
        for (const auto& b : exp.bullets) {
            std::cout << "  - " << b.text << "\n";
            printTags(b.tags);
        }
        std::cout << "\n";
    }

    for (const auto& proj : ar.projects) {
        std::cout << "[Project] " << proj.name << " (" << proj.context << ")\n";
        for (const auto& b : proj.bullets) {
            std::cout << "  - " << b.text << "\n";
            printTags(b.tags);
        }
        std::cout << "\n";
    }

    return 0;
}
