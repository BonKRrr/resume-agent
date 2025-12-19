#include "commands/ResumeDump.hpp"
#include "commands/analyze.hpp"
#include "commands/embed.hpp"
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage:\n"
                  << "  resume-agent resume dump [path]\n";
        return 1;
    }

    std::string cmd = argv[1];
    std::string sub1 = argv[2];

    // Default path if not provided
    std::string path = (argc >= 4) ? argv[3] : "data/abstract_resume.json";

    if (cmd == "resume" && sub1 == "dump") {
        return resumeDump(path);
    }
    if (argc >= 2 && std::string(argv[1]) == "analyze") {
        return cmd_analyze(argc - 1, argv + 1);
    }
    if (argc >= 2 && std::string(argv[1]) == "embed") {
    return cmd_embed(argc - 1, argv + 1);
    }


    std::cerr << "unknown command\n";
    return 1;
}
