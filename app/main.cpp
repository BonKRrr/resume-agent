#include "commands/ResumeDump.hpp"
#include "commands/analyze.hpp"
#include "commands/embed.hpp"
#include "commands/build.hpp"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "usage:\n"
            << "  resume-agent resume dump [path]\n"
            << "  resume-agent analyze [args]\n"
            << "  resume-agent embed [args]\n"
            << "  resume-agent build [args]\n";
        return 1;
    }

    // legacy: resume dump
    if (argc >= 3) {
        std::string cmd = argv[1];
        std::string sub1 = argv[2];

        // Default path if not provided
        std::string path = (argc >= 4) ? argv[3] : "data/abstract_resume.json";

        if (cmd == "resume" && sub1 == "dump") {
            return resumeDump(path);
        }
    }

    // modern subcommands: analyze/embed/build (they parse their own flags)
    if (std::string(argv[1]) == "analyze") {
        return cmd_analyze(argc - 1, argv + 1);
    }
    if (std::string(argv[1]) == "embed") {
        return cmd_embed(argc - 1, argv + 1);
    }
    if (std::string(argv[1]) == "build") {
        return cmd_build(argc - 1, argv + 1);
    }

    std::cerr
        << "unknown command\n"
        << "usage:\n"
        << "  resume-agent resume dump [path]\n"
        << "  resume-agent analyze [args]\n"
        << "  resume-agent embed [args]\n"
        << "  resume-agent build [args]\n";
    return 1;
}
