#include "commands/ResumeDump.hpp"
#include "commands/analyze.hpp"
#include "commands/embed.hpp"
#include "commands/build.hpp"

#include <iostream>
#include <string>

static int print_usage() {
    std::cerr
        << "usage:\n"
        << "  resume-agent resume dump [path]\n"
        << "  resume-agent analyze [args]\n"
        << "  resume-agent embed [args]\n"
        << "  resume-agent build [args]\n"
        << "  resume-agent help\n";
    return 1;
}

static int print_analyze_help() {
    std::cerr
        << "usage:\n"
        << "  resume-agent analyze --role \"<job title>\" [options]\n"
        << "\n"
        << "common:\n"
        << "  --role <str>                 (required)\n"
        << "  --jobs <dir>                 default: data/jobs/raw\n"
        << "  --topk <n>                   default: 15\n"
        << "  --min_score <f>              default: 0.30\n"
        << "  --out <path>                 optional: mirror console output to a file\n"
        << "  --outdir <dir>               default: out\n"
        << "\n"
        << "profile:\n"
        << "  --profile                    write out/profile.json + out/mentions.jsonl\n"
        << "\n"
        << "llm:\n"
        << "  --llm                        enable LLM extraction path\n"
        << "  --llm_model <str>            default: llama3.1:8b\n"
        << "  --llm_cache <dir>            default: out/llm_cache\n"
        << "  --llm_mock <dir>             use mock responses from dir (disables real ollama)\n";
    return 0;
}

static int print_embed_help() {
    std::cerr
        << "usage:\n"
        << "  resume-agent embed [options]\n"
        << "\n"
        << "options:\n"
        << "  --jobs <dir>                 default: data/jobs/raw\n"
        << "  --out <path>                 default: data/embeddings/jobs.bin\n"
        << "  --model <path>               default: models/emb/model.onnx\n"
        << "  --vocab <path>               default: models/emb/vocab.txt\n"
        << "  --max_len <n>                default: 256\n";
    return 0;
}

static int print_build_help() {
    std::cerr
        << "usage:\n"
        << "  resume-agent build [options]\n"
        << "\n"
        << "inputs/outputs:\n"
        << "  --resume <path>              default: data/abstract_resume.json\n"
        << "  --profile <path>             default: out/profile.json\n"
        << "  --outdir <dir>               default: out\n"
        << "  --role <str>                 optional override of role in profile\n"
        << "\n"
        << "semantic matching:\n"
        << "  --semantic                   enable semantic tag->skill matching\n"
        << "  --emb_model <path>           default: models/emb/model.onnx\n"
        << "  --emb_vocab <path>           default: models/emb/vocab.txt\n"
        << "  --semantic_threshold <f>      default: 0.66\n"
        << "  --semantic_topk <n>           default: 1\n"
        << "  --semantic_cache <path>       default: (none)\n"
        << "\n"
        << "selection (only used when NOT --scores_only):\n"
        << "  --scores_only                only write out/bullet_scores.json\n"
        << "  --max_total_bullets <n>       default: 10\n"
        << "  --max_bullets_per_parent <n>  default: 3\n"
        << "  --max_experience_bullets <n>  default: 6\n"
        << "  --max_project_bullets <n>     default: 4\n"
        << "  --min_unique_parents <n>      default: 2\n";
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return print_usage();

    const std::string cmd = argv[1];

    if (cmd == "help") {
        return print_usage();
    }

    // legacy: resume dump
    if (argc >= 3) {
        const std::string sub1 = argv[2];
        const std::string path = (argc >= 4) ? argv[3] : "data/abstract_resume.json";
        if (cmd == "resume" && sub1 == "dump") return resumeDump(path);
    }

    // subcommand help
    if (cmd == "analyze" && (argc >= 3 && std::string(argv[2]) == "--help")) return print_analyze_help();
    if (cmd == "embed"   && (argc >= 3 && std::string(argv[2]) == "--help")) return print_embed_help();
    if (cmd == "build"   && (argc >= 3 && std::string(argv[2]) == "--help")) return print_build_help();

    if (cmd == "analyze") return cmd_analyze(argc - 1, argv + 1);
    if (cmd == "embed")   return cmd_embed(argc - 1, argv + 1);
    if (cmd == "build")   return cmd_build(argc - 1, argv + 1);

    std::cerr << "unknown command\n";
    return print_usage();
}
