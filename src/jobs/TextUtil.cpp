#include "jobs/TextUtil.hpp"
#include <cctype>
#include <unordered_map>

namespace textutil {

std::string normalize(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    bool prev_space = true;

    for (unsigned char ch : s) {
        unsigned char c = static_cast<unsigned char>(std::tolower(ch));

        bool keep =
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') ||
            (c == '+') || (c == '#'); // keeps "c++" and "c#"

        if (keep) {
            out.push_back(static_cast<char>(c));
            prev_space = false;
        } else {
            if (!prev_space) {
                out.push_back(' ');
                prev_space = true;
            }
        }
    }

    // trim trailing space
    if (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

std::vector<std::string> tokenize(const std::string& normalized) {
    std::vector<std::string> tokens;
    std::string cur;

    for (char c : normalized) {
        if (c == ' ') {
            if (!cur.empty()) {
                // drop tiny tokens except "c++"
                if (cur.size() >= 2 || cur == "c++") tokens.push_back(cur);
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        if (cur.size() >= 2 || cur == "c++") tokens.push_back(cur);
    }
    return tokens;
}

std::vector<std::string> normalize_tokens(const std::vector<std::string>& tokens) {
    // single-token synonym folding
    static const std::unordered_map<std::string, std::string> fold = {
        {"dev", "engineer"},
        {"developer", "engineer"},
        {"programmer", "engineer"},
        {"engineering", "engineer"},
        {"eng", "engineer"},
        {"serverside", "backend"},   // sometimes appears as one token
        {"server-side", "backend"}   // (will likely become "server side" anyway)
    };

    std::vector<std::string> out;
    out.reserve(tokens.size());

    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];

        // phrase merging (2-grams)
        if (i + 1 < tokens.size()) {
            const std::string& n = tokens[i + 1];

            if (t == "back" && n == "end") {        // "back end" / "back-end"
                out.push_back("backend");
                ++i;
                continue;
            }
            if (t == "server" && n == "side") {     // "server side" / "server-side"
                out.push_back("backend");
                ++i;
                continue;
            }
        }

        // apply fold map
        auto it = fold.find(t);
        if (it != fold.end()) out.push_back(it->second);
        else out.push_back(t);
    }

    return out;
}

}
