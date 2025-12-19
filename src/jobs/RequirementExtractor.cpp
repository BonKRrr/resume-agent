#include "jobs/RequirementExtractor.hpp"
#include "jobs/TextUtil.hpp"
#include <algorithm>
#include <cctype>
#include <unordered_set>

static void add_unique(std::vector<std::string>& out, std::unordered_set<std::string>& seen, const std::string& item) {
    if (seen.insert(item).second) out.push_back(item);
}

std::string RequirementExtractor::to_lower_ascii(const std::string& s) {
    std::string out = s;
    for (unsigned char& c : reinterpret_cast<std::basic_string<unsigned char>&>(out)) {
        c = static_cast<unsigned char>(std::tolower(c));
    }
    return out;
}

std::string RequirementExtractor::trim(const std::string& s) {
    size_t i = 0, j = s.size();
    while (i < j && std::isspace((unsigned char)s[i])) ++i;
    while (j > i && std::isspace((unsigned char)s[j - 1])) --j;
    return s.substr(i, j - i);
}

std::vector<std::string> RequirementExtractor::split_lines(const std::string& s) {
    std::vector<std::string> lines;
    std::string cur;
    cur.reserve(128);

    for (char ch : s) {
        if (ch == '\r') continue;
        if (ch == '\n') {
            lines.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    lines.push_back(cur);
    return lines;
}

RequirementExtractor::SectionSlices RequirementExtractor::slice_requirement_sections(const std::string& raw_text) {
    // Heuristic:
    // - Find headings like "Requirements", "Qualifications", "What you bring", "Skills"
    // - Capture following lines until a new heading or too many blanks.
    // - Also look for "Preferred" / "Nice to have" headings.
    SectionSlices out;

    auto lines = split_lines(raw_text);

    auto is_heading = [](const std::string& line_lc) -> bool {
        // normalize common heading patterns
        // (keep this simple: it’s Day 2, not a full parser)
        if (line_lc == "requirements" || line_lc == "requirements:" ||
            line_lc == "qualifications" || line_lc == "qualifications:" ||
            line_lc == "skills" || line_lc == "skills:" ||
            line_lc == "what you bring" || line_lc == "what you bring:" ||
            line_lc == "what you will bring" || line_lc == "what you will bring:" ||
            line_lc == "what we're looking for" || line_lc == "what we're looking for:" ||
            line_lc == "what we are looking for" || line_lc == "what we are looking for:" ||
            line_lc == "must have" || line_lc == "must have:" ||
            line_lc == "minimum qualifications" || line_lc == "minimum qualifications:" ||
            line_lc == "required qualifications" || line_lc == "required qualifications:") {
            return true;
        }
        return false;
    };

    auto is_preferred_heading = [](const std::string& line_lc) -> bool {
        if (line_lc == "preferred" || line_lc == "preferred:" ||
            line_lc == "preferred qualifications" || line_lc == "preferred qualifications:" ||
            line_lc == "nice to have" || line_lc == "nice to have:" ||
            line_lc == "bonus" || line_lc == "bonus:" ||
            line_lc == "bonus points" || line_lc == "bonus points:" ||
            line_lc == "assets" || line_lc == "assets:" ) {
            return true;
        }
        return false;
    };

    enum class Mode { None, Must, Preferred };
    Mode mode = Mode::None;
    int blank_run = 0;

    auto push_line = [&](Mode m, const std::string& line) {
        if (m == Mode::Must) {
            out.must += line;
            out.must += "\n";
        } else if (m == Mode::Preferred) {
            out.preferred += line;
            out.preferred += "\n";
        }
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        std::string t = trim(lines[i]);
        std::string lc = to_lower_ascii(t);

        if (t.empty()) {
            blank_run++;
            // stop capturing after a few blank lines to avoid swallowing the whole posting
            if (blank_run >= 3) mode = Mode::None;
            continue;
        }
        blank_run = 0;

        // some headings show as "REQUIREMENTS" etc.
        std::string lc_compact = lc;
        // collapse trailing ":" already handled via equality checks, keep simple

        if (is_heading(lc_compact)) {
            mode = Mode::Must;
            continue;
        }
        if (is_preferred_heading(lc_compact)) {
            mode = Mode::Preferred;
            continue;
        }

        // If line looks like a heading (short, ends with ":"), stop current section.
        if (lc.size() <= 40 && !lc.empty() && lc.back() == ':') {
            mode = Mode::None;
            continue;
        }

        // capture content lines if we're in a section
        push_line(mode, t);
    }

    return out;
}

bool RequirementExtractor::contains_phrase(const std::string& normalized_haystack, const std::string& normalized_phrase) {
    // normalized_* are expected to be space-separated lowercase already (via textutil::normalize)
    // enforce word-ish boundaries by padding spaces
    std::string h = " " + normalized_haystack + " ";
    std::string p = " " + normalized_phrase + " ";
    return h.find(p) != std::string::npos;
}

ExtractedReqs RequirementExtractor::extract(const std::string& raw_text) const {
    // Build two candidate texts:
    // - section text (must/preferred) if present
    // - full text fallback
    auto slices = slice_requirement_sections(raw_text);

    std::string norm_all = textutil::normalize(raw_text);
    std::string norm_must = textutil::normalize(slices.must);
    std::string norm_pref = textutil::normalize(slices.preferred);

    // Day 2: hardcoded lexicon + simple phrase matching.
    // Later (Day 5+) you can swap this for:
    // - config json
    // - OR a GPT adapter that proposes candidates + you verify deterministically.
    struct Item { const char* canon; const char* phrase; };

    struct Cat {
        const char* name;
        std::vector<Item> items;
    };

    const std::vector<Cat> cats = {
        {"languages", {
            {"C++", "c++"},
            {"C", "c"},
            {"C#", "c#"},
            {"Java", "java"},
            {"Python", "python"},
            {"JavaScript", "javascript"},
            {"TypeScript", "typescript"},
            {"SQL", "sql"},
            {"Go", "go"},
            {"Rust", "rust"},
        }},
        {"frameworks", {
            {"gRPC", "grpc"},
            {"Protocol Buffers", "protobuf"},
            {"Boost", "boost"},
            {"Qt", "qt"},
            {"Spring", "spring"},
            {"React", "react"},
            {"Node.js", "node"},
            {"Express", "express"},
        }},
        {"systems", {
            {"Linux", "linux"},
            {"Windows", "windows"},
            {"Multithreading", "multithreading"},
            {"Concurrency", "concurrency"},
            {"Networking", "networking"},
            {"Sockets", "sockets"},
            {"Low latency", "low latency"},
            {"Performance", "performance"},
        }},
        {"tools", {
            {"Git", "git"},
            {"Docker", "docker"},
            {"Kubernetes", "kubernetes"},
            {"CMake", "cmake"},
            {"Bazel", "bazel"},
            {"Jira", "jira"},
        }},
        {"cloud", {
            {"AWS", "aws"},
            {"GCP", "gcp"},
            {"Azure", "azure"},
        }},
        {"databases", {
            {"PostgreSQL", "postgresql"},
            {"PostgreSQL", "postgres"},
            {"MySQL", "mysql"},
            {"MongoDB", "mongodb"},
            {"Redis", "redis"},
        }},
        // a tiny “general” bucket that catches common non-tech postings too
        {"general", {
            {"Communication", "communication"},
            {"Leadership", "leadership"},
            {"Project management", "project management"},
            {"Customer service", "customer service"},
            {"Sales", "sales"},
            {"Marketing", "marketing"},
            {"Social media", "social media"},
            {"Branding", "branding"},
            {"Content creation", "content creation"},
        }},
    };

    ExtractedReqs out;
    out.by_category.reserve(cats.size() + 1);

    // We keep “preferred/nice-to-have” separate, because postings often list extras there.
    std::vector<std::string> nice_to_have;
    std::unordered_set<std::string> nice_seen;

    for (const auto& cat : cats) {
        std::vector<std::string> hits;
        std::unordered_set<std::string> seen;

        for (const auto& it : cat.items) {
            // prefer section hits if sections exist, otherwise full-text is all we have
            bool in_must = !norm_must.empty() && contains_phrase(norm_must, it.phrase);
            bool in_pref = !norm_pref.empty() && contains_phrase(norm_pref, it.phrase);
            bool in_any  = contains_phrase(norm_all, it.phrase);

            if (in_must || (norm_must.empty() && in_any)) {
                add_unique(hits, seen, it.canon);
            } else if (in_pref) {
                add_unique(nice_to_have, nice_seen, it.canon);
            } else if (norm_must.empty() && in_any) {
                // (already handled above, but keep for clarity)
                add_unique(hits, seen, it.canon);
            }
        }

        out.by_category.push_back({cat.name, std::move(hits)});
    }

    if (!nice_to_have.empty()) {
        out.by_category.push_back({"nice_to_have", std::move(nice_to_have)});
    }

    return out;
}
