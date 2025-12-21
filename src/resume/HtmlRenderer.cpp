#include "resume/HtmlRenderer.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace resume {

static std::string html_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 32);
    for (char c : s) {
        switch (c) {
            case '&': out += "&amp;";  break;
            case '<': out += "&lt;";   break;
            case '>': out += "&gt;";   break;
            case '"': out += "&quot;"; break;
            default:  out += c;        break;
        }
    }
    return out;
}

static std::string render_inline_md_bold(const std::string& s) {
    // Converts **bold** to <strong>bold</strong>.
    // Everything else is HTML-escaped.
    std::string out;
    out.reserve(s.size() + 16);

    size_t i = 0;
    while (i < s.size()) {
        if (i + 1 < s.size() && s[i] == '*' && s[i + 1] == '*') {
            size_t j = s.find("**", i + 2);
            if (j != std::string::npos) {
                std::string inner = s.substr(i + 2, j - (i + 2));
                out += "<strong>";
                out += html_escape(inner);
                out += "</strong>";
                i = j + 2;
                continue;
            }
            // If no closing **, fall through and treat literally.
        }

        // Escape one char at a time.
        char c = s[i];
        switch (c) {
            case '&': out += "&amp;";  break;
            case '<': out += "&lt;";   break;
            case '>': out += "&gt;";   break;
            case '"': out += "&quot;"; break;
            default:  out += c;        break;
        }
        ++i;
    }

    return out;
}

static bool starts_with(const std::string& s, const char* prefix) {
    size_t i = 0;
    while (prefix[i]) {
        if (i >= s.size() || s[i] != prefix[i]) return false;
        ++i;
    }
    return true;
}

static bool is_blank(const std::string& s) {
    for (char c : s) {
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') return false;
    }
    return true;
}

static std::string ltrim_copy(const std::string& s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    return s.substr(i);
}

static void flush_paragraph(std::string& html, std::vector<std::string>& lines) {
    if (lines.empty()) return;

    html += "<p>";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) html += "<br/>";
        html += render_inline_md_bold(lines[i]);
    }
    html += "</p>\n";
    lines.clear();
}

std::string render_html_from_markdown(const std::string& md) {
    // Simple, deterministic markdown-ish to HTML converter optimized for Google Docs paste.
    // Supported:
    //   - # / ## / ### headings
    //   - unordered lists starting with "- " or "* "
    //   - paragraphs + line breaks
    //   - inline **bold**
    //
    // Everything else is treated as plain text.

    std::string html;
    html.reserve(md.size() + 4096);

    html += "<!doctype html>\n";
    html += "<html>\n<head>\n";
    html += "<meta charset=\"utf-8\"/>\n";
    html += "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>\n";
    html += "<style>\n";
    html += "  body { font-family: Arial, Helvetica, sans-serif; font-size: 11pt; line-height: 1.35; }\n";
    html += "  h1 { font-size: 18pt; margin: 0 0 8px 0; }\n";
    html += "  h2 { font-size: 13pt; margin: 14px 0 6px 0; }\n";
    html += "  h3 { font-size: 12pt; margin: 10px 0 4px 0; }\n";
    html += "  p  { margin: 0 0 6px 0; }\n";
    html += "  ul { margin: 0 0 8px 22px; padding: 0; }\n";
    html += "  li { margin: 0 0 3px 0; }\n";
    html += "</style>\n";
    html += "</head>\n<body>\n";

    bool in_ul = false;
    std::vector<std::string> para_lines;
    para_lines.reserve(4);

    size_t i = 0;
    while (i <= md.size()) {
        size_t j = md.find('\n', i);
        if (j == std::string::npos) j = md.size();
        std::string line = md.substr(i, j - i);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        i = (j == md.size()) ? j + 1 : j + 1;

        if (is_blank(line)) {
            flush_paragraph(html, para_lines);
            if (in_ul) {
                html += "</ul>\n";
                in_ul = false;
            }
            continue;
        }

        // Headings (inline bold is also supported inside heading text)
        if (starts_with(line, "### ")) {
            flush_paragraph(html, para_lines);
            if (in_ul) { html += "</ul>\n"; in_ul = false; }
            html += "<h3>" + render_inline_md_bold(line.substr(4)) + "</h3>\n";
            continue;
        }
        if (starts_with(line, "## ")) {
            flush_paragraph(html, para_lines);
            if (in_ul) { html += "</ul>\n"; in_ul = false; }
            html += "<h2>" + render_inline_md_bold(line.substr(3)) + "</h2>\n";
            continue;
        }
        if (starts_with(line, "# ")) {
            flush_paragraph(html, para_lines);
            if (in_ul) { html += "</ul>\n"; in_ul = false; }
            html += "<h1>" + render_inline_md_bold(line.substr(2)) + "</h1>\n";
            continue;
        }

        // Unordered list items: "- " or "* "
        {
            std::string t = ltrim_copy(line);
            bool is_li = false;
            std::string item;

            if (t.size() >= 2 && t[0] == '-' && t[1] == ' ') {
                is_li = true;
                item = t.substr(2);
            } else if (t.size() >= 2 && t[0] == '*' && t[1] == ' ') {
                is_li = true;
                item = t.substr(2);
            }

            if (is_li) {
                flush_paragraph(html, para_lines);
                if (!in_ul) {
                    html += "<ul>\n";
                    in_ul = true;
                }
                html += "<li>" + render_inline_md_bold(item) + "</li>\n";
                continue;
            }
        }

        // Default: paragraph text (coalesce consecutive lines into one <p> with <br/>)
        if (in_ul) {
            html += "</ul>\n";
            in_ul = false;
        }
        para_lines.push_back(line);
    }

    flush_paragraph(html, para_lines);
    if (in_ul) html += "</ul>\n";

    html += "</body>\n</html>\n";
    return html;
}

void write_html(const fs::path& path, const std::string& html) {
    try {
        if (path.has_parent_path()) fs::create_directories(path.parent_path());
    } catch (...) {
    }

    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) return;
    out << html;
}

} // namespace resume
