#pragma once

#include <filesystem>
#include <string>

namespace resume {

// Convert a Markdown string (your existing render_markdown output)
// into simple HTML intended for copy/paste into Google Docs.
std::string render_html_from_markdown(const std::string& md);

// Write HTML to disk.
void write_html(const std::filesystem::path& path, const std::string& html);

} // namespace resume
