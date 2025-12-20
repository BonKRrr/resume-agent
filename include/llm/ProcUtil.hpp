#pragma once
#include <string>

namespace procutil {

// Runs a command line and returns captured stdout+stderr (merged).
// Returns "" on failure.
std::string run_capture_stdout(const std::string& cmdline_utf8);

} // namespace procutil
