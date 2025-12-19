#pragma once
#include <string>
#include <vector>

namespace textutil {

// lowercase, keep letters/digits/+/#, turn everything else into spaces, collapse spaces
std::string normalize(const std::string& s);

// split normalized text into tokens, drop very short junk tokens
std::vector<std::string> tokenize(const std::string& normalized);

// synonym folding + phrase merging
std::vector<std::string> normalize_tokens(const std::vector<std::string>& tokens);

}