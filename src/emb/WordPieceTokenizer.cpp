#include "emb/WordPieceTokenizer.hpp"
#include <cctype>
#include <fstream>

bool WordPieceTokenizer::load_vocab(const std::string& vocab_path) {
    std::ifstream in(vocab_path);
    if (!in) return false;

    m_id_to_tok.clear();
    m_tok_to_id.clear();

    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        int64_t id = (int64_t)m_id_to_tok.size();
        m_id_to_tok.push_back(line);
        m_tok_to_id.emplace(line, id);
    }
    return !m_id_to_tok.empty();
}

int64_t WordPieceTokenizer::id_or(int64_t def, const std::string& tok) const {
    auto it = m_tok_to_id.find(tok);
    return it == m_tok_to_id.end() ? def : it->second;
}

bool WordPieceTokenizer::is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool WordPieceTokenizer::is_punct(char c) {
    unsigned char uc = (unsigned char)c;
    return ((uc >= 33 && uc <= 47) || (uc >= 58 && uc <= 64) ||
            (uc >= 91 && uc <= 96) || (uc >= 123 && uc <= 126));
}

std::string WordPieceTokenizer::lower_ascii(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

std::vector<std::string> WordPieceTokenizer::basic_tokenize(const std::string& text) const {
    std::vector<std::string> out;
    std::string s = lower_ascii(text);

    std::string cur;
    auto flush = [&](){
        if (!cur.empty()) { out.push_back(cur); cur.clear(); }
    };

    for (char c : s) {
        if (is_ws(c)) {
            flush();
        } else if (is_punct(c)) {
            flush();
            out.emplace_back(1, c);
        } else {
            cur.push_back(c);
        }
    }
    flush();
    return out;
}

std::vector<std::string> WordPieceTokenizer::wordpiece(const std::string& token) const {
    if (token.empty()) return {"[UNK]"};

    std::vector<std::string> pieces;
    size_t start = 0;

    while (start < token.size()) {
        size_t end = token.size();
        std::string best;

        while (end > start) {
            std::string sub = token.substr(start, end - start);
            if (start > 0) sub = "##" + sub;

            if (m_tok_to_id.find(sub) != m_tok_to_id.end()) {
                best = sub;
                break;
            }
            --end;
        }

        if (best.empty()) return {"[UNK]"};
        pieces.push_back(best);
        start = end;
    }

    return pieces;
}

std::vector<int64_t> WordPieceTokenizer::encode(const std::string& text, size_t max_len) const {
    int64_t cls = cls_id(), sep = sep_id(), unk = unk_id();

    std::vector<int64_t> ids;
    ids.reserve(max_len);
    ids.push_back(cls);

    auto basic = basic_tokenize(text);
    for (const auto& t : basic) {
        auto pieces = wordpiece(t);
        for (const auto& p : pieces) {
            if (ids.size() + 1 >= max_len) break; // keep room for [SEP]
            ids.push_back(id_or(unk, p));
        }
        if (ids.size() + 1 >= max_len) break;
    }

    ids.push_back(sep);
    return ids;
}
