#pragma once
#include <string>
#include <unordered_map>
#include <vector>

class WordPieceTokenizer {
public:
    bool load_vocab(const std::string& vocab_path);

    // Returns token ids including [CLS] ... [SEP], truncated to max_len
    std::vector<int64_t> encode(const std::string& text, size_t max_len) const;

    int64_t pad_id() const { return id_or(-1, "[PAD]"); }
    int64_t unk_id() const { return id_or(-1, "[UNK]"); }
    int64_t cls_id() const { return id_or(-1, "[CLS]"); }
    int64_t sep_id() const { return id_or(-1, "[SEP]"); }

private:
    std::vector<std::string> m_id_to_tok;
    std::unordered_map<std::string, int64_t> m_tok_to_id;

    static bool is_ws(char c);
    static bool is_punct(char c);
    static std::string lower_ascii(std::string s);

    std::vector<std::string> basic_tokenize(const std::string& text) const;
    std::vector<std::string> wordpiece(const std::string& token) const;

    int64_t id_or(int64_t def, const std::string& tok) const;
};
