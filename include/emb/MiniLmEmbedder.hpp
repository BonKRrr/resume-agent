#pragma once
#include "emb/WordPieceTokenizer.hpp"
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

class MiniLmEmbedder {
public:
    bool init(const std::string& model_path, const std::string& vocab_path);

    // L2-normalized embedding
    std::vector<float> embed(const std::string& text, size_t max_len = 256) const;

private:
    WordPieceTokenizer m_tok;

    Ort::Env m_env{ORT_LOGGING_LEVEL_WARNING, "resume-agent"};
    Ort::SessionOptions m_opts;
    std::unique_ptr<Ort::Session> m_session;

    std::string m_in_ids = "input_ids";
    std::string m_in_mask = "attention_mask";
    std::string m_in_type = "token_type_ids";
    std::string m_out_name;
};
