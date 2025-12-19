#include "emb/MiniLmEmbedder.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>

bool MiniLmEmbedder::init(const std::string& model_path, const std::string& vocab_path) {
    if (!m_tok.load_vocab(vocab_path)) {
        std::cerr << "MiniLmEmbedder: failed to load vocab: " << vocab_path << "\n";
        return false;
    }

    try {
        m_opts.SetIntraOpNumThreads(1);
        m_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        // Windows/MSVC wants wide path ctor (you hit this earlier)
        std::wstring wmodel(model_path.begin(), model_path.end());
        m_session = std::make_unique<Ort::Session>(m_env, wmodel.c_str(), m_opts);

        Ort::AllocatorWithDefaultOptions allocator;
        auto name_alloc = m_session->GetOutputNameAllocated(0, allocator);
        m_out_name = name_alloc.get();

        // (optional but super useful) capture input names too:
        auto in0 = m_session->GetInputNameAllocated(0, allocator);
        auto in1 = m_session->GetInputNameAllocated(1, allocator);
        auto in2 = m_session->GetInputNameAllocated(2, allocator);
        m_in_ids = in0.get();
        m_in_mask = in1.get();
        m_in_type = in2.get();

        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "MiniLmEmbedder ORT exception: " << e.what() << "\n";
        std::cerr << "model_path=" << model_path << "\n";
        return false;
    }
}

static void l2_normalize(std::vector<float>& v) {
    double ss = 0.0;
    for (float x : v) ss += (double)x * (double)x;
    if (ss <= 0.0) return;
    double inv = 1.0 / std::sqrt(ss);
    for (float& x : v) x = (float)(x * inv);
}

std::vector<float> MiniLmEmbedder::embed(const std::string& text, size_t max_len) const {
    if (!m_session) return {};

    std::vector<int64_t> ids = m_tok.encode(text, max_len);
    const size_t seq_len = ids.size();

    std::vector<int64_t> mask(seq_len, 1);
    std::vector<int64_t> type_ids(seq_len, 0);

    std::vector<int64_t> shape{1, (int64_t)seq_len};

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value in_ids  = Ort::Value::CreateTensor<int64_t>(mem, ids.data(), ids.size(), shape.data(), shape.size());
    Ort::Value in_mask = Ort::Value::CreateTensor<int64_t>(mem, mask.data(), mask.size(), shape.data(), shape.size());
    Ort::Value in_type = Ort::Value::CreateTensor<int64_t>(mem, type_ids.data(), type_ids.size(), shape.data(), shape.size());

    const char* in_names[3] = { m_in_ids.c_str(), m_in_mask.c_str(), m_in_type.c_str() };
    Ort::Value in_vals[3] = { std::move(in_ids), std::move(in_mask), std::move(in_type) };

    const char* out_names[1] = { m_out_name.c_str() };

    auto outs = m_session->Run(Ort::RunOptions{nullptr}, in_names, in_vals, 3, out_names, 1);

    Ort::Value& out = outs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shp = info.GetShape(); // [1, seq_len, hidden]
    if (shp.size() != 3) return {};

    const int64_t hidden = shp[2];
    const float* data = out.GetTensorData<float>();

    std::vector<float> pooled((size_t)hidden, 0.0f);
    double denom = 0.0;

    // output is contiguous as [1, seq_len, hidden]
    for (size_t t = 0; t < seq_len; ++t) {
        if (mask[t] == 0) continue;
        denom += 1.0;
        const float* row = data + (t * (size_t)hidden);
        for (size_t j = 0; j < (size_t)hidden; ++j) pooled[j] += row[j];
    }

    if (denom > 0.0) {
        float inv = (float)(1.0 / denom);
        for (float& x : pooled) x *= inv;
    }

    l2_normalize(pooled);
    return pooled;
}
