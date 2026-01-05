#pragma once

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();

    bool load_model(const std::string& model_path);

    // Run a single forward pass
    // input_ids: [1, seq_len] - token IDs
    // Returns: logits [1, seq_len, vocab_size]
    std::vector<float> forward(const std::vector<int64_t>& input_ids, bool use_cache = false);

    int get_vocab_size() const { return vocab_size_; }

private:
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::MemoryInfo memory_info_;

    // Model metadata
    int vocab_size_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    // Helper to get output shape
    std::vector<int64_t> get_output_shape(size_t seq_len);
};
