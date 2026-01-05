#pragma once

#include "inference_engine.h"
#include "tokenizer.h"
#include <string>
#include <vector>
#include <random>

struct GenerationConfig {
    int max_length = 50;           // Maximum number of tokens to generate
    float temperature = 1.0f;      // Sampling temperature (higher = more random)
    int top_k = 50;                // Top-k sampling (0 = disabled)
    float top_p = 0.9f;            // Nucleus sampling (1.0 = disabled)
    int eos_token_id = 50256;      // End of sequence token
};

class TextGenerator {
public:
    TextGenerator(InferenceEngine& engine, Tokenizer& tokenizer);
    ~TextGenerator();

    std::string generate(const std::string& prompt, const GenerationConfig& config);

private:
    InferenceEngine& engine_;
    Tokenizer& tokenizer_;
    std::mt19937 rng_;

    // Sampling methods
    int sample_greedy(const std::vector<float>& logits);
    int sample_with_temperature(const std::vector<float>& logits, float temperature);
    int sample_top_k(const std::vector<float>& logits, int k, float temperature);
    int sample_top_p(const std::vector<float>& logits, float p, float temperature);

    // Helper: apply temperature to logits and convert to probabilities
    std::vector<float> softmax(const std::vector<float>& logits, float temperature);

    // Helper: get logits for last token
    std::vector<float> get_last_token_logits(const std::vector<float>& all_logits, size_t seq_len, int vocab_size);
};
