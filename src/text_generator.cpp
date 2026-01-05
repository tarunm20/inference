#include "text_generator.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

TextGenerator::TextGenerator(InferenceEngine& engine, Tokenizer& tokenizer)
    : engine_(engine), tokenizer_(tokenizer) {

    // Seed random number generator
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_.seed(static_cast<unsigned int>(seed));
}

TextGenerator::~TextGenerator() {}

std::vector<float> TextGenerator::get_last_token_logits(
    const std::vector<float>& all_logits,
    size_t seq_len,
    int vocab_size) {

    // all_logits shape: [1, seq_len, vocab_size]
    // We want the logits for the last token: [vocab_size]
    size_t offset = (seq_len - 1) * vocab_size;
    std::vector<float> last_logits(vocab_size);

    for (int i = 0; i < vocab_size; i++) {
        last_logits[i] = all_logits[offset + i];
    }

    return last_logits;
}

std::vector<float> TextGenerator::softmax(const std::vector<float>& logits, float temperature) {
    std::vector<float> probs(logits.size());

    // Apply temperature
    float max_logit = *std::max_element(logits.begin(), logits.end());

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); i++) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum += probs[i];
    }

    // Normalize
    for (size_t i = 0; i < probs.size(); i++) {
        probs[i] /= sum;
    }

    return probs;
}

int TextGenerator::sample_greedy(const std::vector<float>& logits) {
    auto max_it = std::max_element(logits.begin(), logits.end());
    return static_cast<int>(std::distance(logits.begin(), max_it));
}

int TextGenerator::sample_with_temperature(const std::vector<float>& logits, float temperature) {
    auto probs = softmax(logits, temperature);

    // Sample from categorical distribution
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return dist(rng_);
}

int TextGenerator::sample_top_k(const std::vector<float>& logits, int k, float temperature) {
    // Create indices sorted by logit value (descending)
    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
        [&logits](int a, int b) { return logits[a] > logits[b]; });

    // Filter to top-k
    std::vector<float> top_k_logits(k);
    for (int i = 0; i < k; i++) {
        top_k_logits[i] = logits[indices[i]];
    }

    // Sample from top-k
    auto probs = softmax(top_k_logits, temperature);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int selected_idx = dist(rng_);

    return indices[selected_idx];
}

int TextGenerator::sample_top_p(const std::vector<float>& logits, float p, float temperature) {
    // Convert to probabilities
    auto probs = softmax(logits, temperature);

    // Create indices sorted by probability (descending)
    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
        [&probs](int a, int b) { return probs[a] > probs[b]; });

    // Find nucleus (top-p)
    float cumsum = 0.0f;
    size_t nucleus_size = 0;

    for (size_t i = 0; i < indices.size(); i++) {
        cumsum += probs[indices[i]];
        nucleus_size++;
        if (cumsum >= p) break;
    }

    // Sample from nucleus
    std::vector<float> nucleus_probs(nucleus_size);
    float nucleus_sum = 0.0f;

    for (size_t i = 0; i < nucleus_size; i++) {
        nucleus_probs[i] = probs[indices[i]];
        nucleus_sum += nucleus_probs[i];
    }

    // Renormalize
    for (size_t i = 0; i < nucleus_size; i++) {
        nucleus_probs[i] /= nucleus_sum;
    }

    std::discrete_distribution<int> dist(nucleus_probs.begin(), nucleus_probs.end());
    int selected_idx = dist(rng_);

    return indices[selected_idx];
}

std::string TextGenerator::generate(const std::string& prompt, const GenerationConfig& config) {
    std::cout << "Encoding prompt..." << std::endl;

    // Encode the prompt
    std::vector<int> token_ids = tokenizer_.encode(prompt);
    std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());

    std::cout << "Prompt tokens: " << input_ids.size() << std::endl;
    std::cout << "Generating..." << std::endl;

    int vocab_size = engine_.get_vocab_size();

    // Generation loop
    for (int i = 0; i < config.max_length; i++) {
        // Run forward pass
        auto logits = engine_.forward(input_ids);

        if (logits.empty()) {
            std::cerr << "Error: empty logits returned" << std::endl;
            break;
        }

        // Get logits for last token
        auto last_logits = get_last_token_logits(logits, input_ids.size(), vocab_size);

        // Sample next token
        int next_token;
        if (config.temperature == 0.0f) {
            next_token = sample_greedy(last_logits);
        } else if (config.top_k > 0 && config.top_k < vocab_size) {
            next_token = sample_top_k(last_logits, config.top_k, config.temperature);
        } else if (config.top_p < 1.0f) {
            next_token = sample_top_p(last_logits, config.top_p, config.temperature);
        } else {
            next_token = sample_with_temperature(last_logits, config.temperature);
        }

        // Check for EOS token
        if (next_token == config.eos_token_id) {
            std::cout << "\nReached EOS token" << std::endl;
            break;
        }

        // Append to sequence
        input_ids.push_back(next_token);

        // Print token (optional, for debugging)
        std::vector<int> single_token = {next_token};
        std::string token_text = tokenizer_.decode(single_token);
        std::cout << token_text << std::flush;
    }

    std::cout << std::endl;

    // Decode all tokens
    std::vector<int> all_tokens(input_ids.begin(), input_ids.end());
    return tokenizer_.decode(all_tokens);
}
