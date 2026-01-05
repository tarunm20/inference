#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <iostream>
#include <regex>

using json = nlohmann::json;

Tokenizer::Tokenizer() {
    init_byte_encoder();
}

Tokenizer::~Tokenizer() {}

void Tokenizer::init_byte_encoder() {
    // GPT-2 uses a byte-level BPE encoding scheme
    // Create a mapping from bytes to unicode characters
    std::vector<int> byte_list;

    // Printable ASCII and some extended ASCII
    for (int i = 33; i <= 126; i++) byte_list.push_back(i);
    for (int i = 161; i <= 172; i++) byte_list.push_back(i);
    for (int i = 174; i <= 255; i++) byte_list.push_back(i);

    std::vector<int> char_list = byte_list;
    int n = 0;

    // Fill in remaining bytes with shifted values
    for (int b = 0; b < 256; b++) {
        if (std::find(byte_list.begin(), byte_list.end(), b) == byte_list.end()) {
            byte_list.push_back(b);
            char_list.push_back(256 + n);
            n++;
        }
    }

    // Create encoder/decoder mappings
    for (size_t i = 0; i < byte_list.size(); i++) {
        unsigned char byte = static_cast<unsigned char>(byte_list[i]);
        std::string unicode_char(1, static_cast<char>(char_list[i]));
        byte_encoder_[byte] = unicode_char;
        byte_decoder_[unicode_char] = byte;
    }
}

bool Tokenizer::load(const std::string& vocab_path, const std::string& merges_path) {
    // Load vocabulary
    std::ifstream vocab_file(vocab_path);
    if (!vocab_file.is_open()) {
        std::cerr << "Failed to open vocab file: " << vocab_path << std::endl;
        return false;
    }

    json vocab_json;
    vocab_file >> vocab_json;
    vocab_file.close();

    for (auto& [key, value] : vocab_json.items()) {
        vocab_[key] = value.get<int>();
        reverse_vocab_[value.get<int>()] = key;
    }

    std::cout << "Loaded " << vocab_.size() << " tokens from vocabulary" << std::endl;

    // Load merges
    std::ifstream merges_file(merges_path);
    if (!merges_file.is_open()) {
        std::cerr << "Failed to open merges file: " << merges_path << std::endl;
        return false;
    }

    std::string line;
    std::getline(merges_file, line); // Skip first line (version)

    while (std::getline(merges_file, line)) {
        if (line.empty()) continue;

        std::istringstream iss(line);
        std::string first, second;
        iss >> first >> second;

        if (!first.empty() && !second.empty()) {
            merges_.push_back({first, second});
        }
    }
    merges_file.close();

    std::cout << "Loaded " << merges_.size() << " merge rules" << std::endl;

    return true;
}

std::vector<std::string> Tokenizer::split_to_words(const std::string& text) {
    // Split text into words using GPT-2's pattern
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+)");

    std::vector<std::string> words;
    auto words_begin = std::sregex_iterator(text.begin(), text.end(), pattern);
    auto words_end = std::sregex_iterator();

    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        words.push_back((*i).str());
    }

    return words;
}

std::vector<std::string> Tokenizer::get_pairs(const std::vector<std::string>& word) {
    std::vector<std::string> pairs;
    if (word.size() < 2) return pairs;

    for (size_t i = 0; i < word.size() - 1; i++) {
        pairs.push_back(word[i] + " " + word[i + 1]);
    }
    return pairs;
}

std::vector<std::string> Tokenizer::byte_pair_encode(const std::string& token) {
    // Convert token to byte-level representation
    std::vector<std::string> word;
    for (unsigned char c : token) {
        word.push_back(byte_encoder_[c]);
    }

    if (word.size() == 1) return word;

    // Apply BPE merges
    while (true) {
        auto pairs = get_pairs(word);
        if (pairs.empty()) break;

        // Find the highest priority merge
        int best_merge_idx = -1;
        size_t best_merge_rank = merges_.size();

        for (size_t i = 0; i < pairs.size(); i++) {
            std::istringstream iss(pairs[i]);
            std::string first, second;
            iss >> first >> second;

            // Find this pair in merges
            for (size_t j = 0; j < merges_.size(); j++) {
                if (merges_[j].first == first && merges_[j].second == second) {
                    if (j < best_merge_rank) {
                        best_merge_rank = j;
                        best_merge_idx = i;
                    }
                    break;
                }
            }
        }

        if (best_merge_idx == -1) break;

        // Apply the merge
        std::vector<std::string> new_word;
        size_t i = 0;
        while (i < word.size()) {
            if (i == best_merge_idx && i + 1 < word.size()) {
                new_word.push_back(word[i] + word[i + 1]);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = new_word;

        if (word.size() == 1) break;
    }

    return word;
}

std::vector<int> Tokenizer::encode(const std::string& text) {
    std::vector<int> token_ids;

    // Split text into words
    auto words = split_to_words(text);

    // Apply BPE to each word
    for (const auto& word : words) {
        auto bpe_tokens = byte_pair_encode(word);

        for (const auto& token : bpe_tokens) {
            auto it = vocab_.find(token);
            if (it != vocab_.end()) {
                token_ids.push_back(it->second);
            } else {
                std::cerr << "Warning: Unknown token '" << token << "'" << std::endl;
            }
        }
    }

    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
    std::string text;

    for (int token_id : tokens) {
        auto it = reverse_vocab_.find(token_id);
        if (it != reverse_vocab_.end()) {
            text += it->second;
        }
    }

    // Decode from byte-level representation
    // Need to handle multi-byte unicode characters properly
    std::string result;
    size_t i = 0;
    while (i < text.length()) {
        // Try to find the longest matching sequence in byte_decoder
        bool found = false;

        // Check for multi-character sequences first (up to 4 bytes for UTF-8)
        for (int len = std::min(static_cast<int>(text.length() - i), 4); len > 0; len--) {
            std::string char_str = text.substr(i, len);
            auto it = byte_decoder_.find(char_str);
            if (it != byte_decoder_.end()) {
                result += static_cast<char>(it->second);
                i += len;
                found = true;
                break;
            }
        }

        if (!found) {
            // If no match found, just copy the character as-is
            result += text[i];
            i++;
        }
    }

    return result;
}
