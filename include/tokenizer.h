#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    bool load(const std::string& vocab_path, const std::string& merges_path);

    std::vector<int> encode(const std::string& text);
    std::string decode(const std::vector<int>& tokens);

private:
    // Vocabulary: token string -> token ID
    std::unordered_map<std::string, int> vocab_;

    // Reverse vocabulary: token ID -> token string
    std::unordered_map<int, std::string> reverse_vocab_;

    // BPE merges: pair of tokens -> merged token
    std::vector<std::pair<std::string, std::string>> merges_;

    // Helper functions
    std::vector<std::string> byte_pair_encode(const std::string& token);
    std::vector<std::string> split_to_words(const std::string& text);
    std::string bytes_to_unicode_char(unsigned char byte);
    std::vector<std::string> get_pairs(const std::vector<std::string>& word);

    // Byte encoder for handling all possible bytes
    std::unordered_map<unsigned char, std::string> byte_encoder_;
    std::unordered_map<std::string, unsigned char> byte_decoder_;

    void init_byte_encoder();
};
