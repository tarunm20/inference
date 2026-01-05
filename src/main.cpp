#include "tokenizer.h"
#include "inference_engine.h"
#include "text_generator.h"
#include <iostream>
#include <string>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --model <path>       Path to ONNX model (default: models/gpt2/onnx/decoder_model_merged.onnx)\n";
    std::cout << "  --vocab <path>       Path to vocab.json (default: models/gpt2/vocab.json)\n";
    std::cout << "  --merges <path>      Path to merges.txt (default: models/gpt2/merges.txt)\n";
    std::cout << "  --prompt <text>      Prompt text (default: interactive mode)\n";
    std::cout << "  --max-length <n>     Maximum tokens to generate (default: 50)\n";
    std::cout << "  --temperature <f>    Sampling temperature (default: 1.0, use 0 for greedy)\n";
    std::cout << "  --top-k <n>          Top-k sampling (default: 50, use 0 to disable)\n";
    std::cout << "  --top-p <f>          Nucleus sampling (default: 0.9, use 1.0 to disable)\n";
    std::cout << "  --help               Show this help message\n";
}

int main(int argc, char* argv[]) {
    // Default paths
    std::string model_path = "models/gpt2/onnx/decoder_model_merged.onnx";
    std::string vocab_path = "models/gpt2/vocab.json";
    std::string merges_path = "models/gpt2/merges.txt";
    std::string prompt = "";

    // Default generation config
    GenerationConfig config;
    config.max_length = 50;
    config.temperature = 1.0f;
    config.top_k = 50;
    config.top_p = 0.9f;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--merges" && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--max-length" && i + 1 < argc) {
            config.max_length = std::stoi(argv[++i]);
        } else if (arg == "--temperature" && i + 1 < argc) {
            config.temperature = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            config.top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            config.top_p = std::stof(argv[++i]);
        }
    }

    std::cout << "=== C++ Inference Engine ===" << std::endl;
    std::cout << std::endl;

    // Initialize tokenizer
    std::cout << "Loading tokenizer..." << std::endl;
    Tokenizer tokenizer;
    if (!tokenizer.load(vocab_path, merges_path)) {
        std::cerr << "Failed to load tokenizer" << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // Initialize inference engine
    InferenceEngine engine;
    if (!engine.load_model(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    std::cout << std::endl;

    // Create text generator
    TextGenerator generator(engine, tokenizer);

    // Interactive mode or single prompt
    if (prompt.empty()) {
        std::cout << "=== Interactive Mode ===" << std::endl;
        std::cout << "Enter prompts (Ctrl+C to exit)" << std::endl;
        std::cout << std::endl;

        while (true) {
            std::cout << "Prompt: ";
            if (!std::getline(std::cin, prompt)) {
                break;
            }

            if (prompt.empty()) {
                continue;
            }

            std::cout << "\nGenerated text:\n";
            std::cout << "-------------------\n";

            std::string output = generator.generate(prompt, config);

            std::cout << "\n-------------------\n";
            std::cout << std::endl;
        }
    } else {
        // Single prompt mode
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "\nGenerated text:\n";
        std::cout << "-------------------\n";

        std::string output = generator.generate(prompt, config);

        std::cout << "\n-------------------\n";
    }

    return 0;
}
