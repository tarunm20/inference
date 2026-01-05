<<<<<<< HEAD
# inference
LLM inference engine built on C++
=======
# C++ Inference Engine

A simple C++ inference engine for running ONNX models, specifically designed for GPT-2 text generation.

## Features

- **Simple Architecture**: Clean, minimal C++ implementation
- **BPE Tokenization**: Full GPT-2 tokenizer with byte-pair encoding
- **ONNX Runtime**: Efficient model inference using ONNX Runtime
- **Multiple Sampling Methods**:
  - Greedy sampling
  - Temperature-based sampling
  - Top-k sampling
  - Nucleus (top-p) sampling
- **CLI Interface**: Easy-to-use command-line interface

## Project Structure

```
inference/
├── include/
│   ├── tokenizer.h           # BPE tokenizer header
│   ├── inference_engine.h    # ONNX Runtime wrapper
│   └── text_generator.h      # Text generation with sampling
├── src/
│   ├── tokenizer.cpp
│   ├── inference_engine.cpp
│   ├── text_generator.cpp
│   └── main.cpp              # CLI application
├── models/
│   └── gpt2/
│       ├── vocab.json
│       ├── merges.txt
│       └── onnx/
│           └── decoder_model_merged.onnx
└── CMakeLists.txt
```

## Prerequisites

1. **C++17 Compiler** (GCC, Clang, MSVC)
2. **CMake** (>= 3.15)
3. **ONNX Runtime** (download from [GitHub releases](https://github.com/microsoft/onnxruntime/releases))

### Installing ONNX Runtime

**Windows:**
```bash
# Download ONNX Runtime (e.g., v1.16.3)
# Extract to a directory, e.g., C:\onnxruntime
# Set environment variable
set ONNXRUNTIME_DIR=C:\onnxruntime
```

**Linux/Mac:**
```bash
# Download and extract ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
export ONNXRUNTIME_DIR=$(pwd)/onnxruntime-linux-x64-1.16.3
```

## Build Instructions

```bash
# Create build directory
mkdir build
cd build

# Configure (set ONNXRUNTIME_DIR to your ONNX Runtime installation)
cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime

# Build
cmake --build . --config Release

# Run
./inference_engine --prompt "Once upon a time"
```

## Usage

### Basic Usage

```bash
# Interactive mode
./inference_engine

# Single prompt
./inference_engine --prompt "The meaning of life is"

# Custom settings
./inference_engine --prompt "Hello" --max-length 100 --temperature 0.8 --top-k 40
```

### Command-line Options

```
--model <path>       Path to ONNX model (default: models/gpt2/onnx/decoder_model_merged.onnx)
--vocab <path>       Path to vocab.json (default: models/gpt2/vocab.json)
--merges <path>      Path to merges.txt (default: models/gpt2/merges.txt)
--prompt <text>      Prompt text (default: interactive mode)
--max-length <n>     Maximum tokens to generate (default: 50)
--temperature <f>    Sampling temperature (default: 1.0, use 0 for greedy)
--top-k <n>          Top-k sampling (default: 50, use 0 to disable)
--top-p <f>          Nucleus sampling (default: 0.9, use 1.0 to disable)
--help               Show help message
```

### Examples

```bash
# Greedy decoding (deterministic)
./inference_engine --prompt "The quick brown fox" --temperature 0

# Creative generation with high temperature
./inference_engine --prompt "In a galaxy far away" --temperature 1.5 --max-length 100

# Top-k sampling
./inference_engine --prompt "Machine learning is" --top-k 20 --temperature 0.8
```

## Implementation Details

### Tokenizer
- Implements GPT-2's byte-level BPE algorithm
- Handles special characters and unicode properly
- Loads vocabulary and merge rules from JSON/text files

### Inference Engine
- Wraps ONNX Runtime C++ API
- Supports dynamic input shapes
- Single-threaded inference (simple and reliable)

### Text Generator
- Implements multiple sampling strategies
- Softmax with temperature scaling
- Top-k and nucleus (top-p) filtering
- Prints tokens as they're generated (streaming output)

## Performance Notes

This is a **simple** implementation focused on:
- ✅ Correctness and clarity
- ✅ Easy to understand and modify
- ✅ Minimal dependencies

For production use, consider:
- KV-cache optimization for faster generation
- Batched inference
- Quantization (INT8/FP16)
- GPU acceleration

## License

MIT License

## Acknowledgments

- [ONNX Runtime](https://onnxruntime.ai/)
- [nlohmann/json](https://github.com/nlohmann/json)
- OpenAI's GPT-2 model
>>>>>>> dev
