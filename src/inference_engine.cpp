#include "inference_engine.h"
#include <iostream>
#include <algorithm>

InferenceEngine::InferenceEngine()
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
      vocab_size_(50257) { // GPT-2 default vocab size

    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "InferenceEngine");
    session_options_ = std::make_unique<Ort::SessionOptions>();

    // Set optimization level
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
}

InferenceEngine::~InferenceEngine() {}

bool InferenceEngine::load_model(const std::string& model_path) {
    try {
        std::cout << "Loading model from: " << model_path << std::endl;

        // Create session
#ifdef _WIN32
        // Convert to wide string for Windows
        std::wstring wide_path(model_path.begin(), model_path.end());
        session_ = std::make_unique<Ort::Session>(*env_, wide_path.c_str(), *session_options_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
#endif

        // Get input names
        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();

        std::cout << "Model inputs: " << num_input_nodes << std::endl;
        for (size_t i = 0; i < num_input_nodes; i++) {
            auto input_name = session_->GetInputNameAllocated(i, allocator);
            input_names_.push_back(std::string(input_name.get()));
            std::cout << "  Input " << i << ": " << input_names_.back() << std::endl;
        }

        // Get output names
        size_t num_output_nodes = session_->GetOutputCount();
        std::cout << "Model outputs: " << num_output_nodes << std::endl;
        for (size_t i = 0; i < num_output_nodes; i++) {
            auto output_name = session_->GetOutputNameAllocated(i, allocator);
            output_names_.push_back(std::string(output_name.get()));
            std::cout << "  Output " << i << ": " << output_names_.back() << std::endl;
        }

        std::cout << "Model loaded successfully" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int64_t> InferenceEngine::get_output_shape(size_t seq_len) {
    return {1, static_cast<int64_t>(seq_len), static_cast<int64_t>(vocab_size_)};
}

std::vector<float> InferenceEngine::forward(const std::vector<int64_t>& input_ids, bool use_cache) {
    try {
        size_t seq_len = input_ids.size();

        // Prepare input tensor shape: [batch_size=1, seq_len]
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(seq_len)};

        // Create input_ids tensor
        auto input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            const_cast<int64_t*>(input_ids.data()),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Create attention_mask tensor (all ones)
        std::vector<int64_t> attention_mask(seq_len, 1);
        auto attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info_,
            attention_mask.data(),
            attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        // Create use_cache_branch tensor (bool)
        // Note: std::vector<bool> doesn't have .data(), so we use a plain bool
        bool use_cache_value = use_cache;
        std::vector<int64_t> scalar_shape = {1};
        auto use_cache_tensor = Ort::Value::CreateTensor<bool>(
            memory_info_,
            &use_cache_value,
            1,
            scalar_shape.data(),
            scalar_shape.size()
        );

        // Prepare inputs
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_ids_tensor));
        input_tensors.push_back(std::move(attention_mask_tensor));
        input_tensors.push_back(std::move(use_cache_tensor));

        // Prepare input/output names as C strings
        std::vector<const char*> input_names_cstr = {"input_ids", "attention_mask", "use_cache_branch"};

        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_) {
            output_names_cstr.push_back(name.c_str());
        }

        // Run inference
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_cstr.data(),
            input_tensors.data(),
            input_tensors.size(),
            output_names_cstr.data(),
            output_names_cstr.size()
        );

        // Get logits from output
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
        size_t output_size = output_shape_info.GetElementCount();

        // Copy output data
        std::vector<float> logits(output_data, output_data + output_size);

        return logits;

    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return {};
    }
}
