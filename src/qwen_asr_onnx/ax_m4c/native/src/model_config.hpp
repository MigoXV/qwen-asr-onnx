#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace ax_qwen_asr {

constexpr std::size_t kDecoderLayerCount = 28;
constexpr std::size_t kVocabularySize = 151936;
constexpr std::size_t kHiddenSize = 1024;
constexpr std::size_t kEmbeddingElementBytes = 2;
constexpr std::size_t kMaxPrefillTokens = 448;
constexpr std::size_t kPrefillBlockTokens = 64;
constexpr std::size_t kMaxContextTokens = 2047;
constexpr std::size_t kMaxGeneratedTokens = 256;

class ModelConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ModelFileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct ModelConfig {
    std::filesystem::path root;
    std::filesystem::path config_file;
    std::filesystem::path conv_frontend;
    std::filesystem::path encoder;
    std::vector<std::filesystem::path> decoder_layers;
    std::filesystem::path post;
    std::filesystem::path embedding;
    std::filesystem::path tokenizer;
    std::filesystem::path post_config;
    std::string layer_template;
    std::size_t layer_count = 0;
    std::size_t vocabulary_size = 0;
    std::size_t hidden_size = 0;

    static ModelConfig Load(const std::filesystem::path& model_dir);
};

}  // namespace ax_qwen_asr
