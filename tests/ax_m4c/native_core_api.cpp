#include "audio_features.hpp"
#include "model_config.hpp"
#include "prompt.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <string>

namespace {
thread_local std::string g_error;
}

extern "C" int ax_qwen_asr_test_compute_mel(
    const std::int16_t* samples,
    std::size_t sample_count,
    float* output,
    std::size_t output_capacity,
    std::size_t* frame_count,
    std::size_t* required_values) {
    if (!samples || sample_count == 0 || !frame_count || !required_values) return 1;
    try {
        static const ax_qwen_asr::AudioFeatureExtractor extractor;
        const auto mel = extractor.Compute(samples, sample_count);
        *frame_count = mel.frame_count;
        *required_values = mel.values.size();
        if (!output || output_capacity < mel.values.size()) return 2;
        std::copy(mel.values.begin(), mel.values.end(), output);
        g_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        g_error = exception.what();
        return 3;
    }
}

extern "C" int ax_qwen_asr_test_load_config(const char* model_dir) {
    if (!model_dir) return 1;
    try {
        (void)ax_qwen_asr::ModelConfig::Load(model_dir);
        g_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        g_error = exception.what();
        return 2;
    }
}

extern "C" int ax_qwen_asr_test_prompt(
    std::size_t audio_tokens,
    std::uint32_t* output,
    std::size_t output_capacity,
    std::size_t* required_values) {
    if (!required_values) return 1;
    try {
        const auto ids = ax_qwen_asr::BuildPromptIds(audio_tokens);
        *required_values = ids.size();
        if (!output || output_capacity < ids.size()) return 2;
        std::copy(ids.begin(), ids.end(), output);
        g_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        g_error = exception.what();
        return 3;
    }
}

extern "C" const char* ax_qwen_asr_test_last_error() {
    return g_error.c_str();
}
