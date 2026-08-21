#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace ax_qwen_asr {

class AxEngineError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct RunnerMetrics {
    std::size_t sample_count = 0;
    std::size_t mel_frame_count = 0;
    std::size_t audio_token_count = 0;
    std::size_t prompt_token_count = 0;
    std::size_t generated_token_count = 0;
    double preprocess_ms = 0.0;
    double encoder_ms = 0.0;
    double decoder_ttft_ms = 0.0;
    double decoder_ms = 0.0;
    double native_total_ms = 0.0;
};

class QwenAsrRunner {
public:
    explicit QwenAsrRunner(const std::string& model_dir);
    ~QwenAsrRunner();

    QwenAsrRunner(const QwenAsrRunner&) = delete;
    QwenAsrRunner& operator=(const QwenAsrRunner&) = delete;

    void Warmup();
    std::string Transcribe(
        const std::int16_t* samples,
        std::size_t sample_count,
        int sample_rate);
    const RunnerMetrics& LastMetrics() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ax_qwen_asr
