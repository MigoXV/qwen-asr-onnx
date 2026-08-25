#include "qwen_asr_runner.hpp"

#include <filesystem>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ax_qwen_asr {

class QwenAsrRunner::Impl {
public:
    explicit Impl(const std::string& model_dir) {
        if (!std::filesystem::is_directory(model_dir)) {
            throw std::invalid_argument("fake runner model_dir 必须是目录");
        }
    }
    RunnerMetrics metrics;
};

QwenAsrRunner::QwenAsrRunner(const std::string& model_dir)
    : impl_(std::make_unique<Impl>(model_dir)) {}
QwenAsrRunner::~QwenAsrRunner() = default;

void QwenAsrRunner::Warmup() {
    impl_->metrics = RunnerMetrics{};
}

std::string QwenAsrRunner::Transcribe(
    const std::int16_t*,
    std::size_t sample_count,
    int sample_rate,
    const TokenCallback& callback) {
    if (sample_rate != 16000) throw std::invalid_argument("只接受 16000 Hz");
    impl_->metrics.sample_count = sample_count;
    impl_->metrics.mel_frame_count = sample_count / 160 + 1;
    impl_->metrics.audio_token_count = 65;
    impl_->metrics.prompt_token_count = 80;
    impl_->metrics.generated_token_count = 32;
    impl_->metrics.preprocess_ms = 1.0;
    impl_->metrics.encoder_ms = 2.0;
    impl_->metrics.decoder_ttft_ms = 3.0;
    impl_->metrics.decoder_ms = 4.0;
    impl_->metrics.native_total_ms = 7.0;
    const std::vector<std::pair<std::uint32_t, std::string>> tokens = {
        {100, "language "},
        {101, "Chinese"},
        {102, "<asr_text>"},
        {103, "在同一个话题上纠缠不清的人，"},
        {104, "是不会讨女孩子喜欢的。"},
    };
    std::string text;
    std::size_t generated_count = 0;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        text += tokens[index].second;
        generated_count = index + 1;
        if (callback && !callback(tokens[index].first, index, tokens[index].second)) {
            break;
        }
    }
    impl_->metrics.generated_token_count = generated_count;
    return text;
}

const RunnerMetrics& QwenAsrRunner::LastMetrics() const noexcept {
    return impl_->metrics;
}

}  // namespace ax_qwen_asr
