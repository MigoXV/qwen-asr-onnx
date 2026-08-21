#pragma once

#include <cstddef>
#include <complex>
#include <cstdint>
#include <vector>

namespace ax_qwen_asr {

constexpr int kSampleRate = 16000;
constexpr std::size_t kFftSize = 400;
constexpr std::size_t kHopLength = 160;
constexpr std::size_t kMelBinCount = 128;
constexpr std::size_t kMaxMelFrames = 3000;

struct MelFeatures {
    // frame-major: [frame_count, 128]，不含补到 3000 帧的尾部零。
    std::vector<float> values;
    std::size_t frame_count = 0;
};

class AudioFeatureExtractor {
public:
    AudioFeatureExtractor();
    MelFeatures Compute(const std::int16_t* samples, std::size_t sample_count) const;
    void ComputeInto(
        const std::int16_t* samples,
        std::size_t sample_count,
        MelFeatures& output) const;

private:
    void Radix2Fft(std::vector<std::complex<float>>& values, bool inverse) const;

    std::vector<float> window_;
    std::vector<float> mel_filters_;
    std::vector<std::complex<float>> bluestein_chirp_;
    std::vector<std::complex<float>> bluestein_kernel_fft_;
    std::vector<std::complex<float>> fft_twiddles_;
    std::vector<std::size_t> fft_bit_reverse_;
};

std::size_t DownsampledAudioTokenCount(std::size_t mel_frames);

}  // namespace ax_qwen_asr
