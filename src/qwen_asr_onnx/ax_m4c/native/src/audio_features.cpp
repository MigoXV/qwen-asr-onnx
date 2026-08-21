#include "audio_features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ax_qwen_asr {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;
constexpr std::size_t kFrequencyBinCount = kFftSize / 2 + 1;
constexpr std::size_t kBluesteinSize = 1024;

double HertzToMelSlaney(double frequency) {
    constexpr double min_log_hertz = 1000.0;
    constexpr double min_log_mel = 15.0;
    constexpr double log_step = 27.0 / std::log(6.4);
    if (frequency < min_log_hertz) return 3.0 * frequency / 200.0;
    return min_log_mel + std::log(frequency / min_log_hertz) * log_step;
}

double MelToHertzSlaney(double mel) {
    constexpr double min_log_hertz = 1000.0;
    constexpr double min_log_mel = 15.0;
    constexpr double log_step = std::log(6.4) / 27.0;
    if (mel < min_log_mel) return mel * 200.0 / 3.0;
    return min_log_hertz * std::exp(log_step * (mel - min_log_mel));
}

std::size_t ReflectIndex(std::ptrdiff_t index, std::size_t length) {
    if (length <= 1) return 0;
    const std::ptrdiff_t period = static_cast<std::ptrdiff_t>(2 * (length - 1));
    std::ptrdiff_t wrapped = index % period;
    if (wrapped < 0) wrapped += period;
    if (wrapped >= static_cast<std::ptrdiff_t>(length)) wrapped = period - wrapped;
    return static_cast<std::size_t>(wrapped);
}

}  // namespace

AudioFeatureExtractor::AudioFeatureExtractor()
    : window_(kFftSize),
      mel_filters_(kMelBinCount * kFrequencyBinCount),
      bluestein_chirp_(kFftSize),
      bluestein_kernel_fft_(kBluesteinSize),
      fft_twiddles_(kBluesteinSize / 2),
      fft_bit_reverse_(kBluesteinSize) {
    for (std::size_t index = 0; index < kFftSize; ++index) {
        window_[index] = static_cast<float>(
            0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(index) /
                                 static_cast<double>(kFftSize)));
    }
    for (std::size_t index = 0; index < kBluesteinSize; ++index) {
        std::size_t value = index;
        std::size_t reversed = 0;
        for (int bit = 0; bit < 10; ++bit) {
            reversed = (reversed << 1) | (value & 1U);
            value >>= 1;
        }
        fft_bit_reverse_[index] = reversed;
    }
    for (std::size_t index = 0; index < kBluesteinSize / 2; ++index) {
        const double angle = -2.0 * kPi * static_cast<double>(index) /
                             static_cast<double>(kBluesteinSize);
        fft_twiddles_[index] = {
            static_cast<float>(std::cos(angle)),
            static_cast<float>(std::sin(angle)),
        };
    }

    for (std::size_t index = 0; index < kFftSize; ++index) {
        const double squared = static_cast<double>(index) * static_cast<double>(index);
        const double angle = kPi * squared / static_cast<double>(kFftSize);
        bluestein_chirp_[index] = {
            static_cast<float>(std::cos(-angle)),
            static_cast<float>(std::sin(-angle)),
        };
        const std::complex<float> kernel = {
            static_cast<float>(std::cos(angle)),
            static_cast<float>(std::sin(angle)),
        };
        bluestein_kernel_fft_[index] = kernel;
        if (index != 0) bluestein_kernel_fft_[kBluesteinSize - index] = kernel;
    }
    Radix2Fft(bluestein_kernel_fft_, false);

    const double mel_min = HertzToMelSlaney(0.0);
    const double mel_max = HertzToMelSlaney(kSampleRate / 2.0);
    std::vector<double> boundaries(kMelBinCount + 2);
    for (std::size_t index = 0; index < boundaries.size(); ++index) {
        const double ratio = static_cast<double>(index) /
                             static_cast<double>(kMelBinCount + 1);
        boundaries[index] = MelToHertzSlaney(mel_min + (mel_max - mel_min) * ratio);
    }
    for (std::size_t mel = 0; mel < kMelBinCount; ++mel) {
        const double left = boundaries[mel];
        const double center = boundaries[mel + 1];
        const double right = boundaries[mel + 2];
        const double norm = 2.0 / (right - left);
        for (std::size_t bin = 0; bin < kFrequencyBinCount; ++bin) {
            const double frequency = static_cast<double>(bin * kSampleRate) /
                                     static_cast<double>(kFftSize);
            const double lower = (frequency - left) / (center - left);
            const double upper = (right - frequency) / (right - center);
            mel_filters_[mel * kFrequencyBinCount + bin] =
                static_cast<float>(std::max(0.0, std::min(lower, upper)) * norm);
        }
    }
}

void AudioFeatureExtractor::Radix2Fft(
    std::vector<std::complex<float>>& values,
    bool inverse) const {
    for (std::size_t index = 0; index < kBluesteinSize; ++index) {
        const std::size_t reversed = fft_bit_reverse_[index];
        if (index < reversed) std::swap(values[index], values[reversed]);
    }
    for (std::size_t length = 2; length <= kBluesteinSize; length <<= 1) {
        const std::size_t half = length / 2;
        const std::size_t step = kBluesteinSize / length;
        for (std::size_t start = 0; start < kBluesteinSize; start += length) {
            for (std::size_t offset = 0; offset < half; ++offset) {
                std::complex<float> twiddle = fft_twiddles_[offset * step];
                if (inverse) twiddle = std::conj(twiddle);
                const auto even = values[start + offset];
                const auto odd = values[start + offset + half] * twiddle;
                values[start + offset] = even + odd;
                values[start + offset + half] = even - odd;
            }
        }
    }
    if (inverse) {
        const float scale = 1.0F / static_cast<float>(kBluesteinSize);
        for (auto& value : values) value *= scale;
    }
}

MelFeatures AudioFeatureExtractor::Compute(
    const std::int16_t* samples,
    std::size_t sample_count) const {
    MelFeatures result;
    ComputeInto(samples, sample_count, result);
    return result;
}

void AudioFeatureExtractor::ComputeInto(
    const std::int16_t* samples,
    std::size_t sample_count,
    MelFeatures& result) const {
    if (!samples) throw std::invalid_argument("samples 不能为空");
    if (sample_count == 0) throw std::invalid_argument("PCM16 不能为空");

    const std::size_t frame_count = sample_count / kHopLength + 1;
    if (frame_count > kMaxMelFrames) {
        throw std::invalid_argument(
            "音频过长：mel 帧数 " + std::to_string(frame_count) +
            " 超过固定上限 " + std::to_string(kMaxMelFrames));
    }

    result.frame_count = frame_count;
    result.values.resize(frame_count * kMelBinCount);
    std::vector<float> power(kFrequencyBinCount);
    std::vector<std::complex<float>> fft_buffer(kBluesteinSize);
    float maximum = -std::numeric_limits<float>::infinity();
    constexpr std::ptrdiff_t pad = static_cast<std::ptrdiff_t>(kFftSize / 2);

    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        const std::ptrdiff_t source_start =
            static_cast<std::ptrdiff_t>(frame * kHopLength) - pad;
        std::fill(fft_buffer.begin(), fft_buffer.end(), std::complex<float>{});
        for (std::size_t index = 0; index < kFftSize; ++index) {
            const auto sample_index = ReflectIndex(
                source_start + static_cast<std::ptrdiff_t>(index), sample_count);
            const float value =
                (static_cast<float>(samples[sample_index]) / 32768.0F) * window_[index];
            fft_buffer[index] = value * bluestein_chirp_[index];
        }
        Radix2Fft(fft_buffer, false);
        for (std::size_t index = 0; index < kBluesteinSize; ++index) {
            fft_buffer[index] *= bluestein_kernel_fft_[index];
        }
        Radix2Fft(fft_buffer, true);
        for (std::size_t bin = 0; bin < kFrequencyBinCount; ++bin) {
            const auto value = fft_buffer[bin] * bluestein_chirp_[bin];
            power[bin] = std::norm(value);
        }

        for (std::size_t mel = 0; mel < kMelBinCount; ++mel) {
            float energy = 0.0F;
            const float* filter = mel_filters_.data() + mel * kFrequencyBinCount;
            for (std::size_t bin = 0; bin < kFrequencyBinCount; ++bin) {
                energy += filter[bin] * power[bin];
            }
            const float value = std::log10(std::max(energy, 1.0e-10F));
            result.values[frame * kMelBinCount + mel] = value;
            maximum = std::max(maximum, value);
        }
    }

    const float lower = maximum - 8.0F;
    for (float& value : result.values) {
        value = (std::max(value, lower) + 4.0F) / 4.0F;
    }
}

std::size_t DownsampledAudioTokenCount(std::size_t mel_frames) {
    if (mel_frames == 0) return 0;
    std::size_t length = mel_frames;
    for (int index = 0; index < 3; ++index) length = (length - 1) / 2 + 1;
    return length;
}

}  // namespace ax_qwen_asr
