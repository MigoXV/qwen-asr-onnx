#include "qwen_asr_runner.hpp"

#include "audio_features.hpp"
#include "model_config.hpp"
#include "prompt.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if !defined(AX_QWEN_ASR_WITH_AXENGINE)

namespace ax_qwen_asr {

class QwenAsrRunner::Impl {
public:
    explicit Impl(const std::string&) {
        throw AxEngineError("libax_qwen_asr 构建时未启用 AXEngine 设备后端");
    }
    RunnerMetrics metrics;
};

QwenAsrRunner::QwenAsrRunner(const std::string& model_dir)
    : impl_(std::make_unique<Impl>(model_dir)) {}
QwenAsrRunner::~QwenAsrRunner() = default;
void QwenAsrRunner::Warmup() { throw AxEngineError("AXEngine 设备后端不可用"); }
std::string QwenAsrRunner::Transcribe(const std::int16_t*, std::size_t, int) {
    throw AxEngineError("AXEngine 设备后端不可用");
}
const RunnerMetrics& QwenAsrRunner::LastMetrics() const noexcept { return impl_->metrics; }

}  // namespace ax_qwen_asr

#else

#include "ax_model_runner_ax650.hpp"
#include "tokenizer/tokenizer_optimized.hpp"

#include <ax_engine_api.h>
#include <ax_sys_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ax_qwen_asr {
namespace {

using Clock = std::chrono::steady_clock;

double Milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::uint16_t FloatToBf16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<std::uint16_t>(bits >> 16);
}

std::uint16_t FloatToBf16RoundToNearestEven(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(bits >> 16);
}

float Bf16ToFloat(std::uint16_t value) {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

class AxRuntime {
public:
    AxRuntime() {
        const int system_result = AX_SYS_Init();
        if (system_result != 0) {
            throw AxEngineError("AX_SYS_Init 失败：0x" + Hex(system_result));
        }
        system_initialized_ = true;

        AX_ENGINE_NPU_ATTR_T attributes{};
        attributes.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        const int engine_result = AX_ENGINE_Init(&attributes);
        if (engine_result != 0) {
            AX_SYS_Deinit();
            system_initialized_ = false;
            throw AxEngineError("AX_ENGINE_Init 失败：0x" + Hex(engine_result));
        }
        engine_initialized_ = true;
    }

    ~AxRuntime() {
        if (engine_initialized_) AX_ENGINE_Deinit();
        if (system_initialized_) AX_SYS_Deinit();
    }

    AxRuntime(const AxRuntime&) = delete;
    AxRuntime& operator=(const AxRuntime&) = delete;

private:
    static std::string Hex(int value) {
        constexpr char digits[] = "0123456789abcdef";
        std::uint32_t number = static_cast<std::uint32_t>(value);
        std::string output(8, '0');
        for (int index = 7; index >= 0; --index) {
            output[static_cast<std::size_t>(index)] = digits[number & 0xF];
            number >>= 4;
        }
        return output;
    }

    bool system_initialized_ = false;
    bool engine_initialized_ = false;
};

class MappedEmbedding {
public:
    explicit MappedEmbedding(const std::filesystem::path& path) {
        descriptor_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (descriptor_ < 0) throw ModelFileError("无法打开 embedding：" + path.string());
        struct stat status {};
        if (fstat(descriptor_, &status) != 0 || status.st_size <= 0) {
            close(descriptor_);
            descriptor_ = -1;
            throw ModelFileError("无法读取 embedding 尺寸：" + path.string());
        }
        size_ = static_cast<std::size_t>(status.st_size);
        address_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, descriptor_, 0);
        close(descriptor_);
        descriptor_ = -1;
        if (address_ == MAP_FAILED) {
            address_ = nullptr;
            throw ModelFileError("mmap embedding 失败：" + path.string());
        }
    }

    ~MappedEmbedding() {
        if (address_) munmap(address_, size_);
        if (descriptor_ >= 0) close(descriptor_);
    }

    const std::uint16_t* Row(std::uint32_t token) const {
        if (token >= kVocabularySize) {
            throw std::runtime_error("生成 token 超出固定词表：" + std::to_string(token));
        }
        return static_cast<const std::uint16_t*>(address_) +
               static_cast<std::size_t>(token) * kHiddenSize;
    }

private:
    int descriptor_ = -1;
    void* address_ = nullptr;
    std::size_t size_ = 0;
};

const ax_runner_tensor_t& RequireInput(
    ax_runner_ax650& runner,
    int group,
    const std::string& name) {
    const auto& tensor = runner.get_input(group, name);
    if (!tensor.pVirAddr || tensor.phyAddr == 0 || tensor.nSize <= 0) {
        throw AxEngineError("输入 tensor 无设备 buffer：" + name);
    }
    return tensor;
}

const ax_runner_tensor_t& RequireOutput(
    ax_runner_ax650& runner,
    int group,
    const std::string& name) {
    const ax_runner_tensor_t* tensor_pointer = nullptr;
    try {
        tensor_pointer = &runner.get_output(group, name);
    } catch (const std::runtime_error&) {
        // Conv/encoder/post 的导出工具可能保留不同的单输出名称；单输出模型按
        // 唯一输出定位仍是无歧义的。decoder 多输出始终按固定名称校验。
        if (runner.get_num_outputs(group) != 1) throw;
        tensor_pointer = &runner.get_output(group, 0);
    }
    const auto& tensor = *tensor_pointer;
    if (!tensor.pVirAddr || tensor.phyAddr == 0 || tensor.nSize <= 0) {
        throw AxEngineError("输出 tensor 无设备 buffer：" + name);
    }
    return tensor;
}

void RequireBytes(
    const ax_runner_tensor_t& tensor,
    std::size_t expected,
    const std::string& description) {
    if (static_cast<std::size_t>(tensor.nSize) != expected) {
        throw ModelConfigError(
            description + " shape/字节数不符合固定 profile：期望 " +
            std::to_string(expected) + "，实际 " + std::to_string(tensor.nSize));
    }
}

void Run(ax_runner_ax650& runner, int group, const std::string& description) {
    const int result = runner.inference(group);
    if (result != 0) {
        throw AxEngineError(description + " 推理失败，AXEngine 错误码 " + std::to_string(result));
    }
}

void InitializeSession(ax_runner_ax650& runner, const std::filesystem::path& path) {
    const int result = runner.init(path.c_str(), -1);
    if (result != 0) {
        throw AxEngineError("加载 AXModel 失败：" + path.string() +
                            "，错误码 " + std::to_string(result));
    }
    runner.set_auto_sync_before_inference(true);
    runner.set_auto_sync_after_inference(true);
}

std::string DecodeAsrToken(
    const MNN::Transformer::Tokenizer& tokenizer,
    std::uint32_t token) {
    if (token == 151675) return "<non_speech>";
    if (token == 151704) return "<asr_text>";
    if (token >= 151669 && token <= 151703) return {};
    return tokenizer.decode(static_cast<int>(token));
}

std::unique_ptr<MNN::Transformer::Tokenizer> CreateTokenizerQuietly(
    const std::string& path) {
    // tokenizer.axera 的 createTokenizer() 会无条件 printf tokenizer_type。
    // 本后端单进程、单 handle，初始化期间短暂重定向 stdout，避免污染调用者
    // 要求的原始 ASR 文本输出；推理阶段不做任何 stdout 重定向。
    std::fflush(stdout);
    const int saved_stdout = dup(STDOUT_FILENO);
    const int null_output = open("/dev/null", O_WRONLY | O_CLOEXEC);
    if (saved_stdout >= 0 && null_output >= 0) dup2(null_output, STDOUT_FILENO);
    auto* tokenizer = MNN::Transformer::Tokenizer::createTokenizer(path);
    std::fflush(stdout);
    if (saved_stdout >= 0) {
        dup2(saved_stdout, STDOUT_FILENO);
        close(saved_stdout);
    }
    if (null_output >= 0) close(null_output);
    return std::unique_ptr<MNN::Transformer::Tokenizer>(tokenizer);
}

}  // namespace

class QwenAsrRunner::Impl {
public:
    explicit Impl(const std::string& model_dir)
        : config_(ModelConfig::Load(model_dir)),
          runtime_(std::make_unique<AxRuntime>()),
          convolution_(std::make_unique<ax_runner_ax650>()),
          encoder_(std::make_unique<ax_runner_ax650>()),
          post_(std::make_unique<ax_runner_ax650>()),
          embedding_(std::make_unique<MappedEmbedding>(config_.embedding)) {
        mel_.values.reserve(kMaxMelFrames * kMelBinCount);
        prompt_hidden_.resize(kMaxPrefillTokens * kHiddenSize);
        block_hidden_.resize(kPrefillBlockTokens * kHiddenSize);
        decode_hidden_.resize(kHiddenSize);

        tokenizer_ = CreateTokenizerQuietly(config_.tokenizer.string());
        if (!tokenizer_) throw ModelFileError("加载 Qwen3 tokenizer 失败：" + config_.tokenizer.string());
        VerifyPromptTokenizer();

        InitializeSession(*convolution_, config_.conv_frontend);
        ValidateConvolution();
        InitializeSession(*encoder_, config_.encoder);
        ValidateEncoder();

        layers_.reserve(kDecoderLayerCount);
        for (const auto& path : config_.decoder_layers) {
            auto layer = std::make_unique<ax_runner_ax650>();
            InitializeSession(*layer, path);
            ValidateDecoder(*layer);
            layers_.push_back(std::move(layer));
        }
        InitializeSession(*post_, config_.post);
        ValidatePost();
    }

    void Warmup() {
        std::vector<std::int16_t> silence(kSampleRate, 0);
        (void)TranscribeInternal(silence.data(), silence.size(), 1);
    }

    std::string Transcribe(
        const std::int16_t* samples,
        std::size_t sample_count,
        int sample_rate) {
        if (sample_rate != kSampleRate) {
            throw std::invalid_argument(
                "固定 profile 只接受 16000 Hz PCM16，收到 " + std::to_string(sample_rate));
        }
        return TranscribeInternal(samples, sample_count, kMaxGeneratedTokens);
    }

    const RunnerMetrics& LastMetrics() const noexcept { return metrics_; }

private:
    void VerifyPromptTokenizer() {
        for (const auto& expected : {
                 std::pair<const char*, int>{"system", static_cast<int>(kSystemId)},
                 {"user", static_cast<int>(kUserId)},
                 {"assistant", static_cast<int>(kAssistantId)}}) {
            const auto ids = tokenizer_->encode(expected.first);
            if (ids.size() != 1 || ids.front() != expected.second) {
                throw ModelConfigError(
                    std::string("tokenizer 与固定 prompt 不匹配：") + expected.first);
            }
        }
    }

    void ValidateConvolution() {
        if (convolution_->get_num_input_groups() != 1 ||
            convolution_->get_num_output_groups() != 1) {
            throw ModelConfigError("conv_frontend 必须只有一个 shape group");
        }
        RequireBytes(
            RequireInput(*convolution_, 0, "input_features"),
            kMaxMelFrames * kMelBinCount * sizeof(float),
            "conv_frontend.input_features");
        const auto& output = RequireOutput(*convolution_, 0, "output");
        RequireBytes(output, 390 * 896 * sizeof(float), "conv_frontend.output");
    }

    void ValidateEncoder() {
        if (encoder_->get_num_input_groups() != 1 || encoder_->get_num_output_groups() != 1) {
            throw ModelConfigError("encoder 必须只有一个 shape group");
        }
        RequireBytes(
            RequireInput(*encoder_, 0, "input_features"),
            390 * 896 * sizeof(float),
            "encoder.input_features");
        RequireBytes(
            RequireInput(*encoder_, 0, "feature_attention_mask"),
            390,
            "encoder.feature_attention_mask");
        RequireBytes(
            RequireOutput(*encoder_, 0, "output"),
            390 * kHiddenSize * sizeof(float),
            "encoder.output");
    }

    void ValidateDecoder(ax_runner_ax650& layer) {
        if (layer.get_num_input_groups() != 9 || layer.get_num_output_groups() != 9) {
            throw ModelConfigError(
                "decoder 固定要求 9 个 shape group（decode + C64/P448 history），实际 input=" +
                std::to_string(layer.get_num_input_groups()) + " output=" +
                std::to_string(layer.get_num_output_groups()));
        }
        for (int group = 0; group < 9; ++group) {
            const std::size_t tokens = group == 0 ? 1 : kPrefillBlockTokens;
            RequireBytes(
                RequireInput(layer, group, "input"),
                tokens * kHiddenSize * sizeof(std::uint16_t),
                "decoder.input");
            RequireBytes(
                RequireOutput(layer, group, "output"),
                tokens * kHiddenSize * sizeof(std::uint16_t),
                "decoder.output");
            RequireBytes(
                RequireOutput(layer, group, "K_cache_out"),
                tokens * kHiddenSize * sizeof(std::uint16_t),
                "decoder.K_cache_out");
            RequireBytes(
                RequireOutput(layer, group, "V_cache_out"),
                tokens * kHiddenSize * sizeof(std::uint16_t),
                "decoder.V_cache_out");
            const auto& cache_k = RequireInput(layer, group, "K_cache");
            const auto& cache_v = RequireInput(layer, group, "V_cache");
            if (cache_k.nSize != cache_v.nSize) {
                throw ModelConfigError("decoder K/V cache 字节数不一致");
            }
            const std::size_t minimum_history =
                group == 0 ? kMaxContextTokens : std::max(1, (group - 1) * 64);
            if (static_cast<std::size_t>(cache_k.nSize) <
                minimum_history * kHiddenSize * sizeof(std::uint16_t)) {
                throw ModelConfigError("decoder K/V cache 小于固定 history shape");
            }

            const auto& indices = RequireInput(layer, group, "indices");
            const auto& mask = RequireInput(layer, group, "mask");
            if (group == 0) {
                RequireBytes(indices, sizeof(std::uint32_t), "decoder.decode.indices");
                RequireBytes(
                    mask,
                    (kMaxContextTokens + 1) * sizeof(std::uint16_t),
                    "decoder.decode.mask");
            } else {
                RequireBytes(
                    indices,
                    3 * kPrefillBlockTokens * sizeof(std::uint32_t),
                    "decoder.prefill.indices");
                const std::size_t width = static_cast<std::size_t>(group) * kPrefillBlockTokens;
                RequireBytes(
                    mask,
                    kPrefillBlockTokens * width * sizeof(std::uint16_t),
                    "decoder.prefill.mask");
            }
        }
    }

    void ValidatePost() {
        if (post_->get_num_input_groups() != 1 || post_->get_num_output_groups() != 1) {
            throw ModelConfigError("post 必须只有一个 shape group");
        }
        RequireBytes(
            RequireInput(*post_, 0, "input"),
            kHiddenSize * sizeof(std::uint16_t),
            "post.input");
        RequireBytes(
            RequireOutput(*post_, 0, "output"),
            kVocabularySize * sizeof(std::uint16_t),
            "post.output");
    }

    void EncodeAudio(std::size_t& audio_tokens) {
        auto& conv_input = RequireInput(*convolution_, 0, "input_features");
        std::memset(conv_input.pVirAddr, 0, static_cast<std::size_t>(conv_input.nSize));
        std::memcpy(
            conv_input.pVirAddr,
            mel_.values.data(),
            mel_.values.size() * sizeof(float));
        Run(*convolution_, 0, "conv_frontend");

        const auto& conv_output = RequireOutput(*convolution_, 0, "output");
        auto& encoder_input = RequireInput(*encoder_, 0, "input_features");
        std::memcpy(
            encoder_input.pVirAddr,
            conv_output.pVirAddr,
            static_cast<std::size_t>(encoder_input.nSize));
        auto& encoder_mask = RequireInput(*encoder_, 0, "feature_attention_mask");
        std::memset(encoder_mask.pVirAddr, 0, static_cast<std::size_t>(encoder_mask.nSize));
        audio_tokens = DownsampledAudioTokenCount(mel_.frame_count);
        if (audio_tokens > static_cast<std::size_t>(encoder_mask.nSize)) {
            throw ModelConfigError("audio token 数超过 encoder mask shape");
        }
        std::memset(encoder_mask.pVirAddr, 1, audio_tokens);
        Run(*encoder_, 0, "encoder");
    }

    std::vector<std::uint32_t> PreparePrompt(std::size_t audio_tokens) {
        auto ids = BuildPromptIds(audio_tokens);
        const auto& audio = RequireOutput(*encoder_, 0, "output");
        const auto* audio_fp32 = static_cast<const float*>(audio.pVirAddr);
        std::size_t audio_index = 0;
        for (std::size_t index = 0; index < ids.size(); ++index) {
            auto* destination = prompt_hidden_.data() + index * kHiddenSize;
            if (ids[index] == kAudioPadId) {
                const auto* source = audio_fp32 + audio_index * kHiddenSize;
                for (std::size_t hidden = 0; hidden < kHiddenSize; ++hidden) {
                    destination[hidden] = FloatToBf16RoundToNearestEven(source[hidden]);
                }
                ++audio_index;
            } else {
                std::memcpy(
                    destination,
                    embedding_->Row(ids[index]),
                    kHiddenSize * sizeof(std::uint16_t));
            }
        }
        if (audio_index != audio_tokens) {
            throw std::runtime_error("AUDIO_PAD 数量与 encoder 输出不一致");
        }
        return ids;
    }

    void ResetKvCache() {
        for (auto& layer : layers_) {
            auto& cache_k = RequireInput(*layer, 0, "K_cache");
            auto& cache_v = RequireInput(*layer, 0, "V_cache");
            std::memset(cache_k.pVirAddr, 0, static_cast<std::size_t>(cache_k.nSize));
            std::memset(cache_v.pVirAddr, 0, static_cast<std::size_t>(cache_v.nSize));
        }
    }

    std::uint32_t Prefill(std::size_t prompt_tokens) {
        ResetKvCache();
        const std::uint16_t negative = FloatToBf16(-65536.0F);
        const std::uint16_t zero = FloatToBf16(0.0F);
        std::size_t last_valid = 0;

        for (std::size_t start = 0; start < prompt_tokens; start += kPrefillBlockTokens) {
            const std::size_t valid = std::min(kPrefillBlockTokens, prompt_tokens - start);
            const int group = static_cast<int>(start / kPrefillBlockTokens + 1);
            std::fill(block_hidden_.begin(), block_hidden_.end(), 0);
            std::memcpy(
                block_hidden_.data(),
                prompt_hidden_.data() + start * kHiddenSize,
                valid * kHiddenSize * sizeof(std::uint16_t));

            for (auto& layer : layers_) {
                auto& input = RequireInput(*layer, group, "input");
                std::memcpy(input.pVirAddr, block_hidden_.data(), block_hidden_.size() * sizeof(std::uint16_t));

                auto& indices = RequireInput(*layer, group, "indices");
                std::memset(indices.pVirAddr, 0, static_cast<std::size_t>(indices.nSize));
                auto* index_data = static_cast<std::uint32_t*>(indices.pVirAddr);
                const std::size_t rows = static_cast<std::size_t>(indices.nSize) /
                                         (kPrefillBlockTokens * sizeof(std::uint32_t));
                for (std::size_t row = 0; row < rows; ++row) {
                    for (std::size_t token = 0; token < valid; ++token) {
                        index_data[row * kPrefillBlockTokens + token] =
                            static_cast<std::uint32_t>(start + token);
                    }
                }

                auto& mask = RequireInput(*layer, group, "mask");
                auto* mask_data = static_cast<std::uint16_t*>(mask.pVirAddr);
                const std::size_t mask_elements = static_cast<std::size_t>(mask.nSize) / sizeof(std::uint16_t);
                const std::size_t width = mask_elements / kPrefillBlockTokens;
                std::fill(mask_data, mask_data + mask_elements, negative);
                for (std::size_t row = 0; row < valid; ++row) {
                    std::fill(mask_data + row * width, mask_data + row * width + start + row + 1, zero);
                }

                Run(*layer, group, "decoder prefill");

                const auto& output_k = RequireOutput(*layer, group, "K_cache_out");
                const auto& output_v = RequireOutput(*layer, group, "V_cache_out");
                auto& cache_k = RequireInput(*layer, 0, "K_cache");
                auto& cache_v = RequireInput(*layer, 0, "V_cache");
                const std::size_t bytes_per_token =
                    static_cast<std::size_t>(output_k.nSize) / kPrefillBlockTokens;
                const std::size_t offset = start * bytes_per_token;
                const std::size_t copy_bytes = valid * bytes_per_token;
                if (offset + copy_bytes > static_cast<std::size_t>(cache_k.nSize) ||
                    output_k.nSize != output_v.nSize) {
                    throw ModelConfigError("prefill KV 输出与设备 cache shape 不匹配");
                }
                std::memcpy(static_cast<char*>(cache_k.pVirAddr) + offset, output_k.pVirAddr, copy_bytes);
                std::memcpy(static_cast<char*>(cache_v.pVirAddr) + offset, output_v.pVirAddr, copy_bytes);

                const auto& output = RequireOutput(*layer, group, "output");
                std::memcpy(block_hidden_.data(), output.pVirAddr, block_hidden_.size() * sizeof(std::uint16_t));
            }
            last_valid = valid - 1;
        }

        std::memcpy(
            decode_hidden_.data(),
            block_hidden_.data() + last_valid * kHiddenSize,
            kHiddenSize * sizeof(std::uint16_t));
        return PostArgmax(decode_hidden_.data());
    }

    std::uint32_t PostArgmax(const std::uint16_t* hidden) {
        auto& input = RequireInput(*post_, 0, "input");
        std::memcpy(input.pVirAddr, hidden, kHiddenSize * sizeof(std::uint16_t));
        Run(*post_, 0, "post");
        const auto& output = RequireOutput(*post_, 0, "output");
        const auto* logits = static_cast<const std::uint16_t*>(output.pVirAddr);
        std::uint32_t best_token = 0;
        float best_value = -std::numeric_limits<float>::infinity();
        for (std::uint32_t token = 0; token < kVocabularySize; ++token) {
            const float value = Bf16ToFloat(logits[token]);
            if (value > best_value) {
                best_value = value;
                best_token = token;
            }
        }
        return best_token;
    }

    std::uint32_t DecodeStep(std::uint32_t token, std::size_t position) {
        std::memcpy(
            decode_hidden_.data(),
            embedding_->Row(token),
            kHiddenSize * sizeof(std::uint16_t));
        const std::uint16_t negative = FloatToBf16(-65536.0F);
        const std::uint16_t zero = FloatToBf16(0.0F);

        for (auto& layer : layers_) {
            auto& input = RequireInput(*layer, 0, "input");
            std::memcpy(input.pVirAddr, decode_hidden_.data(), kHiddenSize * sizeof(std::uint16_t));

            auto& indices = RequireInput(*layer, 0, "indices");
            *static_cast<std::uint32_t*>(indices.pVirAddr) = static_cast<std::uint32_t>(position);

            auto& mask = RequireInput(*layer, 0, "mask");
            auto* mask_data = static_cast<std::uint16_t*>(mask.pVirAddr);
            const std::size_t elements = static_cast<std::size_t>(mask.nSize) / sizeof(std::uint16_t);
            std::fill(mask_data, mask_data + elements, negative);
            std::fill(mask_data, mask_data + position, zero);
            mask_data[elements - 1] = zero;

            Run(*layer, 0, "decoder decode");

            const auto& output_k = RequireOutput(*layer, 0, "K_cache_out");
            const auto& output_v = RequireOutput(*layer, 0, "V_cache_out");
            auto& cache_k = RequireInput(*layer, 0, "K_cache");
            auto& cache_v = RequireInput(*layer, 0, "V_cache");
            const std::size_t bytes_per_token = static_cast<std::size_t>(output_k.nSize);
            const std::size_t offset = position * bytes_per_token;
            if (offset + bytes_per_token > static_cast<std::size_t>(cache_k.nSize) ||
                output_k.nSize != output_v.nSize) {
                throw ModelConfigError("decode KV 输出与设备 cache shape 不匹配");
            }
            std::memcpy(static_cast<char*>(cache_k.pVirAddr) + offset, output_k.pVirAddr, bytes_per_token);
            std::memcpy(static_cast<char*>(cache_v.pVirAddr) + offset, output_v.pVirAddr, bytes_per_token);

            const auto& output = RequireOutput(*layer, 0, "output");
            std::memcpy(decode_hidden_.data(), output.pVirAddr, kHiddenSize * sizeof(std::uint16_t));
        }
        return PostArgmax(decode_hidden_.data());
    }

    std::string TranscribeInternal(
        const std::int16_t* samples,
        std::size_t sample_count,
        std::size_t generation_limit) {
        if (!samples) throw std::invalid_argument("samples 不能为空");
        if (sample_count == 0) throw std::invalid_argument("PCM16 不能为空");

        const auto total_begin = Clock::now();
        const auto preprocess_begin = total_begin;
        feature_extractor_.ComputeInto(samples, sample_count, mel_);
        const auto preprocess_end = Clock::now();

        const auto encoder_begin = preprocess_end;
        std::size_t audio_tokens = 0;
        EncodeAudio(audio_tokens);
        auto prompt_ids = PreparePrompt(audio_tokens);
        const auto encoder_end = Clock::now();

        const auto decoder_begin = encoder_end;
        std::uint32_t next_token = Prefill(prompt_ids.size());
        const auto first_token_time = Clock::now();
        std::vector<std::uint32_t> generated;
        generated.reserve(generation_limit);
        for (std::size_t index = 0; index < generation_limit; ++index) {
            if (next_token == kImEndId || next_token == kEndOfTextId) break;
            generated.push_back(next_token);
            const std::size_t position = prompt_ids.size() + index;
            if (index + 1 >= generation_limit || position >= kMaxContextTokens - 1) break;
            next_token = DecodeStep(next_token, position);
        }
        const auto decoder_end = Clock::now();

        std::string text;
        text.reserve(generated.size() * 4);
        for (const std::uint32_t token : generated) text += DecodeAsrToken(*tokenizer_, token);
        const auto total_end = Clock::now();

        metrics_.sample_count = sample_count;
        metrics_.mel_frame_count = mel_.frame_count;
        metrics_.audio_token_count = audio_tokens;
        metrics_.prompt_token_count = prompt_ids.size();
        metrics_.generated_token_count = generated.size();
        metrics_.preprocess_ms = Milliseconds(preprocess_begin, preprocess_end);
        metrics_.encoder_ms = Milliseconds(encoder_begin, encoder_end);
        metrics_.decoder_ttft_ms = Milliseconds(decoder_begin, first_token_time);
        metrics_.decoder_ms = Milliseconds(decoder_begin, decoder_end);
        metrics_.native_total_ms = Milliseconds(total_begin, total_end);
        return text;
    }

    ModelConfig config_;
    AudioFeatureExtractor feature_extractor_;
    MelFeatures mel_;
    RunnerMetrics metrics_;
    std::vector<std::uint16_t> prompt_hidden_;
    std::vector<std::uint16_t> block_hidden_;
    std::vector<std::uint16_t> decode_hidden_;

    // runtime_ 必须晚于所有 session 析构，因此在 session 之前声明。
    std::unique_ptr<AxRuntime> runtime_;
    std::unique_ptr<ax_runner_ax650> convolution_;
    std::unique_ptr<ax_runner_ax650> encoder_;
    std::vector<std::unique_ptr<ax_runner_ax650>> layers_;
    std::unique_ptr<ax_runner_ax650> post_;
    std::unique_ptr<MNN::Transformer::Tokenizer> tokenizer_;
    std::unique_ptr<MappedEmbedding> embedding_;
};

QwenAsrRunner::QwenAsrRunner(const std::string& model_dir)
    : impl_(std::make_unique<Impl>(model_dir)) {}

QwenAsrRunner::~QwenAsrRunner() = default;

void QwenAsrRunner::Warmup() { impl_->Warmup(); }

std::string QwenAsrRunner::Transcribe(
    const std::int16_t* samples,
    std::size_t sample_count,
    int sample_rate) {
    return impl_->Transcribe(samples, sample_count, sample_rate);
}

const RunnerMetrics& QwenAsrRunner::LastMetrics() const noexcept {
    return impl_->LastMetrics();
}

}  // namespace ax_qwen_asr

#endif
