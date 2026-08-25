#include "ax_qwen_asr.h"

#include "model_config.hpp"
#include "qwen_asr_runner.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>

struct ax_qwen_asr_handle {
    explicit ax_qwen_asr_handle(const std::string& model_dir)
        : runner(std::make_unique<ax_qwen_asr::QwenAsrRunner>(model_dir)) {}

    std::mutex mutex;
    std::unique_ptr<ax_qwen_asr::QwenAsrRunner> runner;
    std::string last_error;
};

namespace {

thread_local std::string g_error_snapshot;
std::mutex g_lifecycle_mutex;
bool g_has_live_handle = false;

void SetNullHandleError(const std::string& message) {
    g_error_snapshot = message;
}

void SetHandleError(ax_qwen_asr_handle* handle, const std::string& message) {
    if (handle) handle->last_error = message;
    g_error_snapshot = message;
}

int ExceptionCode() {
    try {
        throw;
    } catch (const ax_qwen_asr::ModelConfigError&) {
        return AX_QWEN_ASR_MODEL_CONFIG;
    } catch (const ax_qwen_asr::ModelFileError&) {
        return AX_QWEN_ASR_MODEL_FILE;
    } catch (const ax_qwen_asr::AxEngineError&) {
        return AX_QWEN_ASR_AXENGINE;
    } catch (const std::bad_alloc&) {
        return AX_QWEN_ASR_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    } catch (...) {
        return AX_QWEN_ASR_INTERNAL;
    }
}

std::string ExceptionMessage() {
    try {
        throw;
    } catch (const std::exception& exception) {
        return exception.what();
    } catch (...) {
        return "未知 C++ 异常";
    }
}

void ClearHandleError(ax_qwen_asr_handle* handle) {
    handle->last_error.clear();
    g_error_snapshot.clear();
}

int TranscribePcm16(
    ax_qwen_asr_handle* handle,
    const int16_t* samples,
    size_t sample_count,
    int sample_rate,
    ax_qwen_asr_token_callback callback,
    void* user_data,
    char* output,
    size_t output_capacity,
    size_t* required_size) {
    if (!handle) {
        SetNullHandleError("transcribe handle 不能为空");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    ClearHandleError(handle);
    if (!required_size) {
        SetHandleError(handle, "required_size 不能为空");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    *required_size = 0;
    if (!samples || sample_count == 0) {
        SetHandleError(handle, "samples 不能为空且 sample_count 必须大于 0");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    if (sample_rate != 16000) {
        SetHandleError(handle, "固定 profile 只接受 16000 Hz PCM16");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    if (!output && output_capacity != 0) {
        SetHandleError(handle, "output 为空时 output_capacity 必须为 0");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }

    try {
        ax_qwen_asr::QwenAsrRunner::TokenCallback runner_callback;
        if (callback) {
            runner_callback = [callback, user_data](
                                  std::uint32_t token_id,
                                  std::size_t token_index,
                                  const std::string& delta) {
                return callback(
                           token_id,
                           token_index,
                           delta.data(),
                           delta.size(),
                           user_data) == 0;
            };
        }
        const std::string text = handle->runner->Transcribe(
            samples, sample_count, sample_rate, runner_callback);
        *required_size = text.size() + 1;
        if (!output || output_capacity < *required_size) {
            SetHandleError(
                handle,
                "输出 buffer 不足：需要 " + std::to_string(*required_size) +
                    " 字节，实际 " + std::to_string(output_capacity));
            return AX_QWEN_ASR_BUFFER_TOO_SMALL;
        }
        std::memcpy(output, text.data(), text.size());
        output[text.size()] = '\0';
        return AX_QWEN_ASR_OK;
    } catch (...) {
        const int code = ExceptionCode();
        SetHandleError(handle, ExceptionMessage());
        return code;
    }
}

}  // namespace

extern "C" int ax_qwen_asr_create(
    const char* model_dir,
    ax_qwen_asr_handle** handle) {
    if (!handle) {
        SetNullHandleError("handle 输出参数不能为空");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    *handle = nullptr;
    if (!model_dir || model_dir[0] == '\0') {
        SetNullHandleError("model_dir 不能为空");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }

    {
        std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
        if (g_has_live_handle) {
            SetNullHandleError("当前进程已经存在一个 AX Qwen3-ASR runner");
            return AX_QWEN_ASR_BUSY;
        }
        g_has_live_handle = true;
    }

    try {
        auto created = std::make_unique<ax_qwen_asr_handle>(std::string(model_dir));
        *handle = created.release();
        g_error_snapshot.clear();
        return AX_QWEN_ASR_OK;
    } catch (...) {
        const int code = ExceptionCode();
        SetNullHandleError(ExceptionMessage());
        std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
        g_has_live_handle = false;
        return code;
    }
}

extern "C" int ax_qwen_asr_warmup(ax_qwen_asr_handle* handle) {
    if (!handle) {
        SetNullHandleError("warmup handle 不能为空");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    ClearHandleError(handle);
    try {
        handle->runner->Warmup();
        return AX_QWEN_ASR_OK;
    } catch (...) {
        const int code = ExceptionCode();
        SetHandleError(handle, ExceptionMessage());
        return code;
    }
}

extern "C" int ax_qwen_asr_transcribe_pcm16(
    ax_qwen_asr_handle* handle,
    const int16_t* samples,
    size_t sample_count,
    int sample_rate,
    char* output,
    size_t output_capacity,
    size_t* required_size) {
    return TranscribePcm16(
        handle,
        samples,
        sample_count,
        sample_rate,
        nullptr,
        nullptr,
        output,
        output_capacity,
        required_size);
}

extern "C" int ax_qwen_asr_transcribe_pcm16_stream(
    ax_qwen_asr_handle* handle,
    const int16_t* samples,
    size_t sample_count,
    int sample_rate,
    ax_qwen_asr_token_callback callback,
    void* user_data,
    char* output,
    size_t output_capacity,
    size_t* required_size) {
    return TranscribePcm16(
        handle,
        samples,
        sample_count,
        sample_rate,
        callback,
        user_data,
        output,
        output_capacity,
        required_size);
}

extern "C" int ax_qwen_asr_get_last_metrics(
    ax_qwen_asr_handle* handle,
    ax_qwen_asr_metrics* metrics) {
    if (!handle) {
        SetNullHandleError("metrics handle 不能为空");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    ClearHandleError(handle);
    if (!metrics) {
        SetHandleError(handle, "metrics 不能为空");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    if (metrics->struct_size < sizeof(ax_qwen_asr_metrics)) {
        SetHandleError(handle, "metrics.struct_size 小于当前 ABI 结构体尺寸");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }
    if (metrics->abi_version != AX_QWEN_ASR_ABI_VERSION) {
        SetHandleError(handle, "metrics.abi_version 与 native ABI 不匹配");
        return AX_QWEN_ASR_INVALID_ARGUMENT;
    }

    const auto& source = handle->runner->LastMetrics();
    ax_qwen_asr_metrics result{};
    result.struct_size = sizeof(result);
    result.abi_version = AX_QWEN_ASR_ABI_VERSION;
    result.sample_count = source.sample_count;
    result.mel_frame_count = source.mel_frame_count;
    result.audio_token_count = source.audio_token_count;
    result.prompt_token_count = source.prompt_token_count;
    result.generated_token_count = source.generated_token_count;
    result.preprocess_ms = source.preprocess_ms;
    result.encoder_ms = source.encoder_ms;
    result.decoder_ttft_ms = source.decoder_ttft_ms;
    result.decoder_ms = source.decoder_ms;
    result.native_total_ms = source.native_total_ms;
    std::memcpy(metrics, &result, sizeof(result));
    return AX_QWEN_ASR_OK;
}

extern "C" const char* ax_qwen_asr_last_error(ax_qwen_asr_handle* handle) {
    if (handle) {
        std::lock_guard<std::mutex> lock(handle->mutex);
        g_error_snapshot = handle->last_error;
    }
    return g_error_snapshot.c_str();
}

extern "C" void ax_qwen_asr_destroy(ax_qwen_asr_handle* handle) {
    if (!handle) return;
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        handle->runner.reset();
        handle->last_error.clear();
    }
    delete handle;
    std::lock_guard<std::mutex> lock(g_lifecycle_mutex);
    g_has_live_handle = false;
    g_error_snapshot.clear();
}
