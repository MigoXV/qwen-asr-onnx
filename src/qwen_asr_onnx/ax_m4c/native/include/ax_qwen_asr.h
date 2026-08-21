#ifndef AX_QWEN_ASR_H
#define AX_QWEN_ASR_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define AX_QWEN_ASR_API __declspec(dllexport)
#else
#define AX_QWEN_ASR_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define AX_QWEN_ASR_ABI_VERSION 1u

typedef struct ax_qwen_asr_handle ax_qwen_asr_handle;

typedef enum ax_qwen_asr_error_code {
    AX_QWEN_ASR_OK = 0,
    AX_QWEN_ASR_INVALID_ARGUMENT = 1,
    AX_QWEN_ASR_MODEL_CONFIG = 2,
    AX_QWEN_ASR_MODEL_FILE = 3,
    AX_QWEN_ASR_AXENGINE = 4,
    AX_QWEN_ASR_OUT_OF_MEMORY = 5,
    AX_QWEN_ASR_BUFFER_TOO_SMALL = 6,
    AX_QWEN_ASR_BUSY = 7,
    AX_QWEN_ASR_INTERNAL = 8
} ax_qwen_asr_error_code;

typedef struct ax_qwen_asr_metrics {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t reserved;
    size_t sample_count;
    size_t mel_frame_count;
    size_t audio_token_count;
    size_t prompt_token_count;
    size_t generated_token_count;
    double preprocess_ms;
    double encoder_ms;
    double decoder_ttft_ms;
    double decoder_ms;
    double native_total_ms;
} ax_qwen_asr_metrics;

AX_QWEN_ASR_API int ax_qwen_asr_create(
    const char *model_dir,
    ax_qwen_asr_handle **handle);

AX_QWEN_ASR_API int ax_qwen_asr_warmup(ax_qwen_asr_handle *handle);

AX_QWEN_ASR_API int ax_qwen_asr_transcribe_pcm16(
    ax_qwen_asr_handle *handle,
    const int16_t *samples,
    size_t sample_count,
    int sample_rate,
    char *output,
    size_t output_capacity,
    size_t *required_size);

AX_QWEN_ASR_API int ax_qwen_asr_get_last_metrics(
    ax_qwen_asr_handle *handle,
    ax_qwen_asr_metrics *metrics);

/*
 * 返回的指针指向当前线程的快照，调用者不得释放。该指针在同一线程下一次
 * ax_qwen_asr_* 调用前有效。传入 NULL 时返回最近一次 create 参数/构造错误。
 */
AX_QWEN_ASR_API const char *ax_qwen_asr_last_error(
    ax_qwen_asr_handle *handle);

/* NULL 安全。不得与同一 handle 上尚未返回的裸 C ABI 调用并发执行。 */
AX_QWEN_ASR_API void ax_qwen_asr_destroy(ax_qwen_asr_handle *handle);

#ifdef __cplusplus
}
#endif

#endif
