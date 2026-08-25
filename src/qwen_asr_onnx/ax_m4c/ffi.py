"""仅声明并按需加载 AX Qwen3-ASR 的 C ABI。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cffi import FFI

from .errors import AxQwenAsrLibraryError

ABI_VERSION = 1
DEFAULT_LIBRARY_NAME = "libax_qwen_asr.so"


_CDEF = r"""
typedef struct ax_qwen_asr_handle ax_qwen_asr_handle;

typedef int (*ax_qwen_asr_token_callback)(
    uint32_t token_id,
    size_t token_index,
    const char *utf8_delta,
    size_t utf8_delta_size,
    void *user_data
);

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

int ax_qwen_asr_create(const char *model_dir, ax_qwen_asr_handle **handle);
int ax_qwen_asr_warmup(ax_qwen_asr_handle *handle);
int ax_qwen_asr_transcribe_pcm16(
    ax_qwen_asr_handle *handle,
    const int16_t *samples,
    size_t sample_count,
    int sample_rate,
    char *output,
    size_t output_capacity,
    size_t *required_size
);
int ax_qwen_asr_transcribe_pcm16_stream(
    ax_qwen_asr_handle *handle,
    const int16_t *samples,
    size_t sample_count,
    int sample_rate,
    ax_qwen_asr_token_callback callback,
    void *user_data,
    char *output,
    size_t output_capacity,
    size_t *required_size
);
int ax_qwen_asr_get_last_metrics(
    ax_qwen_asr_handle *handle,
    ax_qwen_asr_metrics *metrics
);
const char *ax_qwen_asr_last_error(ax_qwen_asr_handle *handle);
void ax_qwen_asr_destroy(ax_qwen_asr_handle *handle);
"""


@dataclass(frozen=True)
class NativeBindings:
    ffi: FFI
    lib: object
    path: Path


def default_library_path() -> Path:
    return Path(__file__).resolve().parent / "lib" / DEFAULT_LIBRARY_NAME


@lru_cache(maxsize=4)
def load_native_library(library_path: str | Path | None = None) -> NativeBindings:
    """加载动态库；导入子包本身不会调用此函数。"""

    path = (
        default_library_path()
        if library_path is None
        else Path(library_path).expanduser().resolve(strict=False)
    )
    if not path.is_file():
        raise AxQwenAsrLibraryError(
            f"未找到 AX Qwen3-ASR native 动态库：{path}。请先构建并安装 libax_qwen_asr.so。"
        )

    ffi = FFI()
    ffi.cdef(_CDEF)
    try:
        lib = ffi.dlopen(str(path))
    except OSError as exc:
        raise AxQwenAsrLibraryError(
            f"无法加载 AX Qwen3-ASR native 动态库 {path}：{exc}"
        ) from exc
    try:
        getattr(lib, "ax_qwen_asr_transcribe_pcm16_stream")
    except AttributeError as exc:
        raise AxQwenAsrLibraryError(
            f"AX Qwen3-ASR native 动态库缺少 token streaming ABI，请重新构建：{path}"
        ) from exc
    return NativeBindings(ffi=ffi, lib=lib, path=path)
