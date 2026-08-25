"""AX650 Qwen3-ASR-0.6B 的 Python 高层接口。"""

from __future__ import annotations

import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Any

from .errors import (
    ERROR_CLASS_BY_CODE,
    AxQwenAsrBufferTooSmallError,
    AxQwenAsrError,
    AxQwenAsrInvalidArgumentError,
    AxQwenAsrRuntimeError,
    NativeErrorCode,
)
from .ffi import ABI_VERSION, NativeBindings, load_native_library

_OUTPUT_CAPACITY = 64 * 1024
TokenCallback = Callable[[int, int, str], bool | None]


@dataclass(frozen=True)
class TranscriptionMetrics:
    sample_count: int
    mel_frame_count: int
    audio_token_count: int
    prompt_token_count: int
    generated_token_count: int
    preprocess_ms: float
    encoder_ms: float
    decoder_ttft_ms: float
    decoder_ms: float
    native_total_ms: float


def _destroy_handle(lib: Any, handle: Any) -> None:
    try:
        lib.ax_qwen_asr_destroy(handle)
    except Exception:
        # finalizer 不能把异常泄漏到解释器关闭流程。
        pass


class AxQwenAsr:
    """持有唯一 native handle 的串行 Qwen3-ASR runner。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        _bindings: NativeBindings | None = None,
    ) -> None:
        path = Path(model_dir).expanduser().resolve(strict=True)
        if not path.is_dir():
            raise NotADirectoryError(f"模型路径不是目录：{path}")

        bindings = _bindings or load_native_library()
        self._ffi = bindings.ffi
        self._lib = bindings.lib
        self._model_dir = path
        self._lock = threading.RLock()
        self._closed = False

        handle_out = self._ffi.new("ax_qwen_asr_handle **")
        code = int(
            self._lib.ax_qwen_asr_create(
                str(path).encode("utf-8"),
                handle_out,
            )
        )
        if code != NativeErrorCode.OK:
            self._raise_native_error(code, self._ffi.NULL)
        if handle_out[0] == self._ffi.NULL:
            raise AxQwenAsrRuntimeError("native create 成功但返回了空 handle")

        self._handle = handle_out[0]
        self._finalizer = weakref.finalize(
            self,
            _destroy_handle,
            self._lib,
            self._handle,
        )

    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def warmup(self) -> None:
        with self._lock:
            handle = self._require_open()
            code = int(self._lib.ax_qwen_asr_warmup(handle))
            if code != NativeErrorCode.OK:
                self._raise_native_error(code, handle)

    def transcribe_pcm16(self, pcm: Any, *, sample_rate: int = 16000) -> str:
        return self._transcribe_pcm16(
            pcm,
            sample_rate=sample_rate,
            token_callback=None,
        )

    def transcribe_pcm16_stream(
        self,
        pcm: Any,
        *,
        sample_rate: int = 16000,
        token_callback: TokenCallback,
    ) -> str:
        """逐 token 回调 UTF-8 delta，并返回完整或被回调提前停止的文本。"""
        if not callable(token_callback):
            raise AxQwenAsrInvalidArgumentError("token_callback 必须可调用")
        return self._transcribe_pcm16(
            pcm,
            sample_rate=sample_rate,
            token_callback=token_callback,
        )

    def _transcribe_pcm16(
        self,
        pcm: Any,
        *,
        sample_rate: int,
        token_callback: TokenCallback | None,
    ) -> str:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise AxQwenAsrInvalidArgumentError("sample_rate 必须是整数 16000")
        if sample_rate != 16000:
            raise AxQwenAsrInvalidArgumentError(
                f"AX650 专用后端只接受 16000 Hz PCM16，收到 {sample_rate} Hz"
            )

        try:
            view = memoryview(pcm)
        except TypeError as exc:
            raise AxQwenAsrInvalidArgumentError(
                "pcm 必须实现一维连续 buffer protocol，内容为本机字节序 PCM16"
            ) from exc
        if view.ndim != 1 or not view.c_contiguous:
            raise AxQwenAsrInvalidArgumentError("pcm 必须是一维 C 连续 buffer")
        if view.nbytes == 0:
            raise AxQwenAsrInvalidArgumentError("pcm 不能为空")
        if view.nbytes % 2:
            raise AxQwenAsrInvalidArgumentError("PCM16 字节数必须是 2 的倍数")
        if view.format.startswith(">"):
            raise AxQwenAsrInvalidArgumentError("PCM16 必须使用本机小端字节序")

        byte_view = view.cast("B")
        with self._lock:
            handle = self._require_open()
            samples = self._ffi.from_buffer("int16_t[]", byte_view)
            output = self._ffi.new("char[]", _OUTPUT_CAPACITY)
            required_size = self._ffi.new("size_t *")
            callback_errors: list[BaseException] = []
            native_callback = self._ffi.NULL
            if token_callback is not None:

                @self._ffi.callback(
                    "int(uint32_t, size_t, const char *, size_t, void *)",
                    error=1,
                )
                def on_native_token(
                    token_id: int,
                    token_index: int,
                    utf8_delta: Any,
                    utf8_delta_size: int,
                    user_data: Any,
                ) -> int:
                    del user_data
                    try:
                        delta = bytes(
                            self._ffi.buffer(utf8_delta, utf8_delta_size)
                        ).decode("utf-8", errors="strict")
                        should_continue = token_callback(
                            int(token_id),
                            int(token_index),
                            delta,
                        )
                        return 1 if should_continue is False else 0
                    except BaseException as exc:
                        callback_errors.append(exc)
                        return 1

                native_callback = on_native_token
                native_call = self._lib.ax_qwen_asr_transcribe_pcm16_stream
                native_args = (
                    handle,
                    samples,
                    view.nbytes // 2,
                    sample_rate,
                    native_callback,
                    self._ffi.NULL,
                    output,
                    _OUTPUT_CAPACITY,
                    required_size,
                )
            else:
                native_call = self._lib.ax_qwen_asr_transcribe_pcm16
                native_args = (
                    handle,
                    samples,
                    view.nbytes // 2,
                    sample_rate,
                    output,
                    _OUTPUT_CAPACITY,
                    required_size,
                )
            code = int(native_call(*native_args))
            if callback_errors:
                raise callback_errors[0]
            if code != NativeErrorCode.OK:
                self._raise_native_error(
                    code,
                    handle,
                    required_size=int(required_size[0]),
                )
            return self._ffi.string(output).decode("utf-8", errors="strict")

    def last_metrics(self) -> TranscriptionMetrics:
        with self._lock:
            handle = self._require_open()
            native = self._ffi.new("ax_qwen_asr_metrics *")
            native.struct_size = self._ffi.sizeof("ax_qwen_asr_metrics")
            native.abi_version = ABI_VERSION
            code = int(self._lib.ax_qwen_asr_get_last_metrics(handle, native))
            if code != NativeErrorCode.OK:
                self._raise_native_error(code, handle)
            return TranscriptionMetrics(
                sample_count=int(native.sample_count),
                mel_frame_count=int(native.mel_frame_count),
                audio_token_count=int(native.audio_token_count),
                prompt_token_count=int(native.prompt_token_count),
                generated_token_count=int(native.generated_token_count),
                preprocess_ms=float(native.preprocess_ms),
                encoder_ms=float(native.encoder_ms),
                decoder_ttft_ms=float(native.decoder_ttft_ms),
                decoder_ms=float(native.decoder_ms),
                native_total_ms=float(native.native_total_ms),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._finalizer.alive:
                self._finalizer()
            self._handle = self._ffi.NULL

    def __enter__(self) -> AxQwenAsr:
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def __copy__(self) -> AxQwenAsr:
        raise TypeError("AxQwenAsr 持有唯一 native handle，不能复制")

    def __deepcopy__(self, memo: dict[int, Any]) -> AxQwenAsr:
        raise TypeError("AxQwenAsr 持有唯一 native handle，不能深复制")

    def __reduce__(self) -> Any:
        raise TypeError("AxQwenAsr native handle 不能 pickle")

    def __reduce_ex__(self, protocol: int) -> Any:
        raise TypeError("AxQwenAsr native handle 不能 pickle")

    def _require_open(self) -> Any:
        if self._closed:
            raise AxQwenAsrRuntimeError("AxQwenAsr handle 已关闭")
        return self._handle

    def _last_error(self, handle: Any) -> str:
        ptr = self._lib.ax_qwen_asr_last_error(handle)
        if ptr == self._ffi.NULL:
            return "native 未提供错误详情"
        return self._ffi.string(ptr).decode("utf-8", errors="replace")

    def _raise_native_error(
        self,
        code: int,
        handle: Any,
        *,
        required_size: int = 0,
    ) -> None:
        message = self._last_error(handle)
        try:
            native_code = NativeErrorCode(code)
        except ValueError:
            raise AxQwenAsrError(
                f"未知 native 错误码 {code}：{message}",
                code=code,
            ) from None

        if native_code is NativeErrorCode.BUFFER_TOO_SMALL:
            raise AxQwenAsrBufferTooSmallError(
                f"native 输出缓冲区不足，需要 {required_size} 字节：{message}",
                required_size=required_size,
                code=code,
            )
        error_class = ERROR_CLASS_BY_CODE.get(native_code, AxQwenAsrError)
        raise error_class(message, code=code)
