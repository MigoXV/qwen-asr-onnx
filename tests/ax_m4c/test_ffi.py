from __future__ import annotations

from pathlib import Path

import pytest

from qwen_asr_onnx.ax_m4c.errors import AxQwenAsrLibraryError, NativeErrorCode
from qwen_asr_onnx.ax_m4c.ffi import load_native_library


def test_import_does_not_load_native_library(monkeypatch) -> None:
    import qwen_asr_onnx.ax_m4c as module

    assert module.AxQwenAsr is not None


def test_missing_library_error_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(AxQwenAsrLibraryError, match="未找到"):
        load_native_library(tmp_path / "missing.so")


def test_c_abi_null_and_buffer_contract(fake_bindings, tmp_path: Path) -> None:
    ffi, lib = fake_bindings.ffi, fake_bindings.lib
    required = ffi.new("size_t *")
    sample = ffi.new("int16_t[]", [0])
    assert (
        lib.ax_qwen_asr_transcribe_pcm16(
            ffi.NULL, sample, 1, 16000, ffi.NULL, 0, required
        )
        == NativeErrorCode.INVALID_ARGUMENT
    )
    assert "handle" in ffi.string(lib.ax_qwen_asr_last_error(ffi.NULL)).decode()
    lib.ax_qwen_asr_destroy(ffi.NULL)

    handle = ffi.new("ax_qwen_asr_handle **")
    assert lib.ax_qwen_asr_create(str(tmp_path).encode(), handle) == 0
    tiny = ffi.new("char[]", 4)
    assert (
        lib.ax_qwen_asr_transcribe_pcm16(
            handle[0], sample, 1, 16000, tiny, 4, required
        )
        == NativeErrorCode.BUFFER_TOO_SMALL
    )
    assert required[0] > 4
    assert "buffer" in ffi.string(lib.ax_qwen_asr_last_error(handle[0])).decode()
    lib.ax_qwen_asr_destroy(handle[0])


def test_c_abi_only_allows_one_live_handle(fake_bindings, tmp_path: Path) -> None:
    ffi, lib = fake_bindings.ffi, fake_bindings.lib
    first = ffi.new("ax_qwen_asr_handle **")
    second = ffi.new("ax_qwen_asr_handle **")
    assert lib.ax_qwen_asr_create(str(tmp_path).encode(), first) == 0
    assert lib.ax_qwen_asr_create(str(tmp_path).encode(), second) == NativeErrorCode.BUSY
    assert second[0] == ffi.NULL
    lib.ax_qwen_asr_destroy(first[0])
    assert lib.ax_qwen_asr_create(str(tmp_path).encode(), second) == 0
    lib.ax_qwen_asr_destroy(second[0])
