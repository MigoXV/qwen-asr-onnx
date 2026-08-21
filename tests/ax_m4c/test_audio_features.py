from __future__ import annotations

import numpy as np

from qwen_asr_onnx.inferencers.onnx import (
    compute_mel_spectrogram,
    get_mel_filters,
)


def _native_mel(core_native, pcm: np.ndarray) -> np.ndarray:
    ffi, lib = core_native
    frame_count = pcm.size // 160 + 1
    output = np.empty((frame_count, 128), dtype=np.float32)
    native_frames = ffi.new("size_t *")
    required = ffi.new("size_t *")
    result = lib.ax_qwen_asr_test_compute_mel(
        ffi.from_buffer("int16_t[]", pcm),
        pcm.size,
        ffi.from_buffer("float[]", output),
        output.size,
        native_frames,
        required,
    )
    assert result == 0, ffi.string(lib.ax_qwen_asr_test_last_error()).decode()
    assert native_frames[0] == frame_count
    assert required[0] == output.size
    return output


def test_mel_matches_existing_python_implementation(core_native) -> None:
    sample_count = 16000 // 2 + 37
    time = np.arange(sample_count, dtype=np.float32) / 16000.0
    waveform = 0.42 * np.sin(2 * np.pi * 440.0 * time)
    waveform += 0.07 * np.sin(2 * np.pi * 1234.0 * time)
    pcm = np.clip(np.rint(waveform * 32768.0), -32768, 32767).astype("<i2")

    native = _native_mel(core_native, pcm)
    expected = compute_mel_spectrogram(
        pcm.astype(np.float32) / 32768.0,
        get_mel_filters(),
    ).T
    np.testing.assert_allclose(native, expected, rtol=2e-4, atol=2e-5)


def test_mel_reflect_padding_for_short_pcm(core_native) -> None:
    pcm = np.array([100, -200, 300, -400, 500], dtype="<i2")
    native = _native_mel(core_native, pcm)
    expected = compute_mel_spectrogram(
        pcm.astype(np.float32) / 32768.0,
        get_mel_filters(),
    ).T
    np.testing.assert_allclose(native, expected, rtol=2e-4, atol=2e-5)


def test_mel_rejects_more_than_3000_frames(core_native) -> None:
    ffi, lib = core_native
    pcm = np.zeros(3000 * 160, dtype="<i2")
    frames = ffi.new("size_t *")
    required = ffi.new("size_t *")
    result = lib.ax_qwen_asr_test_compute_mel(
        ffi.from_buffer("int16_t[]", pcm),
        pcm.size,
        ffi.NULL,
        0,
        frames,
        required,
    )
    assert result == 3
    assert "3000" in ffi.string(lib.ax_qwen_asr_test_last_error()).decode()
