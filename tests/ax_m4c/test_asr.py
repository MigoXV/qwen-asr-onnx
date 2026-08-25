from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from qwen_asr_onnx.ax_m4c import AxQwenAsr
from qwen_asr_onnx.ax_m4c.errors import (
    AxQwenAsrBusyError,
    AxQwenAsrInvalidArgumentError,
    AxQwenAsrRuntimeError,
)


RAW_TEXT = "language Chinese<asr_text>在同一个话题上纠缠不清的人，是不会讨女孩子喜欢的。"


def test_transcribe_lifecycle_and_metrics(fake_bindings, tmp_path) -> None:
    pcm = np.zeros(1600, dtype="<i2")
    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        asr.warmup()
        assert asr.transcribe_pcm16(pcm, sample_rate=16000) == RAW_TEXT
        metrics = asr.last_metrics()
        assert metrics.sample_count == 1600
        assert metrics.native_total_ms == 7.0
        assert not asr.closed
    assert asr.closed
    asr.close()
    with pytest.raises(AxQwenAsrRuntimeError, match="已关闭"):
        asr.warmup()


def test_accepts_pcm16_bytes(fake_bindings, tmp_path) -> None:
    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        assert asr.transcribe_pcm16(b"\0\0" * 80) == RAW_TEXT


def test_streaming_callback_preserves_final_text(fake_bindings, tmp_path) -> None:
    events: list[tuple[int, int, str]] = []

    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        result = asr.transcribe_pcm16_stream(
            b"\0\0" * 80,
            token_callback=lambda token_id, index, delta: events.append(
                (token_id, index, delta)
            ),
        )

    assert result == RAW_TEXT
    assert [event[1] for event in events] == list(range(len(events)))
    assert "".join(event[2] for event in events) == RAW_TEXT


def test_streaming_callback_can_stop_at_token_boundary(
    fake_bindings, tmp_path
) -> None:
    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        result = asr.transcribe_pcm16_stream(
            b"\0\0" * 80,
            token_callback=lambda _token_id, index, _delta: index < 2,
        )

    assert result == "language Chinese<asr_text>"


def test_streaming_callback_exception_is_propagated(fake_bindings, tmp_path) -> None:
    def fail(_token_id: int, _index: int, _delta: str) -> bool:
        raise RuntimeError("callback failed")

    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        with pytest.raises(RuntimeError, match="callback failed"):
            asr.transcribe_pcm16_stream(
                b"\0\0" * 80,
                token_callback=fail,
            )


@pytest.mark.parametrize("rate", [8000, 44100])
def test_rejects_non_16k_rate(fake_bindings, tmp_path, rate: int) -> None:
    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        with pytest.raises(AxQwenAsrInvalidArgumentError, match="16000"):
            asr.transcribe_pcm16(b"\0\0", sample_rate=rate)


def test_rejects_bad_pcm_buffers(fake_bindings, tmp_path) -> None:
    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        with pytest.raises(AxQwenAsrInvalidArgumentError, match="不能为空"):
            asr.transcribe_pcm16(b"")
        with pytest.raises(AxQwenAsrInvalidArgumentError, match="2 的倍数"):
            asr.transcribe_pcm16(b"\0")
        non_contiguous = np.zeros(20, dtype=np.int16)[::2]
        with pytest.raises(AxQwenAsrInvalidArgumentError, match="连续"):
            asr.transcribe_pcm16(non_contiguous)


def test_handle_cannot_be_copied_or_pickled(fake_bindings, tmp_path) -> None:
    with AxQwenAsr(tmp_path, _bindings=fake_bindings) as asr:
        with pytest.raises(TypeError):
            copy.copy(asr)
        with pytest.raises(TypeError):
            copy.deepcopy(asr)
        with pytest.raises(TypeError):
            pickle.dumps(asr)


def test_native_busy_error_contains_last_error(fake_bindings, tmp_path) -> None:
    first = AxQwenAsr(tmp_path, _bindings=fake_bindings)
    try:
        with pytest.raises(AxQwenAsrBusyError, match="已经存在"):
            AxQwenAsr(tmp_path, _bindings=fake_bindings)
    finally:
        first.close()
