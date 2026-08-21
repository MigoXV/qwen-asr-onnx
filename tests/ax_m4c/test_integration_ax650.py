from __future__ import annotations

import difflib
import os
import wave
from pathlib import Path

import pytest

from qwen_asr_onnx.ax_m4c import AxQwenAsr


MODEL_ENV = "AX_QWEN_ASR_MODEL_DIR"
AUDIO_ENV = "AX_QWEN_ASR_AUDIO"
REFERENCE = "在同一件事情上纠缠不清的人，是不会讨女孩子喜欢的。"


def _integration_paths() -> tuple[Path, Path]:
    model = os.environ.get(MODEL_ENV)
    audio = os.environ.get(AUDIO_ENV)
    if not model or not audio:
        pytest.skip(f"设置 {MODEL_ENV} 和 {AUDIO_ENV} 后运行 AX650 集成测试")
    return Path(model), Path(audio)


@pytest.mark.ax650
def test_rita_raw_transcription_is_semantically_close() -> None:
    model, audio = _integration_paths()
    with wave.open(str(audio), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        pcm = wav_file.readframes(wav_file.getnframes())

    with AxQwenAsr(model) as asr:
        raw_output = asr.transcribe_pcm16(pcm, sample_rate=16000)
    print(f"AX650 原始输出：{raw_output}")
    suffix = raw_output.split("<asr_text>", 1)[-1]
    similarity = difflib.SequenceMatcher(None, suffix, REFERENCE).ratio()
    assert similarity >= 0.75, (raw_output, similarity)
