from __future__ import annotations

import importlib.util
import wave
from pathlib import Path

import pytest


def _load_example(repository_root: Path):
    path = repository_root / "examples/ax-m4c/infer.py"
    spec = importlib.util.spec_from_file_location("ax_m4c_infer_example", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wav(
    path: Path,
    *,
    channels: int = 1,
    sample_width: int = 2,
    sample_rate: int = 16000,
) -> bytes:
    frames = b"\x01\x00" * channels * 80 if sample_width == 2 else b"\x80" * channels * 80
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)
    return frames


def test_example_reads_valid_pcm16_wav(repository_root: Path, tmp_path: Path) -> None:
    example = _load_example(repository_root)
    path = tmp_path / "valid.wav"
    expected = _write_wav(path)
    assert example.read_pcm16_wav(path) == expected


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"channels": 2}, "单声道"),
        ({"sample_width": 1}, "PCM16"),
        ({"sample_rate": 8000}, "16000"),
    ],
)
def test_example_rejects_wrong_wav_parameters(
    repository_root: Path,
    tmp_path: Path,
    parameters: dict[str, int],
    message: str,
) -> None:
    example = _load_example(repository_root)
    path = tmp_path / "invalid.wav"
    _write_wav(path, **parameters)
    with pytest.raises(ValueError, match=message):
        example.read_pcm16_wav(path)
