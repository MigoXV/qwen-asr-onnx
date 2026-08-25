#!/usr/bin/env python3
"""调用 qwen_asr_onnx.ax_m4c 完成一次 WAV 转写。"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

from qwen_asr_onnx.ax_m4c import AxQwenAsr
from qwen_asr_onnx.inferencers.text.asr_output import parse_asr_output


def read_pcm16_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("只支持单声道 WAV")
        if wav_file.getsampwidth() != 2:
            raise ValueError("只支持 PCM16 WAV")
        if wav_file.getframerate() != 16000:
            raise ValueError("只支持 16000 Hz WAV")
        if wav_file.getcomptype() != "NONE":
            raise ValueError("只支持未压缩 PCM WAV")
        return wav_file.readframes(wav_file.getnframes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="AX650 模型根目录")
    parser.add_argument("wav", type=Path, help="16 kHz 单声道 PCM16 WAV")
    args = parser.parse_args()

    pcm = read_pcm16_wav(args.wav)
    with AxQwenAsr(args.model_dir) as asr:
        raw_output = asr.transcribe_pcm16(pcm, sample_rate=16000)
        _, transcript = parse_asr_output(raw_output)
        print(transcript)


if __name__ == "__main__":
    main()
