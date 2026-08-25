#!/usr/bin/env python3
"""调用 Qwen3-ASR AX650 gRPC 服务转写一个完整 WAV 文件。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import grpc

from qwen_asr_onnx.protos.asr.ux_speech_pb2 import (
    RecognitionConfig,
    StreamingRecognizeRequest,
    StreamingRecognitionConfig,
)
from qwen_asr_onnx.protos.asr.ux_speech_pb2_grpc import UxSpeechStub


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


def requests(
    pcm: bytes,
    language: str,
    *,
    interim_results: bool,
) -> Iterator[StreamingRecognizeRequest]:
    yield StreamingRecognizeRequest(
        streaming_config=StreamingRecognitionConfig(
            config=RecognitionConfig(
                encoding=RecognitionConfig.LINEAR16,
                sample_rate_hertz=16000,
                language_code=language,
            ),
            interim_results=interim_results,
        )
    )
    yield StreamingRecognizeRequest(audio_content=pcm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="16 kHz 单声道 PCM16 WAV")
    parser.add_argument("--target", default="127.0.0.1:50051")
    parser.add_argument("--language", default="", help="可选 ISO/BCP-47 语言代码")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--interim-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="逐 token 返回累计文本；使用 --no-interim-results 只取 final",
    )
    parser.add_argument(
        "--show-timing",
        action="store_true",
        help="把每条响应的到达时间和 final 状态以 JSON 输出到 stderr",
    )
    args = parser.parse_args()

    pcm = read_pcm16_wav(args.wav)
    started = time.monotonic()
    with grpc.insecure_channel(args.target) as channel:
        responses = UxSpeechStub(channel).StreamingRecognize(
            requests(
                pcm,
                args.language,
                interim_results=args.interim_results,
            ),
            timeout=args.timeout,
        )
        for response in responses:
            for result in response.results:
                if args.show_timing:
                    print(
                        json.dumps(
                            {
                                "arrival_ms": round(
                                    (time.monotonic() - started) * 1000.0,
                                    3,
                                ),
                                "is_final": result.is_final,
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                    )
                print(result.alternative.transcript, flush=True)


if __name__ == "__main__":
    main()
