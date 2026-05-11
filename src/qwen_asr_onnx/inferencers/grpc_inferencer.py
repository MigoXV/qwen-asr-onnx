# coding=utf-8
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from qwen_asr_onnx.inferencers.onnx import OnnxAsrPipeline
from qwen_asr_onnx.inferencers.text.transcript_parser import StreamingTranscriptParser


class GrpcInferencer:
    """Adapter that exposes ONNX decoding as gRPC-friendly transcript events."""

    def __init__(self, inferencer: OnnxAsrPipeline) -> None:
        self.inferencer = inferencer

    async def infer(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        language_code: str = "",
        interim_results: bool = False,
        context: str = "",
    ) -> AsyncIterator[tuple[str, str, bool]]:
        transcript_parser = StreamingTranscriptParser(language_code=language_code)

        for delta in self.inferencer.transcribe(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            language=transcript_parser.force_language,
            context=context,
        ):
            transcript, parsed_delta, changed = transcript_parser.push(delta)
            if interim_results and changed:
                yield transcript, parsed_delta, False
                await asyncio.sleep(0)

        yield transcript_parser.transcript, "", True

    def close(self) -> None:
        self.inferencer.close()
