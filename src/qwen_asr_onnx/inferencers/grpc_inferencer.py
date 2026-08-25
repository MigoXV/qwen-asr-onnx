"""AX 推理结果到 gRPC 领域结果的适配。"""

from __future__ import annotations

from dataclasses import dataclass

from qwen_asr_onnx.inferencers.ax_engine import (
    AxInferenceEngine,
    AxInferenceResult,
)
from qwen_asr_onnx.inferencers.language import resolve_language_code
from qwen_asr_onnx.inferencers.text.asr_output import parse_asr_output


@dataclass(frozen=True)
class TranscriptResult:
    transcript: str
    language: str
    inference: AxInferenceResult


class GrpcInferencer:
    """调用 AX engine，并保证模型协议前缀不会进入客户端正文。"""

    def __init__(self, engine: AxInferenceEngine) -> None:
        self.engine = engine

    async def infer(
        self,
        audio_bytes: bytes,
        sample_rate: int,
        language_code: str = "",
    ) -> TranscriptResult:
        inference = await self.engine.transcribe_pcm16(
            audio_bytes,
            sample_rate=sample_rate,
        )
        requested_language = resolve_language_code(language_code)
        language, transcript = parse_asr_output(
            inference.raw_output,
            user_language=requested_language,
        )
        return TranscriptResult(
            transcript=transcript,
            language=language,
            inference=inference,
        )
