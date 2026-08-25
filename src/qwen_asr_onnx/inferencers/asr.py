from __future__ import annotations

from dataclasses import dataclass

from qwen_asr_onnx.engine.client import EngineClient
from qwen_asr_onnx.inferencers.language import resolve_language_code
from qwen_asr_onnx.inferencers.text.asr_output import parse_asr_output
from qwen_asr_onnx.runners.base import RunnerMetrics


@dataclass(frozen=True)
class AsrResult:
    transcript: str
    language: str
    request_id: str
    queue_wait_ms: float
    engine_total_ms: float
    metrics: RunnerMetrics


class AsrInferencer:
    """不感知 protobuf 的 ASR task adapter。"""

    def __init__(self, engine: EngineClient) -> None:
        self._engine = engine

    async def infer(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        language_code: str,
        deadline_monotonic: float | None,
    ) -> AsrResult:
        result = await self._engine.infer(
            pcm,
            sample_rate=sample_rate,
            deadline_monotonic=deadline_monotonic,
        )
        requested_language = resolve_language_code(language_code)
        language, transcript = parse_asr_output(
            result.output.raw_text,
            user_language=requested_language,
        )
        return AsrResult(
            transcript=transcript,
            language=language,
            request_id=result.request_id,
            queue_wait_ms=result.queue_wait_ms,
            engine_total_ms=result.engine_total_ms,
            metrics=result.output.metrics,
        )
