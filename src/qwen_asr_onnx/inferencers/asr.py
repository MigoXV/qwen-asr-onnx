from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass

from qwen_asr_onnx.engine.client import EngineClient
from qwen_asr_onnx.engine.types import InferenceResult, InferenceTokenEvent
from qwen_asr_onnx.inferencers.language import resolve_language_code
from qwen_asr_onnx.inferencers.text.asr_output import (
    AsrOutputStreamParser,
    parse_asr_output,
)
from qwen_asr_onnx.runners.base import RunnerMetrics


@dataclass(frozen=True)
class AsrResult:
    transcript: str
    language: str
    request_id: str
    queue_wait_ms: float
    engine_total_ms: float
    metrics: RunnerMetrics


@dataclass(frozen=True)
class AsrStreamEvent:
    transcript: str
    language: str
    is_final: bool
    request_id: str
    emitted_monotonic: float
    result: AsrResult | None = None


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
        return self._make_result(result, requested_language)

    async def stream(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        language_code: str,
        deadline_monotonic: float | None,
    ) -> AsyncIterator[AsrStreamEvent]:
        """输出累计 interim，最后输出带完整 metrics 的 final 事件。"""
        requested_language = resolve_language_code(language_code)
        parser = AsrOutputStreamParser(requested_language)
        events: asyncio.Queue[InferenceTokenEvent] = asyncio.Queue(maxsize=256)
        inference_task = asyncio.create_task(
            self._engine.infer(
                pcm,
                sample_rate=sample_rate,
                deadline_monotonic=deadline_monotonic,
                token_callback=events.put_nowait,
            )
        )
        try:
            while not inference_task.done():
                event_task = asyncio.create_task(events.get())
                done, _ = await asyncio.wait(
                    {event_task, inference_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if event_task in done:
                    event = event_task.result()
                    parsed = parser.feed(event.text_delta)
                    if parsed is not None:
                        language, transcript = parsed
                        yield AsrStreamEvent(
                            transcript=transcript,
                            language=language,
                            is_final=False,
                            request_id=event.request_id,
                            emitted_monotonic=event.emitted_monotonic,
                        )
                else:
                    event_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await event_task

            while not events.empty():
                event = events.get_nowait()
                parsed = parser.feed(event.text_delta)
                if parsed is not None:
                    language, transcript = parsed
                    yield AsrStreamEvent(
                        transcript=transcript,
                        language=language,
                        is_final=False,
                        request_id=event.request_id,
                        emitted_monotonic=event.emitted_monotonic,
                    )

            result = self._make_result(
                await inference_task,
                requested_language,
            )
            yield AsrStreamEvent(
                transcript=result.transcript,
                language=result.language,
                is_final=True,
                request_id=result.request_id,
                emitted_monotonic=asyncio.get_running_loop().time(),
                result=result,
            )
        finally:
            if not inference_task.done():
                inference_task.cancel()
                with suppress(asyncio.CancelledError):
                    await inference_task

    @staticmethod
    def _make_result(
        result: InferenceResult,
        requested_language: str,
    ) -> AsrResult:
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
