import asyncio

from qwen_asr_onnx.engine.types import InferenceResult, InferenceTokenEvent
from qwen_asr_onnx.inferencers.asr import AsrInferencer
from qwen_asr_onnx.inferencers.text.asr_output import (
    detect_and_fix_repetitions,
    parse_asr_output,
)
from qwen_asr_onnx.runners.base import RunnerMetrics, RunnerOutput


def test_parse_asr_output_strips_protocol_prefix() -> None:
    language, text = parse_asr_output("language Chinese<asr_text>你好")

    assert language == "Chinese"
    assert text == "你好"


def test_parse_asr_output_strips_forced_language_prefix() -> None:
    language, text = parse_asr_output(
        "language English<asr_text>Hello",
        user_language="English",
    )

    assert language == "English"
    assert text == "Hello"


def test_detect_and_fix_repetitions_collapses_long_pattern() -> None:
    assert detect_and_fix_repetitions("哈哈" * 25) == "哈"


def test_asr_inferencer_returns_one_clean_domain_result() -> None:
    class FakeEngineClient:
        async def infer(
            self,
            audio_bytes: bytes,
            *,
            sample_rate: int,
            deadline_monotonic: float | None,
        ):
            assert audio_bytes == b"00"
            assert sample_rate == 16000
            assert deadline_monotonic == 123.0
            return InferenceResult(
                request_id="request-1",
                output=RunnerOutput(
                    raw_text="language Chinese<asr_text>你好",
                    metrics=RunnerMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                ),
                queue_wait_ms=1.0,
                engine_total_ms=2.0,
            )

    async def infer():
        return await AsrInferencer(FakeEngineClient()).infer(
            b"00",
            sample_rate=16000,
            language_code="zh-CN",
            deadline_monotonic=123.0,
        )

    result = asyncio.run(infer())
    assert result.transcript == "你好"
    assert result.language == "Chinese"
    assert result.request_id == "request-1"
    assert "language" not in result.transcript
    assert "<asr_text>" not in result.transcript


def test_asr_inferencer_streams_clean_cumulative_text() -> None:
    class FakeStreamingEngineClient:
        async def infer(
            self,
            audio_bytes: bytes,
            *,
            sample_rate: int,
            deadline_monotonic: float | None,
            token_callback=None,
        ) -> InferenceResult:
            assert token_callback is not None
            for sequence, delta in enumerate(
                ["language Chinese<asr_", "text>你", "好"]
            ):
                token_callback(
                    InferenceTokenEvent(
                        request_id="request-1",
                        sequence=sequence,
                        token_id=100 + sequence,
                        text_delta=delta,
                        emitted_monotonic=float(sequence),
                    )
                )
            return InferenceResult(
                request_id="request-1",
                output=RunnerOutput(
                    raw_text="language Chinese<asr_text>你好",
                    metrics=RunnerMetrics(0, 0, 0, 0, 3, 0, 0, 0, 0, 0),
                ),
                queue_wait_ms=1.0,
                engine_total_ms=2.0,
            )

    async def collect():
        return [
            event
            async for event in AsrInferencer(FakeStreamingEngineClient()).stream(
                b"00",
                sample_rate=16000,
                language_code="",
                deadline_monotonic=None,
            )
        ]

    events = asyncio.run(collect())
    assert [event.transcript for event in events] == ["你", "你好", "你好"]
    assert [event.is_final for event in events] == [False, False, True]
    assert all("language" not in event.transcript for event in events)
    assert events[-1].result is not None
