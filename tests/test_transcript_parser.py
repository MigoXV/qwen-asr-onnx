import asyncio

from qwen_asr_onnx.engine.types import InferenceResult
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
