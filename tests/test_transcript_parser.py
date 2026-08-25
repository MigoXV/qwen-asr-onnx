import asyncio

from qwen_asr_onnx.ax_m4c import TranscriptionMetrics
from qwen_asr_onnx.inferencers.ax_engine import AxInferenceResult
from qwen_asr_onnx.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr_onnx.inferencers.text.asr_output import (
    detect_and_fix_repetitions,
    parse_asr_output,
)


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


def test_grpc_inferencer_returns_one_clean_final_result() -> None:
    class FakeEngine:
        async def transcribe_pcm16(self, audio_bytes: bytes, *, sample_rate: int):
            assert audio_bytes == b"00"
            assert sample_rate == 16000
            return AxInferenceResult(
                raw_output="language Chinese<asr_text>你好",
                metrics=TranscriptionMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            )

    async def infer():
        return await GrpcInferencer(FakeEngine()).infer(
            audio_bytes=b"00",
            sample_rate=16000,
            language_code="zh-CN",
        )

    result = asyncio.run(infer())
    assert result.transcript == "你好"
    assert result.language == "Chinese"
    assert "language" not in result.transcript
    assert "<asr_text>" not in result.transcript
