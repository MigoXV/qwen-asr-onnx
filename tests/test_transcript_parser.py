import asyncio

from qwen_asr_onnx.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr_onnx.inferencers.text.asr_output import (
    detect_and_fix_repetitions,
    parse_asr_output,
)
from qwen_asr_onnx.inferencers.text.transcript_parser import StreamingTranscriptParser


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


def test_streaming_parser_suppresses_incomplete_protocol_prefix() -> None:
    parser = StreamingTranscriptParser(language_code="")

    transcript, delta, changed = parser.push("lang")
    assert transcript == ""
    assert delta == ""
    assert changed is False

    transcript, delta, changed = parser.push("uage Chinese<asr_text>你好")
    assert transcript == "你好"
    assert delta == "你好"
    assert changed is True


def test_detect_and_fix_repetitions_collapses_long_pattern() -> None:
    assert detect_and_fix_repetitions("哈哈" * 25) == "哈"


def test_grpc_inferencer_yields_cleaned_interim_deltas() -> None:
    class FakePipeline:
        def transcribe(self, **kwargs):
            assert kwargs["language"] == "Chinese"
            yield from "language Chinese<asr_text>你好"

        def close(self) -> None:
            return None

    async def collect() -> list[tuple[str, str, bool]]:
        events: list[tuple[str, str, bool]] = []
        async for item in GrpcInferencer(FakePipeline()).infer(
            audio_bytes=b"00",
            sample_rate=16000,
            language_code="zh-CN",
            interim_results=True,
        ):
            events.append(item)
        return events

    events = asyncio.run(collect())

    assert events[-3:-1] == [("你", "你", False), ("你好", "好", False)]
    assert events[-1] == ("你好", "", True)
