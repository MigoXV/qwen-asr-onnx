from __future__ import annotations

from typing import Tuple

from qwen_asr_onnx.inferencers.language import resolve_language_code
from qwen_asr_onnx.inferencers.text.asr_output import parse_asr_output

TranscriptUpdate = Tuple[str, str, bool]


def suppress_incomplete_protocol_prefix(raw_text: str, parsed_text: str) -> str:
    """
    在 gRPC 中间结果中隐藏泄漏的协议前缀。

    流式生成时，模型可能先输出 ``language None`` 或 ``lang`` / ``langute`` 等未完成
    片段。正式转写出现前，这些协议前缀不应下发给客户端。
    """
    text = (parsed_text or "").strip()
    raw = (raw_text or "").strip()
    if not text or not raw:
        return ""

    if "<asr_text>" in raw:
        return parsed_text

    lowered = raw.lower()
    first_token = lowered.split(None, 1)[0]
    if first_token.startswith("lang"):
        return ""
    return parsed_text


class StreamingTranscriptParser:
    """累积模型原始分片，并输出已去除协议前缀的转写更新。"""

    def __init__(self, language_code: str | None = None) -> None:
        self.force_language = resolve_language_code(language_code)
        self.raw_text = ""
        self.transcript = ""

    def push(self, chunk: str) -> TranscriptUpdate:
        self.raw_text += chunk
        _, parsed_text = parse_asr_output(
            self.raw_text,
            user_language=self.force_language,
        )
        parsed_text = suppress_incomplete_protocol_prefix(
            raw_text=self.raw_text,
            parsed_text=parsed_text,
        )
        previous = self.transcript
        delta = (
            parsed_text[len(previous) :]
            if parsed_text.startswith(previous)
            else parsed_text
        )
        self.transcript = parsed_text
        return parsed_text, delta, parsed_text != previous
