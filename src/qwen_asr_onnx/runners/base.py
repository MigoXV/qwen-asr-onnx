from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import Protocol


@dataclass(frozen=True)
class RunnerMetrics:
    sample_count: int
    mel_frame_count: int
    audio_token_count: int
    prompt_token_count: int
    generated_token_count: int
    preprocess_ms: float
    encoder_ms: float
    decoder_ttft_ms: float
    decoder_ms: float
    native_total_ms: float


@dataclass(frozen=True)
class RunnerOutput:
    raw_text: str
    metrics: RunnerMetrics


@dataclass(frozen=True)
class RunnerToken:
    token_id: int
    sequence: int
    text_delta: str


RunnerTokenCallback = Callable[[RunnerToken], bool | None]


class ModelRunner(Protocol):
    """同步后端策略的最小内部契约。"""

    def load(self, model_dir: Path) -> None: ...

    def warmup(self) -> None: ...

    def infer_pcm16(self, pcm: bytes, *, sample_rate: int) -> RunnerOutput: ...

    def infer_pcm16_stream(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        token_callback: RunnerTokenCallback,
    ) -> RunnerOutput: ...

    def close(self) -> None: ...
