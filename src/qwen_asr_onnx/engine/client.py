from __future__ import annotations

from qwen_asr_onnx.engine.core import EngineCore
from collections.abc import Callable

from qwen_asr_onnx.engine.types import (
    EngineSnapshot,
    InferenceResult,
    InferenceTokenEvent,
)


class EngineClient:
    """供 task adapter 使用的窄异步 facade。"""

    def __init__(self, core: EngineCore) -> None:
        self._core = core

    async def infer(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        deadline_monotonic: float | None,
        token_callback: Callable[[InferenceTokenEvent], None] | None = None,
    ) -> InferenceResult:
        return await self._core.infer(
            pcm,
            sample_rate=sample_rate,
            deadline_monotonic=deadline_monotonic,
            token_callback=token_callback,
        )

    def snapshot(self) -> EngineSnapshot:
        return self._core.snapshot()
