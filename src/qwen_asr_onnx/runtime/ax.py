from __future__ import annotations

from enum import Enum
from pathlib import Path

from qwen_asr_onnx.runners.base import ModelRunner, RunnerOutput


class RuntimeState(str, Enum):
    NEW = "new"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


class AxRuntime:
    """拥有 AX runner 生命周期；调用方必须保证线程亲和性。"""

    def __init__(
        self,
        model_dir: Path,
        runner: ModelRunner,
        *,
        warmup: bool,
    ) -> None:
        self._model_dir = model_dir
        self._runner = runner
        self._warmup = warmup
        self._state = RuntimeState.NEW

    @property
    def state(self) -> RuntimeState:
        return self._state

    def start(self) -> None:
        if self._state is not RuntimeState.NEW:
            raise RuntimeError(f"cannot start AX runtime from state {self._state.value}")
        self._state = RuntimeState.LOADING
        try:
            self._runner.load(self._model_dir)
            if self._warmup:
                self._runner.warmup()
        except Exception:
            self._state = RuntimeState.FAILED
            self._runner.close()
            raise
        self._state = RuntimeState.READY

    def execute(self, pcm: bytes, *, sample_rate: int) -> RunnerOutput:
        if self._state is not RuntimeState.READY:
            raise RuntimeError(f"AX runtime is not ready: {self._state.value}")
        return self._runner.infer_pcm16(pcm, sample_rate=sample_rate)

    def close(self) -> None:
        if self._state is RuntimeState.CLOSED:
            return
        self._state = RuntimeState.CLOSING
        try:
            self._runner.close()
        finally:
            self._state = RuntimeState.CLOSED
