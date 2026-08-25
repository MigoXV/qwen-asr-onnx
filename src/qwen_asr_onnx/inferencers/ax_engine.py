"""在专用线程中持有 AX650 runner 的有界异步执行器。"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from qwen_asr_onnx.ax_m4c import AxQwenAsr, TranscriptionMetrics

logger = logging.getLogger(__name__)


class AxEngineError(RuntimeError):
    """AX 执行器自身的生命周期错误。"""


class AxEngineClosedError(AxEngineError):
    """执行器尚未启动或已经关闭。"""


class AxQueueFullError(AxEngineError):
    """AX 执行器已达到 inflight 容量。"""


@dataclass(frozen=True)
class AxInferenceResult:
    raw_output: str
    metrics: TranscriptionMetrics


class AxInferenceEngine:
    """用一个专用线程串行管理唯一的 ``AxQwenAsr`` handle。"""

    def __init__(
        self,
        model_dir: str | Path,
        *,
        warmup: bool = True,
        max_inflight_requests: int = 2,
        runner_factory: Callable[[str | Path], AxQwenAsr] = AxQwenAsr,
    ) -> None:
        if max_inflight_requests < 1:
            raise ValueError("max_inflight_requests must be >= 1")
        self._model_dir = Path(model_dir).expanduser()
        self._warmup = warmup
        self._max_inflight_requests = max_inflight_requests
        self._runner_factory = runner_factory
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ax-qwen-asr",
        )
        self._runner: AxQwenAsr | None = None
        self._pending: set[Future[AxInferenceResult]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._accepting = False

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    async def start(self) -> None:
        if self._started:
            return
        self._loop = asyncio.get_running_loop()
        try:
            future = self._executor.submit(self._initialize_sync)
            await asyncio.wrap_future(future)
        except Exception:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise
        self._started = True
        self._accepting = True

    def _initialize_sync(self) -> None:
        runner = self._runner_factory(self._model_dir)
        try:
            if self._warmup:
                runner.warmup()
        except Exception:
            runner.close()
            raise
        self._runner = runner

    async def transcribe_pcm16(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
    ) -> AxInferenceResult:
        if not self._started or not self._accepting:
            raise AxEngineClosedError("AX inference engine is not accepting requests")
        if len(self._pending) >= self._max_inflight_requests:
            raise AxQueueFullError(
                "AX inference queue is full "
                f"({self._max_inflight_requests} requests including the active request)"
            )

        future = self._executor.submit(
            self._transcribe_sync,
            audio_bytes,
            sample_rate,
        )
        self._pending.add(future)
        future.add_done_callback(self._discard_pending_threadsafe)
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            # 排队中的任务可硬取消；已经进入 AXEngine 的任务只能完成后丢弃结果。
            future.cancel()
            raise

    def _transcribe_sync(
        self,
        audio_bytes: bytes,
        sample_rate: int,
    ) -> AxInferenceResult:
        if self._runner is None:
            raise AxEngineClosedError("AX runner is not initialized")
        raw_output = self._runner.transcribe_pcm16(
            audio_bytes,
            sample_rate=sample_rate,
        )
        return AxInferenceResult(
            raw_output=raw_output,
            metrics=self._runner.last_metrics(),
        )

    def _discard_pending_threadsafe(self, future: Future[AxInferenceResult]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            self._pending.discard(future)
            return
        loop.call_soon_threadsafe(self._pending.discard, future)

    async def close(self, *, grace_seconds: float = 5.0) -> None:
        if not self._started:
            self._executor.shutdown(wait=False, cancel_futures=True)
            return
        self._accepting = False
        deadline = asyncio.get_running_loop().time() + grace_seconds

        while self._pending:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "AX shutdown grace period expired with %d request(s) pending.",
                    len(self._pending),
                )
                break
            await asyncio.sleep(min(0.05, remaining))

        close_future = self._executor.submit(self._close_sync)
        try:
            remaining = max(0.0, deadline - asyncio.get_running_loop().time())
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(close_future)),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            logger.warning("Timed out while closing the AX runner; waiting in background.")
        finally:
            # 超时后仍让已经排队的 _close_sync 执行，确保最终释放 AX handle。
            self._executor.shutdown(wait=False, cancel_futures=False)
            self._started = False

    def _close_sync(self) -> None:
        if self._runner is not None:
            self._runner.close()
            self._runner = None
