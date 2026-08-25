from __future__ import annotations

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from qwen_asr_onnx.engine.types import (
    CancelReason,
    InferenceRequest,
    WorkerOutcome,
)
from qwen_asr_onnx.runners.base import RunnerOutput
from qwen_asr_onnx.runtime.ax import AxRuntime


@dataclass(frozen=True)
class _WorkItem:
    request: InferenceRequest
    callback: Callable[[WorkerOutcome], None]


class AxWorker:
    """唯一 AX 设备所有者；runtime 的全部调用都发生在该线程。"""

    def __init__(
        self,
        runtime_factory: Callable[[], AxRuntime],
        *,
        queue_capacity: int,
        fatal_callback: Callable[[BaseException], None] | None = None,
    ) -> None:
        if queue_capacity < 1:
            raise ValueError("queue_capacity must be >= 1")
        self._runtime_factory = runtime_factory
        self._queue: queue.Queue[_WorkItem] = queue.Queue(maxsize=queue_capacity)
        self._fatal_callback = fatal_callback
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready: asyncio.Future[None] | None = None
        self._closed: asyncio.Future[None] | None = None

    def set_fatal_callback(
        self,
        callback: Callable[[BaseException], None],
    ) -> None:
        self._fatal_callback = callback

    async def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("AX worker has already been started")
        self._loop = asyncio.get_running_loop()
        self._ready = self._loop.create_future()
        self._closed = self._loop.create_future()
        self._thread = threading.Thread(
            target=self._run,
            name="ax-qwen-asr-worker",
            daemon=False,
        )
        self._thread.start()
        await self._ready

    def submit(
        self,
        request: InferenceRequest,
        callback: Callable[[WorkerOutcome], None],
    ) -> None:
        if self._thread is None or self._stop_requested.is_set():
            raise RuntimeError("AX worker is not accepting requests")
        self._queue.put_nowait(_WorkItem(request=request, callback=callback))

    async def close(self) -> None:
        if self._thread is None:
            return
        self._stop_requested.set()
        assert self._closed is not None
        await asyncio.shield(self._closed)
        await asyncio.to_thread(self._thread.join)
        self._thread = None

    def _run(self) -> None:
        runtime: AxRuntime | None = None
        started = False
        fatal: BaseException | None = None
        try:
            runtime = self._runtime_factory()
            runtime.start()
            started = True
            self._notify_future(self._ready, None)

            while not (self._stop_requested.is_set() and self._queue.empty()):
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    self._execute_item(runtime, item)
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            fatal = exc
            if not started:
                self._notify_future(self._ready, exc)
            else:
                if self._fatal_callback is not None:
                    self._notify(self._fatal_callback, exc)
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except BaseException as close_exc:
                    if fatal is None and started and self._fatal_callback is not None:
                        self._notify(self._fatal_callback, close_exc)
            self._notify_future(self._closed, None)

    def _execute_item(self, runtime: AxRuntime, item: _WorkItem) -> None:
        request = item.request
        now = time.monotonic()
        reason = request.cancellation.reason
        if reason is None and (
            request.deadline_monotonic is not None
            and now >= request.deadline_monotonic
        ):
            request.cancellation.cancel(CancelReason.DEADLINE)
            reason = CancelReason.DEADLINE
        if reason is not None:
            self._complete(item, None, None, reason, None)
            return

        started = time.monotonic()
        output = None
        error = None
        try:
            output = runtime.execute(request.pcm, sample_rate=request.sample_rate)
        except BaseException as exc:
            error = exc
        finished = time.monotonic()

        reason = request.cancellation.reason
        if reason is None and (
            request.deadline_monotonic is not None
            and finished >= request.deadline_monotonic
        ):
            request.cancellation.cancel(CancelReason.DEADLINE)
            reason = CancelReason.DEADLINE
        self._complete(item, output, error, reason, started, finished)

    def _complete(
        self,
        item: _WorkItem,
        output: RunnerOutput | None,
        error: BaseException | None,
        reason: CancelReason | None,
        started: float | None,
        finished: float | None = None,
    ) -> None:
        outcome = WorkerOutcome(
            request=item.request,
            output=output,
            error=error,
            skipped_reason=reason,
            started_monotonic=started,
            finished_monotonic=finished or time.monotonic(),
        )
        self._notify(item.callback, outcome)

    def _notify(self, callback, *args) -> None:
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(callback, *args)

    def _notify_future(
        self,
        future: asyncio.Future[None] | None,
        error: BaseException | None,
    ) -> None:
        if future is None:
            return

        def finish() -> None:
            if future.done():
                return
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(error)

        self._notify(finish)
