from __future__ import annotations

import asyncio
import logging
import queue
import time
import uuid
from dataclasses import dataclass
from typing import Callable

from qwen_asr_onnx.engine.errors import (
    EngineDeadlineExceededError,
    EngineResourceExhaustedError,
    EngineUnavailableError,
)
from qwen_asr_onnx.engine.observability import log_event
from qwen_asr_onnx.engine.types import (
    CancelReason,
    CancellationToken,
    EngineSnapshot,
    EngineState,
    InferenceRequest,
    InferenceResult,
    WorkerOutcome,
)
from qwen_asr_onnx.engine.worker import AxWorker

logger = logging.getLogger(__name__)


@dataclass
class _Pending:
    request: InferenceRequest
    future: asyncio.Future[InferenceResult]


class EngineCore:
    """事件循环内的准入、请求关联、deadline、取消和统计核心。"""

    def __init__(self, worker: AxWorker, *, max_inflight_requests: int) -> None:
        if max_inflight_requests < 1:
            raise ValueError("max_inflight_requests must be >= 1")
        self._worker = worker
        self._worker.set_fatal_callback(self.worker_fatal_callback)
        self._max_inflight_requests = max_inflight_requests
        self._state = EngineState.NEW
        self._pending: dict[str, _Pending] = {}
        self._idle = asyncio.Event()
        self._idle.set()
        self._state_listeners: list[Callable[[EngineState], None]] = []
        self._high_water = 0
        self._accepted = 0
        self._completed = 0
        self._rejected = 0
        self._cancelled = 0
        self._deadline_exceeded = 0
        self._errors = 0
        self._discarded_inflight = 0

    @property
    def state(self) -> EngineState:
        return self._state

    def add_state_listener(self, listener: Callable[[EngineState], None]) -> None:
        self._state_listeners.append(listener)

    async def start(self) -> None:
        if self._state is not EngineState.NEW:
            raise RuntimeError(f"cannot start engine from state {self._state.value}")
        self._set_state(EngineState.STARTING)
        try:
            await self._worker.start()
        except Exception:
            self._set_state(EngineState.FAILED)
            raise
        self._set_state(EngineState.READY)
        self._emit_snapshot("engine_ready")

    async def infer(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        deadline_monotonic: float | None,
    ) -> InferenceResult:
        now = time.monotonic()
        if self._state is not EngineState.READY:
            raise EngineUnavailableError(
                f"inference engine is not ready: {self._state.value}"
            )
        if deadline_monotonic is not None and now >= deadline_monotonic:
            self._deadline_exceeded += 1
            raise EngineDeadlineExceededError("request deadline already expired")
        if len(self._pending) >= self._max_inflight_requests:
            self._rejected += 1
            self._emit_snapshot("engine_rejected")
            raise EngineResourceExhaustedError(
                "inference capacity is full "
                f"({self._max_inflight_requests} requests including active)"
            )

        request_id = uuid.uuid4().hex
        request = InferenceRequest(
            request_id=request_id,
            pcm=pcm,
            sample_rate=sample_rate,
            enqueued_monotonic=now,
            deadline_monotonic=deadline_monotonic,
            cancellation=CancellationToken(),
        )
        future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = _Pending(request=request, future=future)
        self._idle.clear()
        self._accepted += 1
        self._high_water = max(self._high_water, len(self._pending))
        try:
            self._worker.submit(request, self._handle_outcome)
        except (queue.Full, RuntimeError) as exc:
            self._pending.pop(request_id, None)
            self._rejected += 1
            if not self._pending:
                self._idle.set()
            raise EngineResourceExhaustedError(str(exc)) from exc

        self._emit_snapshot("engine_accepted", request_id=request_id)
        try:
            if deadline_monotonic is None:
                return await asyncio.shield(future)
            remaining = max(0.0, deadline_monotonic - time.monotonic())
            return await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
        except asyncio.TimeoutError as exc:
            if request.cancellation.cancel(CancelReason.DEADLINE):
                self._deadline_exceeded += 1
            future.cancel()
            self._emit_snapshot("engine_deadline", request_id=request_id)
            raise EngineDeadlineExceededError("request deadline exceeded") from exc
        except asyncio.CancelledError:
            if request.cancellation.cancel(CancelReason.CLIENT):
                self._cancelled += 1
            future.cancel()
            self._emit_snapshot("engine_cancelled", request_id=request_id)
            raise

    async def close(self, *, grace_seconds: float) -> None:
        if self._state is EngineState.CLOSED:
            return
        if self._state is not EngineState.FAILED:
            self._set_state(EngineState.DRAINING)
        deadline = time.monotonic() + grace_seconds
        if self._pending:
            try:
                await asyncio.wait_for(
                    self._idle.wait(),
                    timeout=max(0.0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                for pending in self._pending.values():
                    if pending.request.cancellation.cancel(CancelReason.SHUTDOWN):
                        self._cancelled += 1
                    pending.future.cancel()
                self._emit_snapshot("engine_shutdown_cancel")

        await self._worker.close()
        self._set_state(EngineState.CLOSED)
        self._emit_snapshot("engine_closed")

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            state=self._state,
            current=len(self._pending),
            high_water=self._high_water,
            accepted=self._accepted,
            completed=self._completed,
            rejected=self._rejected,
            cancelled=self._cancelled,
            deadline_exceeded=self._deadline_exceeded,
            errors=self._errors,
            discarded_inflight=self._discarded_inflight,
        )

    def _handle_outcome(self, outcome: WorkerOutcome) -> None:
        pending = self._pending.pop(outcome.request.request_id, None)
        if pending is None:
            return
        future = pending.future
        reason = outcome.skipped_reason
        if reason is not None:
            if outcome.started_monotonic is not None:
                self._discarded_inflight += 1
            if not future.done():
                if reason is CancelReason.DEADLINE:
                    self._deadline_exceeded += 1
                    future.set_exception(
                        EngineDeadlineExceededError("request deadline exceeded")
                    )
                else:
                    self._cancelled += 1
                    future.cancel()
        elif outcome.error is not None:
            self._errors += 1
            if not future.done():
                future.set_exception(outcome.error)
        elif outcome.output is not None:
            self._completed += 1
            if not future.done():
                started = outcome.started_monotonic or outcome.finished_monotonic
                future.set_result(
                    InferenceResult(
                        request_id=outcome.request.request_id,
                        output=outcome.output,
                        queue_wait_ms=(
                            started - outcome.request.enqueued_monotonic
                        )
                        * 1000.0,
                        engine_total_ms=(
                            outcome.finished_monotonic
                            - outcome.request.enqueued_monotonic
                        )
                        * 1000.0,
                    )
                )
        if not self._pending:
            self._idle.set()
        self._emit_snapshot(
            "engine_completed",
            request_id=outcome.request.request_id,
            reason=reason.value if reason else "success",
        )

    def _handle_worker_fatal(self, error: BaseException) -> None:
        self._errors += 1
        self._set_state(EngineState.FAILED)
        for pending in self._pending.values():
            pending.request.cancellation.cancel(CancelReason.WORKER_FAILED)
            if not pending.future.done():
                pending.future.set_exception(
                    EngineUnavailableError(f"AX worker failed: {error}")
                )
        self._pending.clear()
        self._idle.set()
        logger.error(
            "AX worker failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self._emit_snapshot("engine_failed", error=str(error))

    def worker_fatal_callback(self, error: BaseException) -> None:
        self._handle_worker_fatal(error)

    def _set_state(self, state: EngineState) -> None:
        self._state = state
        for listener in tuple(self._state_listeners):
            listener(state)

    def _emit_snapshot(self, event: str, **fields) -> None:
        snapshot = self.snapshot()
        log_event(
            logger,
            event,
            state=snapshot.state.value,
            current=snapshot.current,
            high_water=snapshot.high_water,
            accepted=snapshot.accepted,
            completed=snapshot.completed,
            rejected=snapshot.rejected,
            cancelled=snapshot.cancelled,
            deadline_exceeded=snapshot.deadline_exceeded,
            errors=snapshot.errors,
            discarded_inflight=snapshot.discarded_inflight,
            **fields,
        )
