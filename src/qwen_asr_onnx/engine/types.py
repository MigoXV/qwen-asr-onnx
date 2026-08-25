from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum

from qwen_asr_onnx.runners.base import RunnerOutput


class EngineState(str, Enum):
    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    FAILED = "failed"
    CLOSED = "closed"


class CancelReason(str, Enum):
    CLIENT = "client"
    DEADLINE = "deadline"
    SHUTDOWN = "shutdown"
    WORKER_FAILED = "worker_failed"


class CancellationToken:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: CancelReason | None = None

    def cancel(self, reason: CancelReason) -> bool:
        with self._lock:
            if self._reason is not None:
                return False
            self._reason = reason
            return True

    @property
    def reason(self) -> CancelReason | None:
        with self._lock:
            return self._reason


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    pcm: bytes
    sample_rate: int
    enqueued_monotonic: float
    deadline_monotonic: float | None
    cancellation: CancellationToken


@dataclass(frozen=True)
class InferenceResult:
    request_id: str
    output: RunnerOutput
    queue_wait_ms: float
    engine_total_ms: float


@dataclass(frozen=True)
class WorkerOutcome:
    request: InferenceRequest
    output: RunnerOutput | None
    error: BaseException | None
    skipped_reason: CancelReason | None
    started_monotonic: float | None
    finished_monotonic: float


@dataclass(frozen=True)
class EngineSnapshot:
    state: EngineState
    current: int
    high_water: int
    accepted: int
    completed: int
    rejected: int
    cancelled: int
    deadline_exceeded: int
    errors: int
    discarded_inflight: int
