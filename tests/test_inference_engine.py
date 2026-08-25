from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from qwen_asr_onnx.engine.core import EngineCore
from qwen_asr_onnx.engine.errors import (
    EngineDeadlineExceededError,
    EngineResourceExhaustedError,
)
from qwen_asr_onnx.engine.worker import AxWorker
from qwen_asr_onnx.runners.base import RunnerMetrics, RunnerOutput
from qwen_asr_onnx.runtime.ax import AxRuntime, RuntimeState


METRICS = RunnerMetrics(1, 1, 1, 1, 1, 1, 1, 1, 1, 5)


class BlockingRunner:
    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.thread_ids: list[int] = []
        self.closed = False

    def load(self, model_dir: Path) -> None:
        self.thread_ids.append(threading.get_ident())

    def warmup(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def infer_pcm16(self, pcm: bytes, *, sample_rate: int) -> RunnerOutput:
        self.thread_ids.append(threading.get_ident())
        self.release.wait(timeout=2)
        return RunnerOutput(
            raw_text="language Chinese<asr_text>测试",
            metrics=METRICS,
        )

    def close(self) -> None:
        self.thread_ids.append(threading.get_ident())
        self.closed = True


def _make_core(
    tmp_path: Path,
    runner: BlockingRunner,
    *,
    max_inflight: int = 2,
) -> EngineCore:
    worker = AxWorker(
        lambda: AxRuntime(tmp_path, runner, warmup=True),
        queue_capacity=max_inflight,
    )
    return EngineCore(worker, max_inflight_requests=max_inflight)


def test_runtime_state_machine(tmp_path) -> None:
    release = threading.Event()
    release.set()
    runner = BlockingRunner(release)
    runtime = AxRuntime(tmp_path, runner, warmup=True)

    assert runtime.state is RuntimeState.NEW
    runtime.start()
    assert runtime.state is RuntimeState.READY
    assert runtime.execute(b"\0\0", sample_rate=16000).raw_text.endswith("测试")
    runtime.close()
    runtime.close()
    assert runtime.state is RuntimeState.CLOSED


def test_worker_owns_runtime_on_one_thread_and_core_is_bounded(tmp_path) -> None:
    release = threading.Event()
    runner = BlockingRunner(release)

    async def scenario() -> None:
        core = _make_core(tmp_path, runner)
        await core.start()
        first = asyncio.create_task(
            core.infer(b"\0\0", sample_rate=16000, deadline_monotonic=None)
        )
        second = asyncio.create_task(
            core.infer(b"\0\0", sample_rate=16000, deadline_monotonic=None)
        )
        await asyncio.sleep(0)
        with pytest.raises(EngineResourceExhaustedError):
            await core.infer(
                b"\0\0",
                sample_rate=16000,
                deadline_monotonic=None,
            )
        assert core.snapshot().current == 2
        assert core.snapshot().high_water == 2
        assert core.snapshot().rejected == 1
        release.set()
        await asyncio.gather(first, second)
        await core.close(grace_seconds=1)

    caller_thread = threading.get_ident()
    asyncio.run(scenario())
    assert runner.closed
    assert len(set(runner.thread_ids)) == 1
    assert runner.thread_ids[0] != caller_thread


def test_queued_deadline_and_client_cancellation(tmp_path) -> None:
    release = threading.Event()
    runner = BlockingRunner(release)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        core = _make_core(tmp_path, runner, max_inflight=3)
        await core.start()
        active = asyncio.create_task(
            core.infer(b"\0\0", sample_rate=16000, deadline_monotonic=None)
        )
        deadline = asyncio.create_task(
            core.infer(
                b"\0\0",
                sample_rate=16000,
                deadline_monotonic=loop.time() + 0.02,
            )
        )
        cancelled = asyncio.create_task(
            core.infer(b"\0\0", sample_rate=16000, deadline_monotonic=None)
        )
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        with pytest.raises(EngineDeadlineExceededError):
            await deadline
        release.set()
        await active
        await core.close(grace_seconds=1)
        snapshot = core.snapshot()
        assert snapshot.deadline_exceeded == 1
        assert snapshot.cancelled == 1

    asyncio.run(scenario())
