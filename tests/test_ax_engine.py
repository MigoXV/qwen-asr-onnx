from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from qwen_asr_onnx.ax_m4c import TranscriptionMetrics
from qwen_asr_onnx.inferencers.ax_engine import AxInferenceEngine, AxQueueFullError


METRICS = TranscriptionMetrics(1, 1, 1, 1, 1, 1, 1, 1, 1, 5)


class FakeRunner:
    def __init__(self, model_dir: str | Path, release: threading.Event) -> None:
        self.model_dir = Path(model_dir)
        self.release = release
        self.thread_ids = [threading.get_ident()]
        self.closed = False

    def warmup(self) -> None:
        self.thread_ids.append(threading.get_ident())

    def transcribe_pcm16(self, pcm: bytes, *, sample_rate: int) -> str:
        self.thread_ids.append(threading.get_ident())
        self.release.wait(timeout=2)
        return "language Chinese<asr_text>测试"

    def last_metrics(self) -> TranscriptionMetrics:
        self.thread_ids.append(threading.get_ident())
        return METRICS

    def close(self) -> None:
        self.thread_ids.append(threading.get_ident())
        self.closed = True


def test_engine_owns_runner_on_one_dedicated_thread(tmp_path) -> None:
    release = threading.Event()
    release.set()
    runners: list[FakeRunner] = []

    def factory(model_dir: str | Path) -> FakeRunner:
        runner = FakeRunner(model_dir, release)
        runners.append(runner)
        return runner

    async def scenario() -> None:
        engine = AxInferenceEngine(tmp_path, runner_factory=factory)
        await engine.start()
        result = await engine.transcribe_pcm16(b"\0\0", sample_rate=16000)
        assert result.raw_output.endswith("测试")
        await engine.close()

    caller_thread = threading.get_ident()
    asyncio.run(scenario())
    assert runners[0].closed
    assert len(set(runners[0].thread_ids)) == 1
    assert runners[0].thread_ids[0] != caller_thread


def test_engine_rejects_third_inflight_request(tmp_path) -> None:
    release = threading.Event()

    async def scenario() -> None:
        engine = AxInferenceEngine(
            tmp_path,
            max_inflight_requests=2,
            runner_factory=lambda model_dir: FakeRunner(model_dir, release),
        )
        await engine.start()
        first = asyncio.create_task(
            engine.transcribe_pcm16(b"\0\0", sample_rate=16000)
        )
        second = asyncio.create_task(
            engine.transcribe_pcm16(b"\0\0", sample_rate=16000)
        )
        await asyncio.sleep(0)
        with pytest.raises(AxQueueFullError):
            await engine.transcribe_pcm16(b"\0\0", sample_rate=16000)
        release.set()
        await asyncio.gather(first, second)
        await engine.close()

    asyncio.run(scenario())
