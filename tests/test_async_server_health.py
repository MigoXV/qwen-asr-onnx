from __future__ import annotations

import asyncio
import socket
import threading
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from qwen_asr_onnx.commands.app import ASR_SERVICE_NAME, ServeOptions, run_server


class SlowStartRuntime:
    def __init__(self, release: threading.Event) -> None:
        self.release = release
        self.closed = False

    def start(self) -> None:
        self.release.wait(timeout=2)

    def execute(self, pcm: bytes, *, sample_rate: int):
        raise AssertionError("health test must not execute inference")

    def close(self) -> None:
        self.closed = True


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_async_server_exposes_health_during_runtime_startup(tmp_path) -> None:
    release = threading.Event()
    runtime = SlowStartRuntime(release)
    port = _unused_port()

    async def wait_status(stub, expected: int) -> None:
        deadline = asyncio.get_running_loop().time() + 2
        while True:
            try:
                response = await stub.Check(
                    health_pb2.HealthCheckRequest(service=ASR_SERVICE_NAME),
                    timeout=0.2,
                )
                if response.status == expected:
                    return
            except grpc.aio.AioRpcError:
                pass
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError(f"health status did not become {expected}")
            await asyncio.sleep(0.01)

    async def scenario() -> None:
        options = ServeOptions(
            model=Path(tmp_path),
            host="127.0.0.1",
            port=port,
            warmup=True,
            max_inflight_requests=2,
            shutdown_grace_seconds=1,
        )
        server_task = asyncio.create_task(
            run_server(options, runtime_factory=lambda: runtime)
        )
        async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
            stub = health_pb2_grpc.HealthStub(channel)
            await wait_status(
                stub,
                health_pb2.HealthCheckResponse.NOT_SERVING,
            )
            release.set()
            await wait_status(
                stub,
                health_pb2.HealthCheckResponse.SERVING,
            )
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    assert runtime.closed
