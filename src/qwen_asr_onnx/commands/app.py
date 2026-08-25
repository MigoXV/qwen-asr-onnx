# coding=utf-8
"""Qwen3-ASR AX650 异步 gRPC 服务的 Typer CLI。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import grpc
import typer
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from qwen_asr_onnx.engine.client import EngineClient
from qwen_asr_onnx.engine.core import EngineCore
from qwen_asr_onnx.engine.observability import log_event
from qwen_asr_onnx.engine.types import EngineState
from qwen_asr_onnx.engine.worker import AxWorker
from qwen_asr_onnx.inferencers.asr import AsrInferencer
from qwen_asr_onnx.protos.asr.ux_speech_pb2_grpc import (
    add_UxSpeechServicer_to_server,
)
from qwen_asr_onnx.runners.ax import AxModelRunner
from qwen_asr_onnx.runtime.ax import AxRuntime
from qwen_asr_onnx.servicer.servicer import ASRServicer

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
ASR_SERVICE_NAME = "ux_speech.UxSpeech"
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="qwen-asr",
    help="Qwen3-ASR AX650 gRPC 推理服务。",
    add_completion=False,
)


@dataclass(frozen=True)
class ServeOptions:
    model: Path
    host: str
    port: int
    warmup: bool
    max_inflight_requests: int
    shutdown_grace_seconds: float


def _normalize_log_level(value: str) -> str:
    normalized = str(value).strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        choices = ", ".join(sorted(VALID_LOG_LEVELS))
        raise typer.BadParameter(f"日志级别必须是：{choices}")
    return normalized


async def _set_health(
    health_servicer: health.aio.HealthServicer,
    status: health_pb2.HealthCheckResponse.ServingStatus,
) -> None:
    await health_servicer.set("", status)
    await health_servicer.set(ASR_SERVICE_NAME, status)


async def run_server(
    options: ServeOptions,
    *,
    runtime_factory: Callable[[], AxRuntime] | None = None,
) -> None:
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024),
        ]
    )
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    if runtime_factory is None:
        runtime_factory = lambda: AxRuntime(
            options.model,
            AxModelRunner(),
            warmup=options.warmup,
        )
    worker = AxWorker(
        runtime_factory,
        queue_capacity=options.max_inflight_requests,
    )
    core = EngineCore(
        worker,
        max_inflight_requests=options.max_inflight_requests,
    )
    client = EngineClient(core)
    add_UxSpeechServicer_to_server(
        ASRServicer(AsrInferencer(client)),
        server,
    )

    def on_engine_state(state: EngineState) -> None:
        if state is EngineState.FAILED:
            asyncio.create_task(
                _set_health(
                    health_servicer,
                    health_pb2.HealthCheckResponse.NOT_SERVING,
                )
            )

    core.add_state_listener(on_engine_state)
    address = f"{options.host}:{options.port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"无法绑定 gRPC 地址：{address}")

    started = False
    cancellation: asyncio.CancelledError | None = None
    try:
        await _set_health(
            health_servicer,
            health_pb2.HealthCheckResponse.NOT_SERVING,
        )
        await server.start()
        started = True
        log_event(
            logger,
            "grpc_started_not_ready",
            address=address,
            model=str(options.model),
        )
        await core.start()
        await _set_health(
            health_servicer,
            health_pb2.HealthCheckResponse.SERVING,
        )
        log_event(
            logger,
            "grpc_ready",
            address=address,
            model=str(options.model),
            backend="ax650",
            profile="C64/P448/CTX2047-BF16",
            warmup=options.warmup,
            max_inflight_requests=options.max_inflight_requests,
        )
        await server.wait_for_termination()
    except asyncio.CancelledError as exc:
        cancellation = exc
    finally:
        async def cleanup() -> None:
            try:
                if started:
                    await _set_health(
                        health_servicer,
                        health_pb2.HealthCheckResponse.NOT_SERVING,
                    )
                    await server.stop(grace=options.shutdown_grace_seconds)
            finally:
                await core.close(grace_seconds=options.shutdown_grace_seconds)

        cleanup_task = asyncio.create_task(cleanup())
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
            await cleanup_task
    if cancellation is not None:
        raise cancellation


@app.callback()
def main() -> None:
    """Qwen3-ASR AX650 服务命令组。"""


@app.command()
def serve(
    model: Path = typer.Option(
        ...,
        "--model",
        help="AX650 模型根目录。",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
        envvar="MODEL_PATH",
    ),
    host: str = typer.Option(
        "[::]",
        "--host",
        help="gRPC 监听主机。",
        envvar="GRPC_HOST",
    ),
    port: int = typer.Option(
        50051,
        "--port",
        help="gRPC 监听端口。",
        min=1,
        max=65535,
        envvar="GRPC_PORT",
    ),
    warmup: bool = typer.Option(
        True,
        "--warmup/--no-warmup",
        help="就绪前执行 AX 预热。",
        envvar="QWEN_ASR_AX_WARMUP",
    ),
    max_inflight_requests: int = typer.Option(
        2,
        "--max-inflight-requests",
        help="执行中与排队请求的总容量。",
        min=1,
        envvar="QWEN_ASR_MAX_INFLIGHT_REQUESTS",
    ),
    shutdown_grace_seconds: float = typer.Option(
        5.0,
        "--shutdown-grace-seconds",
        help="停止接收请求后的排空宽限期。",
        min=0.0,
        envvar="QWEN_ASR_SHUTDOWN_GRACE_SECONDS",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="日志级别。",
        envvar="LOG_LEVEL",
        callback=_normalize_log_level,
    ),
) -> None:
    """启动异步 Qwen3-ASR gRPC 服务。"""
    logging.basicConfig(level=getattr(logging, log_level), format=LOG_FORMAT)
    normalized_host = host.strip()
    if not normalized_host:
        raise typer.BadParameter("host 不能为空", param_hint="--host")
    options = ServeOptions(
        model=model,
        host=normalized_host,
        port=port,
        warmup=warmup,
        max_inflight_requests=max_inflight_requests,
        shutdown_grace_seconds=shutdown_grace_seconds,
    )
    asyncio.run(run_server(options))


if __name__ == "__main__":
    app()
