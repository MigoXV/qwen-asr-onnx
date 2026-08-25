# coding=utf-8
"""
Qwen3-ASR gRPC 服务的 Typer CLI 入口。

运行时配置从结构化 YAML 文件加载。

示例：
    python -m qwen_asr_onnx.commands.app serve --config config.yaml
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import grpc
import typer
from omegaconf import OmegaConf

from qwen_asr_onnx.configs import AppConfig
from qwen_asr_onnx.inferencers.ax_engine import AxInferenceEngine
from qwen_asr_onnx.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr_onnx.protos.asr.ux_speech_pb2_grpc import (
    add_UxSpeechServicer_to_server,
)
from qwen_asr_onnx.servicer.servicer import ASRServicer

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    name="qwen-asr",
    help="Qwen3-ASR gRPC 推理服务。",
    add_completion=False,
)


@app.callback()
def main() -> None:
    """Qwen3-ASR AX650 服务命令组。"""


async def run_server(config: AppConfig) -> None:
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", 1024 * 1024),
            ("grpc.max_send_message_length", 1024 * 1024),
        ]
    )
    engine = AxInferenceEngine(
        config.model_path,
        warmup=config.ax.warmup,
        max_inflight_requests=config.ax.max_inflight_requests,
    )
    servicer = ASRServicer(config, GrpcInferencer(engine))
    add_UxSpeechServicer_to_server(servicer, server)
    address = f"[::]:{config.server_port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"Failed to bind gRPC address: {address}")

    try:
        logger.info(
            "Loading AX650 model: model=%s, warmup=%s, max_inflight_requests=%d",
            config.model,
            config.ax.warmup,
            config.ax.max_inflight_requests,
        )
        await engine.start()
        logger.info("AX650 model loaded successfully.")
        await server.start()
        logger.info("gRPC server listening on port %d", config.server_port)
        await server.wait_for_termination()
    finally:
        await server.stop(grace=config.ax.shutdown_grace_seconds)
        await engine.close(grace_seconds=config.ax.shutdown_grace_seconds)


@app.command()
def serve(
    config: Path = typer.Option(
        "examples/ax-m4c/grpc.yaml",
        "--config",
        help="Qwen3-ASR YAML 配置文件路径。",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        envvar="CONFIG_PATH",
    ),
) -> None:
    """启动 Qwen3-ASR gRPC 服务。"""
    logger.info("Loading config from %s", config)
    # 先合并默认结构化配置，再转换为应用配置对象。
    schema = OmegaConf.structured(AppConfig)
    loaded = OmegaConf.load(config)
    merged = OmegaConf.merge(schema, loaded)
    app_config = OmegaConf.to_object(merged)
    logger.info(
        "Config loaded: model=%s, server_port=%d, ax.warmup=%s, "
        "ax.max_inflight_requests=%d, ax.shutdown_grace_seconds=%.1f",
        app_config.model,
        app_config.server_port,
        app_config.ax.warmup,
        app_config.ax.max_inflight_requests,
        app_config.ax.shutdown_grace_seconds,
    )

    asyncio.run(run_server(app_config))


if __name__ == "__main__":
    app()
