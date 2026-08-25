from __future__ import annotations

import asyncio

import grpc
import pytest

from qwen_asr_onnx.ax_m4c import TranscriptionMetrics
from qwen_asr_onnx.configs import AppConfig
from qwen_asr_onnx.inferencers.ax_engine import AxInferenceResult
from qwen_asr_onnx.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr_onnx.protos.asr.ux_speech_pb2 import (
    RecognitionConfig,
    StreamingRecognizeRequest,
    StreamingRecognitionConfig,
)
from qwen_asr_onnx.protos.asr.ux_speech_pb2_grpc import (
    UxSpeechStub,
    add_UxSpeechServicer_to_server,
)
from qwen_asr_onnx.servicer.servicer import ASRServicer, MAX_AUDIO_BYTES


METRICS = TranscriptionMetrics(80, 1, 1, 1, 2, 1, 2, 3, 4, 10)


class FakeEngine:
    async def transcribe_pcm16(self, audio_bytes: bytes, *, sample_rate: int):
        assert audio_bytes == b"\0\0" * 80
        assert sample_rate == 16000
        return AxInferenceResult(
            raw_output="language Chinese<asr_text>干净的识别结果",
            metrics=METRICS,
        )


async def _requests(sample_rate: int = 16000, audio: bytes = b"\0\0" * 80):
    yield StreamingRecognizeRequest(
        streaming_config=StreamingRecognitionConfig(
            config=RecognitionConfig(
                encoding=RecognitionConfig.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code="",
                hotwords=["兼容但忽略"],
            ),
            interim_results=True,
        )
    )
    yield StreamingRecognizeRequest(audio_content=audio)


def test_real_grpc_transport_returns_clean_final_result() -> None:
    async def scenario() -> None:
        server = grpc.aio.server()
        servicer = ASRServicer(
            AppConfig(model="unused"),
            GrpcInferencer(FakeEngine()),
        )
        add_UxSpeechServicer_to_server(servicer, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                responses = [
                    response
                    async for response in UxSpeechStub(channel).StreamingRecognize(
                        _requests()
                    )
                ]
        finally:
            await server.stop(0)

        assert len(responses) == 1
        result = responses[0].results[0]
        assert result.is_final
        assert result.alternative.transcript == "干净的识别结果"
        assert result.alternative.language.code == "Chinese"
        assert not result.alternative.words
        assert "language" not in result.alternative.transcript
        assert "<asr_text>" not in result.alternative.transcript

    asyncio.run(scenario())


def test_real_grpc_transport_rejects_non_16k_audio() -> None:
    async def scenario() -> None:
        server = grpc.aio.server()
        add_UxSpeechServicer_to_server(
            ASRServicer(AppConfig(model="unused"), GrpcInferencer(FakeEngine())),
            server,
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                call = UxSpeechStub(channel).StreamingRecognize(_requests(8000))
                try:
                    async for _ in call:
                        pass
                except grpc.aio.AioRpcError as exc:
                    assert exc.code() is grpc.StatusCode.INVALID_ARGUMENT
                else:
                    raise AssertionError("RPC should reject non-16k audio")
        finally:
            await server.stop(0)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("audio", "message"),
    [
        (b"", "must not be empty"),
        (b"\0", "even number of bytes"),
        (b"\0" * (MAX_AUDIO_BYTES + 2), "exceeds the AX650 limit"),
    ],
)
def test_real_grpc_transport_rejects_invalid_pcm(audio: bytes, message: str) -> None:
    async def scenario() -> None:
        server = grpc.aio.server()
        add_UxSpeechServicer_to_server(
            ASRServicer(AppConfig(model="unused"), GrpcInferencer(FakeEngine())),
            server,
        )
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                call = UxSpeechStub(channel).StreamingRecognize(
                    _requests(audio=audio)
                )
                try:
                    async for _ in call:
                        pass
                except grpc.aio.AioRpcError as exc:
                    assert exc.code() is grpc.StatusCode.INVALID_ARGUMENT
                    assert message in exc.details()
                else:
                    raise AssertionError("RPC should reject invalid PCM")
        finally:
            await server.stop(0)

    asyncio.run(scenario())
