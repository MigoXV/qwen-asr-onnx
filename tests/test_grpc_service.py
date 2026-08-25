from __future__ import annotations

import asyncio

import grpc
import pytest

from qwen_asr_onnx.inferencers.asr import AsrResult, AsrStreamEvent
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
from qwen_asr_onnx.runners.base import RunnerMetrics


METRICS = RunnerMetrics(80, 1, 1, 1, 2, 1, 2, 3, 4, 10)


class FakeInferencer:
    async def infer(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        language_code: str,
        deadline_monotonic: float | None,
    ) -> AsrResult:
        assert audio_bytes == b"\0\0" * 80
        assert sample_rate == 16000
        return AsrResult(
            transcript="干净的识别结果",
            language="Chinese",
            request_id="request-1",
            queue_wait_ms=1.0,
            engine_total_ms=10.0,
            metrics=METRICS,
        )

    async def stream(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        language_code: str,
        deadline_monotonic: float | None,
    ):
        result = await self.infer(
            audio_bytes,
            sample_rate=sample_rate,
            language_code=language_code,
            deadline_monotonic=deadline_monotonic,
        )
        yield AsrStreamEvent(
            transcript="干净的",
            language="Chinese",
            is_final=False,
            request_id=result.request_id,
            emitted_monotonic=1.0,
        )
        yield AsrStreamEvent(
            transcript=result.transcript,
            language=result.language,
            is_final=False,
            request_id=result.request_id,
            emitted_monotonic=2.0,
        )
        yield AsrStreamEvent(
            transcript=result.transcript,
            language=result.language,
            is_final=True,
            request_id=result.request_id,
            emitted_monotonic=3.0,
            result=result,
        )


async def _requests(
    sample_rate: int = 16000,
    audio: bytes = b"\0\0" * 80,
    *,
    interim_results: bool = True,
):
    yield StreamingRecognizeRequest(
        streaming_config=StreamingRecognitionConfig(
            config=RecognitionConfig(
                encoding=RecognitionConfig.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code="",
                hotwords=["兼容但忽略"],
            ),
            interim_results=interim_results,
        )
    )
    yield StreamingRecognizeRequest(audio_content=audio)


def test_real_grpc_transport_returns_clean_stream_and_final_result() -> None:
    async def scenario() -> None:
        server = grpc.aio.server()
        servicer = ASRServicer(FakeInferencer())
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

        assert [
            response.results[0].alternative.transcript for response in responses
        ] == ["干净的", "干净的识别结果", "干净的识别结果"]
        assert [response.results[0].is_final for response in responses] == [
            False,
            False,
            True,
        ]
        for response in responses:
            result = response.results[0]
            assert result.alternative.language.code == "Chinese"
            assert not result.alternative.words
            assert "language" not in result.alternative.transcript
            assert "<asr_text>" not in result.alternative.transcript
        assert not responses[0].results[0].alternative.turn_completed
        assert responses[-1].results[0].alternative.turn_completed

    asyncio.run(scenario())


def test_real_grpc_transport_honors_final_only_config() -> None:
    async def scenario() -> None:
        server = grpc.aio.server()
        add_UxSpeechServicer_to_server(ASRServicer(FakeInferencer()), server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        try:
            async with grpc.aio.insecure_channel(f"127.0.0.1:{port}") as channel:
                responses = [
                    response
                    async for response in UxSpeechStub(channel).StreamingRecognize(
                        _requests(interim_results=False)
                    )
                ]
        finally:
            await server.stop(0)

        assert len(responses) == 1
        assert responses[0].results[0].is_final
        assert responses[0].results[0].alternative.transcript == "干净的识别结果"

    asyncio.run(scenario())


def test_real_grpc_transport_rejects_non_16k_audio() -> None:
    async def scenario() -> None:
        server = grpc.aio.server()
        add_UxSpeechServicer_to_server(
            ASRServicer(FakeInferencer()),
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
            ASRServicer(FakeInferencer()),
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
