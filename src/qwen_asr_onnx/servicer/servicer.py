from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import grpc

from google.protobuf.duration_pb2 import Duration
from qwen_asr_onnx.commands.utils import build_onnx_kwargs
from qwen_asr_onnx.configs import AppConfig
from qwen_asr_onnx.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr_onnx.inferencers.onnx import OnnxAsrPipeline
from qwen_asr_onnx.protos.asr.ux_speech_pb2 import (
    SpeechRecognitionAlternative,
    StreamingRecognizeResponse,
    StreamingRecognizeRequest,
    StreamingRecognitionConfig,
    StreamingRecognitionResult,
    WordInfo,
)
from qwen_asr_onnx.protos.asr.ux_speech_pb2_grpc import UxSpeechServicer

logger = logging.getLogger(__name__)


class ASRServicer(UxSpeechServicer):
    """
    将流式 RPC 按一次性离线 ASR 请求处理。

    客户端仍使用流式协议，但服务端只读取配置消息后的第一段音频，
    并将其视为完整音频。返回一次最终结果后结束 RPC。
    """

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.default_context = str(getattr(config, "context", "") or "")
        self.inferencer = self._load_inferencer(config)

    def close(self) -> None:
        logger.info("Closing ASR servicer resources.")
        self.inferencer.close()

    @staticmethod
    def _load_inferencer(config: AppConfig) -> GrpcInferencer:
        """按应用配置加载 ONNX 推理器。"""
        logger.info(
            "Loading model: model=%s, max_new_tokens=%d, onnx.num_threads=%d, "
            "onnx.quantize=%s",
            config.model,
            config.generation.max_new_tokens,
            config.onnx.num_threads,
            config.onnx.quantize,
        )
        try:
            inferencer = OnnxAsrPipeline(**build_onnx_kwargs(config))
        except Exception as exc:
            logger.error("Failed to load model: %s", exc, exc_info=True)
            raise
        logger.info("Model loaded successfully.")
        return GrpcInferencer(inferencer=inferencer)

    async def StreamingRecognize(
        self,
        request_iterator: AsyncIterator[StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[StreamingRecognizeResponse]:
        """按一次性 ASR 请求处理单个 StreamingRecognize RPC 调用。"""
        # 加载会话配置，无有效配置则直接退出，有则解析
        streaming_config = await self._load_streaming_config(
            request_iterator=request_iterator,
            context=context,
        )
        if streaming_config is None:
            return

        audio_bytes = await self._load_audio_content(
            request_iterator=request_iterator,
            context=context,
        )
        if audio_bytes is None:
            return
        request_context = self._resolve_context(
            default_context=self.default_context,
            hotwords=streaming_config.config.hotwords,
        )
        sample_rate = streaming_config.config.sample_rate_hertz
        audio_duration_seconds = self._calculate_audio_duration_seconds(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
        )
        logger.info(
            "Starting inference: audio_bytes=%d, sample_rate=%d, language_code=%s, "
            "audio_duration_seconds=%.3f, interim_results=%s, has_context=%s",
            len(audio_bytes),
            sample_rate,
            streaming_config.config.language_code,
            audio_duration_seconds,
            streaming_config.interim_results,
            bool(request_context),
        )
        final_transcript = ""
        try:
            async for transcript, delta, is_final in self.inferencer.infer(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                language_code=streaming_config.config.language_code,
                interim_results=streaming_config.interim_results,
                context=request_context,
            ):
                final_transcript = transcript
                # 客户端断开后停止继续推送识别结果。
                if not self._context_is_active(context):
                    logger.info("Client disconnected; stopping response stream.")
                    break
                yield self._make_response(
                    transcript,
                    is_final=is_final,
                    word=delta,
                )
        except asyncio.CancelledError:
            logger.error("StreamingRecognize cancelled by client.", exc_info=True)
            return
        except Exception as exc:
            logger.error("Inference failed: %s", exc, exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"Inference failed: {exc}",
            )
            return
        logger.info(
            "Inference finished: audio_duration_seconds=%.3f, transcript_chars=%d",
            audio_duration_seconds,
            len(final_transcript),
        )

    @staticmethod
    def _calculate_audio_duration_seconds(
        audio_bytes: bytes,
        sample_rate: int,
    ) -> float:
        """按 LINEAR16 PCM 计算音频时长。"""
        if sample_rate <= 0:
            logger.error("Invalid sample_rate_hertz: %d", sample_rate)
            return 0.0
        bytes_per_sample = 2
        return len(audio_bytes) / bytes_per_sample / sample_rate

    @staticmethod
    async def _load_streaming_config(
        request_iterator: AsyncIterator[StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> StreamingRecognitionConfig | None:
        """读取并校验第一条 streaming_config 请求。"""
        config_request = await ASRServicer._anext_or_none(request_iterator)
        if (
            config_request is None
            or config_request.WhichOneof("streaming_request") != "streaming_config"
        ):
            logger.error("First message must contain streaming_config, aborting RPC.")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "First message must contain streaming_config.",
            )
            return None
        return config_request.streaming_config

    @staticmethod
    async def _load_audio_content(
        request_iterator: AsyncIterator[StreamingRecognizeRequest],
        context: grpc.aio.ServicerContext,
    ) -> bytes | None:
        """读取并校验唯一的 audio_content 请求。"""
        audio_request = await ASRServicer._anext_or_none(request_iterator)
        if (
            audio_request is None
            or audio_request.WhichOneof("streaming_request") != "audio_content"
        ):
            logger.error("Second message must contain audio_content, aborting RPC.")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Second message must contain audio_content.",
            )
            return None

        extra_request = await ASRServicer._anext_or_none(request_iterator)
        if extra_request is not None:
            logger.error("Only one audio_content message is supported, aborting RPC.")
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Only a single audio_content message is supported.",
            )
            return None
        return audio_request.audio_content

    @staticmethod
    async def _anext_or_none(
        iterator: AsyncIterator[StreamingRecognizeRequest],
    ) -> StreamingRecognizeRequest | None:
        try:
            return await iterator.__anext__()
        except StopAsyncIteration:
            return None

    @staticmethod
    def _context_is_active(context: grpc.aio.ServicerContext) -> bool:
        cancelled = getattr(context, "cancelled", None)
        if callable(cancelled):
            try:
                return not bool(cancelled())
            except Exception:
                logger.debug("Failed to query context.cancelled()", exc_info=True)

        is_active = getattr(context, "is_active", None)
        if callable(is_active):
            try:
                return bool(is_active())
            except Exception:
                logger.debug("Failed to query context.is_active()", exc_info=True)

        return True

    @staticmethod
    def _resolve_context(default_context: str, hotwords) -> str:
        """按会话热词优先、系统上下文兜底的规则生成模型 context。"""
        hotword_context = " ".join(
            word for word in (str(item).strip() for item in hotwords) if word
        )
        if hotword_context:
            return hotword_context
        return default_context or ""

    @staticmethod
    def _make_response(
        transcript: str, is_final: bool, word: str = ""
    ) -> StreamingRecognizeResponse:
        """构造只包含一个识别结果的 ``StreamingRecognizeResponse``。"""
        return StreamingRecognizeResponse(
            results=[
                StreamingRecognitionResult(
                    alternative=SpeechRecognitionAlternative(
                        transcript=transcript,
                        words=[
                            WordInfo(
                                word=word,
                                start_time=Duration(seconds=0, nanos=0),
                                end_time=Duration(seconds=0, nanos=0),
                            )
                        ],
                    ),
                    is_final=is_final,
                )
            ]
        )
