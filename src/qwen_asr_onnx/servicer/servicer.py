from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import grpc

from qwen_asr_onnx.ax_m4c.errors import (
    AxQwenAsrError,
    AxQwenAsrInvalidArgumentError,
)
from qwen_asr_onnx.configs import AppConfig
from qwen_asr_onnx.inferencers.ax_engine import (
    AxEngineClosedError,
    AxQueueFullError,
)
from qwen_asr_onnx.inferencers.grpc_inferencer import GrpcInferencer
from qwen_asr_onnx.protos.asr.ux_speech_pb2 import (
    LanguageInfo,
    RecognitionConfig,
    SpeechRecognitionAlternative,
    StreamingRecognizeResponse,
    StreamingRecognizeRequest,
    StreamingRecognitionConfig,
    StreamingRecognitionResult,
)
from qwen_asr_onnx.protos.asr.ux_speech_pb2_grpc import UxSpeechServicer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_PCM_SAMPLES = 479_999
MAX_AUDIO_BYTES = MAX_PCM_SAMPLES * 2


class ASRServicer(UxSpeechServicer):
    """
    将流式 RPC 按一次性离线 ASR 请求处理。

    客户端仍使用流式协议，但服务端只读取配置消息后的第一段音频，
    并将其视为完整音频。返回一次最终结果后结束 RPC。
    """

    def __init__(self, config: AppConfig, inferencer: GrpcInferencer) -> None:
        super().__init__()
        self.inferencer = inferencer
        if config.context:
            logger.warning(
                "AX650 fixed prompt does not support context; configured context is ignored."
            )

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
        if streaming_config.interim_results:
            logger.warning(
                "AX650 returns final results only; interim_results is ignored."
            )
        if streaming_config.config.hotwords:
            logger.warning(
                "AX650 fixed prompt does not support hotwords; %d hotword(s) are ignored.",
                len(streaming_config.config.hotwords),
            )
        sample_rate = streaming_config.config.sample_rate_hertz
        audio_duration_seconds = self._calculate_audio_duration_seconds(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
        )
        logger.info(
            "Starting inference: audio_bytes=%d, sample_rate=%d, language_code=%s, "
            "audio_duration_seconds=%.3f, interim_results=%s",
            len(audio_bytes),
            sample_rate,
            streaming_config.config.language_code,
            audio_duration_seconds,
            streaming_config.interim_results,
        )
        try:
            result = await self.inferencer.infer(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                language_code=streaming_config.config.language_code,
            )
            if self._context_is_active(context):
                yield self._make_response(result.transcript, result.language)
            metrics = result.inference.metrics
            logger.info(
                "Inference metrics: preprocess_ms=%.3f, encoder_ms=%.3f, "
                "decoder_ttft_ms=%.3f, decoder_ms=%.3f, native_total_ms=%.3f",
                metrics.preprocess_ms,
                metrics.encoder_ms,
                metrics.decoder_ttft_ms,
                metrics.decoder_ms,
                metrics.native_total_ms,
            )
        except asyncio.CancelledError:
            logger.info("StreamingRecognize cancelled by client.")
            return
        except AxQueueFullError as exc:
            logger.warning("Inference rejected: %s", exc)
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
            return
        except AxQwenAsrInvalidArgumentError as exc:
            logger.warning("Invalid inference input: %s", exc)
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return
        except AxEngineClosedError as exc:
            logger.error("AX inference engine unavailable: %s", exc)
            await context.abort(grpc.StatusCode.UNAVAILABLE, str(exc))
            return
        except AxQwenAsrError as exc:
            logger.error("AX inference failed: %s", exc, exc_info=True)
            await context.abort(
                grpc.StatusCode.INTERNAL,
                f"AX inference failed: {exc}",
            )
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
            len(result.transcript),
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
        streaming_config = config_request.streaming_config
        recognition_config = streaming_config.config
        if recognition_config.encoding not in (
            RecognitionConfig.AUDIO_ENCODING_UNSPECIFIED,
            RecognitionConfig.LINEAR16,
        ):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "Only LINEAR16 audio encoding is supported.",
            )
            return None
        if recognition_config.sample_rate_hertz != SAMPLE_RATE:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"sample_rate_hertz must be {SAMPLE_RATE}.",
            )
            return None
        return streaming_config

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
        audio_content = audio_request.audio_content
        if not audio_content:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "audio_content must not be empty.",
            )
            return None
        if len(audio_content) % 2:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "LINEAR16 audio_content must contain an even number of bytes.",
            )
            return None
        if len(audio_content) > MAX_AUDIO_BYTES:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"audio_content exceeds the AX650 limit of {MAX_AUDIO_BYTES} bytes.",
            )
            return None
        return audio_content

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
    def _make_response(
        transcript: str,
        language: str,
    ) -> StreamingRecognizeResponse:
        """构造不含模型协议前缀的最终响应。"""
        return StreamingRecognizeResponse(
            results=[
                StreamingRecognitionResult(
                    alternative=SpeechRecognitionAlternative(
                        transcript=transcript,
                        words=[],
                        language=LanguageInfo(code=language),
                        turn_completed=True,
                    ),
                    is_final=True,
                )
            ]
        )
