from __future__ import annotations

from pathlib import Path
from typing import Callable

from qwen_asr_onnx.ax_m4c import AxQwenAsr
from qwen_asr_onnx.runners.base import (
    RunnerMetrics,
    RunnerOutput,
    RunnerToken,
    RunnerTokenCallback,
)


class AxModelRunner:
    """把稳定的 ModelRunner 契约适配到现有 AX CFFI facade。"""

    def __init__(
        self,
        *,
        model_factory: Callable[[str | Path], AxQwenAsr] = AxQwenAsr,
    ) -> None:
        self._model_factory = model_factory
        self._model: AxQwenAsr | None = None

    def load(self, model_dir: Path) -> None:
        if self._model is not None:
            raise RuntimeError("AX model runner is already loaded")
        self._model = self._model_factory(model_dir)

    def warmup(self) -> None:
        self._require_model().warmup()

    def infer_pcm16(self, pcm: bytes, *, sample_rate: int) -> RunnerOutput:
        model = self._require_model()
        raw_text = model.transcribe_pcm16(pcm, sample_rate=sample_rate)
        return self._make_output(model, raw_text)

    def infer_pcm16_stream(
        self,
        pcm: bytes,
        *,
        sample_rate: int,
        token_callback: RunnerTokenCallback,
    ) -> RunnerOutput:
        model = self._require_model()

        def on_token(token_id: int, sequence: int, text_delta: str) -> bool | None:
            return token_callback(
                RunnerToken(
                    token_id=token_id,
                    sequence=sequence,
                    text_delta=text_delta,
                )
            )

        raw_text = model.transcribe_pcm16_stream(
            pcm,
            sample_rate=sample_rate,
            token_callback=on_token,
        )
        return self._make_output(model, raw_text)

    @staticmethod
    def _make_output(model: AxQwenAsr, raw_text: str) -> RunnerOutput:
        native = model.last_metrics()
        return RunnerOutput(
            raw_text=raw_text,
            metrics=RunnerMetrics(
                sample_count=native.sample_count,
                mel_frame_count=native.mel_frame_count,
                audio_token_count=native.audio_token_count,
                prompt_token_count=native.prompt_token_count,
                generated_token_count=native.generated_token_count,
                preprocess_ms=native.preprocess_ms,
                encoder_ms=native.encoder_ms,
                decoder_ttft_ms=native.decoder_ttft_ms,
                decoder_ms=native.decoder_ms,
                native_total_ms=native.native_total_ms,
            ),
        )

    def close(self) -> None:
        if self._model is None:
            return
        self._model.close()
        self._model = None

    def _require_model(self) -> AxQwenAsr:
        if self._model is None:
            raise RuntimeError("AX model runner is not loaded")
        return self._model
