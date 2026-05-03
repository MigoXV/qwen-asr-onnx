# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING

from qwen_asr_a733.configs.constants import QUANTIZE_CHOICES, QUANTIZE_INT8


@dataclass
class GenerationConfig:
    max_new_tokens: int = 4096


@dataclass
class OnnxConfig:
    num_threads: int = 0
    quantize: str = QUANTIZE_INT8

    def __post_init__(self) -> None:
        self.quantize = self.normalize_quantize(self.quantize)
        if self.num_threads < 0:
            raise ValueError("onnx.num_threads must be >= 0.")

    @staticmethod
    def normalize_quantize(quantize: str) -> str:
        normalized = (quantize or "").strip().lower()
        if normalized not in QUANTIZE_CHOICES:
            choices = ", ".join(sorted(QUANTIZE_CHOICES))
            raise ValueError(
                f"Invalid quantize value '{quantize}'. Expected one of: {choices}."
            )
        return normalized


@dataclass
class AppConfig:
    model: str = MISSING
    context: str = ""
    server_port: int = 50051
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    onnx: OnnxConfig = field(default_factory=OnnxConfig)

    def __post_init__(self) -> None:
        raw_model = self.model
        model = "" if raw_model is MISSING else str(raw_model or "").strip()
        if not model or model == "???":
            raise ValueError("model must point to the ONNX model root directory.")
        self.model = model

    @property
    def model_path(self) -> Path:
        return Path(self.model).expanduser()
