# coding=utf-8
from __future__ import annotations

import logging
from pathlib import Path

from qwen_asr_onnx.configs import AppConfig

logger = logging.getLogger(__name__)


def build_onnx_kwargs(config: AppConfig) -> dict[str, object]:
    model_root = config.model_path
    tokenizer_path = model_root / "tokenizer.json"
    onnx_dir = model_root / "onnx_models"
    logger.info("Validating model files under %s", model_root)

    if not model_root.exists():
        raise FileNotFoundError(f"Model directory does not exist: {model_root}")
    if not model_root.is_dir():
        raise ValueError(f"Model path must be a directory: {model_root}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
    if not onnx_dir.is_dir():
        raise FileNotFoundError(f"onnx_models directory not found: {onnx_dir}")

    logger.info(
        "Model files validated: tokenizer=%s, onnx_dir=%s",
        tokenizer_path,
        onnx_dir,
    )
    return {
        "model_root": Path(model_root),
        "onnx_dir": onnx_dir,
        "tokenizer_path": tokenizer_path,
        "num_threads": config.onnx.num_threads,
        "quantize": config.onnx.quantize,
        "max_new_tokens": config.generation.max_new_tokens,
    }
