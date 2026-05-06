#!/usr/bin/env python
# coding=utf-8
"""上传本地 ONNX 模型目录到内网 Hugging Face Hub。"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path("model-bin/Daumee/Qwen3-ASR-0.6B-ONNX-CPU")
DEFAULT_AUTHOR = "migo"
DEFAULT_REPO_NAME = "Qwen3-ASR-0.6B-ONNX-CPU"


def load_dotenv(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="上传本地模型目录到内网 Hugging Face Hub。"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"本地模型目录，默认：{DEFAULT_MODEL_DIR}",
    )
    parser.add_argument(
        "--author",
        default=DEFAULT_AUTHOR,
        help=f"Hub 作者或组织名，默认：{DEFAULT_AUTHOR}",
    )
    parser.add_argument(
        "--repo-name",
        default=DEFAULT_REPO_NAME,
        help=f"Hub 仓库名，默认：{DEFAULT_REPO_NAME}",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="内网 Hugging Face Endpoint，默认读取 HF_ENDPOINT。",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hub Token，默认读取 HF_TOKEN。",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="环境变量文件，默认：.env。",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload Qwen3-ASR ONNX CPU model",
        help="提交信息。",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = parse_args()
    load_dotenv(args.env_file)

    model_dir = args.model_dir.expanduser().resolve()
    endpoint = args.endpoint or os.getenv("HF_ENDPOINT")
    token = args.token or os.getenv("HF_TOKEN")
    repo_id = f"{args.author}/{args.repo_name}"

    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在：{model_dir}")
    if not endpoint:
        raise ValueError("缺少 HF_ENDPOINT，请在 .env 中配置或通过 --endpoint 传入。")
    if not token:
        raise ValueError("缺少 HF_TOKEN，请在 .env 中配置或通过 --token 传入。")

    logger.info("Endpoint: %s", endpoint)
    logger.info("Repo: %s", repo_id)
    logger.info("Model dir: %s", model_dir)

    api = HfApi(endpoint=endpoint, token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=args.commit_message,
    )
    logger.info("上传完成：%s/%s", endpoint.rstrip("/"), repo_id)


if __name__ == "__main__":
    main()
