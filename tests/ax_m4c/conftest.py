from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from cffi import FFI

from qwen_asr_onnx.ax_m4c.ffi import NativeBindings, load_native_library


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def native_test_build(repository_root: Path) -> Path:
    source = repository_root / "src/qwen_asr_onnx/ax_m4c/native"
    build = repository_root / "build/ax-m4c-host-tests"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DAX_QWEN_ASR_BUILD_DEVICE=OFF",
            "-DAX_QWEN_ASR_BUILD_HOST_TESTS=ON",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--parallel", "4"],
        check=True,
    )
    return build


@pytest.fixture(scope="session")
def core_native(native_test_build: Path) -> tuple[FFI, object]:
    ffi = FFI()
    ffi.cdef(
        """
        int ax_qwen_asr_test_compute_mel(
            const int16_t *, size_t, float *, size_t, size_t *, size_t *);
        int ax_qwen_asr_test_load_config(const char *);
        int ax_qwen_asr_test_prompt(
            size_t, uint32_t *, size_t, size_t *);
        const char *ax_qwen_asr_test_last_error(void);
        """
    )
    return ffi, ffi.dlopen(str(native_test_build / "libax_qwen_asr_test_core.so"))


@pytest.fixture(scope="session")
def fake_bindings(native_test_build: Path) -> NativeBindings:
    return load_native_library(native_test_build / "libax_qwen_asr_test_abi.so")


@pytest.fixture()
def valid_model_dir(tmp_path: Path) -> Path:
    config = {
        "system_prompt": "you are a helpful assistant.",
        "model_name": "AXERA-TECH/Qwen3-0.6B",
        "url_tokenizer_model": "qwen3_tokenizer.txt",
        "tokenizer_type": "Qwen3",
        "post_config_path": "post_config.json",
        "template_filename_axmodel": "qwen3_asr_p64_l%d_together.axmodel",
        "axmodel_num": 28,
        "filename_post_axmodel": "qwen3_asr_post.axmodel",
        "filename_tokens_embed": "model.embed_tokens.weight.bfloat16.bin",
        "tokens_embed_num": 151936,
        "tokens_embed_size": 1024,
        "use_mmap_load_embed": True,
        "use_mmap_load_layer": True,
        "devices": [0],
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    post_config = {
        "enable_temperature": False,
        "temperature": 0.9,
        "enable_repetition_penalty": False,
        "repetition_penalty": 1.2,
        "penalty_window": 20,
        "enable_top_p_sampling": False,
        "top_p": 0.8,
        "enable_top_k_sampling": False,
        "top_k": 10,
    }
    (tmp_path / "post_config.json").write_text(
        json.dumps(post_config), encoding="utf-8"
    )
    for filename in (
        "conv_frontend.axmodel",
        "encoder.axmodel",
        "qwen3_asr_post.axmodel",
        "qwen3_tokenizer.txt",
    ):
        (tmp_path / filename).write_bytes(b"fixture")
    for index in range(28):
        (tmp_path / f"qwen3_asr_p64_l{index}_together.axmodel").write_bytes(
            b"fixture"
        )
    embedding = tmp_path / "model.embed_tokens.weight.bfloat16.bin"
    with embedding.open("wb") as file:
        file.truncate(151936 * 1024 * 2)
    return tmp_path
