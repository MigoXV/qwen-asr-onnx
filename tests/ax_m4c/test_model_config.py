from __future__ import annotations

import json
from pathlib import Path


def _load(core_native, model_dir: Path) -> tuple[int, str]:
    ffi, lib = core_native
    result = lib.ax_qwen_asr_test_load_config(str(model_dir).encode())
    error = ffi.string(lib.ax_qwen_asr_test_last_error()).decode()
    return result, error


def test_valid_fixed_model_config(core_native, valid_model_dir: Path) -> None:
    result, error = _load(core_native, valid_model_dir)
    assert result == 0, error


def test_missing_layer_is_explicit(core_native, valid_model_dir: Path) -> None:
    (valid_model_dir / "qwen3_asr_p64_l17_together.axmodel").unlink()
    result, error = _load(core_native, valid_model_dir)
    assert result == 2
    assert "l17" in error
    assert "缺少模型文件" in error


def test_rejects_wrong_layer_count(core_native, valid_model_dir: Path) -> None:
    path = valid_model_dir / "config.json"
    config = json.loads(path.read_text())
    config["axmodel_num"] = 27
    path.write_text(json.dumps(config), encoding="utf-8")
    result, error = _load(core_native, valid_model_dir)
    assert result == 2
    assert "axmodel_num" in error


def test_rejects_non_greedy_post_config(core_native, valid_model_dir: Path) -> None:
    path = valid_model_dir / "post_config.json"
    config = json.loads(path.read_text())
    config["enable_top_k_sampling"] = True
    path.write_text(json.dumps(config), encoding="utf-8")
    result, error = _load(core_native, valid_model_dir)
    assert result == 2
    assert "enable_top_k_sampling" in error


def test_rejects_path_traversal(core_native, valid_model_dir: Path) -> None:
    path = valid_model_dir / "config.json"
    config = json.loads(path.read_text())
    config["filename_post_axmodel"] = "../outside.axmodel"
    path.write_text(json.dumps(config), encoding="utf-8")
    result, error = _load(core_native, valid_model_dir)
    assert result == 2
    assert ".." in error


def test_rejects_embedding_size(core_native, valid_model_dir: Path) -> None:
    with (valid_model_dir / "model.embed_tokens.weight.bfloat16.bin").open("r+b") as file:
        file.truncate(128)
    result, error = _load(core_native, valid_model_dir)
    assert result == 2
    assert "embedding 文件尺寸" in error


def test_rejects_symlink_escaping_model_root(core_native, valid_model_dir: Path) -> None:
    outside = valid_model_dir.parent / "outside-post.axmodel"
    outside.write_bytes(b"outside")
    link = valid_model_dir / "post-link.axmodel"
    link.symlink_to(outside)
    path = valid_model_dir / "config.json"
    config = json.loads(path.read_text())
    config["filename_post_axmodel"] = link.name
    path.write_text(json.dumps(config), encoding="utf-8")
    result, error = _load(core_native, valid_model_dir)
    assert result == 2
    assert "逃逸模型根目录" in error


def test_rejects_invalid_json(core_native, valid_model_dir: Path) -> None:
    (valid_model_dir / "config.json").write_text('{"axmodel_num":', encoding="utf-8")
    result, error = _load(core_native, valid_model_dir)
    assert result == 2
    assert "JSON" in error
