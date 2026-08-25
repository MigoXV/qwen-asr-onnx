from __future__ import annotations

from typer.testing import CliRunner

from qwen_asr_onnx.commands import app as app_module


def test_serve_builds_options_from_environment(monkeypatch, tmp_path) -> None:
    captured = []

    async def fake_run_server(options) -> None:
        captured.append(options)

    monkeypatch.setattr(app_module, "run_server", fake_run_server)
    result = CliRunner().invoke(
        app_module.app,
        ["serve"],
        env={
            "MODEL_PATH": str(tmp_path),
            "GRPC_HOST": "127.0.0.1",
            "GRPC_PORT": "50060",
            "QWEN_ASR_AX_WARMUP": "false",
            "QWEN_ASR_MAX_INFLIGHT_REQUESTS": "3",
            "QWEN_ASR_SHUTDOWN_GRACE_SECONDS": "7.5",
            "LOG_LEVEL": "warning",
        },
    )

    assert result.exit_code == 0, result.output
    options = captured[0]
    assert options.model == tmp_path
    assert options.host == "127.0.0.1"
    assert options.port == 50060
    assert options.warmup is False
    assert options.max_inflight_requests == 3
    assert options.shutdown_grace_seconds == 7.5


def test_serve_requires_model_environment(monkeypatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    result = CliRunner().invoke(app_module.app, ["serve"], env={})
    assert result.exit_code == 2
    assert "--model" in result.output


def test_serve_rejects_invalid_log_level(tmp_path) -> None:
    result = CliRunner().invoke(
        app_module.app,
        ["serve"],
        env={"MODEL_PATH": str(tmp_path), "LOG_LEVEL": "verbose"},
    )
    assert result.exit_code == 2
    assert "日志级别" in result.output
