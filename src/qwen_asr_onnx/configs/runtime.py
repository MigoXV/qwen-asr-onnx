# coding=utf-8
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING

@dataclass
class AxConfig:
    """AX650 runtime 与服务容量配置。"""

    warmup: bool = True
    max_inflight_requests: int = 2
    shutdown_grace_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.max_inflight_requests < 1:
            raise ValueError("ax.max_inflight_requests must be >= 1.")
        if self.shutdown_grace_seconds < 0:
            raise ValueError("ax.shutdown_grace_seconds must be >= 0.")


@dataclass
class AppConfig:
    model: str = MISSING
    context: str = ""
    server_port: int = 50051
    ax: AxConfig = field(default_factory=AxConfig)

    def __post_init__(self) -> None:
        raw_model = self.model
        model = "" if raw_model is MISSING else str(raw_model or "").strip()
        if not model or model == "???":
            raise ValueError("model must point to the AX650 model root directory.")
        if not 1 <= self.server_port <= 65535:
            raise ValueError("server_port must be between 1 and 65535.")
        self.model = model
        self.context = str(self.context or "")

    @property
    def model_path(self) -> Path:
        return Path(self.model).expanduser()
