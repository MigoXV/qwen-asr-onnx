#!/usr/bin/env python3
"""AX650 Qwen3-ASR 冷启动、热态耗时和 Linux/NPU 内存基准。"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import threading
import time
import wave
from dataclasses import asdict
from pathlib import Path

from qwen_asr_onnx.ax_m4c import AxQwenAsr


def read_wav(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav_file:
        if (
            wav_file.getnchannels() != 1
            or wav_file.getsampwidth() != 2
            or wav_file.getframerate() != 16000
            or wav_file.getcomptype() != "NONE"
        ):
            raise ValueError("只支持 16 kHz 单声道未压缩 PCM16 WAV")
        frames = wav_file.getnframes()
        return wav_file.readframes(frames), frames / 16000.0


def read_process_memory_kib() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, separator, remainder = line.partition(":")
        if separator and key in {"VmRSS", "VmHWM", "RssAnon", "RssFile"}:
            values[key] = int(remainder.split()[0])
    values["RssTotal"] = values.get("RssAnon", 0) + values.get("RssFile", 0)
    return values


def read_cmm_memory_kib() -> dict[str, int] | None:
    path = Path("/proc/ax_proc/mem_cmm_info")
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    match = re.search(r"total size=(\d+)KB.*?used=(\d+)KB.*?remain=(\d+)KB", text)
    if not match:
        return None
    return {
        "total_kib": int(match.group(1)),
        "used_kib": int(match.group(2)),
        "remain_kib": int(match.group(3)),
    }


class MemoryMonitor:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak = read_process_memory_kib()
        self.cmm_peak_used_kib: int | None = None

    def __enter__(self) -> MemoryMonitor:
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def _sample(self) -> None:
        current = read_process_memory_kib()
        for key, value in current.items():
            self.peak[key] = max(self.peak.get(key, 0), value)
        cmm = read_cmm_memory_kib()
        if cmm:
            used = cmm["used_kib"]
            self.cmm_peak_used_kib = max(self.cmm_peak_used_kib or 0, used)

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            self._sample()


def revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_dir = Path(__file__).resolve().parents[2] / ".git"
        if git_dir.is_file():
            marker = git_dir.read_text().strip()
            if marker.startswith("gitdir:"):
                git_dir = (git_dir.parent / marker.removeprefix("gitdir:").strip()).resolve()
        try:
            head = (git_dir / "HEAD").read_text().strip()
            if head.startswith("ref:"):
                head = (git_dir / head.removeprefix("ref:").strip()).read_text().strip()
            return head[:12]
        except OSError:
            return "unknown"


def dirty_state() -> bool | None:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.warmups < 0 or args.rounds <= 0:
        parser.error("--warmups 必须 >= 0，--rounds 必须 > 0")

    pcm, audio_duration = read_wav(args.wav)
    report: dict[str, object] = {
        "model_dir": str(args.model_dir.expanduser().resolve()),
        "audio": str(args.wav.expanduser().resolve()),
        "audio_duration_s": audio_duration,
        "model_profile": "Qwen3-ASR-0.6B AX650 C64/P448/CTX2047 BF16",
        "code_revision": revision(),
        "code_dirty": dirty_state(),
        "warmup_count": args.warmups,
        "hot_round_count": args.rounds,
    }

    cmm_before = read_cmm_memory_kib()
    with MemoryMonitor() as memory:
        started = time.perf_counter()
        asr = AxQwenAsr(args.model_dir)
        report["cold_start_s"] = time.perf_counter() - started
        try:
            warmup_times = []
            for _ in range(args.warmups):
                started = time.perf_counter()
                asr.warmup()
                warmup_times.append(time.perf_counter() - started)
            report["warmup_times_s"] = warmup_times

            rounds = []
            raw_output = ""
            for _ in range(args.rounds):
                started = time.perf_counter()
                raw_output = asr.transcribe_pcm16(pcm, sample_rate=16000)
                wall_time = time.perf_counter() - started
                rounds.append(
                    {
                        "wall_time_s": wall_time,
                        "rtf": wall_time / audio_duration,
                        "speed_x": audio_duration / wall_time,
                        "native": asdict(asr.last_metrics()),
                    }
                )
            report["raw_output"] = raw_output
            report["rounds"] = rounds
        finally:
            asr.close()

    report["linux_peak_kib"] = memory.peak
    report["cmm_before_kib"] = cmm_before
    report["cmm_peak_used_kib"] = memory.cmm_peak_used_kib
    report["cmm_after_kib"] = read_cmm_memory_kib()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
