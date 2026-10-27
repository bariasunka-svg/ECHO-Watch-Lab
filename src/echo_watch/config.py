# SPDX-License-Identifier: AGPL-3.0-only
"""Paths, pinned artifacts and declared limits for ECHO Watch Lab.

Adapted from the Native foundation archive (src/echo_watch/config.py, tree 3d14d650). Lab changes: the coordinator runs in
``env/ui`` (full official CPython with Tk, no Torch); inference runs only in fresh ``env/gpu`` children; there is no CPU
inference interpreter (CPU is not a qualified substitute). The pinned FFmpeg comes from the UI runtime's imageio-ffmpeg
wheel. ``Limits.max_frame_age_s`` stays 2.0 and is now re-checked at the authoritative commit.

Every path derives from this file's own location. No ``ECHOWATCH_*`` / ``PYTHONPATH`` override is read. Mutable
outputs must resolve inside the root; ``inside_root`` is asserted wherever a caller supplies a directory.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
ENV_UI = ROOT / "env" / "ui"
ENV_GPU = ROOT / "env" / "gpu"
UI_PYTHON = ENV_UI / "python.exe"
UI_PYTHONW = ENV_UI / "pythonw.exe"
GPU_PYTHON = ENV_GPU / "python.exe"
# kept for adopted helpers that expect the name: the coordinator interpreter (never used for inference)
CPU_PYTHON = UI_PYTHON

MODEL_PF = ROOT / "models" / "yoloe-26s-seg-pf.pt"
MODEL_PF_SHA256 = "01605da3f2e0e5325df45d5f5eef8e40b5b5c7d085399be3c33ac638258b150a"
MODEL_VP = ROOT / "models" / "yoloe-26s-seg.pt"
MODEL_VP_SHA256 = "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654"
FFMPEG = ENV_UI / "Lib" / "site-packages" / "imageio_ffmpeg" / "binaries" / "ffmpeg-win-x86_64-v7.1.exe"
FFMPEG_SHA256 = "2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3"
GEMINI_CLIP = ROOT / "inputs" / "gemini_generated_video_448bc16c.mp4"
GEMINI_CLIP_SHA256 = "3a9f089891ae8c08175bad5dec1730df506659777d93777d895ddeb820ba3d07"

OPERATOR_DATA = ROOT / "data" / "operator"
LOG_DIR = ROOT / "logs"

CREATE_NO_WINDOW = 0x08000000   # console children get no window at all

MiB = 1024 ** 2
GiB = 1024 ** 3


class OutsideRoot(ValueError):
    pass


def inside_root(path: str | os.PathLike) -> Path:
    """The resolved path, or ``OutsideRoot`` when it escapes the project root (mutable outputs stay inside)."""
    p = Path(path).resolve()
    try:
        p.relative_to(ROOT)
    except ValueError:
        raise OutsideRoot(f"{p} is outside the project root {ROOT}") from None
    return p


@dataclass(frozen=True)
class Limits:
    """Declared resource and behaviour limits."""

    frame_max_bytes: int = 1920 * 1080 * 3
    worker_reply_max_bytes: int = 8 * MiB
    trace_queue_max: int = 50_000
    clean_queue_max: int = 90          # a few seconds of 15 fps frames; overflow is counted and disclosed
    annotated_queue_max: int = 90
    startup_timeout_s: float = 180.0
    worker_ready_timeout_s: float = 90.0
    request_timeout_s: float = 8.0     # one real inference round trip
    scan_timeout_s: float = 60.0
    stop_drain_s: float = 30.0
    min_free_disk_bytes: int = 2 * GiB
    run_evidence_cap_bytes: int = 10 * GiB
    stale_result_s: float = 1.0        # a displayed result older than this loses live authority
    max_frame_age_s: float = 2.0


@dataclass
class Settings:
    """One engine instance's configuration. ``data_dir`` is a store inside the root."""

    data_dir: Path
    limits: Limits = field(default_factory=Limits)
    device: str = "cuda:0"             # explicit; there is no CPU fallback
    precision: str = "fp32"
    threads: int = 4
    scan_imgsz: int = 640
    scan_conf: float = 0.20
    scan_iou: float = 0.70
    scan_max_det: int = 100
    verify_imgsz: int = 640
    verify_conf: float = 0.10
    verify_iou: float = 0.70
    verify_max_det: int = 100
    record_fps: float = 15.0
    # test hooks: replace the worker command (fault injection / lifecycle regressions). Never set in production.
    worker_cmd: list[str] | None = None

    def __post_init__(self) -> None:
        self.data_dir = inside_root(self.data_dir)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "echo_lab.db"

    def dirs(self) -> dict[str, Path]:
        d = self.data_dir
        return {"scans": d / "scans", "assets": d / "assets", "runs": d / "runs", "logs": d / "logs",
                "ultralytics": d / "runtime" / "ultralytics-config", "tmp": d / "tmp", "exports": d / "exports"}

    def ensure_dirs(self) -> None:
        for p in self.dirs().values():
            p.mkdir(parents=True, exist_ok=True)
