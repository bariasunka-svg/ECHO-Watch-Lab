# SPDX-License-Identifier: AGPL-3.0-only
"""Video file validation and probing behind the pinned FFmpeg.

Adapted from the frozen PatronSight reference src/patronsight/vision/video_source.py (SHA-256
8dc0066e849525936b3c0fd2472e8fb8244953bb14fea80d77670d5f0ca4c90a) and src/patronsight/vision/contracts.py (SHA-256
8a9cfecea877ad4c2360dc554f2c3e1c782761e142ed9f03252ef1827c3da75a), LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only.

Adopted behaviour: ``VideoValidationError``; ``VideoProbe`` (filename, dimensions, reported fps, frame count, timing
mode, effective fps, duration, a human timing note); the explicit three-way timing decision (container presentation
timestamps when strictly increasing, else frame index over the reported rate, else an explicit assumed rate - each
recorded, never silently substituted); approved extensions; ``utc_now_iso``.

Different implementation: the donor probes with OpenCV ``VideoCapture``. That would bring a SECOND native decoder
(OpenCV's bundled opencv_videoio_ffmpeg DLL) into this project's supply picture, and the package pins the decoder to
the single FFmpeg executable. The probe therefore reads the container with that executable (stream line + stream-copy
packet listing). NOT adopted: PrerecordedVideoSource iteration, the entrance-line / alert / crossing contracts
(person-specific), safe_upload_basename (uploads do not exist here).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..guards import popen_windowless
from .ffmpeg_source import parse_framecrc, probe_args

TIMESTAMP_MODE_CONTAINER = "container_pts"
TIMESTAMP_MODE_REPORTED_FPS = "frame_index_over_reported_fps"
TIMESTAMP_MODE_ASSUMED_FPS = "frame_index_over_assumed_fps"
APPROVED_VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"})
ASSUMED_FPS_FALLBACK = 25.0
_STREAM_RE = re.compile(r"Stream #0:\d+.*?Video:.*?\b(\d{2,5})x(\d{2,5})\b.*?(?:,\s*([0-9.]+)\s*fps)?", re.S)
_FPS_RE = re.compile(r"([0-9.]+)\s*fps")


class VideoValidationError(ValueError):
    """A video file is missing, unreadable, unsupported, or has no frames."""


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    filename: str
    width: int
    height: int
    reported_fps: float
    frame_count: int
    timestamp_mode: str
    effective_fps: float
    duration_seconds: float

    def as_dict(self) -> dict[str, object]:
        return {"filename": self.filename, "width": self.width, "height": self.height,
                "reported_fps": round(self.reported_fps, 3), "frame_count": self.frame_count,
                "timestamp_mode": self.timestamp_mode, "effective_fps": round(self.effective_fps, 3),
                "duration_seconds": round(self.duration_seconds, 3)}

    @property
    def timing_note(self) -> str:
        if self.timestamp_mode == TIMESTAMP_MODE_CONTAINER:
            return "Frame times read from the video container's own timestamps."
        if self.timestamp_mode == TIMESTAMP_MODE_REPORTED_FPS:
            return ("Container timestamps were unusable; frame times derived from frame index divided by the "
                    f"reported {self.reported_fps:.3f} fps.")
        return ("Container reported neither usable timestamps nor a frame rate; frame times fall back to an assumed "
                f"{ASSUMED_FPS_FALLBACK:.0f} fps and are approximate.")


def _validate_path(path: Path) -> None:
    if not path.exists() or not path.is_file():
        raise VideoValidationError(f"Video file not found: {path.name}")
    if path.suffix.lower() not in APPROVED_VIDEO_EXTENSIONS:
        raise VideoValidationError(f"Unsupported video type '{path.suffix}'.")


def probe_video(path: str | Path, ffmpeg: Path) -> VideoProbe:
    """Validate *path* and read its dimensions, frame rate, frame count and timing mode with the pinned FFmpeg."""
    path = Path(path)
    _validate_path(path)
    p = popen_windowless([str(ffmpeg), "-hide_banner", "-nostdin", "-i", str(path)], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    try:
        _out, err = p.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate(timeout=5)
        raise VideoValidationError("Video probe timed out.") from None
    text = err.decode("utf-8", "replace")
    m = _STREAM_RE.search(text)
    if not m:
        raise VideoValidationError("Video could not be opened (unreadable or corrupt).")
    width, height = int(m.group(1)), int(m.group(2))
    line = next((ln for ln in text.splitlines() if "Video:" in ln), "")
    fm = _FPS_RE.search(line)
    reported_fps = float(fm.group(1)) if fm else 0.0
    p2 = popen_windowless(probe_args(ffmpeg, str(path)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, _e = p2.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p2.kill()
        p2.communicate(timeout=5)
        raise VideoValidationError("Video packet probe timed out.") from None
    parsed = parse_framecrc(out.decode("utf-8", "replace"))
    if parsed is None or parsed["frames"] <= 0:
        raise VideoValidationError("No frame could be read from the video.")
    pts = parsed["pts_s"]
    strictly_increasing = len(pts) >= 3 and all(pts[i] > pts[i - 1] for i in range(1, len(pts))) and pts[-1] > 0.0
    if strictly_increasing:
        mode, eff = TIMESTAMP_MODE_CONTAINER, reported_fps if reported_fps > 0 else ASSUMED_FPS_FALLBACK
    elif reported_fps > 0:
        mode, eff = TIMESTAMP_MODE_REPORTED_FPS, reported_fps
    else:
        mode, eff = TIMESTAMP_MODE_ASSUMED_FPS, ASSUMED_FPS_FALLBACK
    return VideoProbe(path, path.name, width, height, reported_fps, int(parsed["frames"]), mode, eff,
                      float(parsed["duration_s"]))
