# SPDX-License-Identifier: AGPL-3.0-only
"""Windowless inspection of a WRITTEN recording: decode it with the pinned FFmpeg and compare every decoded frame with the
frame map's signature of the exact pixels that were offered. File existence is never accepted as proof of content.

The decode is a bounded, windowless child (no player, no window): ``-vf scale=16:9:flags=area,format=gray -f rawvideo``
yields 144 bytes per decoded frame, compared with the 16x9 signature recorded when the frame was offered (tolerance
covers lossy coding and colour-space conversion, not a different picture).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from ..guards import popen_windowless
from .recording import SIG_H, SIG_W


def decode_signatures(ffmpeg: Path, video: Path, timeout: float = 300.0) -> list[np.ndarray]:
    p = popen_windowless([str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(video), "-vf",
                          f"scale={SIG_W}:{SIG_H}:flags=area,format=gray", "-f", "rawvideo", "pipe:1"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate(timeout=5)
        raise TimeoutError("recording inspection timed out") from None
    if p.returncode != 0:
        raise RuntimeError(f"decoder failed: {err.decode('utf-8', 'replace')[-300:]}")
    n = SIG_W * SIG_H
    arr = np.frombuffer(out, np.uint8)
    return [arr[i * n:(i + 1) * n] for i in range(len(arr) // n)]


def inspect_recording(ffmpeg: Path, video: Path, framemap: Path, tolerance: float = 10.0) -> dict:
    doc = json.loads(Path(framemap).read_text("utf-8"))
    entries = doc["entries"]
    decoded = decode_signatures(ffmpeg, video)
    rep = {"video": str(video), "bytes": Path(video).stat().st_size, "frames_decoded": len(decoded),
           "frames_mapped": len(entries), "frames_written_claimed": doc.get("frames_written"),
           "count_match": len(decoded) == len(entries) == doc.get("frames_written")}
    diffs = []
    bad = []
    for i, (e, d) in enumerate(zip(entries, decoded)):
        sig = np.frombuffer(bytes.fromhex(e["sig"]), np.uint8)
        diff = float(np.abs(sig.astype(np.int16) - d.astype(np.int16)).mean())
        diffs.append(diff)
        if diff > tolerance:
            bad.append(i)
    rep.update(mean_abs_diff=round(float(np.mean(diffs)), 3) if diffs else None,
               max_abs_diff=round(float(np.max(diffs)), 3) if diffs else None, tolerance=tolerance,
               mismatched_frames=len(bad), first_mismatch=bad[0] if bad else None,
               content_match=bool(diffs) and not bad and rep["count_match"])
    return rep
