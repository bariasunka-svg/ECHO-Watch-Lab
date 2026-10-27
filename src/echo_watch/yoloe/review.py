# SPDX-License-Identifier: AGPL-3.0-only
"""Saved YOLOE test results: bounded listing, paged frame records and exact original-frame retrieval (no inference).

* ``list_runs`` reads at most ``limit`` manifests (newest first); an unreadable manifest is listed as such.
* ``TracePager`` indexes ONLY the byte offsets of frame records (one scan on a worker thread) and reads a page or one
  record on demand, so a long test never sits in memory. Lifecycle records (epochs, gaps, pauses, settles, views) are
  summarized with bounded lists.
* ``verify_source`` re-checks that the file at the recorded path is the ORIGINAL evidence (size, then the full content
  SHA-256); a different file at the same path is reported as an evidence gap, never decoded as if it were the source.
* ``decode_original`` (the small exact-frame adapter) decodes the recorded frame from the original file with the same
  scaling chain as the test: it seeks about one second early, logs each decoded frame's PTS with ``showinfo`` and
  returns ONLY the frame whose PTS equals the recorded one (input ``-ss`` shifts timestamps by exactly the seek value),
  then compares its 16x9 signature with the one saved at analysis time. No match = explicit gap. Nothing is
  re-inferred; the boxes shown are the saved detector records of that frame.
"""
from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import FFMPEG
from ..engine.recording import SIG_H, SIG_W, signature
from ..guards import popen_windowless
from ..sources.fingerprint import fingerprint
from ..sources.ffmpeg_source import parse_output_size
from .source import parse_showinfo

DECODE_TIMEOUT_S = 20.0
SCAN_FRAMES = 240
PTS_TOL_S = 1e-3
SIG_MATCH = 6.0
_SCALE = ("scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")


def list_runs(root: Path, limit: int = 200) -> list[dict]:
    root = Path(root)
    if not root.is_dir():
        return []
    dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for d in dirs:
        row = {"run_id": d.name, "dir": str(d)}
        try:
            m = json.loads((d / "manifest.json").read_text("utf-8"))
        except (OSError, ValueError) as e:
            row.update(status="unreadable", reason=f"manifest unreadable: {type(e).__name__}", complete=False)
            out.append(row)
            continue
        oc = m.get("outcome") or {}
        row.update(source=(m.get("source") or {}).get("name"), started_wall=m.get("started_wall"),
                   status=oc.get("status") or m.get("status"), reason=oc.get("reason"),
                   complete=oc.get("complete"), processed=oc.get("frames_analysed", (m.get("counts") or {}).get("processed")),
                   failure=oc.get("failure"), details_captured=bool(oc.get("details_captured")),
                   settings=((m.get("profile") or {}).get("detector")), manifest=m)
        out.append(row)
    return out


class TracePager:
    def __init__(self, run_dir: Path, max_events: int = 500):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "trace.jsonl"
        self.offsets: list[int] = []
        self.events: list[dict] = []          # bounded summary of lifecycle records
        self.events_omitted = 0
        self.bad_lines = 0
        self.max_events = max_events
        self.indexed = False

    def index(self, cancel: threading.Event | None = None) -> int:
        offsets, events = [], []
        with open(self.path, "rb") as fh:
            pos = 0
            for line in fh:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("cancelled")
                here, pos = pos, pos + len(line)
                if b'"k":"frame"' in line[:200]:
                    offsets.append(here)
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    self.bad_lines += 1
                    continue
                if rec.get("k") in ("paint", "view"):
                    continue
                if len(events) < self.max_events:
                    events.append(rec)
                else:
                    self.events_omitted += 1
        self.offsets, self.events, self.indexed = offsets, events, True
        return len(offsets)

    def __len__(self) -> int:
        return len(self.offsets)

    def record(self, n: int) -> dict:
        with open(self.path, "rb") as fh:
            fh.seek(self.offsets[n])
            return json.loads(fh.readline())

    def page(self, start: int, limit: int = 50) -> list[dict]:
        rows = []
        with open(self.path, "rb") as fh:
            for n in range(max(0, start), min(len(self.offsets), start + limit)):
                fh.seek(self.offsets[n])
                try:
                    r = json.loads(fh.readline())
                except ValueError:
                    rows.append({"n": n, "error": "unreadable record"})
                    continue
                md, c = r.get("media") or {}, r.get("counts") or {}
                rows.append({"n": n, "seq": r.get("seq"), "src_n": md.get("src_n"), "out_n": md.get("out_n"),
                             "pts": md.get("media_pts", md.get("pts")), "verified": md.get("verified"),
                             "detections": len(r.get("detections") or []), "associated": c.get("associated"),
                             "untracked": c.get("untracked"), "epoch": r.get("epoch"),
                             "disposition": r.get("disposition")})
        return rows


def verify_source(manifest: dict, cancelled=lambda: False) -> dict:
    src = manifest.get("source") or {}
    path = src.get("path")
    if not path or not Path(path).is_file():
        return {"ok": False, "why": "The original video is not at its recorded path; frames cannot be shown."}
    p = Path(path)
    if p.stat().st_size != src.get("bytes"):
        return {"ok": False, "why": "A different file (another size) is at the recorded path; it is not this test's source."}
    fp = fingerprint(p, cancelled)
    if fp["sha256"] != src.get("sha256"):
        return {"ok": False, "why": "The file at the recorded path has different content; it is not this test's source."}
    return {"ok": True, "why": None, "path": str(p)}


@dataclass(frozen=True)
class FrameRequest:
    path: str
    pts: float                         # the frame's recorded PTS (decoder timeline of the test)
    size: tuple[int, int]              # processing size of the test
    expect_sig: str | None
    max_w: int = 1280
    max_h: int = 720
    token: object = None


@dataclass(frozen=True)
class ReviewFrame:
    request: FrameRequest
    image: np.ndarray | None           # BGR at processing size (full), or None
    found_pts: float | None
    signature_diff: float | None
    error: str | None = None

    @property
    def matches(self) -> bool:
        return self.image is not None and self.signature_diff is not None and self.signature_diff <= SIG_MATCH


def review_args(ffmpeg: Path, path: str, ss: float, scan_frames: int = SCAN_FRAMES) -> list[str]:
    args = [str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-threads", "1"]
    if ss > 0:
        args += ["-ss", f"{ss:.6f}"]
    return args + ["-i", path, "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", "showinfo," + _SCALE, "-fps_mode",
                   "passthrough", "-frames:v", str(scan_frames), "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]


def decode_original(req: FrameRequest, cancel: threading.Event | None = None, *, ffmpeg: Path = FFMPEG,
                    spawn=popen_windowless) -> ReviewFrame:
    """Stream decoded frames ONE at a time (memory: one frame) until the recorded PTS is found or passed."""
    ss = round(max(0.0, req.pts - 1.0), 6)
    w, h = req.size
    n = w * h * 3
    recs: dict[int, float] = {}
    cond = threading.Condition()
    done = {"stderr": False, "size": None}
    header: list[str] = []
    p = None

    def read_err():
        try:
            for raw in iter(p.stderr.readline, b""):
                line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
                if "Parsed_showinfo_0" in line:
                    r = parse_showinfo(line)
                    if r is not None:
                        with cond:
                            recs[r[0]] = r[1]
                            cond.notify_all()
                elif done["size"] is None:
                    header.append(line.strip())
                    size = parse_output_size(header)
                    if size:
                        with cond:
                            done["size"] = size
                            cond.notify_all()
        except (OSError, ValueError):
            pass
        with cond:
            done["stderr"] = True
            cond.notify_all()

    try:
        p = spawn(review_args(ffmpeg, req.path, ss), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        threading.Thread(target=read_err, name="yoloe-review-stderr", daemon=True).start()
        deadline = time.monotonic() + DECODE_TIMEOUT_S
        with cond:                      # the frame size is checked BEFORE any byte is read as a frame
            while done["size"] is None and not done["stderr"] and time.monotonic() < deadline:
                cond.wait(0.05)
            size = done["size"]
        if size is None:
            return ReviewFrame(req, None, None, None, "the decoder did not report its output size")
        if tuple(size) != (w, h):
            return ReviewFrame(req, None, None, None, f"the decoded frames are {size[0]}x{size[1]}, not the test's "
                               f"processing size {w}x{h} (evidence gap)")
        for k in range(SCAN_FRAMES):
            if cancel is not None and cancel.is_set():
                return ReviewFrame(req, None, None, None, "cancelled")
            buf = p.stdout.read(n)
            if not buf or len(buf) < n:
                why = ("the decoded frames have a different size than the test's processing size" if buf else
                       "no decoded frame has this record's timestamp (evidence gap)")
                return ReviewFrame(req, None, None, None, why)
            with cond:
                while k not in recs and not done["stderr"] and time.monotonic() < deadline:
                    cond.wait(0.05)
                pts = recs.pop(k, None)
            if pts is None:
                return ReviewFrame(req, None, None, None, "the decoder did not report this frame's timestamp")
            if abs(pts + ss - req.pts) <= PTS_TOL_S:
                img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
                diff = None
                if req.expect_sig:
                    a = np.frombuffer(bytes.fromhex(signature(img)), np.uint8).astype(np.int16)
                    b = np.frombuffer(bytes.fromhex(req.expect_sig), np.uint8).astype(np.int16)
                    diff = float(np.abs(a - b).mean()) if a.size == b.size == SIG_W * SIG_H else None
                return ReviewFrame(req, img, pts + ss, diff)
            if pts + ss > req.pts + PTS_TOL_S:
                return ReviewFrame(req, None, None, None, "no decoded frame has this record's timestamp (evidence gap)")
            if time.monotonic() > deadline:
                return ReviewFrame(req, None, None, None, "decoding timed out")
        return ReviewFrame(req, None, None, None, "the recorded frame was not reached within the bounded scan")
    except Exception as e:  # noqa: BLE001 - a visible gap, never a nearby frame
        return ReviewFrame(req, None, None, None, f"{type(e).__name__}: {e}")
    finally:
        if p is not None and p.poll() is None:
            p.kill()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
