# SPDX-License-Identifier: AGPL-3.0-only
"""Controlled FFmpeg decode of an existing local video FILE (EW-01 foundation: files only).

Adapted from the preserved ECHO Watch reference runtime/echowatch/sources/ffmpeg_source.py (SHA-256
25955be86d5d855dfc7f33bedcf6baeb03de5c6841c80f6ba304e1c75bcd0d28; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md).
Adaptation:
* FILE sources only. Network (RTSP/HTTP) and DirectShow (camera/OBS) construction is removed: no camera, OBS,
  microphone or RTSP activation is authorised in this stage, so the code that could build such arguments is absent.
* ``-threads 1`` is applied as an INPUT option for files: the reference's own generated-input comparison showed
  ~0.54-0.57 s pre-wrap decoder gaps with default threading versus ~0.14 s worst gap with one decoder thread
  (review/codex-delivery-review-20261025/REVIEW.md, finding R2). It is a measured stall fix, not a tuning sweep.
* The executable is the pinned imageio-ffmpeg binary at an absolute project path whose SHA-256 is verified before use.
* ``Repeat`` defaults OFF: a finished file raises ``SourceEnded`` (a normal end, never a reconnect loop).
* Frame counters (decoded / overwritten before the consumer read them) and a duplicate-source-frame marker
  (``media['dup_of_prev']``: the decoder's ``fps`` resampler repeated an original-file frame) are added so a repeated
  picture can never be counted as an independent observation.

The same decode carries a documented per-frame side channel. FFmpeg's ``showinfo`` filter logs each frame's ordinal,
PTS and checksum: once before the ``fps`` resampler (original-file frames) and once after it (the frames received).
The k-th raw frame on stdout is output frame k-1, so every received frame is joined to its exact output record by
ordinal and to its original-file frame by checksum and nearest PTS. Replay position comes from that original PTS and a
bounded stream-copy pre-probe of the file's own packets; nothing is inferred from wall time. A record that is not
available yet leaves the frame's media info unavailable (counted), never guessed.
"""
from __future__ import annotations

import collections
import hashlib
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..frames import Epoch, Frame
from ..guards import popen_windowless
from .base import FrameSource, SourceEnded, SourceError, redact

VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts"}
_SIZE_RE = re.compile(r"\b(\d{2,5})x(\d{2,5})\b")
_SHOWINFO_RE = re.compile(r"\[Parsed_showinfo_(\d+) @ [^\]]+\] n:\s*(\d+) pts:\s*(-?\d+) pts_time:(-?[0-9.]+)"
                          r".*? checksum:([0-9A-Fa-f]+)")
_TB_RE = re.compile(r"^#tb 0: (\d+)/(\d+)")
PROBE_TIMEOUT_S = 30.0
HASH_MAX_BYTES = 512 * 2 ** 20
SRC_RECORDS, OUT_RECORDS = 96, 256
MEDIA_WAIT_S = 0.02
SHOWINFO_SRC, SHOWINFO_OUT = 0, 2     # filter-graph positions of the source-side and output-side showinfo instances


def validate_file(location: str) -> str:
    p = Path(location)
    if not p.is_absolute() or not p.is_file():
        raise SourceError("Video file must be an existing absolute path")
    if p.suffix.lower() not in VIDEO_SUFFIXES:
        raise SourceError(f"Unsupported video file type {p.suffix}")
    return str(p)


def build_args(ffmpeg: Path, location: str, fps: float, loop_file: bool = False) -> list[str]:
    args = [str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            "-threads", "1", "-re"] + (["-stream_loop", "-1"] if loop_file else []) + ["-i", location]
    vf = (f"showinfo,fps={fps:g},showinfo,scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    args += ["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    return args


def probe_args(ffmpeg: Path, location: str) -> list[str]:
    """Stream-copy packet listing of the file's video stream (no decode, no output file)."""
    return [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", location, "-map", "0:v:0",
            "-c", "copy", "-f", "framecrc", "-"]


def parse_framecrc(text: str) -> dict | None:
    """Frame count, presentation timestamps and duration of one loop from a framecrc packet listing."""
    tb = None
    pts: list[tuple[int, int]] = []
    for line in text.splitlines():
        m = _TB_RE.match(line)
        if m:
            tb = int(m.group(1)) / int(m.group(2))
            continue
        if line.startswith("0,"):
            f = [x.strip() for x in line.split(",")]
            try:
                pts.append((int(f[2]), int(f[3])))
            except (IndexError, ValueError):
                return None
    if not tb or not pts:
        return None
    pts.sort()
    first, (last, dur) = pts[0][0], pts[-1]
    return {"frames": len(pts), "duration_s": round((last + dur - first) * tb, 6), "first_pts_s": round(first * tb, 6),
            "pts_s": [round((p - first) * tb, 6) for p, _d in pts], "time_base": tb}


def file_identity(path: str) -> dict:
    p = Path(path)
    size = p.stat().st_size
    out = {"name": p.name, "bytes": size, "sha256": None}
    if size <= HASH_MAX_BYTES:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out["sha256"] = h.hexdigest()
    else:
        out["sha256_note"] = f"not hashed (larger than {HASH_MAX_BYTES // 2 ** 20} MiB)"
    return out


def parse_output_size(lines: list[str]) -> tuple[int, int] | None:
    seen_output = False
    for line in lines:
        if line.startswith("Output #0"):
            seen_output = True
        elif seen_output and "Stream #0:0" in line and "Video:" in line:
            m = _SIZE_RE.search(line.split("Video:", 1)[1])
            if m:
                return int(m.group(1)), int(m.group(2))
    return None


class FFmpegFileSource(FrameSource):
    kind = "ffmpeg-file"

    def __init__(self, source_id: str, ffmpeg: Path, location: str, fps: float, loop_file: bool = False):
        super().__init__(source_id)
        self.ffmpeg = ffmpeg
        self.location = validate_file(location)
        self.fps = fps
        self.loop_file = loop_file
        self.media: dict | None = None          # file identity + pre-probed frame count/duration
        self.media_stats = {"joined": 0, "unavailable": 0, "unverified": 0, "src_records": 0, "out_records": 0,
                            "parse_errors": 0, "duplicate_source_frames": 0}
        self._src_recs: collections.deque = collections.deque(maxlen=SRC_RECORDS)
        self._out_recs: dict[int, tuple[float, str]] = {}
        self._ended = False
        self.proc: subprocess.Popen | None = None
        self.size: tuple[int, int] | None = None
        self.stderr_tail: collections.deque[str] = collections.deque(maxlen=60)
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._seq = 0
        self.frames_decoded = 0
        self.frames_overwritten = 0           # decoded frames replaced before the consumer read them
        self._last_src_n: int | None = None
        self._stopped = threading.Event()
        self._error: str | None = None

    def start(self, epoch: Epoch) -> None:
        self.epoch = epoch
        if self._stopped.is_set():
            raise SourceError("cancelled before the decoder started")
        if self.media is None:
            self.media = self._probe_file()
        args = build_args(self.ffmpeg, self.location, self.fps, self.loop_file)
        self.proc = popen_windowless(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        if self._stopped.is_set():  # stop() raced with process creation: never leave an orphan decoder
            self.stop()
            raise SourceError("cancelled while the decoder was starting")
        threading.Thread(target=self._stderr_loop, name="ffmpeg-stderr", daemon=True).start()
        deadline = time.monotonic() + 15
        with self._cond:
            while self.size is None and self._error is None and time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    break
                self._cond.wait(0.2)
        if self.size is None:
            self.stop()
            detail = self._error or "; ".join(list(self.stderr_tail)[-4:]) or "no video stream reported"
            raise SourceError(f"Source did not start: {redact(detail)}")
        threading.Thread(target=self._stdout_loop, name="ffmpeg-frames", daemon=True).start()

    def _probe_file(self) -> dict:
        """Bounded, windowless stream-copy probe of the file's own packets plus its byte identity."""
        info: dict = {"kind": "file", "repeat": self.loop_file}
        try:
            info.update(file_identity(self.location))
        except OSError as e:
            info["identity_error"] = str(e)
        try:
            p = popen_windowless(probe_args(self.ffmpeg, self.location), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, _err = p.communicate(timeout=PROBE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate(timeout=5)
                info["probe_error"] = f"packet probe exceeded {PROBE_TIMEOUT_S:g} s; loop/position unavailable"
                return info
            parsed = parse_framecrc(out.decode("utf-8", "replace"))
            if parsed is None:
                info["probe_error"] = "packet listing could not be read; loop/position unavailable"
            else:
                info.update({k: parsed[k] for k in ("frames", "duration_s", "first_pts_s")})
                info["_pts"] = parsed["pts_s"]
        except OSError as e:
            info["probe_error"] = f"packet probe failed: {e}"
        return info

    def _stderr_loop(self) -> None:
        assert self.proc and self.proc.stderr
        lines: list[str] = []
        for raw in iter(self.proc.stderr.readline, b""):
            text = raw.decode("utf-8", "replace").rstrip()
            if "Parsed_showinfo_" in text:
                m = _SHOWINFO_RE.search(text)
                if m:
                    idx, n, pts_t, ck = int(m.group(1)), int(m.group(2)), float(m.group(4)), m.group(5).upper()
                    with self._cond:
                        if idx == SHOWINFO_SRC:
                            self._src_recs.append((n, pts_t, ck))
                            self.media_stats["src_records"] += 1
                        elif idx == SHOWINFO_OUT:
                            self._out_recs[n] = (pts_t, ck)
                            self.media_stats["out_records"] += 1
                            while len(self._out_recs) > OUT_RECORDS:
                                self._out_recs.pop(next(iter(self._out_recs)))
                        self._cond.notify_all()
                    continue
                if " n:" in text:
                    self.media_stats["parse_errors"] += 1
                    continue
                if "color_range" in text or "config in" in text or "config out" in text:
                    continue
            line = redact(text)
            self.stderr_tail.append(line)
            if self.size is None:
                lines.append(line.strip())
                size = parse_output_size(lines)
                if size:
                    with self._cond:
                        self.size = size
                        self._cond.notify_all()
        with self._cond:
            # once frames flow, only the frame reader classifies the end (after reading the exit code)
            if self._error is None and not self._stopped.is_set() and self.size is None:
                self._error = "decoder exited: " + "; ".join(list(self.stderr_tail)[-3:])
            self._cond.notify_all()

    def _media_for(self, out_n: int) -> dict | None:
        """Join output frame ``out_n`` (0-based) to its side-channel records (caller holds self._cond)."""
        rec = self._out_recs.get(out_n)
        if rec is None:
            self._cond.wait(MEDIA_WAIT_S)          # the log line normally precedes the frame bytes
            rec = self._out_recs.get(out_n)
        if rec is None:
            self.media_stats["unavailable"] += 1
            return None
        out_pts, ck = rec
        m: dict = {"out_n": out_n, "out_pts": round(out_pts, 6), "clock": "ffmpeg output pts (not acquisition time)"}
        best = None
        for sn, sp, sck in self._src_recs:
            if sck == ck and (best is None or abs(sp - out_pts) < abs(best[1] - out_pts)):
                best = (sn, sp)
        if best is None:
            self.media_stats["unverified"] += 1
            m["source"] = "original-file frame not matched in the bounded side channel"
            return m
        sn, sp = best
        m.update({"src_n": sn, "src_pts": round(sp, 6), "clock": "original file pts"})
        md = self.media or {}
        N, D = md.get("frames"), md.get("duration_s")
        if N and D:
            loop = sn // N
            pos = round(sp - (md.get("first_pts_s") or 0.0) - loop * D, 6)
            ordinal = sn % N
            exp = (md.get("_pts") or [None] * N)[ordinal] if ordinal < len(md.get("_pts") or []) else None
            m.update({"loop": loop, "media_n": ordinal, "media_pts": pos,
                      "verified": exp is not None and abs(exp - pos) < 1e-3})
            if not m["verified"]:
                self.media_stats["unverified"] += 1
        if self._last_src_n is not None and sn == self._last_src_n:
            m["dup_of_prev"] = True               # the fps resampler repeated an original frame: not a new observation
            self.media_stats["duplicate_source_frames"] += 1
        self._last_src_n = sn
        self.media_stats["joined"] += 1
        return m

    def _stdout_loop(self) -> None:
        assert self.proc and self.proc.stdout and self.size and self.epoch
        w, h = self.size
        n = w * h * 3
        out = self.proc.stdout
        while not self._stopped.is_set():
            buf = bytearray(n)
            view = memoryview(buf)
            got = 0
            while got < n:
                k = out.readinto(view[got:])
                if not k:
                    break
                got += k
            if got < n:
                break
            self._seq += 1
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            t_m, t_w = time.monotonic(), time.time()
            with self._cond:
                media = self._media_for(self._seq - 1)
            f = Frame(self.epoch, self._seq, t_m, t_w, img, media=media)
            with self._cond:
                if self._latest is not None:
                    self.frames_overwritten += 1
                self._latest = f
                self.frames_decoded += 1
                self._cond.notify_all()
        rc = None
        if self.proc is not None:
            try:
                rc = self.proc.wait(5)
            except subprocess.TimeoutExpired:
                rc = None
        with self._cond:
            if self._error is None and not self._stopped.is_set():
                if not self.loop_file and rc == 0:
                    self._ended = True
                    self._error = "video file ended"
                elif rc not in (0, None):
                    self._error = f"decoder exited (code {rc}): " + "; ".join(list(self.stderr_tail)[-3:])
                else:
                    self._error = "stream ended"
            self._cond.notify_all()

    def read(self, timeout: float) -> Frame | None:
        with self._cond:
            if self._latest is None and self._error is None:
                self._cond.wait(timeout)
            f, self._latest = self._latest, None
            if f is None and self._error:
                if self._ended:
                    raise SourceEnded(self._error)
                raise SourceError(redact(self._error))
            return f

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self._error is None

    def stop(self) -> None:
        self._stopped.set()
        p = self.proc
        if p and p.poll() is None:
            p.kill()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        with self._cond:
            self._cond.notify_all()

    def diagnostics(self) -> dict:
        return {"decoder": "ffmpeg", "size": self.size, "pid": self.proc.pid if self.proc else None,
                "stderr_tail": list(self.stderr_tail)[-8:], "error": self._error,
                "media": {k: v for k, v in (self.media or {}).items() if not k.startswith("_")} or None,
                "side_channel": dict(self.media_stats), "frames_decoded": self.frames_decoded,
                "frames_overwritten": self.frames_overwritten}
