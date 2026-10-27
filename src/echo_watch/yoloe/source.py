# SPDX-License-Identifier: AGPL-3.0-only
"""ORDERED, bounded, back-pressured replay of a local video file for the YOLOE workspace (EW-LAB-06).

The Display Assets source (``sources.ffmpeg_source``) is a paced latest-frame source with an ``fps`` resampler; it is
unchanged and still used there. A tracker evaluated on a file needs every decoded source frame once, in order, so this
source differs deliberately:
* no ``-re`` and no ``fps`` filter: ``-fps_mode passthrough`` writes each decoded frame once with its own timestamp
  (VFR files keep their irregular timing; nothing is duplicated or dropped by a resampler);
* frames go through a SMALL bounded queue with a BLOCKING put: when the consumer is slower (inference) or paused, the
  reader stops reading the pipe, the pipe fills and FFmpeg itself blocks. Nothing is skipped and nothing accumulates;
* one ``showinfo`` before scaling logs each frame's ordinal, PTS, native size and checksum. Output frame k is joined
  to record n == k (passthrough: one output per decoded frame) and its PTS is verified against a bounded stream-copy
  packet listing of the file, giving the ORIGINAL source ordinal. A missing record or an unmatched PTS leaves the
  mapping explicitly unverified; a jump in source ordinals is reported as a GAP of that many frames - never invented;
* pacing is the consumer's job (``MediaClock``): source speed when possible, honest slowdown when not, no burst
  catch-up after a stall or a Pause, paused wall time never charged as media time.
The process is started through an injectable ``spawn`` (default: the windowless owned-child helper) so the parsing,
mapping, back-pressure and end handling are testable with synthetic byte streams and log lines.
"""
from __future__ import annotations

import bisect
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..frames import Epoch, Frame
from ..guards import popen_windowless
from ..sources.base import FrameSource, SourceEnded, SourceError, redact
from ..sources.ffmpeg_source import parse_framecrc, parse_output_size, probe_args, validate_file

_SHOWINFO_RE = re.compile(r"\[Parsed_showinfo_0 @ [^\]]+\] n:\s*(\d+) pts:\s*(-?\d+) pts_time:(-?[0-9.]+)"
                          r".*? checksum:([0-9A-Fa-f]+)")
_SIZE_RE = re.compile(r"\bs:(\d+)x(\d+)\b")
PROBE_TIMEOUT_S = 30.0
START_TIMEOUT_S = 15.0
RECORD_WAIT_S = 0.25
MAX_RECORDS = 1024
PTS_TOL_S = 1e-3
_END = object()


def build_ordered_args(ffmpeg: Path, location: str) -> list[str]:
    vf = ("showinfo,scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    return [str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info", "-threads", "1", "-i", location,
            "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", vf, "-fps_mode", "passthrough", "-pix_fmt", "bgr24",
            "-f", "rawvideo", "pipe:1"]


def parse_showinfo(line: str) -> tuple[int, float, str, tuple[int, int] | None] | None:
    m = _SHOWINFO_RE.search(line)
    if not m:
        return None
    s = _SIZE_RE.search(line)
    return int(m.group(1)), float(m.group(3)), m.group(4).upper(), (int(s.group(1)), int(s.group(2))) if s else None


def probe_packets(ffmpeg: Path, location: str, spawn=popen_windowless, timeout_s: float = PROBE_TIMEOUT_S) -> dict:
    """Bounded stream-copy packet listing: frame count, per-frame PTS (presentation order) and first PTS."""
    info: dict = {}
    try:
        p = spawn(probe_args(ffmpeg, location), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            out, _err = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate(timeout=5)
            return {"probe_error": f"packet probe exceeded {timeout_s:g} s; ordinals cannot be verified"}
        parsed = parse_framecrc(out.decode("utf-8", "replace") if isinstance(out, bytes) else out)
        if parsed is None:
            return {"probe_error": "packet listing could not be read; ordinals cannot be verified"}
        info.update({k: parsed[k] for k in ("frames", "duration_s", "first_pts_s", "time_base")})
        info["_pts"] = parsed["pts_s"]
    except OSError as e:
        info["probe_error"] = f"packet probe failed: {e}"
    return info


def map_frame(out_n: int, rec: tuple | None, probe: dict, last_src_n: int | None) -> dict:
    """Media facts of output frame ``out_n`` from its showinfo record and the packet listing (never guessed)."""
    m: dict = {"out_n": out_n, "verified": False}
    if rec is None:
        m["why"] = "no side-channel record for this frame"
        return m
    pts, ck = rec[0], rec[1]
    m.update(pts=round(pts, 6), checksum=ck)
    pl = probe.get("_pts")
    if not pl:
        m["why"] = probe.get("probe_error") or "no packet listing"
        return m
    media_pts = pts - (probe.get("first_pts_s") or 0.0)
    m["media_pts"] = round(media_pts, 6)
    j = bisect.bisect_left(pl, media_pts - PTS_TOL_S)
    hits = [k for k in (j, j + 1) if k < len(pl) and abs(pl[k] - media_pts) <= PTS_TOL_S]
    if len(hits) != 1:
        m["why"] = "PTS not found once in the packet listing"
        return m
    src_n = hits[0]
    m.update(src_n=src_n, verified=True)
    expected = 0 if last_src_n is None else last_src_n + 1
    if src_n > expected:
        m["gap_before"] = src_n - expected
    elif last_src_n is not None and src_n <= last_src_n:
        m["verified"], m["why"] = False, f"source ordinal {src_n} does not advance past {last_src_n}"
    return m


class OrderedFileSource(FrameSource):
    kind = "ffmpeg-file-ordered"

    def __init__(self, source_id: str, ffmpeg: Path, location: str, *, queue_frames: int = 4, spawn=popen_windowless,
                 probe=None, record_wait_s: float = RECORD_WAIT_S, validate: bool = True):
        super().__init__(source_id)
        self.ffmpeg, self.location = ffmpeg, validate_file(location) if validate else location
        self.spawn = spawn
        self._probe_fn = probe or (lambda: probe_packets(ffmpeg, self.location, spawn))
        self.record_wait_s = record_wait_s
        self.q: "queue.Queue" = queue.Queue(maxsize=max(1, queue_frames))
        self.queue_frames = max(1, queue_frames)
        self.media: dict | None = None
        self.proc = None
        self.size: tuple[int, int] | None = None
        self.native_size: tuple[int, int] | None = None
        self.stderr_tail: list[str] = []
        self._cond = threading.Condition()
        self._recs: dict[int, tuple] = {}
        self._stop = threading.Event()
        self._error: str | None = None
        self._ended = False
        self._last_src_n: int | None = None
        self.stats = {"decoded": 0, "delivered": 0, "unverified": 0, "gaps": 0, "gap_frames": 0, "records": 0,
                      "records_dropped": 0, "parse_errors": 0, "blocked_put_s": 0.0, "max_queue": 0}
        self._threads: list[threading.Thread] = []

    # ---- lifecycle ---------------------------------------------------------------------------------------------
    def start(self, epoch: Epoch) -> None:
        self.epoch = epoch
        if self._stop.is_set():
            raise SourceError("cancelled before the decoder started")
        if self.media is None:
            self.media = self._probe_fn()
        self.proc = self.spawn(build_ordered_args(self.ffmpeg, self.location), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0)
        if self._stop.is_set():
            self.stop()
            raise SourceError("cancelled while the decoder was starting")
        t = threading.Thread(target=self._stderr_loop, name="yoloe-ffmpeg-stderr", daemon=True)
        t.start()
        self._threads.append(t)
        deadline = time.monotonic() + START_TIMEOUT_S
        with self._cond:
            while self.size is None and self._error is None and not self._stop.is_set() and time.monotonic() < deadline:
                if self.proc.poll() is not None and self.size is None:
                    self._cond.wait(0.2)
                    break
                self._cond.wait(0.1)
        if self.size is None:
            detail = self._error or "; ".join(self.stderr_tail[-4:]) or "no video stream reported"
            self.stop()
            raise SourceError(f"Source did not start: {redact(detail)}")
        t = threading.Thread(target=self._stdout_loop, name="yoloe-ffmpeg-frames", daemon=True)
        t.start()
        self._threads.append(t)

    def _stderr_loop(self) -> None:
        lines: list[str] = []
        try:
            for raw in iter(self.proc.stderr.readline, b""):
                text = raw.decode("utf-8", "replace").rstrip() if isinstance(raw, bytes) else raw.rstrip()
                if "Parsed_showinfo_0" in text:
                    rec = parse_showinfo(text)
                    with self._cond:
                        if rec is None:
                            if " n:" in text:
                                self.stats["parse_errors"] += 1
                        else:
                            n, pts, ck, size = rec
                            self._recs[n] = (pts, ck)
                            self.stats["records"] += 1
                            if size and self.native_size is None:
                                self.native_size = size
                            while len(self._recs) > MAX_RECORDS:
                                self._recs.pop(next(iter(self._recs)))
                                self.stats["records_dropped"] += 1
                        self._cond.notify_all()
                    continue
                line = redact(text)
                self.stderr_tail = (self.stderr_tail + [line])[-40:]
                if self.size is None:
                    lines.append(line.strip())
                    size = parse_output_size(lines)
                    if size:
                        with self._cond:
                            self.size = size
                            self._cond.notify_all()
        except (OSError, ValueError):
            pass
        with self._cond:
            if self.size is None and self._error is None and not self._stop.is_set():
                self._error = "decoder exited: " + "; ".join(self.stderr_tail[-3:])
            self._cond.notify_all()

    def _put(self, item) -> bool:
        t0 = time.monotonic()
        while not self._stop.is_set():
            try:
                self.q.put(item, timeout=0.1)
                self.stats["blocked_put_s"] += time.monotonic() - t0
                self.stats["max_queue"] = max(self.stats["max_queue"], self.q.qsize())
                return True
            except queue.Full:
                continue
        return False

    def _stdout_loop(self) -> None:
        w, h = self.size
        n = w * h * 3
        out = self.proc.stdout
        k, got = 0, 0
        while not self._stop.is_set():
            buf = bytearray(n)
            view = memoryview(buf)
            got = 0
            while got < n:
                r = out.readinto(view[got:])
                if not r:
                    break
                got += r
            if got < n:
                break
            img = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            with self._cond:
                end = time.monotonic() + self.record_wait_s
                while k not in self._recs and time.monotonic() < end and not self._stop.is_set():
                    self._cond.wait(0.02)
                rec = self._recs.pop(k, None)
                for old in [x for x in self._recs if x < k]:
                    del self._recs[old]
            media = map_frame(k, rec, self.media or {}, self._last_src_n)
            if media.get("verified"):
                self._last_src_n = media["src_n"]
                if media.get("gap_before"):
                    self.stats["gaps"] += 1
                    self.stats["gap_frames"] += media["gap_before"]
            else:
                self.stats["unverified"] += 1
            self.stats["decoded"] += 1
            f = Frame(self.epoch, k + 1, time.monotonic(), time.time(), img, media=media)
            k += 1
            if not self._put(f):
                return
        rc = None
        try:
            rc = self.proc.wait(5)
        except subprocess.TimeoutExpired:
            rc = None
        with self._cond:
            if self._error is None and not self._stop.is_set():
                if rc == 0 and got == 0:
                    self._ended = True
                    self._error = "video file ended"
                elif rc == 0:
                    self._error = f"decoder output ended inside a frame ({got} of {n} bytes)"
                else:
                    self._error = f"decoder exited (code {rc}): " + "; ".join(self.stderr_tail[-3:])
        self._put(_END)

    def read(self, timeout: float) -> Frame | None:
        """The NEXT frame in order, None if none is ready within ``timeout``; ``SourceEnded`` after the last frame of
        a normally finished file; ``SourceError`` (after every decoded frame was delivered) on a decoder failure."""
        try:
            item = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _END:
            try:
                self.q.put_nowait(_END)          # every later read sees the same end
            except queue.Full:
                pass
            if self._ended:
                raise SourceEnded(self._error or "video file ended")
            raise SourceError(redact(self._error or "the decoder stopped"))
        self.stats["delivered"] += 1
        return item

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self._error is None

    def stop(self) -> None:
        self._stop.set()
        p = self.proc
        if p is not None and p.poll() is None:
            p.kill()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        with self._cond:
            self._cond.notify_all()
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(2.0)

    def diagnostics(self) -> dict:
        md = {k: v for k, v in (self.media or {}).items() if not k.startswith("_")}
        return {"decoder": "ffmpeg ordered passthrough", "size": self.size, "native_size": self.native_size,
                "pid": self.proc.pid if self.proc is not None else None, "error": self._error, "ended": self._ended,
                "media": md or None, "stats": {k: (round(v, 3) if isinstance(v, float) else v)
                                               for k, v in self.stats.items()},
                "queue_frames": self.queue_frames, "stderr_tail": self.stderr_tail[-6:]}


class MediaClock:
    """Source-speed pacing with honest slowdown. ``plan`` returns how long to wait before showing/analysing a frame at
    ``media_pts``. A frame more than ``late_reanchor_s`` behind its due time re-anchors the clock (playback simply
    continues slower - no burst catch-up); ``suspend`` (Pause) drops the anchor so paused wall time is never charged
    as media time. ``speed`` is media seconds advanced per ACTIVE wall second over a sliding window."""

    def __init__(self, late_reanchor_s: float = 0.25, window_s: float = 3.0):
        self.late_reanchor_s, self.window_s = late_reanchor_s, window_s
        self._anchor: tuple[float, float] | None = None
        self._paused_at: float | None = None
        self.paused_total = 0.0
        self.reanchors = {"start": 0, "late": 0, "resume": 0, "pts_backwards": 0}
        self._samples: list[tuple[float, float]] = []      # (active wall, media pts)

    def suspend(self, now: float) -> None:
        if self._paused_at is None:
            self._paused_at = now
        self._anchor = None

    def resume(self, now: float) -> None:
        if self._paused_at is not None:
            self.paused_total += now - self._paused_at
            self._paused_at = None
            self.reanchors["resume"] += 1

    def active(self, now: float) -> float:
        return now - self.paused_total - ((now - self._paused_at) if self._paused_at is not None else 0.0)

    def plan(self, media_pts: float, now: float) -> float:
        if self._anchor is None:
            if not self._samples:
                self.reanchors["start"] += 1
            self._anchor = (now, media_pts)
            return 0.0
        aw, ap = self._anchor
        if media_pts < ap:
            self.reanchors["pts_backwards"] += 1
            self._anchor = (now, media_pts)
            return 0.0
        wait = aw + (media_pts - ap) - now
        if wait < -self.late_reanchor_s:
            self.reanchors["late"] += 1
            self._anchor = (now, media_pts)
            return 0.0
        return max(0.0, wait)

    def shown(self, media_pts: float, now: float) -> None:
        a = self.active(now)
        self._samples.append((a, media_pts))
        while self._samples and a - self._samples[0][0] > self.window_s:
            self._samples.pop(0)

    def speed(self) -> float | None:
        if len(self._samples) < 2:
            return None
        (a0, m0), (a1, m1) = self._samples[0], self._samples[-1]
        return None if a1 - a0 < 0.2 else round((m1 - m0) / (a1 - a0), 3)
