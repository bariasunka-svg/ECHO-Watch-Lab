# SPDX-License-Identifier: AGPL-3.0-only
"""Exact saved-frame retrieval for a decision, and bounded background playback (Tk-free).

``LatestWorker`` is ADOPTED from the frozen PatronSight reference src/patronsight/app/review_player.py (archive
Patsight_PS27D_C3_reference_source.zip, SHA-256 of the ZIP 72efc34a...; LICENSE: 'Original PatSight contributions:
Copyright (c) 2026 Bari Asunka', AGPL-3.0-only): one daemon thread serves only the NEWEST request (rapid seeks
coalesce), results go to an outbox the Tk thread drains, shutdown is bounded. Body unchanged.

Lab decoder (replaces the reference's cv2.VideoCapture): the pinned FFmpeg in a windowless bounded child decodes ONE
frame of a recorded constant-frame-rate file at ``index / fps`` (accurate seek), downscales it, and keeps a small LRU.
``locate_clean_frame`` maps a decision's frame identity to its exact clean-recording output index through the clean
frame map; when the frame map has no entry (dropped, not recorded, output disabled) the result is an explicit
EVIDENCE GAP, never a nearby frame passed off as the decision's own. Every decoded frame is compared with the map's
16x9 signature (supporting evidence, not cryptographic equality).
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from ..config import FFMPEG
from ..engine.recording import SIG_H, SIG_W, signature
from ..guards import popen_windowless

logger = logging.getLogger(__name__)

FRAME_CACHE_SIZE = 48
DECODE_TIMEOUT_S = 20.0


class LatestWorker:
    """One daemon thread that serves only the newest request."""

    def __init__(self, name: str, fn: Callable[[Any, threading.Event], Any]) -> None:
        self._fn = fn
        self._name = name
        self._condition = threading.Condition()
        self._pending: tuple[int, Any] | None = None
        self._results: list[tuple[int, Any]] = []
        self._stop = threading.Event()
        self._cancel_current = threading.Event()
        self._thread: threading.Thread | None = None
        self._busy = False
        self.submitted = 0
        self.completed = 0
        self.coalesced = 0

    def submit(self, token: int, request: Any) -> None:
        """Queue *request*, replacing (and cancelling) any request not yet started."""
        with self._condition:
            if self._stop.is_set():
                return
            if self._pending is not None:
                self.coalesced += 1
            self._pending = (token, request)
            self.submitted += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name=self._name, daemon=True)
                self._thread.start()
            self._condition.notify()

    def cancel_running(self) -> None:
        """Ask the running job (if it checks its event) to stop early."""
        self._cancel_current.set()

    def _loop(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stop.is_set():
                    self._condition.wait(0.5)
                if self._stop.is_set():
                    return
                token, request = self._pending  # type: ignore[misc]
                self._pending = None
                self._busy = True
                self._cancel_current.clear()
            try:
                result = self._fn(request, self._cancel_current)
            except Exception as error:  # noqa: BLE001 - a failed job is reported, never raised into Tk
                logger.warning("%s job failed (%s)", self._name, type(error).__name__)
                result = error
            with self._condition:
                self._busy = False
                if not self._stop.is_set():
                    self._results.append((token, result))
                    self.completed += 1

    def drain(self) -> list[tuple[int, Any]]:
        with self._condition:
            results, self._results = self._results, []
            return results

    @property
    def idle(self) -> bool:
        with self._condition:
            return self._pending is None and not self._busy and not self._results

    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def shutdown(self, timeout: float = 2.0) -> bool:
        """Stop accepting work, cancel the running job and join (bounded). ``True`` when the thread has exited."""
        with self._condition:
            self._stop.set()
            self._pending = None
            self._cancel_current.set()
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()


@dataclass(frozen=True)
class FrameLookup:
    """Where a decision's own analysed pixels live in the clean recording (or why they cannot be shown)."""

    ok: bool
    video: Path | None
    index: int | None
    fps: float | None
    entry: dict | None
    gap: str | None = None


class FrameMaps:
    """Lazily loaded clean/annotated frame maps of one run (loaded on a worker thread; cached)."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._maps: dict[str, dict | None] = {}
        self._ident: dict[str, dict] = {}

    def load(self, which: str) -> dict | None:
        if which not in self._maps:
            p = self.run_dir / f"{which}.framemap.json"
            try:
                self._maps[which] = json.loads(p.read_text("utf-8")) if p.is_file() else None
            except (OSError, ValueError):
                self._maps[which] = None
            if which == "clean" and self._maps[which]:
                self._ident = {tuple(e["ident"]): e for e in self._maps[which]["entries"] if e.get("ident")}
        return self._maps[which]

    def locate_clean_frame(self, frame_ident) -> FrameLookup:
        m = self.load("clean")
        if m is None:
            return FrameLookup(False, None, None, None, None,
                               "No clean recording frame map for this run (clean output not requested, not started or lost).")
        e = self._ident.get(tuple(frame_ident))
        if e is None:
            dropped = any(tuple(d.get("ident") or ()) == tuple(frame_ident) for d in m.get("dropped", []))
            why = ("This decision's frame was dropped by the clean recorder's bounded queue (counted loss)." if dropped else
                   "This decision's frame is not in the clean recording (outside the recorded interval or lost).")
            return FrameLookup(False, self.run_dir / m["video"], None, m.get("fps_nominal"), None, why)
        return FrameLookup(True, self.run_dir / m["video"], int(e["i"]), float(m["fps_nominal"]), e, None)


@dataclass(frozen=True)
class DecodeRequest:
    video: Path
    index: int
    fps: float
    max_w: int
    max_h: int
    expect_sig: str | None = None


@dataclass(frozen=True)
class DecodedFrame:
    request: DecodeRequest
    image: np.ndarray | None          # BGR, at most max_w x max_h (aspect kept)
    source_size: tuple[int, int] | None
    scale: float | None
    signature_diff: float | None      # mean abs difference of the 16x9 grey signature vs the frame map entry
    error: str | None = None


def _probe_size(video: Path) -> tuple[int, int]:
    import re
    p = popen_windowless([str(FFMPEG), "-hide_banner", "-nostdin", "-i", str(video)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    _, err = p.communicate(timeout=DECODE_TIMEOUT_S)
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", err.decode("utf-8", "replace"))
    if not m:
        raise OSError("the recording's video stream could not be read")
    return int(m.group(1)), int(m.group(2))


class FrameDecoder:
    """Decodes single recorded frames on the worker thread that calls :meth:`decode` (bounded LRU of downscaled frames)."""

    def __init__(self, cache_size: int = FRAME_CACHE_SIZE) -> None:
        self._cache: "OrderedDict[tuple, DecodedFrame]" = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._sizes: dict[str, tuple[int, int]] = {}
        self.decoded = 0

    def decode(self, req: DecodeRequest, cancel: threading.Event | None = None) -> DecodedFrame:
        key = (str(req.video), req.index, req.max_w, req.max_h)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        try:
            size = self._sizes.get(str(req.video)) or _probe_size(req.video)
            self._sizes[str(req.video)] = size
            w, h = size
            t = max(0.0, req.index / req.fps)
            args = [str(FFMPEG), "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1", "-ss", f"{t:.6f}",
                    "-i", str(req.video), "-frames:v", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
            p = popen_windowless(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                out, err = p.communicate(timeout=DECODE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                p.kill()
                p.communicate(timeout=5)
                return DecodedFrame(req, None, size, None, None, "decoding timed out")
            if cancel is not None and cancel.is_set():
                return DecodedFrame(req, None, size, None, None, "cancelled")
            if len(out) < w * h * 3:
                return DecodedFrame(req, None, size, None, None, "the recording has no frame at this position")
            img = np.frombuffer(out[:w * h * 3], np.uint8).reshape(h, w, 3)
            diff = None
            if req.expect_sig:
                a = np.frombuffer(bytes.fromhex(signature(img)), np.uint8).astype(np.int16)
                b = np.frombuffer(bytes.fromhex(req.expect_sig), np.uint8).astype(np.int16)
                diff = float(np.abs(a - b).mean()) if a.size == b.size == SIG_W * SIG_H else None
            sc = min(req.max_w / w, req.max_h / h, 1.0)
            small = img if sc >= 1.0 else cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                                                    interpolation=cv2.INTER_AREA)
            res = DecodedFrame(req, small, size, sc, diff)
            self.decoded += 1
        except Exception as e:  # noqa: BLE001 - visible failure
            res = DecodedFrame(req, None, None, None, None, f"{type(e).__name__}: {e}")
        if res.image is not None:
            self._cache[key] = res
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return res


class PlaybackStream:
    """Natural-pace playback of a recorded CFR file from an output index (one bounded, windowless FFmpeg child).

    Frames are decoded already downscaled to the display bound into a SMALL bounded queue (back-pressure, no backlog);
    the Tk thread drains with ``after`` polling and paints each frame at its own index time. ``stop`` kills only this
    child and joins the reader (bounded). A generation token lets the view ignore frames of an older playback."""

    QUEUE = 8

    def __init__(self, video: Path, start_index: int, fps: float, max_w: int, max_h: int, generation: int):
        import queue
        self.video, self.start_index, self.fps, self.generation = Path(video), int(start_index), float(fps), generation
        self.max_w, self.max_h = max_w, max_h
        self.q: "queue.Queue[tuple[int, np.ndarray] | None]" = queue.Queue(maxsize=self.QUEUE)
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.ended = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="review-playback", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            w, h = _probe_size(self.video)
            sc = min(self.max_w / w, self.max_h / h, 1.0)
            ow, oh = max(2, int(w * sc) // 2 * 2), max(2, int(h * sc) // 2 * 2)
            args = [str(FFMPEG), "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1",
                    "-ss", f"{self.start_index / self.fps:.6f}", "-i", str(self.video), "-vf", f"scale={ow}:{oh}:flags=area",
                    "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
            self._proc = popen_windowless(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            n = ow * oh * 3
            idx = self.start_index
            while not self._stop.is_set():
                buf = self._proc.stdout.read(n)
                if len(buf) < n:
                    self.ended = True
                    break
                item = (idx, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3))
                while not self._stop.is_set():
                    try:
                        self.q.put(item, timeout=0.1)
                        break
                    except Exception:  # noqa: BLE001 - queue.Full: wait for the view (back-pressure)
                        continue
                idx += 1
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            p = self._proc
            if p is not None and p.poll() is None:
                p.kill()
                try:
                    p.wait(5)
                except subprocess.TimeoutExpired:
                    pass
            try:
                self.q.put_nowait(None)
            except Exception:  # noqa: BLE001
                pass

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        p = self._proc
        if p is not None and p.poll() is None:
            p.kill()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)
