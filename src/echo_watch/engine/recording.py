# SPDX-License-Identifier: AGPL-3.0-only
"""Clean-source recording, annotated-result recording and the annotated pump, with real frame/time/result maps.

Adapted from the frozen PatronSight reference (LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only):
* ``VideoRecorder``  <- src/patronsight/live_recorder.py (SHA-256 1ed24d9bd0bb9858491ed08b917a64a79633e2dc9abe1e336a481ced8bb4253c):
  a bounded queue on the recorder's own thread so a slow disk/encoder can never slow capture or analysis; a full queue
  drops the NEWEST frame and counts it; ``offer`` never raises into the caller; ``fps`` is an explicit required
  parameter (never invented); a genuine multi-step finalization (recording -> finalizing -> finalized | failed) in which
  ONLY the worker's own exit, after the encoder has really closed the file, sets the terminal state; ``stop()`` never joins.
* ``AnnotatedPump`` <- src/patronsight/live_recording_pump.py (SHA-256 2d1075f68319e5d0b4a59d5ca284d88a1d3cd263ec135a6118bde0e4f7ce829d):
  samples the latest COMMITTED result, renders it exactly once, caches that frame and offers it at a fixed monotonic
  cadence on its own thread; repeating the cached frame between results reproduces "hold the newest result until
  replaced" over real recorded time, and is counted separately (``held``) from a genuinely new result (``fresh``) - the
  two are never conflated and a held frame is never a new observation; a tick overdue by more than a few whole periods is
  skipped and counted, never bursted; ``stop()`` is a genuine offer barrier (a tick's fast commit step takes the same lock).
Different implementation: the encoder is the pinned FFmpeg executable (libx264, raw BGR on a pipe) instead of OpenCV's
VideoWriter, and every written video frame has a MAP ENTRY (output index, kind capture|fresh|held, result/frame identity,
capture time, a 16x9 signature of the exact pixels offered) so the written file can later be inspected against the map.
Requested time, actual recording start and dropped intervals are reported separately and never inferred from surviving
frames. Gaps in capture are listed with their real durations; nothing is compressed without disclosure.
"""
from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..guards import popen_windowless
from ..metrics import RollingRate
from .files import sha256_file

logger = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_FINALIZING = "finalizing"
STATE_FINALIZED = "finalized"
STATE_FAILED = "failed"
_DRAIN_POLL_SECONDS = 0.2
SIG_W, SIG_H = 16, 9


class RecorderError(RuntimeError):
    pass


def signature(image: np.ndarray) -> str:
    """A 16x9 grey signature of the exact BGR pixels (hex), kept in the map to verify the written file later."""
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA).tobytes().hex()


@dataclass
class VideoItem:
    image: np.ndarray
    entry: dict           # map fields supplied by the offering side (kind, identity, capture time, ...)


def encoder_args(ffmpeg: Path, out: Path, size: tuple[int, int], fps: float, crf: int = 23) -> list[str]:
    return [str(ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{size[0]}x{size[1]}", "-framerate", f"{fps:g}", "-i", "pipe:0", "-c:v", "libx264",
            "-preset", "ultrafast", "-crf", str(crf), "-threads", "2", "-pix_fmt", "yuv420p", "-g", str(max(1, int(fps * 2))),
            str(out)]


class VideoRecorder:
    """Writes offered frames to one video file on a background thread (pinned FFmpeg child, raw frames on stdin)."""

    def __init__(self, name: str, output_path: Path, frame_size: tuple[int, int], *, fps: float, ffmpeg: Path,
                 queue_maxsize: int = 90, clock: Callable[[], float] = time.monotonic,
                 launch: Callable[[list[str]], subprocess.Popen] | None = None):
        w, h = int(frame_size[0]), int(frame_size[1])
        if w <= 0 or h <= 0:
            raise ValueError(f"invalid frame size: {frame_size!r}")
        if not (isinstance(fps, (int, float)) and 1.0 <= float(fps) <= 240.0):
            raise RecorderError(f"refusing to record at an implausible frame rate: {fps!r}")
        self.name = name
        self.output_path = Path(output_path)
        self.frame_size = (w, h)
        self.fps = float(fps)
        self.ffmpeg = Path(ffmpeg)
        self._clock = clock
        self._launch = launch or (lambda a: popen_windowless(a, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                                              stderr=subprocess.PIPE))
        self._q: "queue.Queue[VideoItem]" = queue.Queue(maxsize=max(1, int(queue_maxsize)))
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._state_lock = threading.Lock()
        self._state = STATE_IDLE
        self._rate = RollingRate(clock=clock)
        self.frames_offered = 0
        self.frames_written = 0
        self.frames_dropped = 0
        self.frames_size_mismatched = 0
        self.dropped_entries: list[dict] = []
        self.map: list[dict] = []
        self.error: str | None = None
        self.stderr_tail: deque[str] = deque(maxlen=20)
        self.start_requested_mono: float | None = None
        self.started_mono: float | None = None            # actual start (encoder acknowledged)
        self.terminal_mono: float | None = None
        self.terminal_wall: float | None = None
        self.exit_code: int | None = None
        self.sha256: str | None = None
        self.bytes = 0
        self.map_path: Path | None = None

    # ---- state -------------------------------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def is_recording(self) -> bool:
        return self._state == STATE_RECORDING

    @property
    def is_finalizing(self) -> bool:
        return self._state == STATE_FINALIZING

    def start(self) -> None:
        if self._state != STATE_IDLE:
            raise RuntimeError(f"recorder cannot be started from state {self._state!r} - each recorder records exactly once")
        self.start_requested_mono = self._clock()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        args = encoder_args(self.ffmpeg, self.output_path, self.frame_size, self.fps)
        try:
            self._proc = self._launch(args)
        except OSError as e:
            raise RecorderError(f"could not start the encoder: {e}") from e
        threading.Thread(target=self._drain_stderr, args=(self._proc,), name=f"{self.name}-enc-stderr", daemon=True).start()
        time.sleep(0.05)
        if self._proc.poll() is not None:
            raise RecorderError(f"the encoder exited immediately (code {self._proc.returncode}): "
                                f"{' | '.join(self.stderr_tail)}")
        self._rate.reset()
        self._stop_event.clear()
        self.started_mono = self._clock()
        self._state = STATE_RECORDING
        self._thread = threading.Thread(target=self._run, name=f"{self.name}-recorder", daemon=True)
        self._thread.start()

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for raw in iter(proc.stderr.readline, b""):
                self.stderr_tail.append(raw.decode("utf-8", "replace").rstrip()[:300])
        except Exception:  # noqa: BLE001
            pass

    def offer(self, item: VideoItem) -> bool:
        """Non-blocking; never raises. True only if the frame was queued. Refuses the instant finalization begins."""
        if self._state != STATE_RECORDING:
            return False
        self.frames_offered += 1
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            self.frames_dropped += 1
            if len(self.dropped_entries) < 5000:
                self.dropped_entries.append({k: item.entry.get(k) for k in ("ident", "t_mono", "result", "kind")})
            return False
        except Exception:  # pragma: no cover - defensive
            logger.exception("recorder.offer failed unexpectedly")
            return False

    def stop(self) -> None:
        """Request finalization: no more frames are accepted; the worker drains what is queued, closes the encoder and
        exits. NEVER blocks - poll ``state`` for the terminal result."""
        with self._state_lock:
            if self._state == STATE_RECORDING:
                self._state = STATE_FINALIZING
        self._stop_event.set()

    def recorder_fps(self) -> float:
        return self._rate.rate(now=self._clock())

    # ---- worker ------------------------------------------------------------------------------------------
    def _run(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdin is not None
        w, h = self.frame_size
        try:
            while True:
                try:
                    item = self._q.get(timeout=_DRAIN_POLL_SECONDS)
                except queue.Empty:
                    if self._stop_event.is_set():
                        break
                    if proc.poll() is not None:
                        self.error = f"the encoder exited while recording (code {proc.returncode})"
                        break
                    continue
                img = item.image
                if img.shape[:2] != (h, w) or img.ndim != 3:
                    self.frames_size_mismatched += 1     # a renegotiated size must not corrupt the file: skip + count
                    continue
                try:
                    proc.stdin.write(memoryview(np.ascontiguousarray(img)))
                except Exception as e:  # noqa: BLE001
                    self.error = f"recording write failed: {type(e).__name__}: {e}"
                    break
                e = dict(item.entry)
                e["i"] = self.frames_written
                e["sig"] = signature(img)
                self.map.append(e)
                self.frames_written += 1
                self._rate.record(self._clock())
        finally:
            rc = None
            try:
                try:
                    proc.stdin.close()
                except Exception:  # noqa: BLE001
                    pass
                rc = proc.wait(60)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    pass
                self.error = self.error or "the encoder did not close within 60 s and was terminated"
            except Exception as e:  # noqa: BLE001
                self.error = self.error or f"encoder close failed: {e}"
            self.exit_code = rc
            if rc not in (0, None) and not self.error:
                self.error = f"encoder exit code {rc}: {' | '.join(self.stderr_tail)}"
            self.terminal_mono, self.terminal_wall = self._clock(), time.time()
            ok = self.error is None and rc == 0 and self.output_path.is_file() and self.output_path.stat().st_size > 0
            if ok:
                self.bytes = self.output_path.stat().st_size
                try:
                    self.sha256 = sha256_file(self.output_path)
                except OSError as e:
                    self.error, ok = str(e), False
            try:
                self._write_map()
            except Exception as e:  # noqa: BLE001
                self.error = self.error or f"frame map not saved: {e}"
                ok = False
            with self._state_lock:
                self._state = STATE_FINALIZED if ok else STATE_FAILED

    def gaps(self) -> list[dict]:
        """Real missing intervals between consecutive written capture entries (> 1.5 nominal periods)."""
        out, prev = [], None
        for e in self.map:
            t = e.get("t_mono")
            if t is None:
                continue
            if prev is not None and t - prev["t_mono"] > 1.5 / self.fps:
                out.append({"after_i": prev["i"], "from_t_mono": prev["t_mono"], "to_t_mono": t,
                            "seconds": round(t - prev["t_mono"], 4)})
            prev = e
        return out

    def _write_map(self) -> None:
        mp = self.output_path.with_suffix(".framemap.json")
        doc = {"version": 1, "name": self.name, "video": self.output_path.name, "fps_nominal": self.fps,
               "frame_size": list(self.frame_size), "playback": "output frame i is shown at i/fps_nominal seconds of "
               "playback; 't_mono' is the real capture/render time of the content. Gaps are listed, never stretched.",
               "start_requested_mono": self.start_requested_mono, "started_mono": self.started_mono,
               "terminal_mono": self.terminal_mono, "frames_offered": self.frames_offered,
               "frames_written": self.frames_written, "frames_dropped_queue": self.frames_dropped,
               "frames_size_mismatched": self.frames_size_mismatched, "dropped": self.dropped_entries,
               "gaps": self.gaps(), "encoder_exit_code": self.exit_code, "error": self.error,
               "entries": self.map}
        mp.write_text(json.dumps(doc, separators=(",", ":"), default=str), "utf-8")
        self.map_path = mp

    def summary(self) -> dict:
        return {"name": self.name, "state": self._state, "path": str(self.output_path), "fps_nominal": self.fps,
                "frames_offered": self.frames_offered, "frames_written": self.frames_written,
                "frames_dropped": self.frames_dropped, "size_mismatched": self.frames_size_mismatched,
                "bytes": self.bytes, "sha256": self.sha256, "error": self.error, "exit_code": self.exit_code,
                "start_requested_mono": self.start_requested_mono, "started_mono": self.started_mono,
                "terminal_mono": self.terminal_mono, "map": str(self.map_path) if self.map_path else None,
                "gaps": len(self.gaps()) if self._state in (STATE_FINALIZED, STATE_FAILED) else None}


DEFAULT_IDLE_POLL_SECONDS = 0.005
_OVERDUE_THRESHOLD_PERIODS = 3.0
_OVERDUE_THRESHOLD_MIN_SECONDS = 0.1


class AnnotatedPump:
    """Renders and paces annotated frames from the latest COMMITTED result into ``sink`` at ``fps``.

    ``provider()`` returns ``(packet, banner)``: the newest committed packet (or None) and an optional banner text
    ("PAUSED", "SOURCE ENDED", ...) that means the packet no longer has live authority; a bannered frame is a HELD
    frame, drawn without boxes. A held frame is never counted as a result."""

    def __init__(self, *, provider: Callable[[], tuple], render: Callable, sink: VideoRecorder, fps: float,
                 clock: Callable[[], float] = time.monotonic, idle_poll_seconds: float = DEFAULT_IDLE_POLL_SECONDS):
        if not (isinstance(fps, (int, float)) and np.isfinite(fps) and fps > 0):
            raise ValueError(f"pump fps must be a positive, finite number: {fps!r}")
        self._provider, self._render, self._sink = provider, render, sink
        self._period = 1.0 / float(fps)
        self._overdue_threshold = max(self._period * _OVERDUE_THRESHOLD_PERIODS, _OVERDUE_THRESHOLD_MIN_SECONDS)
        self._clock = clock
        self._idle_poll_seconds = float(idle_poll_seconds)
        self._state_lock = threading.Lock()
        self._offer_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = "idle"
        self._rate = RollingRate(clock=clock)
        self.unique_results_rendered = 0
        self.duplicated_frames_offered = 0
        self.skipped_overdue_ticks = 0
        self.skipped_intervals: list[dict] = []
        self.render_ms: deque[float] = deque(maxlen=2000)
        self.error: str | None = None

    def start(self) -> None:
        with self._state_lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("the annotated-recording pump is already running")
            self._stop_event.clear()
            self.state = "running"
            thread = threading.Thread(target=self._run, name="annotated-pump", daemon=True)
            thread.start()
            self._thread = thread

    def stop(self) -> None:
        """Genuine offer barrier: after this returns no tick can BEGIN a new offer (one already in flight finishes)."""
        self._stop_event.set()
        with self._offer_lock:
            pass
        with self._state_lock:
            if self.state == "running":
                self.state = "stopping"

    @property
    def alive(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    @property
    def output_fps(self) -> float:
        return self._rate.rate(now=self._clock())

    def _run(self) -> None:
        last_key = None
        cached: np.ndarray | None = None
        cached_banner = None
        output_index = 0
        start_time = self._clock()
        next_deadline = start_time + self._period
        try:
            while not self._stop_event.is_set():
                now = self._clock()
                if now < next_deadline:
                    self._stop_event.wait(min(next_deadline - now, self._idle_poll_seconds))
                    continue
                overdue = (now - next_deadline) > self._overdue_threshold
                packet, banner, key, rendered = None, None, None, None
                need_render = False
                if not overdue:
                    try:
                        packet, banner = self._provider()
                    except Exception:  # noqa: BLE001
                        packet, banner = None, "provider failed"
                    if packet is not None:
                        key = (packet.packet_id, banner)
                        need_render = key != last_key
                        if need_render:
                            t = time.perf_counter()
                            try:
                                rendered = self._render(packet, banner)
                            except Exception as e:  # noqa: BLE001 - recorded; the previous frame stays
                                self.error = f"render failed: {type(e).__name__}: {e}"
                                need_render = False
                            self.render_ms.append((time.perf_counter() - t) * 1000)
                with self._offer_lock:
                    if self._stop_event.is_set():
                        break
                    if overdue:
                        self.skipped_overdue_ticks += 1
                        self.skipped_intervals.append({"t_mono": next_deadline, "late_s": round(now - next_deadline, 4)})
                        next_deadline = now + self._period
                        continue
                    if packet is not None:
                        if need_render and rendered is not None:
                            cached, last_key, cached_banner = rendered, key, banner
                            kind = "fresh" if banner is None else "held"
                            if banner is None:
                                self.unique_results_rendered += 1
                        else:
                            kind = "held"
                            self.duplicated_frames_offered += 1
                        if cached is not None:
                            entry = {"kind": kind, "result": packet.packet_id, "result_seq": packet.seq,
                                     "ident": list(packet.frame_ident), "frame_t_mono": packet.frame_t_mono,
                                     "t_mono": now, "banner": cached_banner}
                            try:
                                offered = self._sink.offer(VideoItem(cached, entry))
                            except Exception:  # pragma: no cover - defensive
                                logger.exception("annotated-recording sink.offer raised")
                                offered = False
                            if offered:
                                output_index += 1
                                self._rate.record(now)
                    next_deadline += self._period
        finally:
            with self._state_lock:
                self.state = "idle"

    def summary(self) -> dict:
        r = sorted(self.render_ms)
        return {"state": self.state, "fresh_results_rendered": self.unique_results_rendered,
                "held_frames_offered": self.duplicated_frames_offered, "skipped_overdue_ticks": self.skipped_overdue_ticks,
                "render_ms_p95": r[int(0.95 * (len(r) - 1))] if r else None, "error": self.error}
