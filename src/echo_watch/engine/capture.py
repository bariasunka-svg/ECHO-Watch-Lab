# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded capture: one decoded source on its own thread, a latest-frame slot, and non-blocking sinks.

Adapted from the frozen PatronSight reference src/patronsight/live_source.py ``LiveCameraSource`` (SHA-256
cdfed33cc46d32e6e0c7ab51dd9aa8fb5c47246d75b932a7f27cd181d5d13268; LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only). Kept:
* capture never blocks on a slow consumer: it publishes to a single-item latest slot, a clean-recorder sink and an
  inference slot, each non-blocking, and a raising sink can never take down capture;
* the three-state lifecycle idle -> running -> stopping with a bounded ``stop`` that does NOT claim "stopped" while the
  worker thread is still alive; ``start`` is refused while stopping; only the worker's own ``finally`` settles to idle;
* one rolling capture rate shared with the other rates (metrics.RollingRate); ``elapsed_seconds``; a fresh start never
  inherits an earlier run's sink attachments.
Differences: the frame source is the controlled FFmpeg file decoder (no OpenCV VideoCapture, no camera/RTSP); ``start``
returns immediately (the decoder opens on the capture thread, bounded by its own 15 s deadline); a normal end of a
non-repeating file is reported as ``ended`` (never an error or a reconnect); frames the decoder's ``fps`` resampler
repeated (``media['dup_of_prev']``) are NOT published to the inference slot - a repeated picture cannot be an independent
observation - but are still counted and recorded in the ledger.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..frames import Epoch, Frame
from ..metrics import RollingRate
from ..sources.base import FrameSource, SourceEnded, SourceError
from .latest import LatestSlot

logger = logging.getLogger(__name__)

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"


class CaptureSession:
    def __init__(self, source: FrameSource, epoch: Epoch, *, clock: Callable[[], float] = time.monotonic,
                 read_timeout_s: float = 0.5):
        self.source = source
        self.epoch = epoch
        self._clock = clock
        self._read_timeout = read_timeout_s
        self.display_slot: LatestSlot[Frame] = LatestSlot()      # newest captured frame (preview / first-frame checks)
        self.inference_slot: LatestSlot[Frame] = LatestSlot()    # newest frame offered to analysis (latest-only)
        self._rate = RollingRate(clock=clock)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = STATE_IDLE
        self._clean_sink: Callable[[Frame], bool] | None = None
        self._ledger: Callable[[Frame, bool], None] | None = None
        self.error: str | None = None
        self.ended = False
        self.end_reason: str | None = None
        self.frames_captured = 0
        self.frames_duplicate_source = 0
        self.first_frame_mono: float | None = None
        self.last_frame_mono: float | None = None
        self.start_mono: float | None = None
        self.connecting = False

    # ---- lifecycle ------------------------------------------------------------------------------------------
    def attach_clean_sink(self, sink: Callable[[Frame], bool]) -> None:
        with self._lock:
            self._clean_sink = sink

    def attach_ledger(self, fn: Callable[[Frame, bool], None]) -> None:
        with self._lock:
            self._ledger = fn

    @property
    def state(self) -> str:
        t = self._thread
        if self._state == STATE_STOPPING and (t is None or not t.is_alive()):
            self._state = STATE_IDLE
        return self._state

    @property
    def running(self) -> bool:
        return self.state == STATE_RUNNING

    @property
    def stopping(self) -> bool:
        return self.state == STATE_STOPPING

    def start(self) -> None:
        if self.state == STATE_RUNNING:
            raise RuntimeError("capture is already running")
        if self.state == STATE_STOPPING:
            raise RuntimeError("cannot start: the previous capture worker has not finished stopping yet")
        self.error, self.ended, self.end_reason = None, False, None
        self.frames_captured = self.frames_duplicate_source = 0
        self._stop.clear()
        self._rate.reset()
        self.start_mono = self._clock()
        self.connecting = True
        self._state = STATE_RUNNING
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 3.0) -> str:
        """Signal stop and join with a bounded timeout. A timed-out join leaves STATE_STOPPING (never a false 'stopped')."""
        self._stop.set()
        try:
            self.source.stop()                   # unblocks a decoder that is still opening or reading
        except Exception:  # noqa: BLE001
            pass
        t = self._thread
        if t is None:
            self._state = STATE_IDLE
            return self._state
        t.join(timeout=timeout)
        if t.is_alive():
            with self._lock:
                self._state = STATE_STOPPING
            return self._state
        self._state = STATE_IDLE
        return self._state

    # ---- consumption --------------------------------------------------------------------------------------
    @property
    def capture_fps(self) -> float:
        return self._rate.rate(now=self._clock())

    def elapsed_seconds(self) -> float:
        return 0.0 if self.start_mono is None else max(0.0, self._clock() - self.start_mono)

    def peek(self) -> Frame | None:
        return self.display_slot.peek()

    def stats(self) -> dict:
        return {"state": self.state, "frames_captured": self.frames_captured, "capture_fps": round(self.capture_fps, 2),
                "dropped_before_analysis": self.inference_slot.dropped, "duplicate_source_frames": self.frames_duplicate_source,
                "ended": self.ended, "error": self.error, "decoder": self.source.diagnostics()}

    # ---- worker thread ------------------------------------------------------------------------------------
    def _run(self) -> None:
        try:
            try:
                self.source.start(self.epoch)
            except Exception as e:  # noqa: BLE001
                self.error = f"the source could not start: {e}"
                return
            finally:
                self.connecting = False
            while not self._stop.is_set():
                try:
                    f = self.source.read(self._read_timeout)
                except SourceEnded as e:
                    self.ended, self.end_reason = True, str(e)
                    break
                except SourceError as e:
                    self.error = str(e)
                    break
                if f is None:
                    continue
                self.frames_captured += 1
                if self.first_frame_mono is None:
                    self.first_frame_mono = f.t_mono
                self.last_frame_mono = f.t_mono
                self._rate.record(f.t_mono)
                self.display_slot.publish(f)
                dup = bool(f.media and f.media.get("dup_of_prev"))
                if dup:
                    self.frames_duplicate_source += 1
                else:
                    self.inference_slot.publish(f)
                with self._lock:
                    sink, ledger = self._clean_sink, self._ledger
                if sink is not None:
                    try:
                        sink(f)
                    except Exception:  # pragma: no cover - defensive: a recorder failure never stops capture
                        logger.exception("clean sink raised; recording may be broken")
                if ledger is not None:
                    try:
                        ledger(f, dup)
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("capture ledger raised")
        finally:
            try:
                self.source.stop()
            except Exception:  # noqa: BLE001
                pass
            self.inference_slot.close()
            with self._lock:
                self._state = STATE_IDLE
