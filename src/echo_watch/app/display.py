# SPDX-License-Identifier: AGPL-3.0-only
"""Display preparation and UI instrumentation for the workbench (Tk-free; runs off the Tk event thread).

* ``prepare`` turns ONE committed packet's (or setup frame's) own image into a display-sized RGB PPM payload together with the
  packet itself, so the view paints image and boxes of the same packet together (no box ever meets newer pixels).
  It runs on a ``LatestWorker`` (newest request only: paints coalesce to the latest committed result).
* ``UiMetrics`` counts DISTINCT painted packets (repaints and held frames never count), capture-to-paint age, Tk event-
  loop delay, command acknowledgements and the last paint, and writes 1 Hz summaries to a JSONL file on its own thread
  (no file I/O on the Tk thread). These are instruments for the owner's manual run, not verification by themselves.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..metrics import percentile


@dataclass(frozen=True)
class Prepared:
    key: object                       # the request key (generation, order, max size)
    packet: object | None             # the packet whose OWN pixels these are (None: setup image)
    ppm: bytes
    size: tuple[int, int]             # display width, height
    scale: float
    prep_ms: float
    gen: tuple | None = None          # validity generation (app.display_schedule)
    order: int = 0                    # display order inside the generation
    src_size: tuple[int, int] | None = None   # the source image size these pixels came from


def to_ppm(bgr: np.ndarray, max_w: int, max_h: int) -> tuple[bytes, tuple[int, int], float]:
    h, w = bgr.shape[:2]
    sc = max(1e-6, min(max_w / w, max_h / h))
    dw, dh = max(1, int(round(w * sc))), max(1, int(round(h * sc)))
    img = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return b"P6 %d %d 255\n" % (dw, dh) + rgb.tobytes(), (dw, dh), sc


def prepare(req, cancel=None) -> Prepared:
    """``req`` is an app.display_schedule.PrepRequest; runs on the display-prep LatestWorker thread."""
    t0 = time.perf_counter()
    ppm, size, sc = to_ppm(req.image, req.max_w, req.max_h)
    h, w = req.image.shape[:2]
    return Prepared(req.key, req.packet, ppm, size, sc, (time.perf_counter() - t0) * 1000, req.gen, req.order, (w, h))


class UiMetrics:
    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / f"ui-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self._q: "queue.Queue[dict | None]" = queue.Queue(maxsize=600)
        self.dropped = 0
        self.painted: list[float] = []            # distinct packet paints (monotonic) in the current second
        self.paint_age_ms: list[float] = []
        self.loop_delay_ms: list[float] = []
        self.render_ms: list[float] = []
        self.prep_ms: list[float] = []
        self.last_paint_mono: float | None = None
        self.total_painted = 0
        self.last_key = None
        self._last_flush = time.monotonic()
        self._displayed_fps = 0.0
        self._thread = threading.Thread(target=self._run, name="ui-metrics", daemon=True)
        self._thread.start()

    def painted_packet(self, key, capture_mono: float | None, render_ms: float, prep_ms: float) -> bool:
        """Record a paint; only a NEW packet counts as displayed."""
        if key == self.last_key:
            return False
        self.last_key = key
        now = time.monotonic()
        self.painted.append(now)
        self.total_painted += 1
        self.last_paint_mono = now
        if capture_mono is not None:
            self.paint_age_ms.append((now - capture_mono) * 1000)
        self.render_ms.append(render_ms)
        self.prep_ms.append(prep_ms)
        return True

    def loop_delay(self, ms: float) -> None:
        self.loop_delay_ms.append(ms)

    @property
    def displayed_fps(self) -> float:
        return self._displayed_fps

    def tick(self, extra: dict) -> None:
        now = time.monotonic()
        if now - self._last_flush < 1.0:
            return
        span = now - self._last_flush
        self._last_flush = now
        self._displayed_fps = len(self.painted) / span
        rec = {"wall": time.time(), "mono": now, "displayed_packets": len(self.painted),
               "displayed_fps": round(self._displayed_fps, 2),
               "capture_to_paint_ms_p50": round(percentile(self.paint_age_ms, 50), 1) if self.paint_age_ms else None,
               "capture_to_paint_ms_p95": round(percentile(self.paint_age_ms, 95), 1) if self.paint_age_ms else None,
               "event_loop_delay_ms_p95": round(percentile(self.loop_delay_ms, 95), 1) if self.loop_delay_ms else None,
               "event_loop_delay_ms_max": round(max(self.loop_delay_ms), 1) if self.loop_delay_ms else None,
               "paint_render_ms_p95": round(percentile(self.render_ms, 95), 2) if self.render_ms else None,
               "display_prep_ms_p95": round(percentile(self.prep_ms, 95), 2) if self.prep_ms else None,
               "since_last_paint_s": None if self.last_paint_mono is None else round(now - self.last_paint_mono, 3),
               "total_painted": self.total_painted, "writer_dropped": self.dropped, **extra}
        self.painted, self.paint_age_ms, self.loop_delay_ms, self.render_ms, self.prep_ms = [], [], [], [], []
        try:
            self._q.put_nowait(rec)
        except queue.Full:
            self.dropped += 1

    def event(self, kind: str, **kw) -> None:
        try:
            self._q.put_nowait({"wall": time.time(), "mono": time.monotonic(), "event": kind, **kw})
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            while True:
                rec = self._q.get()
                if rec is None:
                    break
                fh.write(json.dumps(rec, default=str) + "\n")
                fh.flush()

    def close(self, timeout: float = 2.0) -> None:
        try:
            self._q.put(None, timeout=1.0)
        except queue.Full:
            pass
        self._thread.join(timeout)
