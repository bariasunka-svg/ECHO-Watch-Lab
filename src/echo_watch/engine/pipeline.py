# SPDX-License-Identifier: AGPL-3.0-only
"""Analysis pipeline: the newest eligible frame -> ONE in-flight real inference -> evaluation -> single commit.

Adapted from the frozen PatronSight reference src/patronsight/live_inference.py ``LiveInferenceWorker`` (SHA-256
7638cabf4e98261c62dfee631f27b60b1b2d818e032ea03503ee0a1132a3dd1e; LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only):
* pending input is a single latest slot: an older unconsumed frame is replaced and counted as skipped; inference may
  skip stale frames but never accumulates a backlog;
* only one model call is ever in flight; the loop is strictly sequential on the model side;
* start() refuses while the previous threads are genuinely alive (thread liveness, never the state string, is the
  authority) and the new epoch is committed atomically with the thread starts; terminal-state writes are guarded by thread
  identity so a stale worker thread can never overwrite a newer epoch's state;
* a result is published only if its epoch is still current; a failure is an honest ``failed`` state, never a crash or a
  partially labelled result; stop() is non-blocking and lets a call already in flight finish.
Differences for the item path: the model is the supervised worker client (visual-prompt detections, no tracker); there
is NO fixed-rate schedule - the loop dispatches the newest eligible frame as soon as the worker is free; and the work is
split into two overlapped stages with a one-slot handoff (stage A: take frame + worker round trip; stage B: item
evaluation + commit) so evaluation of result N overlaps inference of frame N+1. That is still one in-flight INFERENCE and
strictly ordered evaluation/commit; the handoff blocks stage A when stage B is busy (no backlog anywhere).

A frame is eligible only if it belongs to the current control basis' source epoch and was captured at or after the basis'
lower boundary (so a frame captured before a Resume can never become a post-Resume observation). Eligibility is
re-checked by the controller at evaluation and again at commit.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
from dataclasses import dataclass, field

from ..config import Limits, Settings
from ..frames import Frame
from ..metrics import RollingRate
from ..worker.client import WorkerClient
from .contracts import ControlBasis
from .latest import LatestSlot

STATE_IDLE = "idle"
STATE_RUNNING = "running"
STATE_STOPPING = "stopping"
STATE_FAILED = "failed"


@dataclass
class Attempt:
    attempt_id: str
    basis: ControlBasis
    frame: Frame
    t_taken: float = 0.0
    t_dispatch: float = 0.0
    t_reply: float = 0.0
    reply: dict = field(default_factory=dict)


class AnalysisPipeline:
    def __init__(self, *, controller, worker: WorkerClient, slot: LatestSlot, settings: Settings, trace_emit=None):
        self.c = controller
        self.worker = worker
        self.slot = slot
        self.s = settings
        self.limits: Limits = settings.limits
        self.trace_emit = trace_emit or (lambda *a, **k: False)
        self._handoff: "queue.Queue[Attempt | None]" = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.state = STATE_IDLE
        self.error: str | None = None
        self._ids = itertools.count(1)
        self.rate = RollingRate()
        self.attempts = 0
        self.replies = 0
        self.ineligible: dict[str, int] = {}
        self.handoff_waits = 0

    # ---- lifecycle ------------------------------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self) -> None:
        with self._lock:
            if self.alive:
                raise RuntimeError("analysis is already running or has not finished exiting yet")
            self._stop.clear()
            self.state, self.error = STATE_RUNNING, None
            self._threads = [threading.Thread(target=self._infer_loop, name="analysis-infer", daemon=True),
                             threading.Thread(target=self._eval_loop, name="analysis-eval", daemon=True)]
            for t in self._threads:
                t.start()

    def stop(self) -> None:
        """Non-blocking. A call already in flight is allowed to finish; no new call is started."""
        self._stop.set()
        self.slot.close()
        try:
            self._handoff.put_nowait(None)
        except queue.Full:
            pass
        with self._lock:
            if self.state == STATE_RUNNING:
                self.state = STATE_STOPPING

    def _settle(self, failed: str | None = None) -> None:
        with self._lock:
            if failed:
                self.state, self.error = STATE_FAILED, failed
            elif self.state != STATE_FAILED and not any(t.is_alive() for t in self._threads if t is not threading.current_thread()):
                self.state = STATE_IDLE

    def _fail(self, why: str) -> None:
        self._settle(failed=why)
        self._stop.set()
        self.slot.close()
        try:
            self._handoff.put_nowait(None)
        except queue.Full:
            pass
        self.c.pipeline_failed(why)

    # ---- stage A: newest eligible frame -> worker -------------------------------------------------------------
    def _infer_loop(self) -> None:
        me = threading.current_thread()
        try:
            while not self._stop.is_set():
                basis = self.c.dispatch_basis()
                if basis is None:
                    self._stop.wait(0.01)
                    continue
                frame = self.slot.wait_take(0.05)
                if frame is None:
                    continue
                # re-read the basis AFTER taking the frame: the boundary tests use the one in force at dispatch
                basis = self.c.dispatch_basis()
                if basis is None:
                    self._note("paused_or_stopped_before_dispatch", frame)
                    continue
                if frame.epoch.epoch_id != basis.source_epoch_id:
                    self._note("foreign_epoch", frame)
                    continue
                if frame.t_mono < basis.lower_bound_mono:
                    self._note("captured_before_boundary", frame)
                    continue
                att = Attempt(f"att_{next(self._ids):06d}", basis, frame, t_taken=time.monotonic())
                self.attempts += 1
                att.t_dispatch = time.monotonic()
                try:
                    att.reply = self.worker.verify(frame.image, frame.ref, self.limits.request_timeout_s,
                                                   self.s.verify_imgsz, self.s.verify_conf, self.s.verify_iou,
                                                   self.s.verify_max_det)
                except Exception as e:  # noqa: BLE001 - an honest failure: no fabricated result
                    if self._stop.is_set():
                        break
                    self._fail(f"live inference failed: {type(e).__name__}: {e}")
                    return
                att.t_reply = time.monotonic()
                self.replies += 1
                while not self._stop.is_set():
                    try:
                        self._handoff.put(att, timeout=0.05)
                        break
                    except queue.Full:
                        self.handoff_waits += 1
        finally:
            if self.state == STATE_RUNNING and me is not None and self._stop.is_set():
                pass
            try:
                self._handoff.put_nowait(None)
            except queue.Full:
                pass

    def _note(self, reason: str, frame: Frame) -> None:
        self.ineligible[reason] = self.ineligible.get(reason, 0) + 1
        self.c.note_ineligible(frame, reason)

    # ---- stage B: evaluation + single commit ----------------------------------------------------------------------
    def _eval_loop(self) -> None:
        try:
            while True:
                try:
                    att = self._handoff.get(timeout=0.1)
                except queue.Empty:
                    if self._stop.is_set() and not self._threads[0].is_alive():
                        break
                    continue
                if att is None:
                    if self._stop.is_set():
                        break
                    continue
                try:
                    if self.c.evaluate_and_commit(att):
                        self.rate.record()
                except Exception as e:  # noqa: BLE001
                    self._fail(f"item evaluation failed: {type(e).__name__}: {e}")
                    return
        finally:
            self._settle()
