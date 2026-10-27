# SPDX-License-Identifier: AGPL-3.0-only
"""Analysis pipeline lifecycle regressions (donor-derived rules: latest-only input, one call in flight, thread-liveness
authority for start, honest failure, non-blocking stop). A scripted fake worker; no model."""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from echo_watch.config import Settings
from echo_watch.engine.contracts import ControlBasis
from echo_watch.engine.latest import LatestSlot
from echo_watch.engine.pipeline import STATE_FAILED, AnalysisPipeline
from echo_watch.frames import Epoch, Frame
from support import work_dir


class FakeController:
    def __init__(self, epoch: Epoch, lower: float = 0.0):
        self.basis = ControlBasis("b1", "r", 1, 1, lower, epoch.epoch_id, 0, "d", 1, "p", True)
        self.committed: list = []
        self.ineligible: list = []
        self.failed: list = []
        self.lock = threading.Lock()

    def dispatch_basis(self):
        return self.basis

    def note_ineligible(self, frame, reason):
        self.ineligible.append((frame.seq, reason))

    def evaluate_and_commit(self, att):
        with self.lock:
            self.committed.append(att.frame.seq)
        return True

    def pipeline_failed(self, why):
        self.failed.append(why)


class FakeWorker:
    def __init__(self, delay=0.0, fail_at=None):
        self.delay, self.fail_at, self.calls, self.in_flight, self.max_in_flight = delay, fail_at, 0, 0, 0
        self.lock = threading.Lock()

    def verify(self, image, frame_ref, timeout, imgsz, conf, iou, max_det):
        with self.lock:
            self.calls += 1
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            n = self.calls
        try:
            if self.fail_at is not None and n >= self.fail_at:
                raise RuntimeError("worker exploded")
            time.sleep(self.delay)
            return {"detections": [], "infer_s": self.delay}
        finally:
            with self.lock:
                self.in_flight -= 1


def frame(ep, seq, t):
    return Frame(ep, seq, t, 0.0, np.zeros((4, 4, 3), np.uint8))


def make(delay=0.0, fail_at=None, lower=0.0):
    ep = Epoch.begin("s", 1)
    slot = LatestSlot()
    ctl = FakeController(ep, lower)
    w = FakeWorker(delay, fail_at)
    s = Settings(data_dir=work_dir("pipe") / "d")
    return ep, slot, ctl, w, AnalysisPipeline(controller=ctl, worker=w, slot=slot, settings=s)


def test_only_the_newest_frame_is_analysed_and_only_one_call_is_ever_in_flight():
    ep, slot, ctl, w, p = make(delay=0.15)
    p.start()
    t = time.monotonic()
    for k in range(1, 31):                   # a producer far faster than the worker
        slot.publish(frame(ep, k, t + k * 0.001))
        time.sleep(0.01)
    time.sleep(0.5)
    p.stop()
    time.sleep(0.4)
    assert w.max_in_flight == 1                                    # never two model calls at once
    assert 1 <= len(ctl.committed) < 10                            # most frames were replaced, not queued
    assert slot.dropped >= 15 and ctl.committed == sorted(ctl.committed)
    assert ctl.committed[-1] >= 25                                 # the work done was on the NEWEST available frames


def test_start_is_refused_while_threads_are_alive_and_allowed_after_they_exit():
    ep, slot, ctl, w, p = make(delay=0.3)
    p.start()
    with pytest.raises(RuntimeError):
        p.start()
    slot.publish(frame(ep, 1, time.monotonic()))
    time.sleep(0.05)
    p.stop()                                                        # the 0.3 s call is still in flight
    assert p.alive
    with pytest.raises(RuntimeError):
        p.start()
    end = time.monotonic() + 3
    while p.alive and time.monotonic() < end:
        time.sleep(0.02)
    assert not p.alive
    ep2 = Epoch.begin("s", 2)
    p2 = AnalysisPipeline(controller=FakeController(ep2), worker=FakeWorker(), slot=LatestSlot(), settings=p.s)
    p2.start()
    p2.stop()


def test_stop_never_blocks_on_an_in_flight_call_and_that_call_still_completes():
    ep, slot, ctl, w, p = make(delay=0.4)
    p.start()
    slot.publish(frame(ep, 1, time.monotonic()))
    time.sleep(0.1)
    t0 = time.monotonic()
    p.stop()
    assert time.monotonic() - t0 < 0.1
    time.sleep(0.8)
    assert not p.alive and w.in_flight == 0 and w.calls == 1


def test_a_worker_failure_is_an_honest_failed_state_and_never_a_partial_result():
    ep, slot, ctl, w, p = make(delay=0.0, fail_at=3)
    p.start()
    t = time.monotonic()
    for k in range(1, 40):
        slot.publish(frame(ep, k, t + k * 0.01))
        time.sleep(0.02)
        if ctl.failed:
            break
    time.sleep(0.2)
    assert p.state == STATE_FAILED and len(ctl.failed) == 1 and "worker exploded" in ctl.failed[0]
    assert len(ctl.committed) == 2                                  # the two good replies only; the failing call produced nothing


def test_frames_before_the_boundary_or_from_another_epoch_are_never_dispatched():
    ep, slot, ctl, w, p = make(lower=100.0)
    p.start()
    slot.publish(frame(ep, 1, 99.0))                                # captured before the lower boundary
    time.sleep(0.15)
    slot.publish(frame(Epoch.begin("s", 9), 2, 101.0))              # another source epoch
    time.sleep(0.15)
    slot.publish(frame(ep, 3, 101.0))                               # eligible
    time.sleep(0.3)
    p.stop()
    assert ctl.ineligible == [(1, "captured_before_boundary"), (2, "foreign_epoch")]
    assert ctl.committed == [3] and w.calls == 1
