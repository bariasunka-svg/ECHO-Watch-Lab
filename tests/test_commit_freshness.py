# SPDX-License-Identifier: AGPL-3.0-only
"""Defect A (EW-LAB-01 section 4A): freshness and speculative-state isolation at the authoritative commit.

Deterministic counterexamples, no model/decoder/recorder: the REAL ``ManagedItemSession.evaluate_and_commit`` is called
with a fake evaluator whose evaluation advances an injected clock. Required behaviour:
* a frame fresh at evaluation entry but older than ``max_frame_age_s`` (2.0 s) at commit is discarded before ANY
  board / history / latest / last-supported update;
* the discarded evaluation's speculative policy state (supports, confirmations, ambiguity) never reaches a later result;
* Stop / Pause / Resume / reference cutovers reject old work; duplicate or out-of-order results never count twice;
* established ambiguity is never erased to recover from a discarded evaluation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

from echo_watch.config import Settings
from echo_watch.engine.board import ItemBoard
from echo_watch.engine.contracts import PHASE_ACTIVE, ControlBasis, ItemRef
from echo_watch.engine.evaluator import Evaluation
from echo_watch.engine.pipeline import Attempt
from echo_watch.engine.session import ManagedItemSession
from echo_watch.frames import Epoch, Frame
from echo_watch.item_policy import Decision
from support import work_dir


class Clock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@dataclass
class FakeEvaluator:
    """Stands in for ItemEvaluator: ``supports`` is speculative mutable state; ``ambiguity`` is established state."""
    clock: Clock
    advance_s: float = 0.0
    supports: list = field(default_factory=list)
    ambiguity: dict | None = None
    version: int = 0
    last_t: float | None = None

    def fork(self):
        return FakeEvaluator(self.clock, self.advance_s, list(self.supports), None if self.ambiguity is None else dict(self.ambiguity),
                             self.version, self.last_t)

    def adopt(self, other) -> None:
        self.supports, self.ambiguity, self.last_t = other.supports, other.ambiguity, other.last_t
        self.version += 1

    def evaluate(self, frame, dets):
        self.clock.t += self.advance_s              # evaluation takes time on the injected clock
        self.supports.append(frame.t_mono)          # speculative support
        self.last_t = frame.t_mono
        st = "visible" if len(self.supports) >= 3 else "checking"
        d = Decision("it1", st, [10.0, 10.0, 20.0, 20.0], "tk1", 0.9, 0.9, 0.9, None, supports=list(self.supports))
        return Evaluation({"it1": d}, [], {"it1": (True, None)}, {}, {}, 100.0, False, [], {})


class FakeWorker:
    session = 1
    receipt = {"device": "cuda:0"}

    def alive(self):
        return True


def make_session(clock: Clock, ev: FakeEvaluator):
    d = work_dir("commit_freshness")
    s = Settings(data_dir=d)
    sess = ManagedItemSession(settings=s, store=None, worker=FakeWorker(), clock=clock)
    ep = Epoch("ep1", "src", 1, 0.0, 0.0)
    ref = ItemRef("it1", "Phone", "Other", 1, "it1@v1", (10, 10, 20, 20), (64, 48), "", "", "x")
    sess.run_id = "run1"
    sess._slot_epoch = ep
    sess._refs = [ref]
    sess._board = ItemBoard("run1", [ref])
    sess.phase = PHASE_ACTIVE
    sess._evaluator = ev
    sess._basis = ControlBasis("b1", "run1", 1, 1, 0.0, "ep1", 0, "dg", 1, "p", True)
    return sess, ep


def attempt(sess, ep, seq: int, t_mono: float) -> Attempt:
    f = Frame(ep, seq, t_mono, time.time(), np.zeros((48, 64, 3), np.uint8))
    return Attempt(f"att{seq}", sess._basis, f, t_mono, t_mono, t_mono, {"detections": []})


def test_expired_at_commit_is_discarded_and_leaks_nothing(monkeypatch):
    clock = Clock()
    ev = FakeEvaluator(clock, advance_s=2.9)
    sess, ep = make_session(clock, ev)
    monkeypatch.setattr(time, "monotonic", clock)
    a = attempt(sess, ep, 1, clock.t - 0.1)              # 0.1 s old at evaluation entry
    assert sess.evaluate_and_commit(a) is False           # 3.0 s old at commit -> discarded
    assert sess.committed == 0 and sess._latest.peek() is None
    assert sess._board.items["it1"].obs is None and sess._board.items["it1"].last_supported is None
    assert any(k.startswith("expired") for k in sess.discards)
    assert ev.supports == [] and ev.last_t is None        # the rejected speculative support did not survive
    # the next accepted result cannot inherit the discarded support
    ev.advance_s = 0.05
    a2 = attempt(sess, ep, 2, clock.t - 0.05)
    assert sess.evaluate_and_commit(a2) is True
    p = sess._latest.peek()
    assert p is not None and p.items[0].supports == (a2.frame.t_mono,)


def test_fresh_result_commits_once_and_duplicates_never_count_twice(monkeypatch):
    clock = Clock()
    ev = FakeEvaluator(clock, advance_s=0.05)
    sess, ep = make_session(clock, ev)
    monkeypatch.setattr(time, "monotonic", clock)
    a = attempt(sess, ep, 5, clock.t - 0.1)
    assert sess.evaluate_and_commit(a) is True
    dup = attempt(sess, ep, 5, a.frame.t_mono)
    older = attempt(sess, ep, 4, a.frame.t_mono - 0.05)
    assert sess.evaluate_and_commit(dup) is False
    assert sess.evaluate_and_commit(older) is False
    assert sess.committed == 1 and ev.supports == [a.frame.t_mono]


@pytest.mark.parametrize("cutover", ["stop", "pause", "reference"])
def test_cutover_during_evaluation_rejects_old_work(monkeypatch, cutover):
    clock = Clock()
    ev = FakeEvaluator(clock, advance_s=0.05)
    sess, ep = make_session(clock, ev)
    monkeypatch.setattr(time, "monotonic", clock)
    orig = FakeEvaluator.evaluate

    def evaluate_then_cutover(self, frame, dets):
        out = orig(self, frame, dets)
        if cutover == "stop":
            sess._basis = ControlBasis("b2", "run1", 2, 1, 0.0, "ep1", 0, "dg", 1, "p", False)
        elif cutover == "pause":
            sess._basis = ControlBasis("b2", "run1", 2, 1, 0.0, "ep1", 0, "dg", 1, "p", False)
            sess.phase = "paused"
        else:                                            # explicit reference change: new digest, new basis
            sess._basis = ControlBasis("b2", "run1", 2, 1, 0.0, "ep1", 0, "dg2", 1, "p", True)
        return out
    monkeypatch.setattr(FakeEvaluator, "evaluate", evaluate_then_cutover)
    # the fork evaluates (self is not ev) -> triggers the cutover mid-evaluation
    a = attempt(sess, ep, 1, clock.t - 0.1)
    assert sess.evaluate_and_commit(a) is False
    assert sess.committed == 0 and ev.supports == []


def test_resume_boundary_rejects_frames_captured_before_resume(monkeypatch):
    clock = Clock()
    ev = FakeEvaluator(clock, advance_s=0.05)
    sess, ep = make_session(clock, ev)
    monkeypatch.setattr(time, "monotonic", clock)
    sess._basis = ControlBasis("b3", "run1", 3, 2, clock.t, "ep1", 0, "dg", 1, "p", True)    # Resume at clock.t
    a = attempt(sess, ep, 1, clock.t - 0.2)            # captured before Resume
    a = Attempt(a.attempt_id, sess._basis, a.frame, a.t_taken, a.t_dispatch, a.t_reply, a.reply)
    assert sess.evaluate_and_commit(a) is False
    assert sess.committed == 0 and ev.supports == []


def test_established_ambiguity_survives_a_discarded_evaluation(monkeypatch):
    clock = Clock()
    ev = FakeEvaluator(clock, advance_s=2.9, ambiguity={"track_id": "tk9"})
    sess, ep = make_session(clock, ev)
    monkeypatch.setattr(time, "monotonic", clock)
    orig = FakeEvaluator.evaluate

    def erase(self, f, d):
        self.ambiguity = None                           # the speculative evaluation would clear it
        return orig(self, f, d)
    monkeypatch.setattr(FakeEvaluator, "evaluate", erase)
    assert sess.evaluate_and_commit(attempt(sess, ep, 1, clock.t - 0.1)) is False
    assert ev.ambiguity == {"track_id": "tk9"}
