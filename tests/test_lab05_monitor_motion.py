# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-05 matrix 6: a moving supported observation keeps its item identity and its box is drawn on its OWN image;
wrong/ambiguous evidence and old-basis/stale packets never keep a current box or a green state (GUI-free).

The REAL ManagedItemSession.evaluate_and_commit (single commit authority) -> ItemBoard -> ResultPacket -> app.presentation
-> the production tk_app renderers (AST-extracted, recording sinks) are exercised with a fake evaluator that reports
synthetic decisions. This establishes integration invariants of the existing policy/commit path; it does NOT establish
recognition of real moving objects (that remains the existing item_policy tests plus the owner's footage walkthrough).
"""
from __future__ import annotations

import time
import types
from dataclasses import dataclass, field

import numpy as np

from echo_watch.app.canvas_geometry import fit, source_box_to_canvas
from echo_watch.app.presentation import project
from echo_watch.config import Settings
from echo_watch.engine.board import ItemBoard
from echo_watch.engine.contracts import LABEL, PHASE_ACTIVE, PHASE_PAUSED, ControlBasis, ItemRef
from echo_watch.engine.evaluator import Evaluation
from echo_watch.engine.pipeline import Attempt
from echo_watch.engine.session import ManagedItemSession
from echo_watch.frames import Epoch, Frame
from echo_watch.item_policy import Decision
from lab05_view_harness import build
from support import work_dir

GREEN, AMBER = "#4ade80", "#fbbf24"


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


@dataclass
class ScriptedEvaluator:
    """Stands in for ItemEvaluator: returns the scripted decision for each frame sequence number."""
    script: dict
    version: int = 0
    seen: list = field(default_factory=list)

    def fork(self):
        return ScriptedEvaluator(self.script, self.version, list(self.seen))

    def adopt(self, other):
        self.seen = other.seen
        self.version += 1

    def evaluate(self, frame, dets):
        self.seen.append(frame.seq)
        d = self.script[frame.seq]
        return Evaluation({"it1": d}, [], {"it1": (True, None)}, {}, {}, 100.0, False, [], {})


class FakeWorker:
    session = 1
    receipt = {"device": "cuda:0"}

    def alive(self):
        return True


def make_session(clock, script):
    s = Settings(data_dir=work_dir("lab05-motion"))
    sess = ManagedItemSession(settings=s, store=None, worker=FakeWorker(), clock=clock)
    ep = Epoch("ep1", "src", 1, 0.0, 0.0)
    ref = ItemRef("it1", "Phone A01", "Phone", 1, "it1@v1", (100, 100, 60, 120), (640, 360), "", "", "x")
    sess.run_id, sess._slot_epoch, sess._refs = "run1", ep, [ref]
    sess._board = ItemBoard("run1", [ref])
    sess.phase = PHASE_ACTIVE
    sess._evaluator = ScriptedEvaluator(script)
    sess._basis = ControlBasis("b1", "run1", 1, 1, 0.0, "ep1", 0, "dg", 1, "p", True)
    return sess, ep


def commit(sess, ep, seq, t):
    img = np.full((360, 640, 3), seq % 250, np.uint8)               # each frame's pixels are distinguishable
    f = Frame(ep, seq, t, time.time(), img)
    ok = sess.evaluate_and_commit(Attempt(f"att{seq}", sess._basis, f, t, t, t, {"detections": []}))
    return ok, img


def view_for(pk, snap_basis, clock, phase="active"):
    ctl = types.SimpleNamespace(phase=phase, tray=["it1"], tray_profiles={"it1": {"name": "Phone A01", "ref_version": 1}},
                                library=[], selected="item:it1", stopped=False, view_step="monitor", candidates=[],
                                scan_status="done",
                                engine=types.SimpleNamespace(session=types.SimpleNamespace(live_events=lambda *a: [])))
    from echo_watch.app.controller import WorkbenchController
    ctl.tray_rows = lambda snap: WorkbenchController.tray_rows(ctl, snap)
    vclock = types.SimpleNamespace(t=clock.t, monotonic=lambda: vclock.t, perf_counter=lambda: vclock.t)
    v, _ns, _ = build(ctl, vclock)
    v._snapshot = types.SimpleNamespace(run_id="run1", basis_id=snap_basis, items=[])
    v._painted = types.SimpleNamespace(packet=pk)
    v._fit = fit(640, 360, 640, 360)
    return v, vclock


def boxes(v):
    """Item outlines (state-coloured, unfilled); label chips are filled and excluded."""
    return [(tuple(round(x, 1) for x in a), k.get("outline")) for a, k in v.canvas.of("rect")
            if k.get("outline") in (GREEN, AMBER) and "fill" not in k]


def test_a_moving_supported_item_keeps_its_id_and_its_box_is_drawn_from_each_packets_own_image(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    path = {k: [100.0 + 12 * k, 100.0 + 5 * k, 60.0, 120.0] for k in range(1, 9)}   # 84 px of movement
    script = {k: Decision("it1", "visible" if k >= 3 else "checking", path[k], f"trk{1 + k // 4}", 0.9, 0.9, 0.9, None,
                          supports=[float(k)]) for k in path}
    sess, ep = make_session(clock, script)
    seen_boxes = []
    for k in range(1, 9):
        clock.t += 0.066
        ok, img = commit(sess, ep, k, clock.t - 0.01)
        assert ok
        pk = sess._latest.peek()
        o = pk.item("it1")
        assert o.item_id == "it1" and o.box == tuple(path[k]) and np.array_equal(pk.image, img)   # own pixels
        rows, auth, _ = project([{"item_id": "it1", "name": "Phone A01", "state": "saved", "label": "Saved",
                                  "crop_path": None, "last_supported_wall": None, "live": False, "box": None}],
                                pk, types.SimpleNamespace(run_id="run1", basis_id="b1", items=[]), "active", clock.t, LABEL)
        assert auth == "painted" and rows[0]["box"] == tuple(path[k]) and rows[0]["packet"] == ("run1", pk.packet_id)
        v, _vc = view_for(pk, "b1", clock)
        v._overlay()
        x, y, w, h = path[k]
        want = tuple(round(c, 1) for c in source_box_to_canvas(v._fit, (x, y, x + w, y + h)))
        drawn = boxes(v)
        assert [b for b, _ in drawn] == [want]                      # exactly one box, at THIS packet's position
        assert drawn[0][1] == (GREEN if o.state == "visible" else AMBER)
        seen_boxes.append(want)
    assert len(set(seen_boxes)) == 8                                # the box followed the item, not a fixed zone
    assert {p.track_id for p in [sess._latest.peek().item("it1")]} == {"trk3"}   # the local track id is diagnostic only


def test_wrong_or_ambiguous_evidence_is_uncertain_without_a_box_or_green_state(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    script = {1: Decision("it1", "visible", [100.0, 100.0, 60.0, 120.0], "trk1", 0.9, 0.9, 0.9, None, supports=[1.0]),
              2: Decision("it1", "uncertain", [300.0, 200.0, 60.0, 120.0], None, 0.5, 0.5, 0.5,
                          "a resembling look-alike; identity cannot be assigned",
                          alternatives=[{"why": "look-alike", "app": 0.5}])}
    sess, ep = make_session(clock, script)
    for k in (1, 2):
        clock.t += 0.066
        assert commit(sess, ep, k, clock.t - 0.01)[0]
    pk = sess._latest.peek()
    o = pk.item("it1")
    assert o.state == "uncertain" and o.box is None and o.alternatives      # the alternative is not the item
    v, _ = view_for(pk, "b1", clock)
    rows, auth, _ = v._presentation()
    v._render_tray(rows)
    v._render_inspector(rows)
    assert rows[0]["box"] is None and v.tray.rows["it1"]["tone"] == "pending"
    assert v.tray.rows["it1"]["values"][0] == "Match uncertain" and v.tile_lbls["visible"][0].cget("text") == "0"
    v._overlay()
    assert boxes(v) == []
    assert "No current supported view" in v.insp_cap_cur.cget("text")
    assert sess._board.items["it1"].last_supported["box"] == [100.0, 100.0, 60.0, 120.0]   # historical, not redrawn


def test_old_basis_and_stale_packets_never_keep_a_current_box_or_green_state(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(time, "monotonic", clock)
    script = {k: Decision("it1", "visible", [100.0 + k, 100.0, 60.0, 120.0], "trk1", 0.9, 0.9, 0.9, None,
                          supports=[float(k)]) for k in range(1, 5)}
    sess, ep = make_session(clock, script)
    for k in (1, 2, 3):
        clock.t += 0.066
        assert commit(sess, ep, k, clock.t - 0.01)[0]
    pk = sess._latest.peek()
    # Pause: a new basis; the painted packet belongs to the old one
    sess.phase = PHASE_PAUSED
    v, vc = view_for(pk, "b2", clock, phase="paused")
    rows, auth, _ = v._presentation()
    v._render_tray(rows)
    assert auth == "lifecycle" and rows[0]["box"] is None and v.tray.rows["it1"]["tone"] == "neutral"
    v._overlay()
    assert boxes(v) == [] and any("PAUSED" in (t or "") for t in v.canvas.texts())
    # an attempt captured under the old basis cannot commit after the cutover
    sess._basis = ControlBasis("b2", "run1", 2, 1, clock.t, "ep1", 0, "dg", 1, "p", False)
    old = Attempt("att4", ControlBasis("b1", "run1", 1, 1, 0.0, "ep1", 0, "dg", 1, "p", True),
                  Frame(ep, 4, clock.t, time.time(), np.zeros((360, 640, 3), np.uint8)), clock.t, clock.t, clock.t,
                  {"detections": []})
    assert sess.evaluate_and_commit(old) is False and sess._latest.peek() is pk
    # Resume under a new basis: the old image stays but has no authority (awaiting) until a current one is painted
    v, vc = view_for(pk, "b3", clock)
    rows, auth, _ = v._presentation()
    assert auth == "awaiting" and rows[0]["box"] is None and rows[0]["display_tone"] == "neutral"
    v._overlay()
    assert boxes(v) == []
    # the current basis, but the image grows old: stale, neutral, no box
    v, vc = view_for(pk, "b1", clock)
    vc.t = pk.frame_t_mono + 1.5
    rows, auth, _ = v._presentation()
    v._render_tray(rows)
    assert auth == "painted" and not rows[0]["live"] and rows[0]["box"] is None
    assert v.tray.rows["it1"]["values"][0].startswith("Stale") and v.tray.rows["it1"]["tone"] == "neutral"
    v._overlay()
    assert boxes(v) == [] and any("STALE" in (t or "") for t in v.canvas.texts())
