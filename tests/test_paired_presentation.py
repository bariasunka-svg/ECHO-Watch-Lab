# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-03 R3: one paired presentation for image, boxes, tray rows and selected-item fields (GUI-free).

The REAL WorkbenchWindow methods _current_display / _monitor_frame / _presentation / _tray_rows /
_publish_presentation are extracted by AST and driven with the production scheduler, the real image preparer and the
real controller row projection. No Tk/Tcl object is created. These tests establish pairing and ordering after every
callback, not rendered timing.
"""
from __future__ import annotations

import ast
import types
from pathlib import Path

import numpy as np

import echo_watch.app.tk_app as tk_mod
from echo_watch.app.controller import WorkbenchController
from echo_watch.app.display import UiMetrics, prepare
from echo_watch.app.display_schedule import DisplayScheduler
from echo_watch.app.presentation import packet_identity
from support import work_dir

IMG = np.zeros((36, 64, 3), np.uint8)
METHODS = ("_current_display", "_monitor_frame", "_presentation", "_tray_rows", "_publish_presentation")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


def obs(item, state, box=(1.0, 2.0, 10.0, 12.0)):
    return types.SimpleNamespace(item_id=item, state=state, reason=None, name=item.upper(), last_supported=None,
                                 box=box if state in ("visible", "checking") else None)


def packet(seq, run, basis, t, states):
    return types.SimpleNamespace(packet_id=f"res_{seq:06d}", seq=seq, run_id=run, image=IMG, frame_t_mono=t, size=(64, 36),
                                 basis=types.SimpleNamespace(basis_id=basis),
                                 items=[obs(i, s) for i, s in states.items()])


class Worker:
    def __init__(self):
        self.pending, self.ready = None, []

    def submit(self, tok, req):
        self.pending = (tok, req)

    def finish(self):
        if self.pending is not None:
            tok, req = self.pending
            self.ready.append((tok, prepare(req)))
            self.pending = None

    def drain(self):
        r, self.ready = self.ready, []
        return r


class Harness:
    def __init__(self):
        src = Path(tk_mod.__file__).read_text("utf-8")
        cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "WorkbenchWindow")
        fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
        self.clock = Clock()
        ns = {k: getattr(tk_mod, k) for k in ("PHASE_IDLE", "packet_gen", "packet_request", "setup_gen", "setup_request",
                                               "T", "project", "LABEL", "STALE_S")}
        ns["time"] = self.clock                          # the view's monotonic clock, deterministic here
        exec(compile(ast.Module(body=fns, type_ignores=[]), "tk_app.py", "exec"), ns)
        self.ns = ns
        self.w = Worker()
        self.latest = None
        self.published = []
        ctl = types.SimpleNamespace(phase="active", setup_frame=None, gen={"source": 1}, tray=["a", "b"],
                                    tray_profiles={"a": {"name": "A"}, "b": {"name": "B"}},
                                    engine=types.SimpleNamespace(latest_result=lambda: self.latest))
        ctl.tray_rows = lambda snap: WorkbenchController.tray_rows(ctl, snap)
        v = types.SimpleNamespace(prep=self.w, schedule=DisplayScheduler(self.w), _painted=None, c=ctl,
                                  _canvas_size=lambda: (64, 36), _snapshot=None, _overlay=lambda: None,
                                  canvas=types.SimpleNamespace(delete=lambda *a: None, create_text=lambda *a, **k: None))
        for name in METHODS:
            setattr(v, name, (lambda n: (lambda *a: ns[n](v, *a)))(name))
        v._paint = lambda res: setattr(v, "_painted", res)
        v._render_tray = lambda rows: self.published.append(("tray", rows))
        v._render_inspector = lambda rows: self.published.append(("inspector", rows))
        self.v = v

    def snap(self, run, basis, items=()):
        self.v._snapshot = types.SimpleNamespace(run_id=run, basis_id=basis, items=list(items))

    def tick(self, finish=True):
        if finish:
            self.w.finish()
        self.ns["_monitor_frame"](self.v)
        rows, auth, _key = self.v._presentation()
        return rows, auth

    def check_pairing(self):
        """At every completed callback: live rows belong to the painted packet; otherwise no live authority."""
        rows, auth, _ = self.v._presentation()
        pk = self.v._painted.packet if self.v._painted is not None else None
        for r in rows:
            if r.get("live") or r.get("packet"):
                assert pk is not None and r["packet"] == packet_identity(pk)
                assert r["state"] == next(o.state for o in pk.items if o.item_id == r["item_id"])
            else:
                assert not r.get("live") and r.get("box") is None
        if self.published:
            assert self.published[-1][1] == rows or self.published[-2][1] == rows
        return rows, auth


def test_old_painted_basis_and_new_visible_snapshot_never_mix():
    h = Harness()
    h.snap("run1", "b1")
    h.latest = packet(10, "run1", "b1", h.clock.t, {"a": "not_seen", "b": "visible"})
    h.tick(finish=False)
    h.tick()                                                   # painted: run1/b1 packet 10
    rows, auth = h.check_pairing()
    assert auth == "painted" and [r["state"] for r in rows] == ["not_seen", "visible"]
    # Resume: new basis; the session already has a newer Visible packet, its preparation is delayed
    newer = packet(11, "run1", "b2", h.clock.t + 0.05, {"a": "visible", "b": "visible"})
    h.snap("run1", "b2", [types.SimpleNamespace(item_id="a", state="visible", reason=None, last_supported=None)])
    h.latest = newer
    h.clock.t += 0.05
    rows, auth = h.tick(finish=False)                          # preparation submitted, not finished
    h.check_pairing()
    assert auth == "awaiting" and all(not r["live"] and r["state"] == "awaiting" for r in rows)
    assert all(r["label"].startswith("Cannot assess - awaiting") for r in rows)
    rows, auth = h.tick()                                      # the current image is painted: rows follow it
    h.check_pairing()
    assert auth == "painted" and [r["state"] for r in rows] == ["visible", "visible"]


def test_first_start_without_an_eligible_paint_is_awaiting():
    h = Harness()
    h.snap("run1", "b1", [types.SimpleNamespace(item_id="a", state="visible", reason=None, last_supported=None)])
    rows, auth = h.tick(finish=False)
    h.check_pairing()
    assert auth == "awaiting" and not any(r["live"] for r in rows)


def test_new_run_and_source_change_reject_old_images_for_rows():
    h = Harness()
    h.snap("run1", "b1")
    h.latest = packet(500, "run1", "b1", h.clock.t, {"a": "visible", "b": "visible"})
    h.tick(finish=False)
    h.tick()
    h.snap("run2", "c1")                                       # Stop + Start: a new run, numbers restart at 1
    h.latest = packet(1, "run2", "c1", h.clock.t, {"a": "not_seen", "b": "visible"})
    rows, auth = h.tick(finish=False)
    h.check_pairing()
    assert auth == "awaiting"
    rows, auth = h.tick()
    h.check_pairing()
    assert auth == "painted" and rows[0]["packet"] == ("run2", "res_000001") and rows[0]["state"] == "not_seen"


def test_transitions_faster_than_the_chrome_cadence_publish_in_the_same_callback():
    h = Harness()
    h.snap("run1", "b1")
    seq = ["visible", "not_seen", "visible", "not_seen", "visible"]   # changes every 66 ms (15 Hz), < 200 ms chrome
    h.latest = packet(1, "run1", "b1", h.clock.t, {"a": seq[0], "b": "visible"})
    h.tick(finish=False)
    for i, st in enumerate(seq[1:], start=2):
        h.tick()                                               # paints packet i-1, publishes it in this callback
        rows, auth = h.check_pairing()
        painted = h.v._painted.packet
        assert auth == "painted" and rows[0]["state"] == painted.items[0].state
        assert h.published[-1][0] == "inspector" and h.published[-1][1][0]["state"] == painted.items[0].state
        h.clock.t += 0.066
        h.latest = packet(i, "run1", "b1", h.clock.t, {"a": st, "b": "visible"})


def test_stale_expiry_without_arrivals_drops_live_authority_everywhere():
    h = Harness()
    h.snap("run1", "b1")
    h.latest = packet(1, "run1", "b1", h.clock.t, {"a": "visible", "b": "visible"})
    h.tick(finish=False)
    h.tick()
    rows, _ = h.check_pairing()
    assert all(r["live"] for r in rows)
    n = len(h.published)
    h.clock.t += 1.5                                           # no new frame arrives
    rows, auth = h.tick()
    h.check_pairing()
    assert auth == "painted" and not any(r["live"] or r["box"] for r in rows) and len(h.published) > n


def test_pause_and_stop_show_lifecycle_without_live_authority():
    h = Harness()
    h.snap("run1", "b1")
    h.latest = packet(1, "run1", "b1", h.clock.t, {"a": "visible", "b": "visible"})
    h.tick(finish=False)
    h.tick()
    h.v.c.phase = "paused"
    h.snap("run1", "b2", [types.SimpleNamespace(item_id=i, state="paused", reason="paused by the operator",
                                                last_supported=None) for i in ("a", "b")])
    rows, auth = h.tick()
    h.check_pairing()
    assert auth == "lifecycle" and {r["state"] for r in rows} == {"paused"} and not any(r["live"] for r in rows)
    h.v.c.phase = "stopping"                                   # the session board marks the items Stopped
    h.snap("run1", "b3", [types.SimpleNamespace(item_id=i, state="stopped", reason="the session stopped",
                                                last_supported=None) for i in ("a", "b")])
    rows, auth = h.tick()
    assert auth == "lifecycle" and {r["label"] for r in rows} == {"Stopped"} and not any(r["live"] for r in rows)
    h.snap("run1", "b3")                                       # no board rows yet: the phase itself is shown
    rows, auth = h.tick()
    assert auth == "lifecycle" and {r["label"] for r in rows} == {"Finalizing"}


def test_original_starvation_schedule_still_paints_every_completion_with_paired_rows():
    h = Harness()
    h.snap("run1", "b1")
    paints = 0
    for i in range(1, 101):
        before = h.v._painted
        h.latest = packet(i, "run1", "b1", h.clock.t, {"a": "visible" if i % 2 else "not_seen", "b": "visible"})
        h.tick()
        paints += h.v._painted is not before
        h.check_pairing()
        h.clock.t += 0.066
    assert paints == 99


def test_display_metrics_count_full_run_and_packet_identity():
    m = UiMetrics(work_dir("ui-metrics-identity"))
    a = types.SimpleNamespace(run_id="run1", packet_id="res_000001")
    b = types.SimpleNamespace(run_id="run2", packet_id="res_000001")
    assert m.painted_packet(packet_identity(a), 0.0, 1.0, 1.0)
    assert not m.painted_packet(packet_identity(a), 0.0, 1.0, 1.0)      # repaint/resize of the same packet
    assert m.painted_packet(packet_identity(b), 0.0, 1.0, 1.0)          # equal result number, different run
    assert m.total_painted == 2
    m.close()
