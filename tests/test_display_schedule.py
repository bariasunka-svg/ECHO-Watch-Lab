# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-02 repair A: latest-completed paired display scheduling (GUI-free; no Tk, no decoding).

The production component app.display_schedule.DisplayScheduler (used by WorkbenchWindow._monitor_frame) is driven
with a deterministic single-slot worker under the review's schedules. These tests establish ORDERING and PROGRESS,
not desktop FPS.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from echo_watch.app.display import prepare
from echo_watch.app.display_schedule import (DisplayScheduler, live_authority, packet_gen, packet_request, setup_gen,
                                             setup_request)

IMG = np.zeros((36, 64, 3), np.uint8)


def packet(seq, run="run1", basis="b1", t=100.0):
    return types.SimpleNamespace(packet_id=f"res_{seq:06d}", seq=seq, run_id=run, image=IMG, frame_t_mono=t,
                                 basis=types.SimpleNamespace(basis_id=basis), size=(64, 36))


class Worker:
    """LatestWorker semantics: one pending request (replaced by a newer submit) and completions drained by the view."""

    def __init__(self):
        self.pending, self.ready = None, []

    def submit(self, tok, req):
        self.pending = (tok, req)

    def finish(self):
        if self.pending is not None:
            tok, req = self.pending
            self.ready.append((tok, prepare(req)))      # the real preparation function (CPU conversion)
            self.pending = None

    def drain(self):
        r, self.ready = self.ready, []
        return r


def run(schedule, ticks, gen=packet_gen("run1", "b1"), finish_every=1, size=lambda i: (64, 36)):
    w = Worker()
    s = DisplayScheduler(w)
    painted = []
    for i in range(ticks):
        if i % finish_every == 0:
            w.finish()
        pk = schedule(i)
        want = None if pk is None else packet_request(pk, *size(i))
        res = s.tick(want, gen, w.drain())
        if res is not None:
            painted.append(res.order)
    return painted, s


def test_review_schedule_completed_old_and_new_arrival_every_tick_paints_progressively():
    painted, s = run(lambda i: packet(i + 1), 100)
    assert len(painted) == 99 and painted == sorted(painted)      # the review's case: previously 0 paints
    assert painted[-1] == 99


def test_preparation_slower_than_capture_still_advances():
    painted, s = run(lambda i: packet(i + 1), 120, finish_every=3)
    assert len(painted) >= 39 and painted == sorted(painted) and painted[-1] >= 120 - 3   # within one preparation period


def test_burst_and_out_of_order_completions_paint_newest_once_and_never_regress():
    w = Worker()
    s = DisplayScheduler(w)
    gen = packet_gen("run1", "b1")
    comps = [(1, prepare(packet_request(packet(5), 64, 36))), (2, prepare(packet_request(packet(9), 64, 36))),
             (3, prepare(packet_request(packet(7), 64, 36)))]
    res = s.tick(None, gen, comps)
    assert res.order == 9 and s.dropped["superseded_by_newer_completion"] == 2
    late = [(4, prepare(packet_request(packet(8), 64, 36)))]
    assert s.tick(None, gen, late) is None and s.dropped["older_than_painted"] == 1


def test_idle_without_new_frames_neither_repaints_nor_resubmits():
    w = Worker()
    s = DisplayScheduler(w)
    gen, pk = packet_gen("run1", "b1"), packet(3)
    s.tick(packet_request(pk, 64, 36), gen, [])
    w.finish()
    assert s.tick(packet_request(pk, 64, 36), gen, w.drain()).order == 3
    for _ in range(20):
        assert s.tick(packet_request(pk, 64, 36), gen, w.drain()) is None
    assert s.submitted == 1 and w.pending is None


@pytest.mark.parametrize("change", ["stop", "pause", "resume", "new_run", "source"])
def test_cutover_rejects_pre_cutover_results_and_accepts_the_new_generation(change):
    w = Worker()
    s = DisplayScheduler(w)
    old = packet_gen("run1", "b1")
    s.tick(packet_request(packet(50), 64, 36), old, [])
    w.finish()
    late = w.drain()                                            # completed just after the cutover
    new = {"stop": packet_gen("run1", "b2"), "pause": packet_gen("run1", "b2"), "resume": packet_gen("run1", "b3"),
           "new_run": packet_gen("run2", "c1"), "source": setup_gen(2, 777)}[change]
    assert s.tick(None, new, late) is None and s.dropped["outside_current_generation"] == 1
    if change == "source":
        frame = types.SimpleNamespace(image=IMG)
        req = setup_request(2, frame, 64, 36)
        req = req.__class__(req.key[:0] + (new, 0, 64, 36), new, 0, None, IMG, 64, 36)
    else:
        req = packet_request(packet(1, run=new[1], basis=new[2]), 64, 36)
    s.tick(req, new, [])
    w.finish()
    assert s.tick(req, new, w.drain()).order == req.order        # lower order, new generation: painted


def test_result_numbers_restart_per_run_and_identity_is_the_generation():
    w = Worker()
    s = DisplayScheduler(w)
    g1, g2 = packet_gen("run1", "b1"), packet_gen("run2", "c1")
    s.tick(packet_request(packet(500), 64, 36), g1, [])
    w.finish()
    assert s.tick(None, g1, w.drain()).order == 500
    run2_first = packet(1, run="run2", basis="c1")             # same packet_id style numbers restart at 1
    s.tick(packet_request(run2_first, 64, 36), g2, [])
    w.finish()
    assert s.tick(None, g2, w.drain()).order == 1


def test_repeated_resize_keeps_progress_and_each_image_has_its_own_scale():
    sizes = [(64, 36), (128, 72), (96, 54), (200, 113)]
    painted, s = run(lambda i: packet(i + 1), 40, size=lambda i: sizes[i % 4])
    assert len(painted) == 39 and painted == sorted(painted)
    res = prepare(packet_request(packet(3), 128, 72))
    assert res.size == (128, 72) and res.scale == 2.0 and res.src_size == (64, 36)


def test_resize_of_the_same_packet_repaints_without_counting_a_new_packet():
    w = Worker()
    s = DisplayScheduler(w)
    gen, pk = packet_gen("run1", "b1"), packet(4)
    s.tick(packet_request(pk, 64, 36), gen, [])
    w.finish()
    first = s.tick(packet_request(pk, 128, 72), gen, w.drain())
    w.finish()
    second = s.tick(packet_request(pk, 128, 72), gen, w.drain())
    assert first.order == second.order == 4 and first.size != second.size
    assert first.packet.packet_id == second.packet.packet_id     # the view's metrics count distinct packet ids


def test_live_authority_requires_current_basis_active_phase_and_freshness():
    pk = packet(1, t=100.0)
    assert live_authority(pk, "run1", "b1", "active", 100.9)
    assert not live_authority(pk, "run1", "b1", "active", 101.2)       # stale: > 1 s
    assert not live_authority(pk, "run1", "b2", "active", 100.1)       # pre-Pause/Resume basis
    assert not live_authority(pk, "run2", "b1", "active", 100.1)       # previous run
    assert not live_authority(pk, "run1", "b1", "paused", 100.1)
    assert not live_authority(None, "run1", "b1", "active", 100.1)


def test_painted_image_carries_its_own_packet():
    w = Worker()
    s = DisplayScheduler(w)
    gen, pk = packet_gen("run1", "b1"), packet(7)
    s.tick(packet_request(pk, 64, 36), gen, [])
    w.finish()
    res = s.tick(packet_request(packet(8), 64, 36), gen, w.drain())
    assert res.packet is pk and res.order == pk.seq                    # pixels and observations of ONE packet


def test_window_monitor_frame_integration_advances_under_the_review_schedule():
    """Caller/receiver integration: the REAL WorkbenchWindow._current_display/_monitor_frame (extracted by AST, no Tk)
    drive the production scheduler with the review's schedule."""
    import ast
    from pathlib import Path
    import echo_watch.app.tk_app as tk_mod
    src = Path(tk_mod.__file__).read_text("utf-8")
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "WorkbenchWindow")
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ("_current_display", "_monitor_frame")]
    ns = {k: getattr(tk_mod, k) for k in ("PHASE_IDLE", "packet_gen", "packet_request", "setup_gen", "setup_request", "T")}
    exec(compile(ast.Module(body=fns, type_ignores=[]), "tk_app.py", "exec"), ns)
    w = Worker()
    state = {"pk": packet(1)}
    paints = []
    view = types.SimpleNamespace(prep=w, schedule=DisplayScheduler(w), _painted=None, _canvas_size=lambda: (64, 36),
                                 _snapshot=types.SimpleNamespace(run_id="run1", basis_id="b1"),
                                 c=types.SimpleNamespace(phase="active", setup_frame=None, gen={"source": 1},
                                                         engine=types.SimpleNamespace(latest_result=lambda: state["pk"])))
    view.canvas = types.SimpleNamespace(delete=lambda *a: None, create_text=lambda *a, **k: None)
    view._overlay = lambda: None
    view._current_display = lambda: ns["_current_display"](view)
    view._publish_presentation = lambda: None          # presentation publishing is covered by test_paired_presentation
    view._paint = lambda res: (paints.append(res.order), setattr(view, "_painted", res))
    for i in range(100):
        w.finish()
        state["pk"] = packet(i + 1)
        ns["_monitor_frame"](view)
    assert len(paints) == 99 and paints == sorted(paints)
    view._snapshot = types.SimpleNamespace(run_id="run1", basis_id="b2")      # Pause: new basis, old packet stays
    w.finish()
    ns["_monitor_frame"](view)
    assert len(paints) == 99 and view._painted is not None                  # kept on screen (overlay marks it), not repainted
