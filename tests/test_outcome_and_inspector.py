# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-04 C1 (durable operation identity governs the outcome) and C2 (the ACTUAL tray/inspector renderers obey
presentation authority). GUI-free: synthetic stores; the real WorkbenchWindow methods are AST-extracted and executed
with the module's own globals, writing into non-widget sinks (no Tk/Tcl object is created). Hooks/barriers are test-only.
"""
from __future__ import annotations

import ast
import sqlite3
import types
from pathlib import Path

import numpy as np

import echo_watch.app.tk_app as tk_mod
from echo_watch.app.controller import WorkbenchController
from echo_watch.app.correction_flow import CorrectionFlow
from echo_watch.app.display import prepare
from echo_watch.app.display_schedule import DisplayScheduler
from echo_watch.app.presentation import packet_identity
from echo_watch.engine.store import Store
from echo_watch.review import activity
from echo_watch.review.corrections import Corrections
from test_corrections_durable import new_store, originals


def counts(st):
    return (st.one("SELECT COUNT(*) AS n FROM corrections")["n"], st.one("SELECT COUNT(*) AS n FROM correction_confirmations")["n"])


def tx_read_fails(cor, also_after=False):
    """The same-ID lookup INSIDE the pending-write transaction raises; optionally the post-rollback read too."""
    real = Corrections._readback
    state = {"rolled_back": False}

    def read(conn, op):
        if conn is not None:
            state["rolled_back"] = True
            raise sqlite3.OperationalError("synthetic read failure inside the transaction")
        if also_after and state["rolled_back"]:
            raise sqlite3.OperationalError("synthetic read failure after rollback")
        return real(cor, conn, op)
    return read


def with_concurrent_save(cor, other, do_other):
    """Barrier: A has found the operation absent; B saves+confirms it; then A's transaction runs (and its read fails)."""
    real_write = Corrections._write
    seen = []

    def barrier(fn):
        if not seen:
            seen.append(do_other(other))
            cor._readback = tx_read_fails(cor)
        return real_write(cor, fn)
    cor._write = barrier
    return seen


# ---- C1 ---------------------------------------------------------------------------------------------------------------
def test_rollback_after_concurrent_same_id_save_reports_the_durable_confirmed_operation():
    st, path = new_store()
    before = originals(st)
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    st_b = Store(path)
    seen = with_concurrent_save(cor, Corrections(st_b), lambda o: o.apply_same_item("op", pv, "a", False))
    out = cor.apply_same_item("op", pv, "a", False)
    del cor._write, cor._readback
    assert seen[0].status == "confirmed" and out.status == "confirmed" and out.correction_id == seen[0].correction_id
    assert "rolled back" in out.message and "already saved by another session" in out.message
    msg, tone = CorrectionFlow(cor, lambda *a: None).on_result(("same", "op", 0, out))
    assert "nothing changed" not in msg.lower() and "not saved" not in msg.lower() and tone == "success"
    assert counts(st) == (1, 1) and cor.fold().effective("b") == "a"
    assert activity.page(st, cor, item_id="a")["total"] == 1
    st_b.close()
    st.close()
    st2 = Store(path)                                          # reopen
    assert Corrections(st2).fold().effective("b") == "a" and originals(st2) == before
    st2.close()


def test_undo_rollback_after_concurrent_same_id_undo_reports_the_undo_in_effect():
    st, path = new_store()
    cor = Corrections(st)
    same = cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    st_b = Store(path)
    seen = with_concurrent_save(cor, Corrections(st_b), lambda o: o.undo("u", same.correction_id))
    out = cor.undo("u", same.correction_id)
    del cor._write, cor._readback
    assert seen[0].status == "confirmed" and out.status == "confirmed" and out.kind == "undo"
    assert "already saved by another session" in out.message and counts(st) == (2, 2) and cor.fold().effective("b") == "b"
    msg, _ = CorrectionFlow(cor, lambda *a: None).on_result(("undo", "u", 0, out))
    assert "not saved" not in msg.lower()
    st_b.close()


def test_failure_of_the_post_rollback_read_is_uncertain_with_the_operation_id():
    st, path = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    st_b = Store(path)
    other = Corrections(st_b)
    real_write = Corrections._write
    seen = []

    def barrier(fn):
        if not seen:
            seen.append(other.apply_same_item("op", pv, "a", False))
            cor._readback = tx_read_fails(cor, also_after=True)
        return real_write(cor, fn)
    cor._write = barrier
    out = cor.apply_same_item("op", pv, "a", False)
    del cor._write, cor._readback
    assert out.status == "uncertain" and "op" in out.message and "could not be read afterwards" in out.message
    msg, tone = CorrectionFlow(cor, lambda *a: None).on_result(("same", "op", 0, out))
    assert "Reconcile" in msg and "not saved" not in msg.lower() and tone == "warning"
    assert counts(st) == (1, 1)                                # B's operation stands
    st_b.close()


def test_genuinely_absent_operation_after_rollback_is_not_saved_then_same_id_retry_saves_once():
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    cor._readback = tx_read_fails(cor)
    out = cor.apply_same_item("op", pv, "a", False)
    del cor._readback
    assert out.status == "not_saved" and "wrote nothing" in out.message and "no saved operation op was found" in out.message
    assert counts(st) == (0, 0) and cor.fold().effective("b") == "b"
    assert cor.apply_same_item("op", pv, "a", False).status == "confirmed" and counts(st) == (1, 1)


def mismatch_on_second_read(cor):
    real, n = Corrections._readback, [0]

    def read(conn, op):
        row = real(cor, conn, op)
        n[0] += 1
        return dict(row, request_sha256="0" * 64) if (n[0] == 2 and row is not None) else row
    return read


def test_mismatch_on_a_confirmed_operation_is_uncertain_and_never_revokes_it():
    st, _ = new_store()
    before = originals(st)
    cor = Corrections(st)
    cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    rows_before = cor.rows()
    cor._readback = mismatch_on_second_read(cor)
    out = cor.reconcile("op")
    del cor._readback
    assert out.status == "uncertain" and "does not revoke a confirmation" in out.message
    assert cor.fold().effective("b") == "a" and counts(st) == (1, 1) and cor.rows() == rows_before   # nothing repaired
    assert originals(st) == before
    assert cor.reconcile("op").status == "confirmed"           # trustworthy evidence resolves it (same id)


def test_mismatch_on_a_confirmed_undo_is_uncertain():
    st, _ = new_store()
    cor = Corrections(st)
    same = cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    cor.undo("u", same.correction_id)
    cor._readback = mismatch_on_second_read(cor)
    out = cor.reconcile("u")
    del cor._readback
    assert out.status == "uncertain" and cor.fold().effective("b") == "b" and counts(st) == (2, 2)
    assert cor.reconcile("u").status == "confirmed"


def test_mismatch_while_pending_stays_pending_without_effect():
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    cor._confirm_row = lambda conn, row: (_ for _ in ()).throw(sqlite3.OperationalError("process boundary"))
    assert cor.apply_same_item("op", pv, "a", False).status == "uncertain"
    del cor._confirm_row
    cor._readback = mismatch_on_second_read(cor)
    out = cor.reconcile("op")
    del cor._readback
    assert out.status == "pending" and "no confirmation exists" in out.message and cor.fold().effective("b") == "b"
    assert counts(st) == (1, 0)
    assert cor.reconcile("op").status == "confirmed" and counts(st) == (1, 1)


# ---- C2: actual renderers ---------------------------------------------------------------------------------------------
IMG = np.zeros((36, 64, 3), np.uint8)


class FakeTree:
    """Non-widget sink with the Treeview calls _render_tray uses."""

    def __init__(self):
        self.rows, self.order = {}, []

    def get_children(self):
        return tuple(self.order)

    def exists(self, iid):
        return iid in self.rows

    def delete(self, *iids):
        for i in iids:
            self.rows.pop(i, None)
            self.order.remove(i)

    def insert(self, parent, index, iid, text, values, **kw):
        self.rows[iid] = {"text": text, "values": tuple(values), "image": kw.get("image", "")}
        self.order.insert(index, iid)

    def item(self, iid, option=None, **kw):
        if kw:
            self.rows[iid].update({k: (tuple(v) if k == "values" else v) for k, v in kw.items()})
            return None
        return self.rows[iid][option]

    def index(self, iid):
        return self.order.index(iid)

    def move(self, iid, parent, index):
        self.order.remove(iid)
        self.order.insert(index, iid)


class Clock:
    t = 1000.0

    def monotonic(self):
        return self.t


def obs(item, state):
    return types.SimpleNamespace(item_id=item, state=state, reason=None, name=item.upper(), last_supported={"t_wall": 50.0},
                                 box=(1.0, 2.0, 10.0, 12.0) if state in ("visible", "checking") else None)


def packet(seq, basis, t, states, run="run1"):
    return types.SimpleNamespace(packet_id=f"res_{seq:06d}", seq=seq, run_id=run, image=IMG, frame_t_mono=t, size=(64, 36),
                                 basis=types.SimpleNamespace(basis_id=basis), items=[obs(i, s) for i, s in states.items()])


class Worker:
    def __init__(self):
        self.pending, self.ready = None, []

    def submit(self, tok, req):
        self.pending = (tok, req)

    def finish(self):
        if self.pending is not None:
            self.ready.append((self.pending[0], prepare(self.pending[1])))
            self.pending = None

    def drain(self):
        r, self.ready = self.ready, []
        return r


RENDER = ("_current_display", "_monitor_frame", "_presentation", "_tray_rows", "_publish_presentation", "_render_tray",
          "_render_inspector")


class View:
    """The real view methods, wired to non-widget sinks."""

    def __init__(self):
        src = Path(tk_mod.__file__).read_text("utf-8")
        cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "WorkbenchWindow")
        fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in RENDER]
        self.clock = Clock()
        self.badge = {}
        ns = dict(vars(tk_mod))
        ns.update(time=self.clock, set_enabled=lambda *a: None,
                  set_pill=lambda obj, text, tone: self.badge.update(text=text, tone=tone))
        exec(compile(ast.Module(body=fns, type_ignores=[]), "tk_app.py", "exec"), ns)
        self.w = Worker()
        self.latest = None
        ctl = types.SimpleNamespace(phase="active", setup_frame=None, gen={"source": 1}, tray=["a", "b"], stopped=False,
                                    tray_profiles={"a": {"name": "A", "current_ref_version": 1}, "b": {"name": "B"}},
                                    library=[], selected="item:a",
                                    engine=types.SimpleNamespace(latest_result=lambda: self.latest,
                                                                 session=types.SimpleNamespace(live_events=lambda *a: [])))
        ctl.tray_rows = lambda snap: WorkbenchController.tray_rows(ctl, snap)
        labels = {}
        v = types.SimpleNamespace(prep=self.w, schedule=DisplayScheduler(self.w), _painted=None, c=ctl, _snapshot=None,
                                  _canvas_size=lambda: (64, 36), _overlay=lambda: None, tray=FakeTree(),
                                  tray_empty=types.SimpleNamespace(pack=lambda **k: None, pack_forget=lambda: None),
                                  _thumb=lambda *a: None, _sync_tree_selection=lambda: None, _inspector_for="item:a",
                                  name_var=types.SimpleNamespace(set=lambda v: None),
                                  insp_thumb=types.SimpleNamespace(configure=lambda **k: None),
                                  insp_state="badge", insp_text="detail", insp_activity="activity",
                                  b_rename=None, b_remove=None, b_all_activity=None,
                                  _set_lbl=lambda obj, text: labels.__setitem__(obj, text),
                                  canvas=types.SimpleNamespace(delete=lambda *a: None, create_text=lambda *a, **k: None))
        for name in RENDER:
            setattr(v, name, (lambda n: (lambda *a: ns[n](v, *a)))(name))
        v._paint = lambda res: setattr(v, "_painted", res)
        self.v, self.ns, self.labels = v, ns, labels

    def snap(self, basis, items=(), run="run1"):
        self.v._snapshot = types.SimpleNamespace(run_id=run, basis_id=basis, items=list(items))

    def tick(self, finish=True):
        if finish:
            self.w.finish()
        self.ns["_monitor_frame"](self.v)

    def tray(self, item):
        return self.v.tray.rows[item]["values"][0]

    def assert_paired(self):
        rows, _auth, _ = self.v._presentation()
        pk = self.v._painted.packet if self.v._painted is not None else None
        for r in rows:
            if r.get("live"):
                assert r["packet"] == packet_identity(pk)
                assert self.badge["tone"] != "neutral" or r["item_id"] != self.v.c.selected[5:] or r["state"] == "cannot_assess"


def live_view(states_a="visible"):
    h = View()
    h.snap("b1")
    h.latest = packet(1, "b1", h.clock.t, {"a": states_a, "b": "visible"})
    h.tick(finish=False)
    h.tick()
    return h


def test_fresh_visible_is_live_success_in_tray_and_inspector():
    h = live_view()
    assert h.badge == {"text": "Visible", "tone": "success"} and h.tray("a") == "Visible"
    assert h.labels["detail"].endswith("Last supported: " + tk_mod._wall(50.0))
    h.assert_paired()


def test_same_image_older_than_one_second_is_stale_neutral_everywhere():
    h = live_view()
    h.clock.t += 1.5                                           # no new frame arrives
    h.tick()
    assert h.badge["tone"] == "neutral" and h.badge["text"].startswith("Stale - last image 1.5 s old showed Visible")
    assert h.tray("a") == h.badge["text"]                      # tray and inspector say the same thing
    assert "(historical; the shown image is stale and has no live authority)" in h.labels["detail"]
    h.assert_paired()


def test_stale_not_visible_is_not_danger_and_not_charged_as_absence_wording():
    h = live_view("not_seen")
    assert h.badge == {"text": "Not visible", "tone": "danger"}
    h.clock.t += 2.0
    h.tick()
    assert h.badge["tone"] == "neutral" and h.badge["text"] == "Stale - last image 2.0 s old showed Not visible"
    rows, _, _ = h.v._presentation()
    assert rows[0]["state"] == "not_seen"                      # the original observation is unchanged


def test_old_basis_with_new_basis_waiting_and_pause_resume():
    h = live_view()
    h.v.c.phase = "paused"
    h.snap("b2", [types.SimpleNamespace(item_id=i, state="paused", reason="paused", last_supported=None) for i in "ab"])
    h.tick()
    assert h.badge == {"text": "Paused", "tone": "neutral"} and h.tray("a") == "Paused"
    h.v.c.phase = "active"                                     # Resume: new basis, a newer packet is not painted yet
    h.snap("b3", [types.SimpleNamespace(item_id="a", state="visible", reason=None, last_supported=None)])
    h.latest = packet(2, "b3", h.clock.t, {"a": "visible", "b": "visible"})
    h.tick(finish=False)
    assert h.badge["tone"] == "neutral" and h.badge["text"] == "Cannot assess - awaiting the current image"
    assert h.tray("a") == h.badge["text"]
    h.tick()                                                   # the fresh current image returns
    assert h.badge == {"text": "Visible", "tone": "success"} and h.tray("a") == "Visible"
    h.assert_paired()


def test_selected_item_switch_while_stale_uses_the_same_authority():
    h = live_view("checking")
    h.clock.t += 1.2
    h.tick()
    h.v.c.selected = "item:b"
    h.ns["_render_inspector"](h.v)
    assert h.badge["text"].startswith("Stale - last image 1.2 s old showed Visible") and h.badge["tone"] == "neutral"
    h.v.c.selected = "item:a"
    h.ns["_render_inspector"](h.v)
    assert h.badge["text"].startswith("Stale - last image 1.2 s old showed Checking") and h.badge["tone"] == "neutral"


def test_continuous_arrival_schedule_still_progresses_with_real_renderers():
    h = View()
    h.snap("b1")
    painted = set()
    for i in range(1, 61):
        h.latest = packet(i, "b1", h.clock.t, {"a": "visible" if i % 3 else "not_seen", "b": "visible"})
        h.tick()
        if h.v._painted is not None:
            painted.add(packet_identity(h.v._painted.packet))
            rows, _, _ = h.v._presentation()
            assert h.tray("a") == rows[0]["display_label"] and h.badge["text"] == rows[0]["display_label"]
        h.clock.t += 0.066
    assert len(painted) == 59
