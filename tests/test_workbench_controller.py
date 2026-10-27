# SPDX-License-Identifier: AGPL-3.0-only
"""GUI-free regression tests of the workbench command/view-model state and the review/correction backends.

No Tcl/Tk is constructed: the controller is exercised with a real Store/Library on a test data directory and a fake
engine/session/worker (no model). These are command/view-model tests, NOT rendered verification.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from echo_watch.app.controller import WorkbenchController
from echo_watch.app.tasks import TaskRunner
from echo_watch.config import Settings
from echo_watch.engine.contracts import PHASE_ACTIVE, PHASE_IDLE, CaptureOptions, OutputOutcome, SessionOutcome
from echo_watch.engine.library import Library
from echo_watch.engine.store import Store
from echo_watch.fixtures.synthetic_scene import SyntheticScene
from echo_watch.frames import Epoch, Frame
from echo_watch.review import activity
from echo_watch.review.corrections import Corrections, CorrectionError
from echo_watch.review.export import export_run
from echo_watch.review.frames import FrameMaps
from echo_watch.review.runs import build_index, read_decision
from support import work_dir


class FakeWorker:
    receipt = {"model_sha256": "pf", "device": "cuda:0"}
    session = 1

    def __init__(self, scene: SyntheticScene, delay: float = 0.0):
        self.scene, self.delay, self.started = scene, delay, False

    def alive(self):
        return self.started

    def scan(self, image_path, imgsz, conf, iou, max_det, timeout_s, area=None):
        time.sleep(self.delay)
        return {"candidates": [{"box": self.scene.item_box_px(n), "label": n, "confidence": 0.9, "polygon": None}
                               for n in ("phone_a", "speaker", "tablet")], "infer_s": 0.01, "device": "cuda:0",
                "_session": 1}


class FakeSession:
    def __init__(self):
        self.phase = PHASE_IDLE
        self.last_outcome = None
        self.discards = {}
        self.started = []

    @property
    def busy(self):
        return self.phase != PHASE_IDLE

    def live_events(self, item_id, limit=8):
        return []

    def snapshot(self):
        return None

    def start_driver(self):
        pass

    def stop(self, **kw):
        self.phase = PHASE_IDLE
        out = OutputOutcome(True, "finalized")
        self.last_outcome = SessionOutcome("finalized", "operator", "run_x", None, CaptureOptions(), None, out, out, out)


class FakeEngine:
    def __init__(self, d: Path, delay=0.0):
        self.s = Settings(data_dir=d)
        self.s.ensure_dirs()
        self.store = Store(self.s.db_path)
        self.library = Library(self.s, self.store)
        self.scene = SyntheticScene(size=(1280, 720))
        self.worker = FakeWorker(self.scene, delay)
        self.session = FakeSession()
        self.start_calls = []

    def start_worker(self):
        self.worker.started = True
        return {"event": "ready", "device": "cuda:0"}

    def open_setup_source(self, path, fps, source_id="setup", timeout_s=30.0):
        img = self.scene.render(0.0)
        return Frame(Epoch("ep0", source_id, 0, 0.0, 0.0), 1, time.monotonic(), time.time(), img)

    def scan(self, frame, source_id):
        return self.library.scan(frame, source_id, self.worker)

    def start(self, options, item_ids, source, **kw):          # EW-LAB-05: + allowance_s / setup binding keywords
        self.start_calls.append((options, list(item_ids), source))
        self.session.phase = PHASE_ACTIVE

    def latest_result(self):
        return None

    def shutdown(self, wait_s=30.0):
        return {}


def settle(ctl, cond, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        ctl.pump()
        if cond():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture()
def ctl(tmp_path):
    import uuid
    d = work_dir(f"controller-{uuid.uuid4().hex[:8]}")
    eng = FakeEngine(d)
    video = d / "clip.mp4"
    video.write_bytes(b"x")
    c = WorkbenchController(eng, TaskRunner(3), probe=lambda p: {"effective_fps": 15.0, "width": 1280, "height": 720})
    c._video = str(video)
    yield c
    c.tasks.shutdown(1.0)
    eng.store.close()


def ready_scan(c):
    assert c.choose_source(c._video)
    assert settle(c, lambda: c.source_status == "ready")
    assert c.scan()
    assert settle(c, lambda: c.scan_status == "done")


def test_launch_is_stopped_and_selection_does_not_start(ctl):
    assert ctl.primary_status()[0] == "Stopped" and ctl.stopped
    ready_scan(ctl)
    assert ctl.engine.start_calls == [] and ctl.tray == []
    assert len(ctl.candidates) == 3 and ctl.engine.worker.started       # the worker started for Scan, not at launch


def test_late_scan_result_cannot_repopulate_a_changed_source(ctl):
    ctl.engine.worker.delay = 0.3
    assert ctl.choose_source(ctl._video)
    assert settle(ctl, lambda: ctl.source_status == "ready")
    ctl.scan()
    assert settle(ctl, lambda: ctl.scan_status == "running" or ctl.engine.worker.started)
    ctl.choose_source(ctl._video)                                  # source changed while the scan is in flight
    assert settle(ctl, lambda: ctl.discarded_completions, timeout=5)
    assert ctl.candidates == [] and ctl.scan_status in ("idle",)


def test_remember_is_idempotent_and_adds_to_this_run_not_by_library(ctl):
    ready_scan(ctl)
    ctl.toggle_candidate(ctl.candidates[0].candidate_id, True)
    assert ctl.remember()
    assert not ctl.remember()                                      # refused while pending
    assert settle(ctl, lambda: not ctl.remember_pending)
    assert len(ctl.tray) == 1
    first = ctl.tray[0]
    ctl.toggle_candidate(ctl.candidates[0].candidate_id, True)
    assert ctl.remember()                                          # the SAME selection again: same operation id
    assert settle(ctl, lambda: not ctl.remember_pending)
    assert ctl.tray == [first]
    assert len(ctl.engine.library.profiles()) == 1
    assert ctl.selected == f"item:{first}"
    rows = ctl.tray_rows(None)
    assert rows[0]["label"] == "Saved, not monitoring"


def test_library_is_separate_from_the_run_and_stop_clears_only_the_tray(ctl):
    ready_scan(ctl)
    for c in ctl.candidates[:2]:
        ctl.toggle_candidate(c.candidate_id, True)
    ctl.remember()
    assert settle(ctl, lambda: not ctl.remember_pending and len(ctl.library) == 2)
    ctl.tray, ctl.tray_profiles = [], {}
    ctl.refresh_library()
    assert settle(ctl, lambda: len(ctl.library) == 2)
    assert ctl.tray == []                                          # opening/refreshing Library adds nothing
    assert ctl.add_to_run(ctl.library[0]["item_id"]) and len(ctl.tray) == 1
    assert ctl.start()
    assert not ctl.start()                                         # repeated Start while pending/active: refused
    assert settle(ctl, lambda: ctl.engine.session.phase == PHASE_ACTIVE and not ctl.start_pending)
    assert len(ctl.engine.start_calls) == 1
    assert not ctl.add_to_run(ctl.library[1]["item_id"])           # membership changes are stopped-only
    assert ctl.stop()
    assert settle(ctl, lambda: ctl.last_outcome is not None)
    assert ctl.tray == [] and len(ctl.library) == 2


def test_settings_pending_applied_running(ctl):
    ctl.set_pending(record_annotated=False)
    st = ctl.settings_status()
    assert st["record_annotated"]["pending"] is False and st["record_annotated"]["applied"] is True
    assert st["record_annotated"]["dirty"]
    ctl.apply_settings()
    assert ctl.settings_status()["record_annotated"]["applied"] is False
    assert ctl.settings_status()["record_annotated"]["running"] is None


def test_manual_region_requires_scan_and_correction_validates(ctl):
    assert not ctl.add_manual_region((10, 10, 50, 50))
    ready_scan(ctl)
    assert ctl.add_manual_region((10, 10, 50, 50))
    assert settle(ctl, lambda: any(c.manual for c in ctl.candidates))
    cid = ctl.candidates[0].candidate_id
    assert not ctl.correct_region(cid, (1270, 700, 50, 50))
    assert ctl.correct_region(cid, (100, 100, 60, 90)) and ctl.candidate(cid).corrected


# ---- corrections ------------------------------------------------------------------------------------------------------
def _two_profiles(ctl):
    ready_scan(ctl)
    for c in ctl.candidates[:2]:
        ctl.toggle_candidate(c.candidate_id, True)
    ctl.remember()
    assert settle(ctl, lambda: not ctl.remember_pending)
    return [p["item_id"] for p in ctl.engine.library.profiles()]


def test_same_item_save_readback_duplicate_retry_undo_and_originals(ctl):
    """EW-LAB-02 format 2: outcomes are values; a retry of the same operation id returns the saved record."""
    a, b = _two_profiles(ctl)
    st = ctl.engine.store
    with st.tx() as c:
        c.execute("INSERT INTO run_events(run_id,item_id,wall,kind,payload) VALUES('r1',?,?,'first_confirmed','{}')", (b, 100.0))
    before = [dict(r) for r in st.query("SELECT * FROM run_events")]
    cor = Corrections(st)
    pv = cor.preview(a, b)
    assert pv.compatible and not pv.conflicts                      # both profiles share the clip's content scope
    r1 = cor.apply_same_item("op-1", pv, a, False)
    r2 = cor.apply_same_item("op-1", pv, a, False)                  # retry: the saved record, not a duplicate
    assert r1.status == r2.status == "confirmed" and r1.correction_id == r2.correction_id and len(cor.rows()) == 1
    g = cor.fold()
    assert g.effective(b) == a and sorted(g.members(a)) == sorted([a, b])
    page = activity.page(st, cor, item_id=a)
    assert page["total"] == 1 and page["rows"][0]["original_item"] == b and page["rows"][0]["corrected"]
    with pytest.raises(CorrectionError):
        cor.apply_same_item("op-2", pv, a, False)                   # the old comparison cannot be replayed under a new id
    u = cor.undo("op-u", r1.correction_id)
    assert u.status == "confirmed" and cor.undo("op-u", r1.correction_id).correction_id == u.correction_id
    assert cor.fold().effective(b) == b
    hist = cor.history(a)
    assert any(h["kind"] == "undo" for h in hist) and any(h.get("undone") for h in hist)
    assert [dict(r) for r in st.query("SELECT * FROM run_events")] == before   # originals unchanged


def test_failed_readback_is_not_effective_and_conflicts_need_acknowledgement(ctl, monkeypatch):
    a, b = _two_profiles(ctl)
    st = ctl.engine.store
    with st.tx() as c:
        for item, s0 in ((a, 0.0), (b, 0.5)):
            c.execute("INSERT INTO run_items(run_id,item_id,ref_version) VALUES('r9',?,1)", (item,))
            c.execute("INSERT INTO run_intervals(interval_id,run_id,item_id,state,start_mono,end_mono,closed) "
                      "VALUES(?,?,?,'visible',?,?,1)", (f"iv{item}", "r9", item, s0, s0 + 2.0))
    cor = Corrections(st)
    pv = cor.preview(a, b)
    assert pv.conflicts and pv.conflicts[0]["seen_together_s"] == 1.5
    with pytest.raises(CorrectionError):
        cor.apply_same_item("op-c", pv, a, False)
    monkeypatch.setattr(Corrections, "_readback", lambda self, conn, op: None)       # read-back fails (test-only)
    out = cor.apply_same_item("op-c", pv, a, True)
    assert out.status == "uncertain" and "outcome unknown" in out.message and cor.fold().effective(b) == b


# ---- review: index, exact clean-frame lookup, evidence gaps, export --------------------------------------------------
def _fake_run(d: Path) -> Path:
    run = d / "runs" / "run_t"
    run.mkdir(parents=True)
    recs = [{"n": 1, "k": "session", "t": 1.0, "w": 1.0, "event": "start_requested"},
            {"n": 2, "k": "frame", "t": 1.1, "w": 1.1, "ident": ["ep", 1], "t_mono": 1.1, "media_pts": 0.0, "dup": False},
            {"n": 3, "k": "frame", "t": 1.2, "w": 1.2, "ident": ["ep", 2], "t_mono": 1.2, "media_pts": 0.066, "dup": False},
            {"n": 4, "k": "result", "t": 1.3, "w": 1.3, "seq": 1, "packet_id": "res_000001", "frame": ["ep", 2],
             "frame_t_mono": 1.2, "frame_t_wall": 1.2, "media": {"media_pts": 0.066, "media_n": 1},
             "items": [{"item_id": "it", "state": "visible", "label": "Visible", "box": [1, 2, 30, 40], "reason": None,
                        "app": 0.9, "admission": "support"}],
             "candidates": [{"idx": 0, "box": [1, 2, 30, 40], "conf": 0.9, "app": {"it": 0.9}, "admission": {"it": "support"}}],
             "timing": {"capture_to_commit_ms": 50, "stage_ms": {"evaluate": 3}}, "worker": {"device": "cuda:0"}},
            {"n": 6, "k": "discard", "t": 1.4, "w": 1.4, "ident": ["ep", 1], "reason": "expired_at_commit"}]
    (run / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", "utf-8")
    (run / "clean.framemap.json").write_text(json.dumps({"video": "clean.mp4", "fps_nominal": 15.0, "gaps": [], "dropped": [],
                                                          "entries": [{"i": 0, "ident": ["ep", 2], "sig": "00"}]}), "utf-8")
    return run


def test_review_index_exact_decision_frame_and_gaps():
    d = work_dir(f"review_index-{time.time_ns()}")
    run = _fake_run(d)
    idx = build_index("run_t", run)
    assert len(idx.rows) == 1 and idx.status == "partial"          # sequence gap 4 -> 6 is disclosed
    rec = read_decision(idx, idx.rows[0])
    assert rec["packet_id"] == "res_000001"
    maps = FrameMaps(run)
    ok = maps.locate_clean_frame(["ep", 2])
    assert ok.ok and ok.index == 0
    gap = maps.locate_clean_frame(["ep", 1])
    assert not gap.ok and "not in the clean recording" in gap.gap
    assert build_index("run_t", run).rows == idx.rows                # cached projection, same rows


def test_export_projects_without_recomputing():
    d = work_dir(f"review_export-{time.time_ns()}")
    run = _fake_run(d)
    rep = export_run(run)
    out = Path(rep["dir"])
    frames = (out / "frames.csv").read_text("utf-8").splitlines()
    assert any("analysed,result 1" in x for x in frames) and any("discard,expired_at_commit" in x for x in frames)
    dec = (out / "decisions.csv").read_text("utf-8").splitlines()
    assert len(dec) == 2 and ",support," in dec[1]
    assert json.loads((out / "manifest.json").read_text("utf-8"))["recomputed"] is False


def test_gui_free_startup_never_creates_a_tk_root():
    import sys
    from echo_watch.app import launch
    cfg = launch.startup_config()
    assert cfg["tk_loaded"] is False and cfg["title"]
    tk = sys.modules.get("tkinter")
    assert tk is None or getattr(tk, "_default_root", None) is None


# ---- EW-LAB-02 repair B at the controller ---------------------------------------------------------------------------
def test_source_scope_binds_to_setup_scan_and_items(ctl):
    ready_scan(ctl)
    scope = ctl.source_info["scope"]
    assert scope.startswith("file-sha256:") and ctl.setup_frame.epoch.source_id == scope
    ctl.toggle_candidate(ctl.candidates[0].candidate_id, True)
    ctl.remember()
    assert settle(ctl, lambda: not ctl.remember_pending)
    assert {p["item_id"]: p for p in ctl.engine.library.profiles()}
    row = ctl.engine.store.one("SELECT source_id FROM items")
    assert row["source_id"] == scope and ctl.engine.store.one("SELECT source_id FROM scans")["source_id"] == scope


def test_late_old_source_preparation_is_discarded(ctl, tmp_path):
    other = Path(ctl._video).with_name("other.mp4")
    other.write_bytes(b"different bytes" * 100)
    slow = {"first": True}

    def probe(p):
        if slow["first"]:
            slow["first"] = False
            time.sleep(0.3)
        return {"effective_fps": 15.0}
    ctl._probe = probe
    assert ctl.choose_source(ctl._video)
    assert ctl.choose_source(str(other))                     # a newer selection while the first is preparing
    assert settle(ctl, lambda: ctl.source_status == "ready")
    time.sleep(0.4)
    ctl.pump()
    assert ctl.source_info["path"] == str(other)
    assert any(d["kind"] == "source" for d in ctl.discarded_completions)      # the late first preparation was dropped


def test_source_mutation_after_selection_invalidates_setup(ctl):
    assert ctl.choose_source(ctl._video)
    assert settle(ctl, lambda: ctl.source_status == "ready")
    time.sleep(0.02)
    Path(ctl._video).write_bytes(b"replaced content")
    assert ctl.scan()
    assert settle(ctl, lambda: ctl.source_status == "failed")
    assert ctl.setup_frame is None and ctl.source_info is None and ctl.candidates == []
    assert "changed" in ctl.messages[-1].text
