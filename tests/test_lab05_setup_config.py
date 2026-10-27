# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-05 matrix 5: versioned setup artifacts, exact reference/config binding and the review allowance (GUI-free).

Real SetupStore / Library / ItemEngine.start boundary / ManagedItemSession.start / ItemBoard / ItemTimeline on synthetic
stores; fake worker and session; numpy codec stubs; ``prepare_member`` stubbed (no image decode). No process, model,
GPU or media is involved.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest

from echo_watch.config import Settings
from echo_watch.engine import session as session_mod
from echo_watch.engine.board import ItemBoard
from echo_watch.engine.contracts import PHASE_IDLE, CaptureOptions
from echo_watch.engine.library import LibraryError
from echo_watch.engine.session import ManagedItemSession, SessionError, SourceSpec
from echo_watch.engine.setups import SetupError, SetupStore, build, validate_allowance
from echo_watch.engine.store import Store
from echo_watch.item_policy import ItemTimeline, Sample
from lab05_support import Rig, settle
from support import work_dir


@pytest.fixture()
def rig(monkeypatch):
    r = Rig("lab05-setup", monkeypatch)
    yield r
    r.close()


def _ready(rig, labels=("phone", "mug")):
    rig.discover(rig.a)
    for lab in labels:
        rig.ctl.toggle_candidate(rig.cid(lab), True)
    rig.remember()
    assert rig.ctl.view_step == "configure"


def _hash_tree(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*"))
            if p.is_file()}


# ---- the artifact itself ---------------------------------------------------------------------------------------------
def test_allowance_is_a_whole_number_of_seconds_from_1_to_3600():
    assert validate_allowance(1) == 1 and validate_allowance(3600) == 3600 and validate_allowance("60") == 60
    for bad in (0, 3601, 1.5, "abc", None, True, -5):
        with pytest.raises(SetupError):
            validate_allowance(bad)


def test_save_is_versioned_atomic_never_overwrites_and_unchanged_returns_the_same_version(rig):
    c = rig.ctl
    _ready(rig)
    assert c.save_configuration()
    assert settle(c, lambda: c.setup_status == "saved")
    s1 = c.setup
    assert s1.version == 1 and Path(s1.path).is_file() and not list(Path(s1.path).parent.glob("*.tmp"))
    doc = json.loads(Path(s1.path).read_text("utf-8"))
    assert doc["schema"] == "echo-watch.display-setup" and doc["review_allowance_s"] == 60
    assert doc["outputs"] == {"clean_source": True, "record_annotated": True, "decision_trace": True, "repeat": False}
    assert [(i["item_id"], i["ref_version"]) for i in doc["items"]] == [(i, c.member_versions[i]) for i in c.tray]
    assert doc["source"]["scope"] == c.source_info["scope"] and doc["model"]["discovery_model_sha256"] == "pf-test-model"
    assert c.save_configuration() and settle(c, lambda: c.setup_status == "saved")
    assert c.setup.version == 1                                  # unchanged: no new version
    c.set_allowance(120)
    c.set_pending(record_annotated=False)
    assert c.setup_dirty()
    assert c.save_configuration() and settle(c, lambda: c.setup_status == "saved" and c.setup.version == 2)
    assert c.setups.versions(s1.setup_id) == [1, 2]
    v1 = c.setups.load(s1.setup_id, 1)
    assert v1.allowance_s == 60 and v1.outputs["record_annotated"] is True     # old versions keep what they saved
    assert c.applied_allowance == 120 and c.applied.record_annotated is False


def test_load_validates_schema_fields_digest_and_links_without_altering_saved_files(rig):
    c = rig.ctl
    _ready(rig)
    assert c.save_configuration() and settle(c, lambda: c.setup_status == "saved")
    s = c.setup
    root = c.setups.root
    base = json.loads(Path(s.path).read_text("utf-8"))

    def write_version(v, doc):
        p = root / s.setup_id / f"v{v}.json"
        p.write_text(json.dumps(doc), "utf-8")
        return p
    write_version(2, {**base, "version": 2, "surprise": 1})
    write_version(3, {**base, "version": 3, "schema_version": 9})
    write_version(4, {**base, "version": 4, "review_allowance_s": 7})            # edited outside: digest mismatch
    (root / s.setup_id / "v5.json").write_text("{not json", "utf-8")
    for v, msg in ((2, "unknown fields"), (3, "unsupported version"), (4, "digest"), (5, "could not be read")):
        with pytest.raises(SetupError, match=msg):
            c.setups.load(s.setup_id, v)
    before = _hash_tree(root)
    # a broken link: the saved crop of one reference changed on disk -> visible failure, nothing repaired
    item = c.tray[0]
    crop = Path(c.tray_profiles[item]["crop_path"])
    crop.write_bytes(crop.read_bytes() + b"tamper")
    c.new_setup()
    assert c.load_setup(s.setup_id, 1)
    assert settle(c, lambda: c.setup_status == "failed")
    assert "cannot be used as it is" in c.setup_error and "changed on disk" in c.setup_error
    assert c.tray == [] and _hash_tree(root) == before          # membership unchanged, saved files untouched
    lst = c.setups.list_latest()
    assert lst[0]["error"]                                       # the newest (corrupt) version is reported, not hidden


def test_a_setup_of_another_recording_is_not_loaded(rig):
    c = rig.ctl
    _ready(rig)
    assert c.save_configuration() and settle(c, lambda: c.setup_status == "saved")
    sid = c.setup.setup_id
    rig.prepare(rig.b)
    assert c.load_setup(sid)
    assert settle(c, lambda: c.setup_status == "failed")
    assert "different file content" in c.setup_error and c.tray == []


# ---- Start binds the exact configuration -------------------------------------------------------------------------------
def test_start_saves_then_binds_exact_versions_allowance_outputs_and_setup_identity(rig):
    c = rig.ctl
    _ready(rig, ("phone",))
    item = c.tray[0]
    c.set_allowance(15)
    c.set_pending(clean_source=False, repeat=True)
    run = rig.start()
    s = c.setup
    assert run["refs"] == [(item, 1)] and run["allowance_s"] == 15
    assert run["options"].clean_source is False and run["source"].repeat is True
    assert run["source"].source_id == c.source_info["scope"] and run["source"].size == (640, 360)
    assert run["setup"] == s.identity() and run["setup"]["version"] == s.version and run["setup"]["scope"] == s.scope
    assert c.running_settings["review_allowance_s"] == 15 and c.running_setup["setup_id"] == s.setup_id
    # pending edits during the run never change the running configuration
    c.set_allowance(900)
    c.set_pending(decision_trace=False)
    assert c.running_settings["review_allowance_s"] == 15 and c.running_settings["decision_trace"] is True
    assert c.allowance_status() == {"pending": 900, "applied": 15, "running": 15, "dirty": True}


def test_start_uses_the_captured_version_even_after_a_newer_reference_exists(rig):
    c = rig.ctl
    _ready(rig, ("phone",))
    item = c.tray[0]
    lib = rig.eng.library
    v2 = lib.update_reference("op-v2", item, c.scan_id, rig.cid("phone"))
    assert v2.ref_version == 2
    run = rig.start()
    assert run["refs"] == [(item, 1)]                            # exactly what the setup captured


def test_storage_failure_prevents_start_and_keeps_the_selection(rig, monkeypatch):
    c = rig.ctl
    _ready(rig)
    monkeypatch.setattr(SetupStore, "save", lambda self, doc, sid=None: (_ for _ in ()).throw(OSError("disk full")))
    assert c.start()
    assert settle(c, lambda: not c.start_pending)
    assert rig.eng.session.started == [] and len(c.tray) == 2 and c.view_step == "configure"
    assert "could not be saved, so monitoring did not start" in c.messages[-1].text
    assert c.setup_status == "failed"


def test_engine_start_boundary_revalidates_versions_crops_and_setup_binding(rig):
    c = rig.ctl
    _ready(rig, ("phone",))
    eng, item = rig.eng, c.tray[0]
    src = SourceSpec(c.source_info["scope"], c.source_info["path"], 15.0, size=(640, 360))
    with pytest.raises(LibraryError, match="version is missing"):
        eng.start(CaptureOptions(), [(item, 7)], src)
    with pytest.raises(LibraryError, match="different recording"):
        eng.start(CaptureOptions(), [(item, 1)], SourceSpec("file-sha256:" + "f" * 64, "x.mp4", 15.0, size=(640, 360)))
    with pytest.raises(LibraryError, match="exactly these reference versions"):
        eng.start(CaptureOptions(), [(item, 1)], src, setup={"scope": src.source_id, "items": [[item, 2]]})
    crop = Path(c.tray_profiles[item]["crop_path"])
    data = crop.read_bytes()
    crop.write_bytes(data + b"x")
    with pytest.raises(LibraryError, match="changed on disk"):
        eng.start(CaptureOptions(), [(item, 1)], src)
    crop.write_bytes(data)
    assert eng.session.started == []
    eng.start(CaptureOptions(), [(item, 1)], src, allowance_s=30, setup={"scope": src.source_id, "items": [[item, 1]]})
    assert eng.session.started[-1]["refs"] == [(item, 1)] and eng.session.started[-1]["allowance_s"] == 30


# ---- session: the allowance reaches the real board/timeline; binding and metadata ---------------------------------------
class SessionWorker:
    session = 1
    receipt = {"device": "cuda:0", "model_sha256": "pf-test-model"}

    def alive(self):
        return True

    def set_refs(self, payload, digest):
        raise RuntimeError("test worker: no reference preparation")

    def health(self):
        return {"state": "test"}


def _session(rig, monkeypatch):
    monkeypatch.setattr(session_mod, "prepare_member", lambda ref, amb=None: object())
    d = work_dir("lab05-session")
    s = Settings(data_dir=d)
    s.ensure_dirs()
    st = Store(s.db_path)
    return ManagedItemSession(settings=s, store=st, worker=SessionWorker()), st


def _settle(sess):
    if sess._prep_thread is not None:
        sess._prep_thread.join(5)
    end = time.monotonic() + 5
    while sess.phase != PHASE_IDLE and time.monotonic() < end:
        sess.poll()
        time.sleep(0.01)
    return sess.last_outcome


@pytest.mark.parametrize("allowance", [None, 7])
def test_session_start_passes_default_or_chosen_allowance_to_every_timeline_and_records_it(rig, monkeypatch, allowance):
    c = rig.ctl
    _ready(rig)
    refs = rig.eng.library.resolve_refs([(i, c.member_versions[i]) for i in c.tray],
                                        rig.eng.library.scan_scope(c.scan_id))
    sess, st = _session(rig, monkeypatch)
    setup = {"setup_id": "setup_0123456789abcdef", "version": 3, "digest": "d", "name": "Display 01",
             "scope": refs[0].item_id and c.source_info["scope"], "items": [[r.item_id, r.ref_version] for r in refs],
             "review_allowance_s": allowance or 60}
    kw = {} if allowance is None else {"allowance_s": allowance, "setup": setup}
    sess.start(CaptureOptions(False, False, False), refs, SourceSpec(c.source_info["scope"], "a.mp4", 15.0), **kw)
    want = 60.0 if allowance is None else 7.0
    assert {i: s.timeline.allowance_s for i, s in sess._board.items.items()} == {r.item_id: want for r in refs}
    row = json.loads(st.one("SELECT options FROM runs WHERE run_id=?", (sess.run_id,))["options"])
    assert row["review_allowance_s"] == want and row["setup"] == (None if allowance is None else setup)
    assert {k: row[k] for k in ("record_annotated", "clean_source", "decision_trace")} == \
        {"record_annotated": False, "clean_source": False, "decision_trace": False}
    run_dir = sess.run_dir
    out = _settle(sess)
    assert out is not None and out.kind == "rolled_back"        # the fake worker refused; nothing kept running
    man = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    assert man["review_allowance_s"] == want and man["setup"] == (None if allowance is None else setup)
    st.close()


def test_session_refuses_an_invalid_allowance_or_a_setup_that_names_other_versions(rig, monkeypatch):
    c = rig.ctl
    _ready(rig, ("phone",))
    refs = rig.eng.library.resolve_refs([(c.tray[0], 1)], rig.eng.library.scan_scope(c.scan_id))
    sess, st = _session(rig, monkeypatch)
    src = SourceSpec(c.source_info["scope"], "a.mp4", 15.0)
    for bad in (0, 3601):
        with pytest.raises(SessionError, match="between 1 and 3600"):
            sess.start(CaptureOptions(False, False, False), refs, src, allowance_s=bad)
    with pytest.raises(SessionError, match="does not match the exact references"):
        sess.start(CaptureOptions(False, False, False), refs, src, setup={"items": [[c.tray[0], 2]]})
    assert sess.phase == PHASE_IDLE and sess.run_id is None
    st.close()


def test_allowance_uses_the_existing_supported_time_semantics_only():
    """Only eligible Not-visible time is charged; uncertain / paused time freezes the counter (existing policy)."""
    def run(allowance, cats):
        tl = ItemTimeline("it", allowance)
        ev = []
        for k, cat in enumerate(cats):
            ev += tl.sample(Sample(100.0 + k, 1000.0 + k, "ep", cat, k))
        return tl, [e["kind"] for e in ev]
    seq = ["visible"] * 3 + ["not_seen"] * 6
    tl, kinds = run(5, seq)
    assert "review_requested" in kinds and tl.gap["charged_s"] >= 5
    _, kinds60 = run(60, seq)
    assert "gap_started" in kinds60 and "review_requested" not in kinds60     # default allowance not reached
    mixed = ["visible"] * 3 + ["not_seen", "not_seen"] + ["uncertain"] * 4 + ["paused"] * 4 + ["not_seen"]
    tl2, kinds2 = run(5, mixed)
    assert "review_requested" not in kinds2 and tl2.gap["charged_s"] < 5      # unknown/paused time never counts
    board = ItemBoard("r", [], allowance_s=9)
    assert board.items == {}
