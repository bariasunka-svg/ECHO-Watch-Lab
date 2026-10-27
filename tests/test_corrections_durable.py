# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-02 repairs B and C: content-based replay scope, transactional preview revalidation and durable
pending/confirmed correction + Undo operations (GUI-free; synthetic stores and files only; no model or decoder).

Failure injection is test-only monkeypatching of private steps; nothing is left enabled in production code.
Each fault case checks fold(), activity projections and reopen - not only that an exception was raised.
"""
from __future__ import annotations

import hashlib
import shutil
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from echo_watch.engine.store import Store
from echo_watch.review import activity
from echo_watch.review import corrections as cm
from echo_watch.review.corrections import Corrections, CorrectionError, OperationConflict, StalePreview
from echo_watch.sources import fingerprint as fpm
from support import work_dir

S1 = "file-sha256:" + "1" * 64
S2 = "file-sha256:" + "2" * 64


def new_store(scopes=(S1, S1), ref_scopes=None, legacy=False) -> tuple[Store, Path]:
    d = work_dir(f"cor-{uuid.uuid4().hex[:8]}")
    path = d / "store.sqlite"
    st = Store(path)
    ref_scopes = ref_scopes or scopes
    with st.tx() as c:
        for ident, src, rsrc in (("a", scopes[0], ref_scopes[0]), ("b", scopes[1], ref_scopes[1])):
            c.execute("INSERT INTO scans(scan_id,source_id,status) VALUES(?,?,'succeeded')", ("scan_" + ident, rsrc))
            c.execute("INSERT INTO items(item_id,name,status,current_ref_version,source_id) VALUES(?,?,?,1,?)",
                      (ident, ident.upper(), "active", src))
            c.execute("INSERT INTO item_references(item_id,version,scan_id,frame_width,frame_height,crop_sha256) "
                      "VALUES(?,1,?,1280,720,?)", (ident, "scan_" + ident, ident * 8))
        c.execute("INSERT INTO run_events(run_id,item_id,wall,kind,payload) VALUES('r1','b',100.0,'first_confirmed','{}')")
    return st, path


def originals(st: Store) -> list:
    return [[dict(r) for r in st.query(f"SELECT * FROM {t} ORDER BY rowid")] for t in ("run_events", "items", "item_references")]


# ---- B: scope ------------------------------------------------------------------------------------------------------
def test_different_bytes_equal_resolution_are_refused():
    st, _ = new_store((S1, S2))
    pv = Corrections(st).preview("a", "b")
    assert not pv.compatible and any("different recordings" in p for p in pv.problems)
    with pytest.raises(CorrectionError):
        Corrections(st).apply_same_item("op", pv, "a", False)
    assert Corrections(st).rows() == []


def test_identical_bytes_at_another_path_are_one_scope():
    d = work_dir(f"fp-{uuid.uuid4().hex[:8]}")
    (d / "x.mp4").write_bytes(b"same content" * 1000)
    (d / "copy" ).mkdir()
    (d / "copy" / "y.mp4").write_bytes(b"same content" * 1000)
    (d / "z.mp4").write_bytes(b"other content" * 1000)
    f1, f2, f3 = (fpm.fingerprint(d / "x.mp4", chunk=4096), fpm.fingerprint(d / "copy" / "y.mp4", chunk=4096),
                  fpm.fingerprint(d / "z.mp4"))
    assert f1["scope"] == f2["scope"] != f3["scope"] and fpm.is_known_scope(f1["scope"])
    assert f1["sha256"] == hashlib.sha256(b"same content" * 1000).hexdigest()
    st, _ = new_store((f1["scope"], f2["scope"]))
    assert Corrections(st).preview("a", "b").compatible


def test_unknown_legacy_scope_is_refused_and_preserved():
    st, _ = new_store(("setup", S1))
    before = originals(st)
    pv = Corrections(st).preview("a", "b")
    assert not pv.compatible and any("no known source scope" in p for p in pv.problems)
    assert originals(st) == before


def test_reference_from_another_recording_makes_the_scope_unknown():
    st, _ = new_store((S1, S1), ref_scopes=(S1, S2))
    pv = Corrections(st).preview("a", "b")
    assert not pv.compatible and any("no known source scope" in p for p in pv.problems)


def test_fingerprint_detects_mutation_and_cancellation(monkeypatch):
    d = work_dir(f"fp-{uuid.uuid4().hex[:8]}")
    f = d / "v.mp4"
    f.write_bytes(b"0" * 10000)
    states = iter([(10000, 1), (10001, 2)])
    monkeypatch.setattr(fpm, "file_state", lambda p: next(states))
    with pytest.raises(fpm.SourceChanged):
        fpm.fingerprint(f)
    monkeypatch.undo()
    with pytest.raises(fpm.Cancelled):
        fpm.fingerprint(f, cancelled=lambda: True)
    st = fpm.fingerprint(f)["state"]
    fpm.verify_unchanged(f, st)
    time.sleep(0.02)
    f.write_bytes(b"1" * 10001)
    with pytest.raises(fpm.SourceChanged):
        fpm.verify_unchanged(f, st)


@pytest.mark.parametrize("change", ["reference", "conflict", "grouping"])
def test_preview_is_revalidated_inside_the_write_and_a_new_operation_id_cannot_bypass(change):
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    with st.tx() as c:
        if change == "reference":
            c.execute("UPDATE item_references SET crop_sha256='changed' WHERE item_id='b'")
        elif change == "conflict":
            for item, s0 in (("a", 0.0), ("b", 0.5)):
                c.execute("INSERT INTO run_items(run_id,item_id,ref_version) VALUES('r9',?,1)", (item,))
                c.execute("INSERT INTO run_intervals(interval_id,run_id,item_id,state,start_mono,end_mono,closed) "
                          "VALUES(?,?,?,'visible',?,?,1)", ("iv" + item, "r9", item, s0, s0 + 2.0))
        else:
            c.execute("INSERT INTO items(item_id,name,status,current_ref_version,source_id) VALUES('c','C','active',1,?)", (S1,))
            c.execute("INSERT INTO scans(scan_id,source_id,status) VALUES('scan_c',?,'succeeded')", (S1,))
            c.execute("INSERT INTO item_references(item_id,version,scan_id,frame_width,frame_height,crop_sha256) "
                      "VALUES('c',1,'scan_c',1280,720,'cc')")
    if change == "grouping":
        assert cor.apply_same_item("op-other", cor.preview("b", "c"), "c", False).status == "confirmed"
    for op in ("op-1", "op-2"):
        with pytest.raises(StalePreview):
            cor.apply_same_item(op, pv, "a", True)
    assert all(r["op_id"] not in ("op-1", "op-2") for r in cor.rows())


# ---- C: durable operations -------------------------------------------------------------------------------------------
def test_failed_initial_write_is_reported_unsaved_and_changes_nothing(monkeypatch):
    st, _ = new_store()
    cor = Corrections(st)

    def boom(*a, **k):
        raise sqlite3.OperationalError("disk I/O error")
    monkeypatch.setattr(Corrections, "_insert_pending", boom)
    out = cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    assert out.status == "not_saved" and not out.effective and cor.rows() == [] and cor.fold().effective("b") == "b"


def test_missing_readback_is_uncertain_ineffective_then_reconciled_without_a_second_operation(monkeypatch):
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    monkeypatch.setattr(Corrections, "_readback", lambda self, conn, op: None)
    out = cor.apply_same_item("op", pv, "a", False)
    assert out.status == "uncertain" and "NOT applied" not in out.message and "outcome unknown" in out.message
    monkeypatch.undo()
    assert cor.fold().effective("b") == "b"                                   # message and durable state agree
    assert activity.page(st, cor, item_id="a")["total"] == 0
    rec = cor.reconcile("op")
    assert rec.status == "confirmed" and len(cor.rows()) == 1 and cor.fold().effective("b") == "a"
    assert activity.page(st, cor, item_id="a")["rows"][0]["corrected"]


def test_mismatched_readback_stays_pending_and_visible(monkeypatch):
    st, _ = new_store()
    cor = Corrections(st)
    real = Corrections._readback

    def wrong(self, conn, op):
        r = real(self, conn, op)
        return None if r is None else dict(r, request_sha256="0" * 64)
    monkeypatch.setattr(Corrections, "_readback", wrong)
    out = cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    monkeypatch.undo()
    assert out.status == "pending" and not out.effective and cor.fold().effective("b") == "b"
    h = cor.history("a")
    assert h[0]["status"] == "pending" and "NOT confirmed" in h[0]["status_text"]


def test_process_boundary_after_pending_save_then_reopen_and_reconcile(monkeypatch):
    st, path = new_store()
    before = originals(st)
    cor = Corrections(st)

    def crash(self, conn, row):
        raise sqlite3.OperationalError("process ended before confirmation")
    monkeypatch.setattr(Corrections, "_confirm_row", crash)
    out = cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    monkeypatch.undo()
    assert out.status == "uncertain" and cor.fold().effective("b") == "b"
    st.close()
    st2 = Store(path)                                                          # a new process opens the store
    cor2 = Corrections(st2)
    assert cor2.fold().effective("b") == "b" and cor2.history("a")[0]["status"] == "pending"
    assert cor2.reconcile("op").status == "confirmed" and cor2.fold().effective("b") == "a"
    assert len(cor2.rows()) == 1 and st2.one("SELECT COUNT(*) AS n FROM correction_confirmations")["n"] == 1
    assert originals(st2) == before
    st2.close()


def test_durable_confirmation_with_a_lost_acknowledgement_reconciles_on_same_id_retry(monkeypatch):
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    real = Corrections._write
    calls = {"n": 0}

    def lost_ack(self, fn):
        calls["n"] += 1
        real(self, fn)                                  # really committed ...
        if calls["n"] == 2:                             # ... but the confirmation acknowledgement is lost
            raise cm._CommitUncertain(RuntimeError("acknowledgement lost"))
    monkeypatch.setattr(Corrections, "_write", lost_ack)
    out = cor.apply_same_item("op", pv, "a", False)
    monkeypatch.undo()
    assert out.status == "uncertain" and "NOT applied" not in out.message      # never denies a possibly durable effect
    assert cor.fold().effective("b") == "a"                                   # it IS durable
    again = cor.apply_same_item("op", pv, "a", False)                          # same-id retry = reconciliation
    assert again.status == "confirmed" and again.correction_id == out.correction_id
    assert len(cor.rows()) == 1 and st.one("SELECT COUNT(*) AS n FROM correction_confirmations")["n"] == 1


def test_concurrent_identical_retries_create_one_operation():
    st, path = new_store()
    pv = Corrections(st).preview("a", "b")
    results, errors = [], []

    def go():
        try:
            s = Store(path)                                      # independent connection, like another process
            results.append(Corrections(s).apply_same_item("op", pv, "a", False))
            s.close()
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    ts = [threading.Thread(target=go) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    assert not errors and {r.status for r in results} == {"confirmed"}
    assert len({r.correction_id for r in results}) == 1
    cor = Corrections(st)
    assert len(cor.rows()) == 1 and st.one("SELECT COUNT(*) AS n FROM correction_confirmations")["n"] == 1


def test_operation_id_is_bound_to_its_kind_and_request():
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    first = cor.apply_same_item("op", pv, "a", False)
    with pytest.raises(OperationConflict):
        cor.apply_same_item("op", pv, "b", False)                     # same id, different primary
    with pytest.raises(OperationConflict):
        cor.undo("op", first.correction_id)                          # same id, different kind
    assert len(cor.rows()) == 1


def test_undo_failure_retry_and_single_effective_undo(monkeypatch):
    st, _ = new_store()
    before = originals(st)
    cor = Corrections(st)
    same = cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    real = Corrections._readback
    monkeypatch.setattr(Corrections, "_readback", lambda self, conn, op: None if op == "undo-1" else real(self, conn, op))
    out = cor.undo("undo-1", same.correction_id)
    monkeypatch.undo()
    assert out.status == "uncertain" and cor.fold().effective("b") == "a"      # pending Undo has no effect
    with pytest.raises(CorrectionError):
        cor.undo("undo-2", same.correction_id)                                # a second Undo is refused
    again = cor.undo("undo-1", same.correction_id)
    assert again.status == "confirmed" and cor.fold().effective("b") == "b"
    hist = cor.history("a")
    assert [h["status"] for h in hist] == ["confirmed", "confirmed"] and hist[0]["undone"]
    assert originals(st) == before


def test_synthetic_v1_store_copy_keeps_legacy_rows_effective_and_undoable():
    st, path = new_store()
    with st.tx() as c:                                   # a format-1 row, as the earlier build wrote it
        c.execute("INSERT INTO corrections(correction_id,op_id,wall,kind,actor,payload,reverses) VALUES(?,?,?,?,?,?,NULL)",
                  ("cor_v1", "same_v1", 50.0, "same_item", "operator",
                   '{"a":"a","b":"b","primary":"a","conflicts":[],"acknowledged":false,"token":"0:None"}'))
    st.close()
    orig_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    copy = path.with_name("copy.sqlite")
    shutil.copyfile(path, copy)
    st2 = Store(copy)
    cor = Corrections(st2)                                # additive migration on the COPY only
    assert hashlib.sha256(path.read_bytes()).hexdigest() == orig_hash
    cols = [r["name"] for r in st2.query("PRAGMA table_info(corrections)")]
    assert "format" in cols and "request_sha256" in cols
    assert cor.fold().effective("b") == "a"              # legacy rule: effective as saved
    h = cor.history("a")
    assert h[0]["status"] == "legacy" and "not readback-confirmed" in h[0]["status_text"]
    assert cor.reconcile("same_v1").status == "legacy"
    u = cor.undo("undo-v1", "cor_v1")
    assert u.status == "confirmed" and cor.fold().effective("b") == "b"
    Corrections(st2)                                      # the migration is idempotent
    st2.close()


def test_no_failure_injection_switch_in_production():
    src = Path(cm.__file__).read_text("utf-8")
    assert "inject" not in src.lower() and "os.environ" not in src
