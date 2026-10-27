# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-03 R1 (truthful durable outcomes through store, retry/reconcile and UI boundaries) and R2 (comparisons bind the
complete affected effective groups). Synthetic stores only; fault hooks/barriers live in this test code only.
Each case checks durable rows/confirmations, fold(), Activity and reopen - not only the returned status."""
from __future__ import annotations

import sqlite3
import types

import pytest

from echo_watch.app.correction_flow import CorrectionFlow, describe
from echo_watch.engine.store import Store
from echo_watch.review import activity
from echo_watch.review.corrections import Corrections, CorrectionError, StalePreview
from test_corrections_durable import S1, new_store, originals


def add_item(st: Store, ident: str, scope: str = S1, crop: str | None = None) -> None:
    with st.tx() as c:
        c.execute("INSERT INTO scans(scan_id,source_id,status) VALUES(?,?,'succeeded')", ("scan_" + ident, scope))
        c.execute("INSERT INTO items(item_id,name,status,current_ref_version,source_id) VALUES(?,?,?,1,?)",
                  (ident, ident.upper(), "active", scope))
        c.execute("INSERT INTO item_references(item_id,version,scan_id,frame_width,frame_height,crop_sha256) "
                  "VALUES(?,1,?,1280,720,?)", (ident, "scan_" + ident, crop or ident * 8))


def covisible(st: Store, run: str, a: str, b: str, a_iv=(0.0, 2.0), b_iv=(0.5, 2.5)) -> None:
    with st.tx() as c:
        for item, (s0, s1) in ((a, a_iv), (b, b_iv)):
            c.execute("INSERT OR IGNORE INTO run_items(run_id,item_id,ref_version) VALUES(?,?,1)", (run, item))
            c.execute("INSERT INTO run_intervals(interval_id,run_id,item_id,state,start_mono,end_mono,closed) "
                      "VALUES(?,?,?,'visible',?,?,1)", (f"iv-{run}-{item}", run, item, s0, s1))


def counts(st: Store) -> tuple[int, int]:
    return (st.one("SELECT COUNT(*) AS n FROM corrections")["n"],
            st.one("SELECT COUNT(*) AS n FROM correction_confirmations")["n"])


def raising_readback(cor: Corrections, when_row: bool = True):
    real = Corrections._readback

    def rb(conn, op):
        row = real(cor, conn, op)
        if (row is not None) == when_row:
            raise sqlite3.OperationalError("synthetic readback I/O error")
        return row
    return rb


# ---- R1: truthful outcomes ----------------------------------------------------------------------------------------------
def test_raised_read_error_after_pending_save_is_uncertain_never_nothing_saved():
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    cor._readback = raising_readback(cor)              # fails only once the pending row exists
    out = cor.apply_same_item("op-1", pv, "a", False)
    del cor._readback
    assert out.status == "uncertain" and "op-1" in out.message and "Nothing is assumed" in out.message
    assert counts(st) == (1, 0) and cor.fold().effective("b") == "b"              # saved, pending, no effect
    flow = CorrectionFlow(cor, lambda *a: None)
    msg, _tone = flow.on_result(("same", "op-1", 0, out))
    assert "nothing was" not in msg.lower() and "op-1" in msg
    rec = cor.reconcile("op-1")
    assert rec.status == "confirmed" and counts(st) == (1, 1) and cor.fold().effective("b") == "a"


def test_raised_read_error_before_pending_save_is_uncertain_then_same_id_retry_saves_once():
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    cor._readback = raising_readback(cor, when_row=False)     # the very first lookup fails
    out = cor.apply_same_item("op-2", pv, "a", False)
    del cor._readback
    assert out.status == "uncertain" and counts(st) == (0, 0)
    again = cor.apply_same_item("op-2", pv, "a", False)
    assert again.status == "confirmed" and counts(st) == (1, 1)


def test_read_error_on_retry_of_a_confirmed_operation_keeps_uncertainty_and_effect():
    st, _ = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    assert cor.apply_same_item("op-3", pv, "a", False).status == "confirmed"
    cor._readback = raising_readback(cor)
    out = cor.apply_same_item("op-3", pv, "a", False)
    del cor._readback
    assert out.status == "uncertain" and cor.fold().effective("b") == "a" and counts(st) == (1, 1)
    msg, _ = CorrectionFlow(cor, lambda *a: None).on_result(("same", "op-3", 0, out))
    assert "refused" not in msg.lower() and "nothing was" not in msg.lower() and "op-3" in msg


def test_raw_exception_reaching_the_ui_keeps_operation_identity_and_uncertainty():
    flow = CorrectionFlow(types.SimpleNamespace(), lambda *a: None)
    msg, tone = flow.on_result(("same", "op-x", 0, sqlite3.OperationalError("disk I/O error")))
    assert "Outcome unknown for operation op-x" in msg and "nothing was" not in msg.lower() and tone == "warning"
    msg, tone = flow.on_result(("same", "op-x", 0, CorrectionError("review and acknowledge the conflicts")))
    assert msg.startswith("Refused - nothing was written by this request") and tone == "danger"


def test_deterministic_interleaving_A_reads_pending_B_confirms_A_writes():
    st, path = new_store()
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    cor._confirm_row = lambda conn, row: (_ for _ in ()).throw(sqlite3.OperationalError("process boundary"))
    assert cor.apply_same_item("race", pv, "a", False).status == "uncertain"       # pending, unconfirmed
    del cor._confirm_row
    st_b = Store(path)
    other = Corrections(st_b)
    real_write = Corrections._write
    seen = []

    def barrier(fn):                       # B confirms the SAME operation after A observed it unconfirmed
        seen.append(other.reconcile("race"))
        return real_write(cor, fn)
    cor._write = barrier
    out = cor.reconcile("race")
    del cor._write
    assert seen[0].status == "confirmed"
    assert out.status == "confirmed" and "NOT applied" not in out.message           # never pending for a confirmed op
    assert counts(st) == (1, 1) and cor.fold().effective("b") == "a"
    assert activity.page(st, cor, item_id="a")["total"] == 1
    st_b.close()


def test_undo_raised_read_error_is_uncertain_and_same_id_retry_confirms():
    st, _ = new_store()
    cor = Corrections(st)
    same = cor.apply_same_item("op", cor.preview("a", "b"), "a", False)
    cor._readback = raising_readback(cor)
    out = cor.undo("undo-1", same.correction_id)
    del cor._readback
    assert out.status == "uncertain" and cor.fold().effective("b") == "a" and counts(st) == (2, 1)
    assert cor.undo("undo-1", same.correction_id).status == "confirmed" and cor.fold().effective("b") == "b"
    assert counts(st) == (2, 2)


def test_confirmed_then_undone_reconciles_as_undone_and_is_never_redone():
    st, path = new_store()
    before = originals(st)
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    same = cor.apply_same_item("op", pv, "a", False)
    cor.undo("u", same.correction_id)
    for c in (cor, Corrections(Store(path))):            # also after reopening
        rec = c.reconcile("op")
        assert rec.status == "confirmed_undone" and rec.durable and not rec.effective and "UNDONE" in rec.message
        retry = c.apply_same_item("op", pv, "a", False)
        assert retry.status == "confirmed_undone" and c.fold().effective("b") == "b"
        h = {x["correction_id"]: x for x in c.history("a")}
        assert h[same.correction_id]["undone"] and not h[same.correction_id]["in_effect"]
        assert "NOT in effect" in h[same.correction_id]["status_text"]
        assert activity.page(st, c, item_id="a")["total"] == 0          # b's event is b's own again
        rowb = activity.page(st, c, item_id="b")["rows"][0]
        assert rowb["original_item"] == "b" and rowb["corrected"] is False
        assert c.reconcile("u").status == "confirmed"                               # the Undo itself is in effect
    assert counts(st) == (2, 2) and originals(st) == before
    msg, _ = CorrectionFlow(cor, lambda *a: None).on_result(("reconcile", "op", 0, cor.reconcile("op")))
    assert "UNDONE" in msg and "in effect" in msg and "Activity shows the effective grouping" not in msg


def test_ui_completion_for_an_older_comparison_never_clears_the_newer_one():
    st, _ = new_store()
    add_item(st, "c")
    cor = Corrections(st)
    jobs = []
    flow = CorrectionFlow(cor, lambda kind, tok, fn: jobs.append((kind, tok, fn)))
    flow.compare("a", "b")
    flow.on_preview(jobs[-1][1], True, jobs[-1][2](), None)
    flow.save("a", False)
    first_op = flow.request.op_id
    pending_save = jobs[-1][2]
    flow.compare("a", "c")                                    # a newer comparison before the save completes
    flow.on_preview(jobs[-1][1], True, jobs[-1][2](), None)
    msg, _ = flow.on_result(pending_save())                   # the older save completes now
    assert "in effect" in msg and flow.preview is not None and flow.preview.b == "c" and flow.request is None
    flow.save("a", False)
    assert flow.request.op_id != first_op                     # the newer comparison has its own operation identity
    # a stale preview result of the OLD comparison is ignored
    assert flow.on_preview(("corr", 1), True, cor.preview("a", "b"), None) is None


def test_retry_repeats_the_captured_request_and_a_changed_request_is_refused():
    st, _ = new_store()
    cor = Corrections(st)
    jobs = []
    flow = CorrectionFlow(cor, lambda kind, tok, fn: jobs.append(fn))
    flow.compare("a", "b")
    flow.on_preview(("corr", flow.gen), True, jobs[-1](), None)
    cor_insert = Corrections._insert_pending
    Corrections._insert_pending = lambda *a, **k: (_ for _ in ()).throw(sqlite3.OperationalError("disk full"))
    try:
        flow.save("a", False)
        msg, _ = flow.on_result(jobs[-1]())
    finally:
        Corrections._insert_pending = cor_insert
    assert "wrote nothing" in msg and "no saved operation" in msg and flow.request is not None
    notice = flow.save("b", False)                            # operator flips the primary before retrying
    assert notice and "original request" in notice
    msg, _ = flow.on_result(jobs[-1]())
    assert "in effect" in msg and cor.fold().effective("b") == "a"                 # primary stayed a
    with pytest.raises(CorrectionError):
        cor.apply_same_item(flow.retry_op() or "op-none", cor.preview("a", "b"), "b", False)


# ---- R2: complete affected groups ---------------------------------------------------------------------------------------
def test_membership_added_with_unchanged_primary_makes_the_old_comparison_stale():
    st, _ = new_store()
    add_item(st, "c")
    cor = Corrections(st)
    old = cor.preview("a", "b")
    assert cor.apply_same_item("b-c", cor.preview("b", "c"), "b", False).status == "confirmed"   # b stays primary
    fresh = cor.preview("a", "b")
    assert fresh.binding != old.binding and fresh.group_b == ("b", "c") and fresh.primary_b == "b"
    with pytest.raises(StalePreview):
        cor.apply_same_item("a-b-old", old, "a", False)
    assert cor.fold().members("a") == ["a"]
    text = describe(fresh)
    assert "C (c)" in text and "Second group (2)" in text and "all 3 profiles" in text
    assert cor.apply_same_item("a-b-new", fresh, "a", False).status == "confirmed"
    assert cor.fold().members("a") == ["a", "b", "c"]


def test_merge_then_undo_restoring_primaries_still_invalidates():
    st, _ = new_store()
    add_item(st, "c")
    cor = Corrections(st)
    old = cor.preview("a", "b")
    bc = cor.apply_same_item("b-c", cor.preview("b", "c"), "b", False)
    cor.undo("u-bc", bc.correction_id)
    back = cor.preview("a", "b")
    assert (back.primary_a, back.primary_b, back.group_b) == (old.primary_a, old.primary_b, old.group_b)
    assert back.binding != old.binding
    with pytest.raises(StalePreview):
        cor.apply_same_item("a-b-old", old, "a", False)


@pytest.mark.parametrize("change", ["reference", "scope"])
def test_third_member_reference_or_scope_change_invalidates(change):
    st, _ = new_store()
    add_item(st, "c")
    cor = Corrections(st)
    cor.apply_same_item("b-c", cor.preview("b", "c"), "b", False)
    pv = cor.preview("a", "b")
    with st.tx() as c:
        if change == "reference":
            c.execute("UPDATE item_references SET crop_sha256='new' WHERE item_id='c'")
        else:
            c.execute("UPDATE items SET source_id='setup' WHERE item_id='c'")
    with pytest.raises((StalePreview, CorrectionError)):
        cor.apply_same_item("a-b", pv, "a", False)
    assert cor.fold().members("a") == ["a"]


def test_new_conflict_involving_a_third_member_invalidates_and_is_shown():
    st, _ = new_store()
    add_item(st, "c")
    cor = Corrections(st)
    cor.apply_same_item("b-c", cor.preview("b", "c"), "b", False)
    pv = cor.preview("a", "b")
    assert not pv.conflicts
    covisible(st, "r7", "a", "c")
    with pytest.raises(StalePreview):
        cor.apply_same_item("a-b", pv, "a", True)
    fresh = cor.preview("a", "b")
    assert fresh.conflicts and fresh.conflicts[0]["items"] == ["a", "c"]
    assert "CONFLICT: run r7: A (a) and C (c)" in describe(fresh)
    with pytest.raises(CorrectionError):
        cor.apply_same_item("a-b2", fresh, "a", False)       # needs acknowledgement
    assert cor.apply_same_item("a-b3", fresh, "a", True).status == "confirmed"


def test_conflict_change_hidden_by_rounding_still_invalidates():
    st, _ = new_store()
    covisible(st, "r8", "a", "b", (0.0, 2.0), (0.5, 2.5))
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    with st.tx() as c:
        c.execute("UPDATE run_intervals SET end_mono=2.001 WHERE interval_id='iv-r8-a'")
    fresh = cor.preview("a", "b")
    assert fresh.conflicts[0]["seen_together_s"] == pv.conflicts[0]["seen_together_s"]   # same rounded display
    assert fresh.binding != pv.binding
    with pytest.raises(StalePreview):
        cor.apply_same_item("a-b", pv, "a", True)


def test_unrelated_correction_neither_invalidates_nor_expands_the_request():
    st, _ = new_store()
    add_item(st, "d")
    add_item(st, "e")
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    cor.apply_same_item("d-e", cor.preview("d", "e"), "d", False)
    out = cor.apply_same_item("a-b", pv, "a", False)
    assert out.status == "confirmed" and cor.fold().members("a") == ["a", "b"]
    again = cor.apply_same_item("a-b", pv, "a", False)            # identical same-id retry after its own effect
    assert again.status == "confirmed" and again.correction_id == out.correction_id and counts(st) == (2, 2)


def test_confirmation_transaction_rechecks_group_binding():
    st, path = new_store()
    add_item(st, "c")
    cor = Corrections(st)
    pv = cor.preview("a", "b")
    cor._confirm_row = lambda conn, row: (_ for _ in ()).throw(sqlite3.OperationalError("process boundary"))
    assert cor.apply_same_item("a-b", pv, "a", False).status == "uncertain"          # pending a-b
    del cor._confirm_row
    cor.apply_same_item("b-c", cor.preview("b", "c"), "b", False)                    # group grows meanwhile
    out = cor.reconcile("a-b")
    assert out.status == "pending" and "NOT applied" in out.message and cor.fold().members("a") == ["a"]
