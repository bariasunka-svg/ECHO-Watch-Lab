# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-05 matrix 1-3: repeat-run enrollment, source compatibility and immutable enrollment commands (GUI-free).

Real Library / Store / controller / ItemEngine.start boundary on synthetic stores; a fake scan worker and session;
numpy codec stubs (no media codec). These establish enrollment and scope semantics, not recognition.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from echo_watch.engine import compat
from echo_watch.engine.library import LibraryError, request_digest
from lab05_support import FakeEngine, Rig, codec_stubs, frame, settle
from support import work_dir

S1 = "file-sha256:" + "a" * 64
S2 = "file-sha256:" + "b" * 64


@pytest.fixture()
def lib(monkeypatch):
    codec_stubs(monkeypatch)
    eng = FakeEngine(work_dir(f"lab05-lib-{time.time_ns()}"))
    yield eng
    eng.store.close()


def scan(eng, scope, seed=0):
    r = eng.library.scan(frame(scope, seed), scope, eng.worker)
    assert r.status == "succeeded"
    return r, {p.label: p for p in r.proposals}


def counts(eng):
    q = eng.store.one
    return {"items": q("SELECT COUNT(*) n FROM items")["n"], "refs": q("SELECT COUNT(*) n FROM item_references")["n"],
            "remember": q("SELECT COUNT(*) n FROM decisions WHERE kind='remember'")["n"]}


# ---- library: repeat run, separate identity, scope isolation -------------------------------------------------------
def test_rescan_same_source_explicit_reuse_keeps_ids_and_versions_without_a_new_profile(lib):
    r1, p1 = scan(lib, S1)
    first = lib.library.enroll("op-1", r1.scan_id, [{"candidate_id": p1["phone"].candidate_id, "name": "Phone A01"}], [])
    assert first.status == "confirmed" and first.new_count == 1
    item = first.items[0]
    r2, p2 = scan(lib, S1)
    assert p2["phone"].overlaps_item == item.item_id            # a compatible suggestion, nothing more
    before = counts(lib)
    o = lib.library.enroll("op-2", r2.scan_id, [], [{"item_id": item.item_id, "ref_version": 1,
                                                     "candidate_id": p2["phone"].candidate_id}])
    assert o.status == "confirmed" and o.reused_count == 1 and o.new_count == 0
    assert [(i.item_id, i.ref_version, i.disposition) for i in o.items] == [(item.item_id, 1, "reused")]
    after = counts(lib)
    assert after["items"] == before["items"] and after["refs"] == before["refs"]      # memory unchanged
    assert after["remember"] == before["remember"] + 1                                  # the decision is durable
    assert "Saved, not monitoring" in o.message and "Visible" not in o.message


def test_distinct_item_at_old_coordinates_is_allowed_and_is_a_separate_identity(lib):
    r1, p1 = scan(lib, S1)
    a = lib.library.enroll("op-a", r1.scan_id, [{"candidate_id": p1["phone"].candidate_id, "name": "Phone A"}], [])
    r2, p2 = scan(lib, S1)
    b = lib.library.enroll("op-b", r2.scan_id, [{"candidate_id": p2["phone"].candidate_id, "name": "Phone B"}], [])
    assert a.confirmed and b.confirmed and a.items[0].item_id != b.items[0].item_id
    assert lib.library.item_ref(b.items[0].item_id).box == lib.library.item_ref(a.items[0].item_id).box


def test_other_recording_of_identical_size_is_isolated_for_hints_reuse_replacement_and_start(lib):
    r1, p1 = scan(lib, S1)
    item = lib.library.enroll("op-1", r1.scan_id, [{"candidate_id": p1["phone"].candidate_id, "name": "Phone"}],
                              []).items[0].item_id
    r2, p2 = scan(lib, S2, seed=9)
    assert all(p.overlaps_item is None for p in p2.values())                  # no resolution-only hint
    ok, bad = lib.library.saved_for_scope(lib.library.scan_scope(r2.scan_id))
    assert ok == [] and [b["compat_code"] for b in bad] == ["other_source"]
    o = lib.library.enroll("op-x", r2.scan_id, [], [{"item_id": item, "ref_version": 1}])
    assert o.status == "rejected" and "different recording" in o.message
    with pytest.raises(LibraryError, match="different recording"):
        lib.library.update_reference("op-r", item, r2.scan_id, p2["phone"].candidate_id)
    fresh = lib.library.enroll("op-f", r2.scan_id, [{"candidate_id": p2["phone"].candidate_id, "name": "Phone 2"}], [])
    assert fresh.confirmed and fresh.items[0].item_id != item
    from echo_watch.engine.contracts import CaptureOptions
    from echo_watch.engine.session import SourceSpec
    with pytest.raises(LibraryError, match="different recording"):
        lib.start(CaptureOptions(), [(item, 1)], SourceSpec(S2, "b.mp4", 15.0, size=(640, 360)))
    assert lib.session.started == []


def test_unknown_legacy_scope_is_readable_but_never_compatible_even_when_equal(lib):
    r1, p1 = scan(lib, "setup")
    legacy = lib.library.enroll("op-l", r1.scan_id, [{"candidate_id": p1["mug"].candidate_id, "name": "Old mug"}], [])
    assert legacy.confirmed                                     # fresh enrollment still possible
    r2, p2 = scan(lib, "setup")
    assert all(p.overlaps_item is None for p in p2.values())
    ok, bad = lib.library.saved_for_scope(compat.Scope("setup", (640, 360), "pf-test-model"))
    assert ok == [] and bad[0]["compat_code"] == "unscoped_source"
    ok, bad = lib.library.saved_for_scope(compat.Scope(S1, (640, 360), "pf-test-model"))
    assert ok == [] and bad[0]["compat_code"] == "legacy_scope" and "history stays readable" in bad[0]["compat_reason"]
    assert any(p["item_id"] == legacy.items[0].item_id for p in lib.library.profiles())     # still in the Library


def test_size_and_model_identity_are_part_of_compatibility():
    sc = compat.Scope(S1, (640, 360), "m1")
    assert compat.check(S1, (640, 360), "m1", sc).ok
    assert compat.check(S1, (1280, 720), "m1", sc).code == "size"
    assert compat.check(S1, (640, 360), "m2", sc).code == "model"
    assert compat.check(S1, (640, 360), None, sc).code == "model"
    assert compat.check(S2, (640, 360), "m1", sc).code == "other_source"


# ---- library: one immutable command, all-or-nothing, truthful outcomes ----------------------------------------------
def _mixed(lib):
    r1, p1 = scan(lib, S1)
    saved = lib.library.enroll("op-s", r1.scan_id, [{"candidate_id": p1["mug"].candidate_id, "name": "Mug"}], [])
    r2, p2 = scan(lib, S1)
    new = [{"candidate_id": p2["phone"].candidate_id, "name": "Phone"}]
    reuse = [{"item_id": saved.items[0].item_id, "ref_version": 1, "candidate_id": p2["mug"].candidate_id}]
    return r2, new, reuse, saved.items[0].item_id


def test_mixed_batch_write_failure_is_all_or_nothing_and_retry_saves_once(lib, monkeypatch):
    r2, new, reuse, mug = _mixed(lib)
    before = counts(lib)
    real = type(lib.library)._write_enrollment

    def boom(self, c, *a, **k):
        real(self, c, *a, **k)                                  # rows inserted inside the transaction...
        raise sqlite3.OperationalError("disk I/O error")         # ...then the commit fails -> rolled back
    monkeypatch.setattr(type(lib.library), "_write_enrollment", boom)
    o = lib.library.enroll("op-m", r2.scan_id, new, reuse)
    assert o.status == "not_saved" and "wrote nothing" in o.message
    assert counts(lib) == before                                # no partial new item, no decision
    assert len(list(lib.s.dirs()["assets"].glob("*/ref_v1.png"))) == 1     # only the earlier mug crop; no orphan
    monkeypatch.setattr(type(lib.library), "_write_enrollment", real)
    o2 = lib.library.enroll("op-m", r2.scan_id, new, reuse)    # retry: the SAME operation id
    assert o2.confirmed and o2.new_count == 1 and o2.reused_count == 1
    assert counts(lib)["items"] == before["items"] + 1 and counts(lib)["remember"] == before["remember"] + 1
    o3 = lib.library.enroll("op-m", r2.scan_id, new, reuse)    # repeated click after success: no duplicate
    assert o3.confirmed and [i.item_id for i in o3.items] == [i.item_id for i in o2.items]
    assert counts(lib)["items"] == before["items"] + 1


def test_failed_readback_after_commit_is_uncertain_then_reconciles_the_same_operation(lib, monkeypatch):
    r2, new, reuse, _ = _mixed(lib)
    real = type(lib.library)._read_enrollment
    calls = {"n": 0}

    def flaky(self, op):
        calls["n"] += 1
        if calls["n"] == 2:                                     # the post-commit readback fails once
            raise sqlite3.OperationalError("database is locked")
        return real(self, op)
    monkeypatch.setattr(type(lib.library), "_read_enrollment", flaky)
    o = lib.library.enroll("op-u", r2.scan_id, new, reuse)
    assert o.status == "uncertain" and "committed" in o.message and "SAME operation" in o.message
    n_items = counts(lib)["items"]
    rec = lib.library.enroll_outcome("op-u", o.digest)
    assert rec.confirmed and rec.new_count == 1 and rec.reused_count == 1
    again = lib.library.enroll("op-u", r2.scan_id, new, reuse)
    assert again.confirmed and counts(lib)["items"] == n_items      # reconcile never creates anything


def test_rollback_when_another_attempt_saved_the_same_operation_reports_it_confirmed(lib, monkeypatch):
    r2, new, reuse, _ = _mixed(lib)
    first = lib.library.enroll("op-race", r2.scan_id, new, reuse)
    real = type(lib.library)._read_enrollment
    seen = {"n": 0}

    def late(self, op):                                         # the first check misses the concurrent save
        seen["n"] += 1
        return None if seen["n"] == 1 else real(self, op)
    monkeypatch.setattr(type(lib.library), "_read_enrollment", late)
    o = lib.library.enroll("op-race", r2.scan_id, new, reuse)   # its decision insert hits the UNIQUE op id
    assert o.confirmed and "already saved" in o.message
    assert [i.item_id for i in o.items] == [i.item_id for i in first.items]
    assert counts(lib)["remember"] == 2                          # the mug save and the first attempt; no second record


def test_operation_id_is_bound_to_its_request(lib):
    r2, new, reuse, _ = _mixed(lib)
    lib.library.enroll("op-b", r2.scan_id, new, reuse)
    before = counts(lib)
    o = lib.library.enroll("op-b", r2.scan_id, new, [])
    assert o.status == "conflict" and counts(lib) == before
    assert request_digest(r2.scan_id, new, reuse) != request_digest(r2.scan_id, new, [])


def test_invalid_selections_are_rejected_without_writes(lib):
    r2, new, reuse, mug = _mixed(lib)
    before = counts(lib)
    cid = new[0]["candidate_id"]
    bad = [([], []),
           ([{"candidate_id": cid, "name": ""}], []),
           ([{"candidate_id": "cand_nope", "name": "x"}], []),
           ([{"candidate_id": cid, "name": "x", "box": [630, 350, 5, 5]}], []),
           ([{"candidate_id": cid, "name": "x"}, {"candidate_id": cid, "name": "y"}], []),
           ([], [reuse[0], dict(reuse[0], candidate_id=None)]),
           ([{"candidate_id": reuse[0]["candidate_id"], "name": "x"}], reuse),
           ([], [dict(reuse[0], ref_version=2)]),
           ([], [{"item_id": "item_missing", "ref_version": 1}])]
    for i, (n, r) in enumerate(bad):
        o = lib.library.enroll(f"op-bad-{i}", r2.scan_id, n, r)
        assert o.status == "rejected", (i, o)
    assert counts(lib) == before


def test_stale_saved_reference_and_explicit_replacement_stay_distinct_from_a_new_identity(lib):
    r1, p1 = scan(lib, S1)
    item = lib.library.enroll("op-1", r1.scan_id, [{"candidate_id": p1["phone"].candidate_id, "name": "Phone"}],
                              []).items[0].item_id
    r2, p2 = scan(lib, S1)
    chosen_v = lib.library.saved_for_scope(lib.library.scan_scope(r2.scan_id))[0][0]["current_ref_version"]
    v2 = lib.library.update_reference("op-rep", item, r2.scan_id, p2["phone"].candidate_id)
    assert v2.item_id == item and v2.ref_version == 2 and lib.library.item_ref(item, 1).ref_version == 1
    o = lib.library.enroll("op-stale", r2.scan_id, [], [{"item_id": item, "ref_version": chosen_v}])
    assert o.status == "rejected" and "changed to v2" in o.message
    assert lib.library.enroll("op-ok", r2.scan_id, [], [{"item_id": item, "ref_version": 2}]).confirmed


# ---- controller: the full repeat-run and new-video journeys ---------------------------------------------------------
@pytest.fixture()
def rig(monkeypatch):
    r = Rig("lab05-ctl", monkeypatch)
    yield r
    r.close()


def test_matrix1_same_source_repeat_run_reuses_ids_versions_and_membership_only_explicitly(rig):
    c = rig.ctl
    rig.discover(rig.a)
    for lab in ("phone", "mug"):
        c.toggle_candidate(rig.cid(lab), True)
    assert {x.disposition for x in c.candidates if x.checked} == {"new"}
    rig.remember()
    assert c.enroll_state == "confirmed" and c.view_step == "configure"
    first = dict(c.member_versions)
    run = rig.start()
    assert sorted(run["refs"]) == sorted(first.items())          # the exact captured versions reach the session
    assert run["allowance_s"] == 60 and run["setup"]["items"] == [list(x) for x in run["refs"]]
    rig.stop()
    assert c.tray == [] and c.member_versions == {} and c.view_step == "monitor"
    c.refresh_library()
    c.refresh_setups()
    assert settle(c, lambda: not c.tasks.busy("library") and not c.tasks.busy("setups"))
    assert c.tray == []                                         # nothing refills membership implicitly
    assert c.new_setup() and c.view_step == "discover"
    rig.discover()                                              # a fresh discovery of the SAME recording
    assert {s["item_id"] for s in c.saved_items} == set(first)
    for lab in ("phone", "mug"):
        c.toggle_candidate(rig.cid(lab), True)
    assert {x.disposition for x in c.candidates if x.checked} == {"choose"}   # overlap suggests; operator decides
    assert not c.remember() and "choose Use saved item" in c.messages[-1].text
    for lab in ("phone", "mug"):
        cnd = c.candidate(rig.cid(lab))
        assert c.set_disposition(cnd.candidate_id, "reuse", cnd.matches[0]["item_id"])
    n_profiles = len(rig.eng.library.profiles())
    rig.remember()
    assert c.enroll_state == "confirmed" and dict(c.member_versions) == first
    assert len(rig.eng.library.profiles()) == n_profiles and "0 new, 2 reused" in c.enroll_message
    run2 = rig.start()
    assert sorted(run2["refs"]) == sorted(first.items())


def test_reuse_this_setup_after_stop_returns_to_configure_with_the_same_ids_and_starts_nothing(rig):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    rig.remember()
    c.set_allowance(45)
    run = rig.start()
    rig.stop(kind=("finalized", "source_ended"))
    assert c.finished["ended_by"] == "source_ended" and c.tray == []
    n = len(rig.eng.session.started)
    assert c.reuse_setup()
    assert settle(c, lambda: c.setup_status == "saved" and c.tray)
    assert c.view_step == "configure" and c.pending_allowance == 45
    assert [(i, c.member_versions[i]) for i in c.tray] == run["refs"]
    assert len(rig.eng.session.started) == n                    # reuse never starts processing


def test_new_video_of_identical_size_enrolls_fresh_items_and_cannot_inherit_old_identities(rig):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    rig.remember()
    old = set(c.tray)
    rig.discover(rig.b)                                         # source change clears membership/drafts
    assert c.tray == [] and c.saved_items == [] and {b["compat_code"] for b in c.saved_blocked} == {"other_source"}
    c.toggle_candidate(rig.cid("phone"), True)
    assert c.candidate(rig.cid("phone")).disposition == "new"   # same coordinates, no overlap suggestion
    rig.remember()
    assert c.enroll_state == "confirmed" and set(c.tray).isdisjoint(old)
    c.refresh_library()
    assert settle(c, lambda: len(c.library) == 2)
    assert not c.add_to_run(next(iter(old)))                    # an old identity cannot be added on video B
    assert "different recording" in c.messages[-1].text
    run = rig.start()
    assert set(i for i, _ in run["refs"]).isdisjoint(old)


def test_same_bytes_at_another_path_keeps_saved_items_compatible(rig):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("mug"), True)
    rig.remember()
    item = c.tray[0]
    rig.discover(rig.a_copy)
    assert [s["item_id"] for s in c.saved_items] == [item]
    assert c.toggle_saved(item) and c.selection_counts()["reused"] == 1


def test_source_switch_while_saving_reports_the_durable_old_outcome_without_attaching_it(rig, monkeypatch):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    gate = threading.Event()
    real = rig.eng.library.enroll

    def slow(*a, **k):
        gate.wait(5)
        return real(*a, **k)
    monkeypatch.setattr(rig.eng.library, "enroll", slow)
    assert c.remember()
    rig.prepare(rig.b)                                          # the operator switches video while saving
    assert not c.remember_pending                               # the new recording's Select is not blocked
    gate.set()
    assert settle(c, lambda: any(d["kind"] == "remember" for d in c.discarded_completions))
    assert c.tray == [] and c.enroll_state == "idle"
    assert any("earlier save for the previous recording completed" in m.text for m in c.messages)
    assert any(d["kind"] == "remember" for d in c.discarded_completions)
    assert settle(c, lambda: len(c.library) == 1)                # durable in the Library (old scope)


def test_uncertain_save_blocks_new_operations_and_reconciles_with_the_same_id(rig, monkeypatch):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    lib_cls = type(rig.eng.library)
    real = lib_cls._read_enrollment
    calls = {"n": 0}

    def flaky(self, op):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return real(self, op)
    monkeypatch.setattr(lib_cls, "_read_enrollment", flaky)
    rig.remember()
    assert c.enroll_state == "uncertain" and c.tray == [] and c.view_step == "select"
    op = c.enroll_cmd.op_id
    c.toggle_candidate(rig.cid("mug"), True)                    # an edit cannot bypass reconciliation
    rig.remember()
    assert c.enroll_cmd.op_id == op and c.enroll_state == "confirmed"
    assert len(c.tray) == 1 and len(rig.eng.library.profiles()) == 1


def test_a_failed_screen_update_is_distinguished_from_a_failed_save(rig, monkeypatch):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    monkeypatch.setattr(type(c), "_adopt_enrollment", lambda self, out: (_ for _ in ()).throw(KeyError("view")))
    rig.remember()
    assert c.ui_failures and "saved in the Library" in c.messages[-1].text and "could not be updated" in c.messages[-1].text
    assert len(rig.eng.library.profiles()) == 1


def test_repeated_clicks_and_retry_after_a_failed_write_use_one_operation(rig, monkeypatch):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    lib_cls = type(rig.eng.library)
    real = lib_cls._write_enrollment
    fail = {"once": True}

    def once(self, conn, *a, **k):
        if fail.pop("once", False):
            raise sqlite3.OperationalError("disk full")
        return real(self, conn, *a, **k)
    monkeypatch.setattr(lib_cls, "_write_enrollment", once)
    assert c.remember()
    assert not c.remember()                                     # refused while pending
    assert settle(c, lambda: not c.remember_pending)
    assert c.enroll_state == "not_saved" and c.tray == [] and c.candidate(rig.cid("phone")).checked
    op = c.enroll_cmd.op_id
    rig.remember()
    assert c.enroll_state == "confirmed" and c.enroll_cmd.op_id == op
    assert rig.eng.store.one("SELECT COUNT(*) n FROM decisions WHERE op_id=?", (op,))["n"] == 1


def test_removing_a_selection_removes_membership_only(rig):
    c = rig.ctl
    rig.discover(rig.a)
    c.toggle_candidate(rig.cid("phone"), True)
    rig.remember()
    item = c.tray[0]
    assert c.remove_from_run(item) and c.tray == []
    assert any(p["item_id"] == item for p in rig.eng.library.profiles())
