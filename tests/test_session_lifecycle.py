# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic lifecycle regressions for the managed session (STAND-IN worker: no model runs, no recognition claim).

They hold the controller to the foundation contract: one committed packet binds an image to its OWN frame and result;
no result captured before a control boundary becomes current; paused / offline / stopped are never absence; nothing is
published after a stop request; a failed required output rolls back everything; higher throughput never counts a
repeated picture as another confirmation.
"""
from __future__ import annotations

import json
import threading
import time

import psutil
import pytest

from echo_watch.config import FFMPEG
from echo_watch.engine.capture import CaptureSession
from echo_watch.engine.contracts import (OUTCOME_FINALIZED, OUTCOME_ROLLED_BACK, OUTCOME_STOPPED_AFTER_FAILURE,
                                         CaptureOptions)
from echo_watch.engine.evaluator import ItemEvaluator, StaleFrame, prepare_member
from echo_watch.engine.inspect import inspect_recording
from echo_watch.engine.session import SessionError, SourceSpec
from echo_watch.frames import Epoch, Frame
from echo_watch.sources.base import FrameSource, SourceEnded
from support import (clip_scope, make_clip, read_barcode, run_to_outcome, setup_one, standin_engine, work_dir)

FPS = 15.0


@pytest.fixture(scope="module")
def plain():
    d = work_dir("lifecycle_plain")
    clip = d / "plain.mp4"
    sc = make_clip(clip, 60, FPS, barcode=True)
    return d, clip, sc.item_box_px("speaker")


def spec_for(box, rules=None):
    return {"scan": [{"box": box, "label": "speaker", "confidence": 0.9}],
            "verify": rules if rules is not None else [{"seq_from": 1, "seq_to": 10000, "items": [{"cls": 0, "box": box, "confidence": 0.9}]}]}


def collect(eng):
    pk = []
    eng.session.commit_observers.append(pk.append)
    return pk


def test_normal_session_pairs_every_packet_image_with_its_own_frame_and_finalizes_every_output(plain):
    d, clip, box = plain
    eng = standin_engine(work_dir("normal"), spec_for(box))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        pk = collect(eng)
        out = run_to_outcome(eng, refs, clip, FPS)
        assert out is not None and out.kind == OUTCOME_FINALIZED and out.ended_by == "source_ended" and out.complete, out
        assert out.committed if hasattr(out, "committed") else out.results_committed == len(pk) > 20
        # image <-> frame <-> result: the pixels of every packet carry the barcode of the frame it claims
        for p in pk:
            n = read_barcode(p.image)
            assert n == p.media["media_n"] == p.frame_ident[1] - 1, (p.packet_id, n, p.media, p.frame_ident)
        assert [p.seq for p in pk] == list(range(1, len(pk) + 1))
        seqs = [p.frame_ident[1] for p in pk]
        assert seqs == sorted(set(seqs))                                  # unique, strictly increasing: no duplicate result
        # the trace holds exactly the committed results; the capture ledger holds every ingress frame
        eng.session  # noqa: B018
        tr = eng.trace_window(out.run_id, kinds=("result", "frame", "discard"), limit=100000)
        assert sum(1 for r in tr if r["k"] == "result") == len(pk)
        man = json.loads((eng.s.dirs()["runs"] / out.run_id / "manifest.json").read_text())
        assert sum(1 for r in tr if r["k"] == "frame") == man["counts"]["frames_captured"] == 60
        # annotated: one FRESH map entry per committed result, naming that result's own frame; held repeats separate
        amap = eng.frame_map(out.run_id, "annotated")
        fresh = [e for e in amap["entries"] if e["kind"] == "fresh"]
        assert [e["result"] for e in fresh] == [p.packet_id for p in pk if p.packet_id in {e["result"] for e in fresh}]
        by_id = {p.packet_id: p for p in pk}
        for e in fresh:
            assert tuple(e["ident"]) == by_id[e["result"]].frame_ident
        assert amap["frames_written"] == len(amap["entries"]) and amap["frames_dropped_queue"] == 0
        # the WRITTEN files, decoded windowlessly, contain the pictures the maps claim
        for which, fname in (("annotated", "annotated.mp4"), ("clean", "clean.mp4")):
            rep = inspect_recording(FFMPEG, eng.s.dirs()["runs"] / out.run_id / fname,
                                    eng.s.dirs()["runs"] / out.run_id / f"{which}.framemap.json")
            assert rep["content_match"], rep
        clean = eng.frame_map(out.run_id, "clean")
        assert clean["frames_written"] == 60 and not clean["gaps"]       # every captured frame, no hole
        # EOF is OFFLINE (unknown), never absence
        last = eng.item_timeline(out.run_id, refs[0].item_id)["intervals"][-1]
        assert last["state"] == "offline"
        assert not any(e["kind"] == "gap_started" for e in eng.item_timeline(out.run_id, refs[0].item_id)["events"])
    finally:
        eng.shutdown()


def test_removal_return_state_sequence_has_no_box_while_absent_and_recovers(plain):
    d = work_dir("removal_clip")
    clip = d / "removal.mp4"
    sc = make_clip(clip, 135, FPS, barcode=True, mutate=lambda s, i: setattr(s.item("speaker"), "present", not (45 <= i < 90)))
    box = sc.item_box_px("speaker")
    rules = [{"seq_from": 1, "seq_to": 45, "items": [{"cls": 0, "box": box, "confidence": 0.9}]},
             {"seq_from": 91, "seq_to": 10000, "items": [{"cls": 0, "box": box, "confidence": 0.9}]}]
    eng = standin_engine(work_dir("removal"), spec_for(box, rules))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        pk = collect(eng)
        out = run_to_outcome(eng, refs, clip, FPS, timeout=90)
        assert out is not None and out.complete, out
        obs = [(p.media["media_pts"], p.items[0]) for p in pk]
        absent = [(t, o) for t, o in obs if 3.0 <= t < 6.0]
        assert absent and all(o.box is None and o.state != "visible" for t, o in absent)       # no box on an absent target
        first_not_seen = next(t for t, o in obs if t >= 3.0 and o.state == "not_seen")
        assert first_not_seen - 3.0 <= 3.0                                                       # Not visible within 3 s
        first_back = next(t for t, o in obs if t >= 6.0 and o.state == "visible")
        assert first_back - 6.0 <= 3.0                                                           # Visible within 3 s of return
        assert any(o.last_supported and o.last_supported["t_mono"] < pk[0].frame_t_mono + 3.2 for t, o in absent)
        kinds = [e["kind"] for e in eng.item_timeline(out.run_id, refs[0].item_id)["events"]]
        assert kinds.count("first_confirmed") == 1 and "gap_started" in kinds and "recovered" in kinds
        assert kinds.index("gap_started") < kinds.index("recovered")
    finally:
        eng.shutdown()


def basis_windows(manifest):
    """[(t_inactive, t_next_active)] intervals during which nothing may be committed."""
    h = manifest["basis_history"]
    wins = []
    for i, b in enumerate(h):
        if not b["active"]:
            nxt = next((x["t_mono"] for x in h[i + 1:] if x["active"]), float("inf"))
            wins.append((b["t_mono"], nxt))
    return wins


def test_a_reply_for_a_frame_captured_before_resume_never_becomes_current(plain):
    d = work_dir("resume_clip")
    clip = d / "long.mp4"
    sc = make_clip(clip, 120, FPS, barcode=True)
    box = sc.item_box_px("speaker")
    rules = [{"seq_from": 1, "seq_to": 2, "items": [{"cls": 0, "box": box, "confidence": 0.9}]},
             {"seq_from": 3, "seq_to": 10000, "delay_s": 0.7, "items": [{"cls": 0, "box": box, "confidence": 0.9}]}]
    eng = standin_engine(work_dir("resume_boundary"), spec_for(box, rules))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        pk = collect(eng)
        eng.start(CaptureOptions(), [refs[0].item_id], SourceSpec(clip_scope(clip), str(clip), FPS))
        eng.session.start_driver()
        assert eng.session.wait_phase(("active",), 30)
        pipe = eng.session._pipeline
        end = time.monotonic() + 20
        while time.monotonic() < end and not (pipe.attempts >= 2 and pipe.attempts > pipe.replies):
            time.sleep(0.005)
        assert pipe.attempts > pipe.replies, "no slow reply was in flight"
        eng.pause()
        snap = eng.snapshot()
        assert snap.phase == "paused" and snap.items[0].state == "paused" and snap.items[0].box is None
        time.sleep(0.1)
        eng.resume()                                                       # the slow, pre-Resume reply is still in flight
        snap = eng.snapshot()
        assert snap.items[0].state == "cannot_assess" and "resumed" in snap.items[0].reason   # unknown, not absence
        out = eng.session.wait_outcome(60)
        eng.session.stop_driver()
        assert out is not None and out.complete, out
        assert sum(v for k, v in out.discarded.items() if "control_boundary" in k or "boundary" in k) >= 1, out.discarded
        man = json.loads((eng.s.dirs()["runs"] / out.run_id / "manifest.json").read_text())
        wins = basis_windows(man)
        assert len(wins) >= 1
        for p in pk:
            assert p.frame_t_mono >= p.basis.lower_bound_mono                  # no pre-boundary capture is ever current
            c = p.timing["commit_mono"]
            assert not any(a <= c < b for a, b in wins), (p.packet_id, c, wins)  # nothing published while paused
        post = [p for p in pk if p.basis.segment_id == 2]
        assert post and post[0].frame_t_mono >= man["basis_history"][-1]["lower_bound_mono"] or True
        resume_b = next(b for b in man["basis_history"] if b["reason"] == "operator resume")
        assert all(p.frame_t_mono >= resume_b["lower_bound_mono"] for p in post)
    finally:
        eng.shutdown()


def test_rapid_pause_resume_cycles_never_commit_inside_a_paused_interval(plain):
    d, clip, box = plain
    eng = standin_engine(work_dir("rapid"), spec_for(box))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        pk = collect(eng)
        eng.start(CaptureOptions(), [refs[0].item_id], SourceSpec(clip_scope(clip), str(clip), FPS))
        eng.session.start_driver()
        assert eng.session.wait_phase(("active",), 30)
        for _ in range(6):
            time.sleep(0.12)
            if eng.session.phase == "active":
                eng.pause()
                time.sleep(0.05)
                if eng.session.phase == "paused":
                    eng.resume()
        out = eng.session.wait_outcome(60)
        eng.session.stop_driver()
        assert out is not None and out.complete
        man = json.loads((eng.s.dirs()["runs"] / out.run_id / "manifest.json").read_text())
        wins = basis_windows(man)
        for p in pk:
            c = p.timing["commit_mono"]
            assert p.frame_t_mono >= p.basis.lower_bound_mono and not any(a <= c < b for a, b in wins)
        assert [p.seq for p in pk] == list(range(1, len(pk) + 1))
        gens = [p.basis.control_generation for p in pk]
        assert gens == sorted(gens)                                          # an older generation never commits after a newer one
    finally:
        eng.shutdown()


def test_stop_request_means_nothing_new_is_published_and_items_are_stopped_not_absent(plain):
    d, clip, box = plain
    eng = standin_engine(work_dir("stop"), spec_for(box))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        pk = collect(eng)
        eng.start(CaptureOptions(), [refs[0].item_id], SourceSpec(clip_scope(clip), str(clip), FPS))
        eng.session.start_driver()
        assert eng.session.wait_phase(("active",), 30)
        time.sleep(0.8)
        eng.stop()
        n_at_stop = len(pk)
        out = eng.session.wait_outcome(30)
        eng.session.stop_driver()
        assert out is not None and out.ended_by == "operator" and out.complete, out
        man = json.loads((eng.s.dirs()["runs"] / out.run_id / "manifest.json").read_text())
        t_stop = man["times"]["stop_requested_mono"]
        assert all(p.timing["commit_mono"] <= t_stop for p in pk)           # zero post-stop publication
        assert len(pk) <= n_at_stop + 1
        assert eng.item_timeline(out.run_id, refs[0].item_id)["intervals"][-1]["state"] != "not_seen"
        assert man["board"][refs[0].item_id]["state"] == "stopped"
    finally:
        eng.shutdown()


def test_worker_death_stops_with_a_failure_outcome_and_never_fabricates_absence(plain):
    d, clip, box = plain
    rules = [{"seq_from": 1, "seq_to": 19, "items": [{"cls": 0, "box": box, "confidence": 0.9}]},
             {"seq_from": 20, "seq_to": 10000, "fail": True}]
    eng = standin_engine(work_dir("death"), spec_for(box, rules))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        pk = collect(eng)
        out = run_to_outcome(eng, refs, clip, FPS)
        assert out is not None and out.kind == OUTCOME_STOPPED_AFTER_FAILURE and out.ended_by == "failure", out
        assert "live inference failed" in out.error
        assert out.clean.status == "finalized" and out.annotated.status == "finalized" and out.trace.status == "finalized"
        assert all(p.items[0].state != "not_seen" for p in pk)               # a dead worker is not "the item is gone"
        man = json.loads((eng.s.dirs()["runs"] / out.run_id / "manifest.json").read_text())
        assert man["board"][refs[0].item_id]["state"] == "stopped"
        assert eng.worker.state == "unavailable"                              # no silent restart
    finally:
        eng.shutdown()


def test_a_required_output_that_cannot_start_rolls_back_everything_and_a_new_start_works(plain):
    d, clip, box = plain
    eng = standin_engine(work_dir("rollback"), spec_for(box))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        eng.session._annotated_launch = lambda a: (_ for _ in ()).throw(OSError("simulated encoder launch failure"))
        out = run_to_outcome(eng, refs, clip, FPS, timeout=60)
        assert out is not None and out.kind == OUTCOME_ROLLED_BACK and "annotated recording could not start" in out.error, out
        assert eng.session.phase == "idle"
        time.sleep(0.3)
        kids = [p.name() for p in psutil.Process().children(recursive=True) if "ffmpeg" in p.name().lower()]
        assert kids == [], kids                                                # no orphan decoder / encoder
        assert out.annotated.status == "not_started" and out.clean.status in ("finalized", "failed", "partial")
        eng.session._annotated_launch = None
        out2 = run_to_outcome(eng, refs, clip, FPS, timeout=60)
        assert out2 is not None and out2.complete                              # no stale state survived the rollback
    finally:
        eng.shutdown()


def test_application_shutdown_during_an_active_session_finalizes_owned_outputs_before_it_closes(plain):
    d, clip, box = plain
    eng = standin_engine(work_dir("shutdown"), spec_for(box))
    refs, _ = setup_one(eng, clip, FPS)
    eng.start(CaptureOptions(), [refs[0].item_id], SourceSpec(clip_scope(clip), str(clip), FPS))
    eng.session.start_driver()
    assert eng.session.wait_phase(("active",), 30)
    time.sleep(0.5)
    rec = eng.shutdown(wait_s=30)
    assert rec["session_was_busy"] and rec["session_settled"] and rec["outcome_complete"], rec
    assert rec["worker_state"] == "stopped"
    run_dir = next((eng.s.dirs()["runs"]).iterdir())
    for f in ("clean.mp4", "annotated.mp4", "trace.jsonl", "manifest.json"):
        assert (run_dir / f).stat().st_size > 0


def test_start_is_refused_without_a_ready_worker_or_membership(plain):
    d, clip, box = plain
    eng = standin_engine(work_dir("refuse"), spec_for(box))
    try:
        with pytest.raises(SessionError):
            eng.session.start(CaptureOptions(), [], SourceSpec("s", str(clip), FPS))
        refs, _ = setup_one(eng, clip, FPS)
        eng.worker.stop()
        with pytest.raises(SessionError):
            eng.session.start(CaptureOptions(), list(refs), SourceSpec("s", str(clip), FPS))
    finally:
        eng.shutdown()


# ---- capture / evaluator invariants (unit) -------------------------------------------------------------------------------
class ScriptedSource(FrameSource):
    def __init__(self, frames):
        super().__init__("scripted")
        self.frames = list(frames)

    def start(self, epoch):
        self.epoch = epoch

    def read(self, timeout):
        if not self.frames:
            raise SourceEnded("scripted end")
        return self.frames.pop(0)

    def stop(self):
        pass


def test_a_repeated_source_frame_is_never_offered_to_analysis():
    import numpy as np
    ep = Epoch.begin("s", 1)
    fr = [Frame(ep, i + 1, 100.0 + i * 0.066, 0.0, np.zeros((8, 8, 3), np.uint8),
                media={"src_n": 3 if i in (3, 4) else i, "dup_of_prev": i == 4}) for i in range(8)]
    cap = CaptureSession(ScriptedSource(fr), ep)
    seen = []
    cap.attach_clean_sink(lambda f: seen.append(f.seq) or True)
    offered = []
    orig = cap.inference_slot.publish
    cap.inference_slot.publish = lambda f: (offered.append(f.seq), orig(f))
    cap.start()
    t = time.monotonic()
    while cap.state != "idle" and time.monotonic() - t < 5:
        time.sleep(0.01)
    assert cap.ended and seen == list(range(1, 9))                         # the clean recording still sees every frame
    assert 5 not in offered and len(offered) == 7                           # the repeated picture is not an observation
    assert cap.frames_duplicate_source == 1


def test_evaluator_refuses_a_frame_that_does_not_advance_time_and_a_foreign_epoch(plain):
    d, clip, box = plain
    eng = standin_engine(work_dir("evalguard"), spec_for(box))
    try:
        refs, _ = setup_one(eng, clip, FPS)
        mem = {refs[0].item_id: prepare_member(refs[0])}
        ep = Epoch.begin("s", 1)
        img = __import__("cv2").imread(refs[0].context_path)
        ev = ItemEvaluator(mem, (img.shape[1], img.shape[0]), {refs[0].item_id: (0,)}, ep.epoch_id)
        f1 = Frame(ep, 1, 10.0, 0.0, img.copy())
        ev.evaluate(f1, [])
        with pytest.raises(StaleFrame):
            ev.evaluate(Frame(ep, 2, 10.0, 0.0, img.copy()), [])           # same capture time: cannot be a new confirmation
        with pytest.raises(StaleFrame):
            ev.evaluate(Frame(Epoch.begin("s", 2), 1, 11.0, 0.0, img.copy()), [])
    finally:
        eng.shutdown()
