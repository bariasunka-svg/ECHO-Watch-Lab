# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 matrix 3 and 6 (session part): the REAL YoloeSession, TraceWriter and production YoloeCore (in-process,
spy detector, scripted tracker) over a scripted ordered source. Ordered processing, pacing and honest slowdown, VFR,
gaps, Pause/Resume, Stop, EOF drain, restart epochs, commit-time rejection with exact restore or acknowledged epoch
reset, failures and evidence fidelity/caps. No model, GPU, FFmpeg, Tk or owner data."""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections import namedtuple
from pathlib import Path

import pytest

from echo_watch.config import Limits
from echo_watch.yoloe.session import YoloeError, YoloeSession
from lab06_support import InProcessClient, ScriptSource, media_frames, read_trace, source_file, wait_for, work_dir

Usage = namedtuple("Usage", "total used free")


def make(name, frames, client=None, **kw):
    d = work_dir(name)
    info = source_file(d)
    src = ScriptSource(frames)
    client = client or InProcessClient()
    s = YoloeSession(runs_dir=d / "yoloe-runs", client=client, source_factory=lambda path: src, **kw)
    return s, client, src, info


def run_to_end(s, info, timeout=20.0):
    s.start(info)
    assert wait_for(lambda: s.phase == "idle" and s.last_outcome is not None, timeout)
    return s.last_outcome


def frames_of(trace):
    return [r for r in trace if r["k"] == "frame"]


def test_every_frame_once_in_order_eof_drains_and_evidence_keeps_every_detection():
    s, client, src, info = make("lab06-eof", media_frames(12, fps=200))
    out = run_to_end(s, info)
    assert out.status == "completed" and out.complete and out.counts["processed"] == 12 == out.counts["published"]
    det = client.detect_calls()
    assert [c[2] for c in det] == [str(k) for k in range(1, 13)]
    assert [c[3] for c in det] == [None] + [str(k) for k in range(1, 12)]          # accept chain k-1 -> k
    assert client.calls[-1] == ("settle", det[0][1], "12", "accept")             # EOF: last result accepted
    pk = s.latest_packet()
    assert pk.seq == 12 and pk.media["out_n"] == 11 and int(pk.image[0, 0, 0]) == 11 # its OWN pixels
    tr = read_trace(out.run_dir)
    fr = frames_of(tr)
    assert [r["media"]["src_n"] for r in fr] == list(range(12)) and all(r["disposition"] == "published" for r in fr)
    for r in fr:                                                      # raw detector output, never the tracked subset
        assert [d["i"] for d in r["detections"]] == [0, 1] and r["detections"][1]["score"] == 0.15
        assert r["detections"][1]["name"] == "bottle" and "1" not in r["assoc"] and r["counts"]["untracked"] == 1
        assert r["sig"] and r["size"] == [64, 48] and r["tracker"]["rows"]
    man = json.loads((Path(out.run_dir) / "manifest.json").read_text("utf-8"))
    assert man["schema"] == "echo-watch-lab/yoloe-run/1" and man["outcome"]["status"] == "completed"
    assert man["source"]["sha256"] == info.sha256 and man["source"]["scope"].startswith("file-sha256:")
    assert man["profile"]["detector"] == {"imgsz": 640, "conf": 0.1, "iou": 0.7, "max_det": 300}
    assert man["profile"]["tracker"]["with_reid"] is False and man["worker"]["end2end"] is True
    assert man["coordinate_transform"]["processing_to_native"] == [2.0, 2.0]
    assert man["trace"]["complete"] and man["trace"]["lines"] == len(tr) and man["counts"]["processed"] == 12
    assert man["tracking"]["track_buffer_unit"].startswith("tracker updates")
    assert tr[0]["k"] == "start" and tr[-1]["k"] == "end" and any(r["k"] == "epoch" for r in tr)


def test_source_speed_pacing_and_vfr_timestamps_are_followed():
    pts = [0.0, 0.05, 0.30, 0.32, 0.60]
    s, _c, _src, info = make("lab06-vfr", media_frames(5, pts=pts))
    t0 = time.monotonic()
    out = run_to_end(s, info)
    took = time.monotonic() - t0
    assert out.status == "completed" and 0.55 <= took < 2.5
    commits = [r["mono"] for r in frames_of(read_trace(out.run_dir))]
    gaps = [b - a for a, b in zip(commits, commits[1:])]
    assert gaps[1] > 0.2 and gaps[3] > 0.2 and gaps[2] < 0.15        # the irregular source timing, not a fixed rate


def test_slow_inference_slows_playback_but_skips_nothing():
    s, client, _src, info = make("lab06-slow", media_frames(10, fps=50), InProcessClient(delay_s=0.05))
    out = run_to_end(s, info)
    assert out.counts["processed"] == 10 and [c[2] for c in client.detect_calls()] == [str(k) for k in range(1, 11)]
    sp = s.snapshot()["playback_speed"]
    assert sp is not None and sp < 0.6                                 # ~0.4x of source speed, reported as such


def test_gaps_are_recorded_with_the_trackers_actual_handling():
    s, _c, _src, info = make("lab06-gap", media_frames(6, fps=200, gaps={3: 2}))
    out = run_to_end(s, info)
    tr = read_trace(out.run_dir)
    gap = [r for r in tr if r["k"] == "gap"]
    assert len(gap) == 1 and gap[0]["missing_source_frames"] == 2 and "ONE update" in gap[0]["tracker_handling"]
    assert out.counts["gaps"] == 1 and out.counts["gap_frames"] == 2


def test_pause_freezes_source_and_tracker_and_resume_continues_without_catch_up():
    s, client, src, info = make("lab06-pause", media_frames(40, fps=20))
    s.start(info)
    assert wait_for(lambda: s.counts["published"] >= 4, 5)
    assert s.pause()
    time.sleep(0.15)
    reads, calls = src.reads, len(client.detect_calls())
    gen = s.control_gen
    time.sleep(0.5)
    assert src.reads == reads and len(client.detect_calls()) == calls      # nothing consumed or tracked while paused
    assert s.phase == "paused" and not s.pause()
    assert s.resume() and s.control_gen == gen + 1
    t_res = time.monotonic()
    assert wait_for(lambda: s.counts["published"] >= calls + 4, 5)
    took = time.monotonic() - t_res
    assert took >= 0.12                                    # 4 more frames at 20 fps are paced, not burst out
    keys = [int(c[2]) for c in client.detect_calls()]
    assert keys == list(range(1, len(keys) + 1))           # continuous: the paused position continues
    srcs = [r["media"]["src_n"] for r in frames_of(read_trace(s.run_dir)) if "media" in r]
    assert srcs == list(range(len(srcs)))
    assert s._clock.paused_total >= 0.5
    s.stop()
    assert s.wait_idle(5)


def test_stop_is_bounded_and_does_not_consume_the_rest_of_the_video():
    s, client, src, info = make("lab06-stop", media_frames(400, fps=20))
    s.start(info)
    assert wait_for(lambda: s.counts["published"] >= 3, 5)
    t0 = time.monotonic()
    assert s.stop()
    assert s.wait_idle(3) and time.monotonic() - t0 < 1.5
    out = s.last_outcome
    assert out.status == "stopped" and out.complete and src.stopped and out.counts["processed"] < 30
    man = json.loads((Path(out.run_dir) / "manifest.json").read_text("utf-8"))
    assert man["outcome"]["reason"] == "stopped (operator)"


def test_a_result_in_flight_at_stop_is_not_published_and_its_update_is_rejected():
    client = InProcessClient()
    s, client, src, info = make("lab06-stop-inflight", media_frames(50, fps=200), client)
    client.gate = threading.Event()
    client.gate.set()
    s.start(info)
    assert wait_for(lambda: s.counts["published"] >= 2, 5)
    client.gate.clear()
    assert wait_for(lambda: s._inflight_since is not None, 2)
    published = s.counts["published"]
    s.stop()
    client.gate.set()
    assert s.wait_idle(5)
    fr = frames_of(read_trace(s.run_dir))
    assert fr[-1]["disposition"] == "not_published_stop_requested" and s.counts["published"] == published
    assert ("settle", fr[-1]["epoch"], str(fr[-1]["seq"]), "reject") in client.calls
    assert s.latest_packet().seq == fr[-2]["seq"]


def test_restart_is_a_new_run_with_a_new_tracking_epoch_and_tracker_instance():
    d = work_dir("lab06-restart")
    info = source_file(d)
    client = InProcessClient()
    s = YoloeSession(runs_dir=d / "runs", client=client, source_factory=lambda p: ScriptSource(media_frames(4, fps=500)))
    first = run_to_end(s, info)
    pk1 = s.latest_packet()
    second = run_to_end(s, info)
    pk2 = s.latest_packet()
    assert first.run_id != second.run_id and pk1.epoch != pk2.epoch and pk2.epoch.startswith(second.run_id)
    assert len(client.trackers) == 2 and client.trackers[0] is not client.trackers[1]
    assert [c[1] for c in client.calls if c[0] == "reset"] == [pk1.epoch, pk2.epoch]
    assert client.calls.count(("start",)) == 1                   # the worker stays; the tracker does not


def test_a_commit_time_rejection_restores_the_exact_tracker_state_and_continues():
    client = InProcessClient()
    client.shape_at = "3"
    s, client, _src, info = make("lab06-reject", media_frames(6, fps=500), client)
    out = run_to_end(s, info)
    fr = frames_of(read_trace(out.run_dir))
    assert fr[2]["disposition"].startswith("rejected: image shape") and out.counts["rejected"] == 1
    assert ("settle", fr[2]["epoch"], "3", "reject") in client.calls
    assert client.calls[client.calls.index(("settle", fr[2]["epoch"], "3", "reject")) + 1][3] is None  # nothing to accept
    assert client.trackers[0].frame_id == 5 and len(client.trackers) == 1     # 6 updates - 1 restored
    assert out.counts["published"] == 5 and out.counts["epoch_resets"] == 0


def test_a_failed_restore_resets_the_epoch_with_an_acknowledged_continuity_break():
    client = InProcessClient()
    client.shape_at = "2"
    s, client, _src, info = make("lab06-reset", media_frames(5, fps=500), client)
    orig = client._new_tracker

    def first_fails():
        t = orig()
        t.fail_restore = len(client.trackers) == 1
        return t
    client.core.tracker_factory = first_fails
    out = run_to_end(s, info)
    assert out.counts["epoch_resets"] == 1 and len(client.trackers) == 2
    assert s.resets[-1]["reason"] == "restore_failed" and s.resets[-1]["acknowledged"]
    eps = [r.get("epoch") for r in frames_of(read_trace(out.run_dir))]
    assert eps[0].endswith("/e1") and eps[-1].endswith("/e2")                 # never silently joined


def test_worker_failure_fails_the_run_and_keeps_the_partial_evidence():
    client = InProcessClient()
    client.fail_at = "4"
    s, client, _src, info = make("lab06-workerfail", media_frames(10, fps=500), client)
    out = run_to_end(s, info)
    assert out.status == "failed" and "worker" in out.reason and out.complete
    assert out.counts["published"] == 3
    man = json.loads((Path(out.run_dir) / "manifest.json").read_text("utf-8"))
    assert man["outcome"]["status"] == "failed" and man["errors"]


def test_a_decoder_failure_mid_run_is_a_failed_outcome_after_the_frames_it_delivered():
    d = work_dir("lab06-srcfail")
    info = source_file(d)
    s = YoloeSession(runs_dir=d / "r", client=InProcessClient(),
                     source_factory=lambda p: ScriptSource(media_frames(8, fps=500), error_after=5))
    out = run_to_end(s, info)
    assert out.status == "failed" and out.failure["stage"] == "decoding" and "decoder exited" in out.reason
    assert out.counts["published"] == 5 and "after 5 analysed frames" in out.headline()


def test_preparation_refuses_low_disk_and_a_changed_source_before_starting_the_worker():
    s, client, _src, info = make("lab06-disk", media_frames(3), disk_usage=lambda p: Usage(1, 1, 10))
    out = run_to_end(s, info)
    assert out.status == "failed" and "free disk" in out.reason and ("start",) not in client.calls
    s2, client2, _src2, info2 = make("lab06-changed", media_frames(3))
    Path(info2.path).write_bytes(b"different bytes now")
    out2 = run_to_end(s2, info2)
    assert out2.status == "failed" and "changed" in out2.reason and ("start",) not in client2.calls


def test_worker_start_failure_is_a_visible_failed_outcome():
    s, client, _src, info = make("lab06-nostart", media_frames(3), InProcessClient(ready=False))
    out = run_to_end(s, info)
    assert out.status == "failed" and "could not start" in out.reason and not s.busy


def test_the_evidence_cap_stops_the_run_with_an_explicit_incomplete_outcome():
    calls = {"n": 0}

    def disk(p):
        calls["n"] += 1
        return Usage(10 * 2**30, 0, 10 * 2**30 if calls["n"] < 2 else 2**20)
    s, _c, _src, info = make("lab06-cap", media_frames(200, fps=5000), disk_usage=disk)
    out = run_to_end(s, info, 30)
    assert out.status == "failed" and not out.complete and "evidence" in out.reason
    assert out.counts["processed"] < 200
    man = json.loads((Path(out.run_dir) / "manifest.json").read_text("utf-8"))
    assert man["trace"]["complete"] is False and "free disk" in man["trace"]["error"]


def test_view_changes_and_paints_are_recorded_separately_from_detector_records():
    s, client, _src, info = make("lab06-view", media_frames(30, fps=50))
    s.start(info)
    assert wait_for(lambda: s.counts["published"] >= 3, 5)
    pk = s.latest_packet()
    assert s.note_painted(pk, time.monotonic()) and not s.note_painted(pk, time.monotonic())
    n_pred = len(client.model.calls)
    s.note_view("detections", 0.5)
    assert len(client.model.calls) - n_pred <= 2                          # a view change never runs the model
    s.stop()
    s.wait_idle(5)
    tr = read_trace(s.run_dir)
    assert [r for r in tr if r["k"] == "view"][0]["shown_conf"] == 0.5
    assert len([r for r in tr if r["k"] == "paint"]) == 1 and s.counts["displayed"] == 1
    assert all("shown_conf" not in r for r in frames_of(tr))


def test_a_second_start_while_busy_is_refused():
    s, _c, _src, info = make("lab06-busy", media_frames(50, fps=20))
    s.start(info)
    with pytest.raises(YoloeError):
        s.start(info)
    s.stop()
    assert s.wait_idle(5)


def test_stop_while_the_worker_is_still_starting_ends_that_worker_without_waiting_for_it():
    client = InProcessClient()
    gate = threading.Event()
    orig = client.start

    def slow_start():
        client.state = "starting"
        gate.wait(10)
        if client.state != "starting":
            return {"event": "fatal", "error": "worker stopped while starting"}
        return orig()
    client.start = slow_start

    def release(wait_s=5.0):
        client.state = "not started"
        client.released += 1
        gate.set()
    client.release = release
    s, client, _src, info = make("lab06-stop-starting", media_frames(5), client)
    s.start(info)
    assert wait_for(lambda: client.state == "starting", 2)
    t0 = time.monotonic()
    s.stop()
    assert s.wait_idle(3) and time.monotonic() - t0 < 1.0 and client.released == 1
    assert s.last_outcome.status == "stopped" and s.last_outcome.counts["processed"] == 0
