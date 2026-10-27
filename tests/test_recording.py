# SPDX-License-Identifier: AGPL-3.0-only
"""Recorder finalization semantics, the annotated pump's fresh/held accounting, and written-output inspection."""
from __future__ import annotations

import subprocess
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from echo_watch.config import FFMPEG
from echo_watch.engine.contracts import ControlBasis, ItemObservation, ResultPacket
from echo_watch.engine.inspect import inspect_recording
from echo_watch.engine.recording import (AnnotatedPump, STATE_FAILED, STATE_FINALIZED, RecorderError, VideoItem,
                                         VideoRecorder, signature)
from echo_watch.engine.render import render_result
from support import work_dir


def img(k: int, w=320, h=180) -> np.ndarray:
    a = np.zeros((h, w, 3), np.uint8)
    a[:, :] = (k * 7 % 255, 90, 200 - k * 3 % 200)
    cv2.rectangle(a, (10 + k, 20), (60 + k, 90), (255, 255, 255), -1)
    return a


def wait_state(rec, states, timeout=30.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and rec.state not in states:
        time.sleep(0.02)
    return rec.state


def test_recorder_finalizes_only_after_the_encoder_really_closed_and_stop_never_blocks():
    d = work_dir("rec_ok")
    rec = VideoRecorder("t", d / "o.mp4", (320, 180), fps=15, ffmpeg=FFMPEG)
    rec.start()
    for k in range(30):
        assert rec.offer(VideoItem(img(k), {"kind": "capture", "ident": ["e", k], "t_mono": k / 15.0}))
    t0 = time.monotonic()
    rec.stop()
    assert time.monotonic() - t0 < 0.5                                   # stop() only signals
    assert rec.state in ("finalizing", STATE_FINALIZED)                  # never 'finalized' by the stop call itself
    assert rec.offer(VideoItem(img(0), {})) is False                     # no frame is accepted once finalization began
    assert wait_state(rec, (STATE_FINALIZED, STATE_FAILED)) == STATE_FINALIZED
    s = rec.summary()
    assert s["frames_written"] == 30 and s["sha256"] and s["bytes"] > 0 and s["exit_code"] == 0
    rep = inspect_recording(FFMPEG, d / "o.mp4", d / "o.framemap.json")
    assert rep["count_match"] and rep["content_match"], rep               # the written file really contains those pictures


def test_recorder_queue_overflow_drops_newest_counts_and_marks_finalized_file_partial_in_outcome():
    d = work_dir("rec_drop")

    class SlowProc:
        def __init__(self):
            self.stdin = SimpleNamespace(write=lambda b: time.sleep(0.05), close=lambda: None)
            self.stderr = SimpleNamespace(readline=lambda: b"")
            self.returncode = 0

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass
    rec = VideoRecorder("slow", d / "o.mp4", (320, 180), fps=15, ffmpeg=FFMPEG, queue_maxsize=3, launch=lambda a: SlowProc())
    rec.start()
    ok = sum(rec.offer(VideoItem(img(k), {"kind": "capture", "ident": ["e", k], "t_mono": k * 0.01})) for k in range(40))
    assert rec.frames_dropped == 40 - ok and rec.frames_dropped > 0       # counted, never silent
    assert len(rec.dropped_entries) == rec.frames_dropped
    rec.stop()
    wait_state(rec, (STATE_FINALIZED, STATE_FAILED))
    assert rec.frames_written == ok - rec.frames_size_mismatched


def test_recorder_encoder_failure_ends_failed_not_finalized():
    d = work_dir("rec_fail")
    rec = VideoRecorder("bad", d / "o.mp4", (320, 180), fps=15, ffmpeg=FFMPEG)
    rec._launch = lambda a: subprocess.Popen(["cmd", "/c", "exit 3"], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.PIPE, creationflags=0x08000000)
    with pytest.raises(RecorderError):
        rec.start()                                                       # a required output that cannot start refuses

    rec2 = VideoRecorder("bad2", d / "o2.mp4", (320, 180), fps=15, ffmpeg=FFMPEG)
    rec2.start()
    rec2.offer(VideoItem(img(1), {}))
    rec2._proc.kill()                                                     # the encoder dies while recording
    rec2.offer(VideoItem(img(2), {}))
    assert wait_state(rec2, (STATE_FINALIZED, STATE_FAILED)) == STATE_FAILED and rec2.error


def test_recorder_refuses_implausible_fps_and_size():
    d = work_dir("rec_args")
    with pytest.raises(RecorderError):
        VideoRecorder("x", d / "o.mp4", (320, 180), fps=0, ffmpeg=FFMPEG)
    with pytest.raises(ValueError):
        VideoRecorder("x", d / "o.mp4", (0, 180), fps=15, ffmpeg=FFMPEG)


def test_frame_map_lists_real_capture_gaps():
    d = work_dir("rec_gaps")
    rec = VideoRecorder("gaps", d / "o.mp4", (320, 180), fps=15, ffmpeg=FFMPEG)
    rec.start()
    t = 0.0
    for k in range(20):
        t += 0.0667 if k != 10 else 0.9                                    # one real 0.9 s hole in capture
        rec.offer(VideoItem(img(k), {"kind": "capture", "ident": ["e", k], "t_mono": t}))
    rec.stop()
    wait_state(rec, (STATE_FINALIZED, STATE_FAILED))
    g = rec.gaps()
    assert len(g) == 1 and g[0]["seconds"] == pytest.approx(0.9, abs=1e-3) and g[0]["after_i"] == 9


# ---- annotated pump ----------------------------------------------------------------------------------------------------
def packet(seq: int, state="visible") -> ResultPacket:
    basis = ControlBasis("b", "r", 1, 1, 0.0, "ep", 0, "d", 1, "item-memory-v2", True)
    o = ItemObservation("i1", "speaker", 1, state, None, (20.0, 20.0, 60.0, 60.0) if state == "visible" else None)
    return ResultPacket(f"res_{seq:06d}", seq, "r", basis, ("ep", seq), float(seq), float(seq), (320, 180), img(seq), None,
                        (), (o,), {}, {}, {})


def test_pump_counts_fresh_results_and_held_repeats_separately_and_stop_is_an_offer_barrier():
    d = work_dir("pump")
    rec = VideoRecorder("ann", d / "a.mp4", (320, 180), fps=30, ffmpeg=FFMPEG)
    rec.start()
    cur = {"p": packet(1), "b": None}
    pump = AnnotatedPump(provider=lambda: (cur["p"], cur["b"]), render=render_result, sink=rec, fps=30)
    pump.start()
    time.sleep(0.35)
    cur["p"] = packet(2)
    time.sleep(0.35)
    cur["b"] = "PAUSED"                                                    # live authority lost: a HELD, boxless frame
    time.sleep(0.2)
    pump.stop()
    n_at_stop = rec.frames_offered
    time.sleep(0.2)
    assert rec.frames_offered == n_at_stop                                 # no offer can begin after stop() returned
    rec.stop()
    wait_state(rec, (STATE_FINALIZED, STATE_FAILED))
    s = pump.summary()
    assert s["fresh_results_rendered"] == 2                                # two distinct committed results, rendered once each
    assert s["held_frames_offered"] > 5                                    # repeats between results are counted apart
    kinds = [e["kind"] for e in rec.map]
    assert kinds.count("fresh") == 2 and "held" in kinds
    assert all(e["result"] in ("res_000001", "res_000002") for e in rec.map)
    banners = [e["banner"] for e in rec.map]
    assert "PAUSED" in banners and banners[0] is None


def test_render_draws_boxes_only_for_live_boxed_states_and_never_for_held_or_absent():
    p = packet(5)
    live = render_result(p)
    held = render_result(p, "PAUSED")
    assert not np.array_equal(live, p.image) and not np.array_equal(held, live)
    # a box colour (cyan) appears on the item edge in the live render but not in the held render
    cyan = np.array([238, 211, 34])
    edge_live = (np.abs(live[80, 20:80].astype(int) - cyan).sum(axis=1) < 30).any()      # bottom edge: clear of the panel
    edge_held = (np.abs(held[80, 20:80].astype(int) - cyan).sum(axis=1) < 30).any()
    assert edge_live and not edge_held
    gone = render_result(packet(6, "not_seen"))
    assert not (np.abs(gone[80, 20:80].astype(int) - cyan).sum(axis=1) < 30).any()       # an absent target gets no box


def test_signature_is_stable_and_discriminates():
    a, b = img(1), img(40)
    assert signature(a) == signature(a.copy()) and signature(a) != signature(b)
