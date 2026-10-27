# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 matrix 3 (source part): ordered passthrough decode arguments, showinfo/PTS mapping (VFR, gaps,
unverified), bounded back-pressure, end/failure ordering, Stop cleanup and source-speed pacing. A scripted decoder
process stands in for FFmpeg (no media is decoded)."""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from echo_watch.frames import Epoch
from echo_watch.sources.base import SourceEnded, SourceError
from echo_watch.yoloe import source as S
from lab06_support import wait_for

W, H = 16, 12        # FFmpeg output sizes are parsed as 2-5 digit dimensions (as real videos have)
N = W * H * 3


def showinfo(n, pts, ck="ABCD0123", size=(32, 24)):
    return (f"[Parsed_showinfo_0 @ 000001] n:{n:4d} pts:{int(pts * 1000):7d} pts_time:{pts:g} duration:1 "
            f"duration_time:0.04 fmt:yuv420p cl:left sar:1/1 s:{size[0]}x{size[1]} i:P iskey:0 type:P "
            f"checksum:{ck} plane_checksum:[1 2 3]")


class FakeProc:
    """A decoder process: stdout bytes (frames), stderr lines (output header + showinfo), exit code."""

    def __init__(self, frames: int, pts=None, rc=0, partial: int = 0, skip_records=()):
        body = b"".join(bytes([k % 256]) * N for k in range(frames)) + b"\x01" * partial
        self.stdout = io.BytesIO(body)
        lines = ["Input #0, mov,mp4", "Output #0, rawvideo, to 'pipe:1':",
                 f"  Stream #0:0: Video: rawvideo (BGR[24] / 0x18524742), bgr24, {W}x{H}, q=2-31"]
        for k in range(frames):
            if k not in skip_records:
                lines.append(showinfo(k, (pts[k] if pts else k / 25)))
        self.stderr = io.BytesIO(("\n".join(lines) + "\n").encode())
        self.rc, self.killed, self.pid = rc, False, 4242
        self.args = None

    def poll(self):
        return -9 if self.killed else None

    def wait(self, timeout=None):
        return -9 if self.killed else self.rc

    def kill(self):
        self.killed = True


def spawn_of(proc):
    def spawn(args, **kw):
        proc.args = args
        return proc
    return spawn


def source(proc, probe=None, q=2):
    pr = probe if probe is not None else {"frames": 50, "first_pts_s": 0.0, "_pts": [k / 25 for k in range(50)]}
    s = S.OrderedFileSource("t", Path("ffmpeg.exe"), "C:/v.mp4", queue_frames=q, spawn=spawn_of(proc),
                            probe=lambda: dict(pr), validate=False, record_wait_s=0.05)
    s.start(Epoch.begin("t", 0))
    return s


def drain(s, limit=200):
    out = []
    while len(out) < limit:
        f = s.read(1.0)
        if f is not None:
            out.append(f)
    return out


def test_ordered_args_have_no_resampler_no_realtime_flag_and_log_before_scaling():
    a = S.build_ordered_args(Path("ff.exe"), "C:/v.mp4")
    vf = a[a.index("-vf") + 1]
    assert "-re" not in a and "fps=" not in vf and a[a.index("-fps_mode") + 1] == "passthrough"
    assert vf.startswith("showinfo,scale=") and vf.count("showinfo") == 1
    assert a[a.index("-f") + 1] == "rawvideo" and a[a.index("-pix_fmt") + 1] == "bgr24"


def test_showinfo_parsing_and_vfr_mapping_to_verified_source_ordinals():
    n, pts, ck, size = S.parse_showinfo(showinfo(3, 0.12))
    assert (n, pts, ck, size) == (3, 0.12, "ABCD0123", (32, 24))
    assert S.parse_showinfo("[Parsed_showinfo_0 @ 1] n: 4 pts:NOPTS pts_time:NOPTS") is None
    vfr = [0.0, 0.04, 0.3, 0.31, 0.9]
    probe = {"first_pts_s": 0.0, "_pts": vfr}
    last, ords = None, []
    for k, t in enumerate(vfr):
        m = S.map_frame(k, (t, "C"), probe, last)
        assert m["verified"] and "gap_before" not in m
        last = m["src_n"]
        ords.append((m["src_n"], m["media_pts"]))
    assert ords == list(enumerate(vfr))


def test_a_missing_source_frame_is_a_counted_gap_and_unknown_timing_is_unverified():
    probe = {"first_pts_s": 1.0, "_pts": [0.0, 0.04, 0.08, 0.12, 0.16]}
    m = S.map_frame(1, (1.12, "C"), probe, 0)                 # decoded frame 1 is source frame 3
    assert m["src_n"] == 3 and m["gap_before"] == 2 and m["media_pts"] == pytest.approx(0.12)
    u = S.map_frame(2, (1.555, "C"), probe, 3)
    assert u["verified"] is False and "src_n" not in u and "not found" in u["why"]
    back = S.map_frame(3, (1.04, "C"), probe, 3)
    assert back["verified"] is False and "does not advance" in back["why"]
    assert S.map_frame(0, None, probe, None)["why"] == "no side-channel record for this frame"
    assert S.map_frame(0, (0.0, "C"), {"probe_error": "no listing"}, None)["why"] == "no listing"


def test_frames_are_delivered_once_in_order_with_their_own_media_facts():
    proc = FakeProc(6)
    s = source(proc)
    frames = []
    with pytest.raises(SourceEnded):
        while True:
            f = s.read(1.0)
            if f is not None:
                frames.append(f)
    assert [f.media["out_n"] for f in frames] == list(range(6)) and [int(f.image[0, 0, 0]) for f in frames] == list(range(6))
    assert all(f.media["verified"] for f in frames) and [f.seq for f in frames] == list(range(1, 7))
    with pytest.raises(SourceEnded):                       # every later read sees the same end
        s.read(0.1)
    d = s.diagnostics()
    assert d["size"] == (W, H) and d["native_size"] == (32, 24) and d["stats"]["delivered"] == 6
    s.stop()


def test_a_slow_consumer_back_pressures_the_decoder_instead_of_skipping_or_buffering():
    proc = FakeProc(40)
    s = source(proc, q=2)
    assert wait_for(lambda: s.stats["decoded"] >= 3, 2.0)
    time.sleep(0.3)
    assert s.stats["decoded"] <= 3                         # queue (2) + the one frame waiting to be put
    got = []
    with pytest.raises(SourceEnded):
        while True:
            f = s.read(1.0)
            if f is not None:
                got.append(f.media["out_n"])
                time.sleep(0.002)
    assert got == list(range(40)) and s.stats["max_queue"] <= 2 and s.stats["blocked_put_s"] > 0.2
    s.stop()


def test_a_decoder_failure_is_reported_after_every_decoded_frame_and_a_truncated_frame_is_not_a_normal_end():
    s = source(FakeProc(3, rc=1))
    got = []
    with pytest.raises(SourceError) as e:
        while True:
            f = s.read(1.0)
            if f is not None:
                got.append(f.media["out_n"])
    assert got == [0, 1, 2] and not isinstance(e.value, SourceEnded) and "code 1" in str(e.value)
    s2 = source(FakeProc(2, partial=10))
    with pytest.raises(SourceError) as e2:
        drain(s2)
    assert not isinstance(e2.value, SourceEnded) and "inside a frame" in str(e2.value)


def test_missing_side_channel_records_leave_frames_explicitly_unverified():
    s = source(FakeProc(4, skip_records={2}))
    frames = []
    with pytest.raises(SourceEnded):
        while True:
            f = s.read(1.0)
            if f is not None:
                frames.append(f)
    assert [f.media["verified"] for f in frames] == [True, True, False, True]
    assert frames[2].media["why"] == "no side-channel record for this frame" and s.stats["unverified"] == 1


def test_stop_is_bounded_kills_only_its_child_and_does_not_consume_the_rest():
    proc = FakeProc(500)
    s = source(proc, q=2)
    s.read(1.0)
    t0 = time.monotonic()
    s.stop()
    assert time.monotonic() - t0 < 2.5 and proc.killed and s.stats["decoded"] < 10
    assert all(not t.is_alive() for t in s._threads)


# ---- pacing ---------------------------------------------------------------------------------------------------------------
def test_media_clock_paces_at_source_speed_and_slows_honestly_without_bursts():
    c = S.MediaClock(late_reanchor_s=0.25)
    assert c.plan(0.0, 100.0) == 0.0
    assert c.plan(0.1, 100.02) == pytest.approx(0.08)
    # processing fell 1 s behind: re-anchor, no catch-up burst for the following frames
    assert c.plan(0.2, 101.2) == 0.0 and c.reanchors["late"] == 1
    assert c.plan(0.3, 101.21) == pytest.approx(0.09)
    assert c.plan(0.15, 101.4) == 0.0 and c.reanchors["pts_backwards"] == 1


def test_paused_wall_time_is_never_charged_as_media_time():
    c = S.MediaClock()
    t = 0.0
    for k in range(5):
        c.plan(k * 0.1, t)
        c.shown(k * 0.1, t)
        t += 0.2                                          # inference twice as slow as the source
    assert c.speed() == pytest.approx(0.5, abs=0.01)
    c.suspend(t)
    t += 30.0                                             # a long pause
    c.resume(t)
    assert c.plan(0.5, t) == 0.0 and c.reanchors["resume"] == 1
    c.shown(0.5, t)
    assert c.speed() == pytest.approx(0.5, abs=0.05) and c.paused_total == pytest.approx(30.0)
    assert c.active(t) == pytest.approx(t - 30.0)
