# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 matrix 6 (review part): bounded listing, offset-indexed paging, original-source verification and the
exact-frame adapter's selection by recorded PTS (a scripted decoder stands in for FFmpeg; no media is decoded)."""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import pytest

from echo_watch.engine.recording import signature
from echo_watch.yoloe import review as R
from echo_watch.yoloe.evidence import TraceWriter, atomic_json
from lab06_support import work_dir


def make_run(root, run_id, frames=7, status="completed", complete=True):
    d = root / run_id
    d.mkdir(parents=True)
    w = TraceWriter(d / "trace.jsonl")
    w.put({"t": 0, "mono": 0, "k": "start"})
    for k in range(frames):
        w.put({"t": 0, "mono": k, "k": "frame", "seq": k + 1, "epoch": f"{run_id}/e1",
               "media": {"out_n": k, "src_n": k, "pts": k / 10, "media_pts": k / 10, "verified": True},
               "detections": [{"i": 0, "xyxy": [1, 2, 3, 4], "score": 0.5, "cls": 1, "name": "cup"}],
               "counts": {"associated": 1, "untracked": 0}, "disposition": "published"})
        if k == 3:
            w.put({"t": 0, "mono": k, "k": "gap", "seq": k + 1, "missing_source_frames": 2})
            w.note({"t": 0, "mono": k, "k": "paint", "seq": k + 1})
    tr = w.close()
    atomic_json(d / "manifest.json", {"run_id": run_id, "started_wall": time.time(), "source": {"name": "a.mp4"},
                                     "outcome": {"status": status, "complete": complete}, "counts": {"processed": frames},
                                     "profile": {"detector": {"conf": 0.1}}, "trace": tr})
    return d


def test_listing_is_bounded_newest_first_and_reports_unreadable_manifests():
    root = work_dir("lab06-list")
    a = make_run(root, "yrun_a")
    b = make_run(root, "yrun_b", status="stopped", complete=False)
    bad = root / "yrun_c"
    bad.mkdir()
    (bad / "manifest.json").write_text("{not json", "utf-8")
    now = time.time()
    for k, d in enumerate((a, b, bad)):                       # explicit, distinct modification times (newest last)
        os.utime(d, (now - 30 + k, now - 30 + k))
    rows = R.list_runs(root, limit=10)
    assert [r["run_id"] for r in rows][:1] == ["yrun_c"] and rows[0]["status"] == "unreadable"
    assert {r["run_id"]: r["status"] for r in rows[1:]} == {"yrun_b": "stopped", "yrun_a": "completed"}
    assert len(R.list_runs(root, limit=2)) == 2 and R.list_runs(root / "missing") == []


def test_pager_indexes_offsets_only_and_reads_pages_and_records_on_demand():
    d = make_run(work_dir("lab06-pager"), "yrun_p", frames=130)
    p = R.TracePager(d)
    assert p.index() == 130 and len(p.offsets) == 130 and all(isinstance(o, int) for o in p.offsets)
    assert [e["k"] for e in p.events] == ["start", "gap", "end"] or [e["k"] for e in p.events] == ["start", "gap"]
    page = p.page(100, 50)
    assert [r["n"] for r in page] == list(range(100, 130)) and page[0]["seq"] == 101 and page[0]["src_n"] == 100
    assert p.record(129)["seq"] == 130 and p.page(500, 50) == []


def test_source_verification_rejects_a_different_file_at_the_recorded_path():
    d = work_dir("lab06-verify")
    v = d / "v.mp4"
    v.write_bytes(b"A" * 100)
    import hashlib
    man = {"source": {"path": str(v), "bytes": 100, "sha256": hashlib.sha256(b"A" * 100).hexdigest()}}
    assert R.verify_source(man)["ok"] is True
    v.write_bytes(b"B" * 100)                                 # same path, same size, other content
    out = R.verify_source(man)
    assert out["ok"] is False and "different content" in out["why"]
    v.write_bytes(b"B" * 7)
    assert "another size" in R.verify_source(man)["why"]
    v.unlink()
    assert "not at its recorded path" in R.verify_source(man)["why"]


class Proc:
    def __init__(self, frames, pts, w=16, h=12):
        self.frames = frames
        self.stdout = io.BytesIO(b"".join(f.tobytes() for f in frames))
        lines = ["Output #0, rawvideo, to 'pipe:1':", f"  Stream #0:0: Video: rawvideo, bgr24, {w}x{h}, q=2-31"]
        lines += [f"[Parsed_showinfo_0 @ 01] n:{k:4d} pts:{int(t * 1000)} pts_time:{t:g} duration:1 s:32x24 "
                  f"checksum:ABCD plane_checksum:[1]" for k, t in enumerate(pts)]
        self.stderr = io.BytesIO(("\n".join(lines) + "\n").encode())
        self.killed = False

    def poll(self):
        return -9 if self.killed else 0

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


def frames(n, w=16, h=12):
    out = []
    for k in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[:, : (k + 1) * 2] = 200
        out.append(img)
    return out


def test_the_exact_frame_is_selected_by_recorded_pts_after_the_seek_offset():
    fr = frames(6)
    seen = {}

    def spawn(args, **kw):
        seen["args"] = args
        return Proc(fr, [0.88, 0.92, 0.96, 1.0, 1.04, 1.08])            # timestamps restart at the seek point
    req = R.FrameRequest("C:/v.mp4", 3.12, (16, 12), signature(fr[3]))      # ss = 2.12, so 1.0 + 2.12 == 3.12
    out = R.decode_original(req, spawn=spawn)
    assert out.image is not None and out.found_pts == pytest.approx(3.12) and out.signature_diff == 0.0 and out.matches
    assert np.array_equal(out.image, fr[3])
    a = seen["args"]
    assert a[a.index("-ss") + 1] == "2.120000" and a[a.index("-fps_mode") + 1] == "passthrough"
    assert a[a.index("-vf") + 1].startswith("showinfo,scale=")


def test_no_matching_timestamp_is_an_explicit_gap_never_a_nearby_frame():
    fr = frames(4)
    out = R.decode_original(R.FrameRequest("C:/v.mp4", 0.05, (16, 12), None),
                            spawn=lambda a, **k: Proc(fr, [0.0, 0.04, 0.08, 0.12]))
    assert out.image is None and "evidence gap" in out.error
    out2 = R.decode_original(R.FrameRequest("C:/v.mp4", 0.08, (20, 12), None),
                             spawn=lambda a, **k: Proc(fr, [0.0, 0.04, 0.08, 0.12]))
    assert out2.image is None and "not the test's processing size 20x12" in out2.error
    other = frames(4)
    other[2][:] = 30
    out3 = R.decode_original(R.FrameRequest("C:/v.mp4", 0.08, (16, 12), signature(fr[2])),
                             spawn=lambda a, **k: Proc(other, [0.0, 0.04, 0.08, 0.12]))
    assert out3.image is not None and not out3.matches and out3.signature_diff > R.SIG_MATCH


def test_trace_writer_reports_exactly_what_it_wrote_and_refuses_after_failure():
    d = work_dir("lab06-writer")
    w = TraceWriter(d / "trace.jsonl", queue_max=4)
    for k in range(10):
        assert w.put({"k": "frame", "n": k})
    rep = w.close()
    data = (d / "trace.jsonl").read_bytes()
    import hashlib
    assert rep["lines"] == 10 and rep["bytes"] == len(data) and rep["sha256"] == hashlib.sha256(data).hexdigest()
    assert rep["complete"] and not w.put({"k": "late"})
    assert [json.loads(x)["n"] for x in data.decode().splitlines()] == list(range(10))
