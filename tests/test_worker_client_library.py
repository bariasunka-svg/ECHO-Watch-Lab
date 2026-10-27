# SPDX-License-Identifier: AGPL-3.0-only
"""Worker client supervision and the Scan / Select / Remember library (STAND-IN worker: no model runs)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from echo_watch.config import CPU_PYTHON, Settings
from echo_watch.engine.library import LibraryError
from echo_watch.frames import Epoch, Frame
from echo_watch.worker.client import WorkerClient, WorkerSpec, WorkerUnavailable
from support import standin_cmd, standin_engine, work_dir


def spec_with(d: Path, script: str | None = None, device="cuda:0", startup=15.0) -> WorkerSpec:
    cmd = [str(CPU_PYTHON), "-I", "-c", script] if script else standin_cmd(d / "spec.json")
    return WorkerSpec(CPU_PYTHON, Path("m"), "x", None, None, d / "cfg", device, "fp32", 1, d / "home", cmd, startup)


STUB_HEAD = """
import sys, json, os, time
def emit(o):
    sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
"""


def test_start_ping_stop_and_session_identity():
    d = work_dir("wc_basic")
    (d / "spec.json").write_text(json.dumps({"scan": [], "verify": []}))
    c = WorkerClient(spec_with(d))
    r = c.start()
    assert r["event"] == "ready" and c.state == "ready" and c.session == 1 and c.pid() is not None
    pong = c.ping()
    assert pong["_session"] == 1 and "_ipc_out_s" in pong and pong["_ipc_out_s"] >= 0
    pid = c.pid()
    c.stop()
    assert c.state == "stopped" and c.pid() is None
    import psutil
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    with pytest.raises(WorkerUnavailable):
        c.ping()


def test_child_environment_is_minimal_but_can_answer_getuser_and_keeps_every_cache_inside_the_project():
    """Regression for the first real GPU start: torch's CUDA build calls getpass.getuser(); the minimal child environment
    must therefore carry a user name, while no inherited PYTHON*/ECHOWATCH_* variable and no profile path leaks in."""
    import subprocess
    from echo_watch.config import GPU_PYTHON
    from echo_watch.worker.client import child_env
    home = work_dir("wc_env") / "home"
    env = child_env(home)
    assert env.get("USERNAME")
    assert not any(k.startswith(("PYTHON", "ECHOWATCH", "PIP_")) for k in env)
    assert all(str(home) in v for k, v in env.items() if k in ("USERPROFILE", "HOME", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TORCH_HOME"))
    r = subprocess.run([str(GPU_PYTHON), "-I", "-c", "import getpass; print(getpass.getuser())"], env=env, capture_output=True,
                       text=True, timeout=60, creationflags=0x08000000)
    assert r.returncode == 0 and r.stdout.strip(), r.stderr


def test_device_mismatch_is_fatal_and_never_a_cpu_substitute():
    d = work_dir("wc_dev")
    script = STUB_HEAD + 'emit({"event":"ready","device":"cpu"}); sys.stdin.read()'
    c = WorkerClient(spec_with(d, script, device="cuda:0"))
    r = c.start()
    assert r["event"] == "fatal" and "no CPU substitute" in r["error"] and c.state == "unavailable"
    with pytest.raises(WorkerUnavailable):
        c.ping()


def test_a_missed_deadline_kills_only_its_child_and_leaves_the_worker_unavailable_without_a_silent_restart():
    d = work_dir("wc_timeout")
    script = STUB_HEAD + 'emit({"event":"ready","device":"cuda:0"})\nfor l in sys.stdin:\n    time.sleep(30)\n'
    c = WorkerClient(spec_with(d, script))
    assert c.start()["event"] == "ready"
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        c.request({"cmd": "ping"}, 0.4)
    assert time.monotonic() - t0 < 3
    assert c.state == "unavailable" and c.pid() is None and c.failures == 1
    with pytest.raises(WorkerUnavailable):
        c.request({"cmd": "ping"}, 1.0)
    assert c.session == 1                                                    # nothing restarted behind the caller's back


def test_a_reply_that_belongs_to_another_job_is_never_attributed_to_this_request():
    d = work_dir("wc_stale")
    script = STUB_HEAD + ('emit({"event":"ready","device":"cuda:0"})\n'
                          'for l in sys.stdin:\n'
                          '    req = json.loads(l)\n'
                          '    emit({"event":"pong","job_id":"j-old","marker":"stale"})\n'
                          '    emit({"event":"pong","job_id":req["job_id"],"marker":"mine","recv_wall":time.time(),"done_wall":time.time()})\n')
    c = WorkerClient(spec_with(d, script))
    c.start()
    assert c.ping()["marker"] == "mine"
    c.stop()


def test_worker_that_exits_mid_request_is_reported_with_its_stderr_tail():
    d = work_dir("wc_exit")
    script = STUB_HEAD + ('emit({"event":"ready","device":"cuda:0"})\n'
                          'sys.stdin.readline()\nsys.stderr.write("boom CUDA out of memory\\n"); sys.stderr.flush()\nsys.exit(9)\n')
    c = WorkerClient(spec_with(d, script))
    c.start()
    with pytest.raises(RuntimeError) as e:
        c.ping(5.0)
    assert "exited" in str(e.value) or "closed" in str(e.value)
    assert c.state == "unavailable" and "CUDA out of memory" in " ".join(c.stderr)


# ---- library -----------------------------------------------------------------------------------------------------------
def frame_with_two_boxes():
    img = np.full((720, 1280, 3), 120, np.uint8)
    cv2.rectangle(img, (100, 100), (260, 300), (30, 200, 60), -1)
    cv2.rectangle(img, (500, 400), (700, 560), (200, 60, 30), -1)
    return Frame(Epoch.begin("s", 1), 1, 1.0, 1.0, img)


def lib_engine(name, scan):
    return standin_engine(work_dir(name), {"scan": scan, "verify": []})


def test_scan_retains_the_exact_frame_and_marks_duplicates_and_never_creates_items_by_itself():
    scan = [{"box": [100, 100, 160, 200], "label": "a", "confidence": 0.9},
            {"box": [104, 102, 160, 200], "label": "a2", "confidence": 0.5},      # near-duplicate (IoU > 0.8)
            {"box": [500, 400, 200, 160], "label": "b", "confidence": 0.8}]
    eng = lib_engine("lib_scan", scan)
    try:
        eng.start_worker()
        fr = frame_with_two_boxes()
        r = eng.scan(fr, "src")
        assert r.status == "succeeded" and len(r.proposals) == 3
        data = Path(r.frame_path).read_bytes()
        assert hashlib.sha256(data).hexdigest() == r.frame_sha256
        back = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        assert np.array_equal(back, fr.image)                                   # lossless: exactly the analysed pixels
        dups = [p for p in r.proposals if p.duplicate_of]
        assert len(dups) == 1 and dups[0].label == "a2"
        assert eng.library.list_items() == []                                   # nothing is an item until Remember
    finally:
        eng.shutdown()


def test_remember_is_idempotent_validated_and_versioned():
    scan = [{"box": [100, 100, 160, 200], "label": "a", "confidence": 0.9},
            {"box": [500, 400, 200, 160], "label": "b", "confidence": 0.8}]
    eng = lib_engine("lib_remember", scan)
    try:
        eng.start_worker()
        r = eng.scan(frame_with_two_boxes(), "src")
        pa, pb = r.proposals
        refs = eng.remember("op-1", r.scan_id, [{"candidate_id": pa.candidate_id, "name": "Alpha"}])
        again = eng.remember("op-1", r.scan_id, [{"candidate_id": pa.candidate_id, "name": "Alpha"}])
        assert refs[0].item_id == again[0].item_id and len(eng.library.list_items()) == 1      # a retried op never duplicates
        # EW-LAB-05: overlap with a remembered item no longer forbids a distinct item at that place
        other = eng.remember("op-2", r.scan_id, [{"candidate_id": pa.candidate_id, "name": "Alpha again"}])
        assert other[0].item_id != refs[0].item_id and len(eng.library.list_items()) == 2
        with pytest.raises(LibraryError):
            eng.remember("op-3", r.scan_id, [{"candidate_id": "cand_nope", "name": "x"}])
        with pytest.raises(LibraryError):
            eng.remember("op-4", r.scan_id, [{"candidate_id": pb.candidate_id, "name": ""}])
        with pytest.raises(LibraryError):
            eng.remember("op-5", r.scan_id, [{"candidate_id": pb.candidate_id, "name": "B", "box": [1270, 710, 5, 5]}])
        two = eng.remember("op-6", r.scan_id, [{"candidate_id": pb.candidate_id, "name": "Beta"}])
        assert len(eng.library.list_items()) == 3 and two[0].ref_key.endswith("@v1")
        # explicit versioned reference change: v2 exists, v1 is preserved, the worker key differs
        r2 = eng.scan(frame_with_two_boxes(), "src")
        v2 = eng.library.update_reference("op-7", refs[0].item_id, r2.scan_id, r2.proposals[0].candidate_id)
        assert v2.ref_version == 2 and v2.ref_key != refs[0].ref_key
        assert eng.library.item_ref(refs[0].item_id, 1).ref_key == refs[0].ref_key
        kinds = [x["kind"] for x in eng.store.query("SELECT kind FROM decisions ORDER BY wall")]
        assert kinds.count("remember") == 3 and "update_reference" in kinds and "scan" in kinds     # append-only decisions
        # later scans hint proposals overlapping a COMPATIBLE remembered item (same content scope); an unknown scope
        # string such as "src" is never compatible (EW-LAB-05 engine.compat)
        r3 = eng.scan(frame_with_two_boxes(), "src")
        assert not any(p.overlaps_item for p in r3.proposals)
    finally:
        eng.shutdown()


def test_a_failed_scan_is_recorded_and_returns_no_fabricated_proposals():
    eng = lib_engine("lib_fail", [])
    try:
        eng.start_worker()
        eng.worker.stop()
        eng.worker._stopped = False
        eng.worker.state = "unavailable"
        r = eng.scan(frame_with_two_boxes(), "src")
        assert r.status == "failed" and r.proposals == () and r.error
        row = eng.store.one("SELECT status, error FROM scans WHERE scan_id=?", (r.scan_id,))
        assert row["status"] == "failed" and row["error"]
    finally:
        eng.shutdown()
