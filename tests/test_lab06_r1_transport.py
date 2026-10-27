# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 R1 diagnostic transport through the SHARED worker client (and its YOLOE subclass) with the process/pipe
boundary mocked: ``popen_windowless`` returns scripted processes - no child process, worker, model or FFmpeg runs.
Covers a structured worker fatal, a multi-line exception ending in generic advice, abrupt EOF, nonzero exit, startup
deadline, stderr flood, per-attempt isolation (a previous child's late output), restart, and the unchanged Display
Assets client behaviour (messages, stderr deque, health shape)."""
from __future__ import annotations

import io
import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

import echo_watch.worker.client as WC
from echo_watch.config import Settings
from echo_watch.worker.client import WorkerClient, WorkerSpec
from lab06_support import work_dir

ADVICE = ["AttributeError: ",
          "            'IterableSimpleNamespace' object has no attribute 'half'. This may be caused by a modified or out "
          "of date ultralytics",
          "            'default.yaml' file.",
          "Please update your code with 'pip install -U ultralytics' and if necessary replace",
          r"            E:\ECHO Watch Lab\env\gpu\Lib\site-packages\ultralytics\cfg\default.yaml with the latest version from",
          "            https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml",
          "            "]
TRACE = ["Traceback (most recent call last):", '  File "E:\\ECHO Watch Lab\\src\\echo_watch\\yoloe\\worker.py", line 229, in main',
         "    \"half\": pa.half,"] + ADVICE


class Out:
    def __init__(self, lines, until=None):
        self.lines, self.until = list(lines), until

    def readline(self, n=-1):
        if self.lines:
            return self.lines.pop(0)
        if self.until is not None:
            self.until.wait(10)
        return ""


class Err:
    def __init__(self, lines, gate=None, after=()):
        self.lines, self.gate, self.after = list(lines), gate, list(after)

    def __iter__(self):
        for ln in self.lines:
            yield ln + "\n"
        if self.gate is not None:
            self.gate.wait(10)
            for ln in self.after:
                yield ln + "\n"


class DiesOnRequest(io.StringIO):
    """stdin that ends the scripted child as soon as a request line arrives (death mid-request)."""

    def __init__(self, ended):
        super().__init__()
        self.ended = ended

    def flush(self):
        if self.getvalue():
            self.ended.set()


class Proc:
    def __init__(self, out=(), err=(), rc=0, hold=None, exit_after=None, err_gate=None, err_after=()):
        self.killed = threading.Event()
        self.stdout = Out([x if x.endswith("\n") else x + "\n" for x in out], hold if hold is not None else None)
        self.stderr = Err(err, err_gate, err_after)
        self.stdin = io.StringIO()
        self.rc, self.pid, self.hold, self.t0, self.exit_after = rc, 4321, hold, time.monotonic(), exit_after

    @property
    def returncode(self):
        return self.poll()

    def poll(self):
        if self.killed.is_set():
            return -9
        if self.exit_after is not None and time.monotonic() - self.t0 >= self.exit_after:
            return self.rc
        if not self.stdout.lines and (self.hold is None or self.hold.is_set()):
            return self.rc
        return None

    def kill(self):
        self.killed.set()
        if self.hold is not None:
            self.hold.set()

    def wait(self, timeout=None):
        end = time.monotonic() + (timeout or 0)
        while self.poll() is None:
            if time.monotonic() > end:
                raise subprocess.TimeoutExpired("fake", timeout)
            time.sleep(0.01)
        return self.poll()


def J(obj):
    return json.dumps(obj)


@pytest.fixture()
def procs(monkeypatch):
    queue = []
    monkeypatch.setattr(WC, "popen_windowless", lambda args, **kw: queue.pop(0))
    return queue


def client(cls=WorkerClient, timeout=3.0):
    d = work_dir("lab06r1-transport")
    return cls(WorkerSpec(Path("unused"), Path("m.pt"), "sha", None, None, d, cmd_override=["scripted"],
                          startup_timeout_s=timeout))


def worker_fatal():
    """The REAL bootstrap's fatal for the owner's failure shape (an AttributeError with the library's advice text)."""
    from test_lab06_r1_startup import run
    import pinned_ultralytics as PU
    P = PU.load()
    err = AttributeError("\n".join(ADVICE[1:]))
    _out, ev = run(P, model={"predict_error": err})
    return ev[0]


def test_a_structured_worker_fatal_reaches_the_start_result_with_stage_exception_and_traceback(procs):
    fatal = worker_fatal()
    procs.append(Proc(out=[J(fatal)], err=["WARNING 'half' is deprecated and will be removed. Use 'quantize' instead."],
                      rc=2))
    c = client()
    r = c.start()
    f = r["failure"]
    assert r["event"] == "fatal" and r["error"].startswith("worker did not start: WorkerError: model warm-up")
    assert (f["stage"], f["exc_type"]) == ("warmup", "AttributeError") and "has no attribute 'half'" in f["message"]
    assert f["traceback"].startswith("Traceback") and f["before_ready"] is True and f["attempt"] == 1
    assert "deprecated" in f["stderr"]["text"] and f["stderr"]["truncated"] is False
    assert c.state == "unavailable" and c.receipt is None and c.session == 0 and c.last_failure is f


def test_a_multiline_exception_ending_in_generic_advice_keeps_the_exception_on_eof(procs):
    procs.append(Proc(out=[], err=TRACE, rc=1))
    c = client()
    r = c.start()
    f = r["failure"]
    assert "closed its output" in r["error"]                                 # old prefix kept
    assert "AttributeError: 'IterableSimpleNamespace' object has no attribute 'half'" in r["error"]
    assert f["category"] == "worker_eof" and f["exit_code"] == 1 and "Traceback" in f["stderr"]["text"]
    assert "pa.half" in f["stderr"]["text"] and not f["stderr"]["truncated"]


def test_nonzero_exit_without_eof_reports_the_exit_code(procs):
    hold = threading.Event()
    procs.append(Proc(out=[], err=["Fatal Python error: Aborted"], rc=9, hold=hold, exit_after=0.2))
    c = client()
    r = c.start()
    hold.set()
    assert "inference worker exited (code 9)" in r["error"] and "Fatal Python error: Aborted" in r["error"]
    assert r["failure"]["category"] == "worker_exit" and r["failure"]["exit_code"] == 9


def test_a_startup_deadline_kills_only_this_child_and_reports_the_deadline(procs):
    hold = threading.Event()
    p = Proc(out=[], err=["loading model..."], hold=hold)
    procs.append(p)
    c = client(timeout=0.4)
    t0 = time.monotonic()
    r = c.start()
    assert time.monotonic() - t0 < 3.0 and p.killed.is_set() and c.state == "unavailable"
    assert r["failure"]["category"] == "deadline" and "deadline exceeded" in r["error"]
    assert "loading model" in r["failure"]["stderr"]["text"]


def test_a_stderr_flood_is_bounded_with_explicit_truncation_and_keeps_the_final_cause(procs):
    flood = [f"noise {k:06d} " + "x" * 480 for k in range(20_000)]                    # about 10 MB
    procs.append(Proc(out=[], err=flood + ["RuntimeError: the real cause"], rc=1))
    c = client()
    r = c.start()
    se = r["failure"]["stderr"]
    assert se["truncated"] and se["dropped_chars"] > 9_000_000 and len(se["text"]) <= WC.STDERR_CAPTURE_CHARS + 200
    assert "noise 000000" in se["text"] and se["text"].rstrip().endswith("RuntimeError: the real cause")
    assert "characters of worker stderr truncated" in se["text"] and "RuntimeError: the real cause" in r["error"]
    assert len(c.stderr) == 60                                               # the old short deque stays bounded too


def test_a_previous_attempts_late_output_cannot_reach_the_next_attempt(procs):
    late_eof, late_err = threading.Event(), threading.Event()
    old = Proc(out=[], err=["attempt one starting"], hold=late_eof, err_gate=late_err, err_after=["OLD ATTEMPT LINE"])
    old.kill = lambda: old.killed.set()              # its pipes stay open after the kill (a slow OS teardown)
    procs.append(old)
    c = client(timeout=0.3)
    assert c.start()["failure"]["category"] == "deadline"
    procs.append(Proc(out=[J({"event": "ready", "device": "cuda:0"}),
                           J({"event": "pong", "job_id": "j1", "recv_wall": 0, "done_wall": 0})], hold=threading.Event()))
    assert c.start()["event"] == "ready" and c.session == 1 and c.attempts == 2
    late_eof.set()                                    # the OLD reader now sees EOF ...
    late_err.set()                                    # ... and the OLD child writes stderr
    time.sleep(0.2)
    assert c.ping(2.0)["event"] == "pong"             # the new attempt is not told "closed its output"
    assert "OLD ATTEMPT LINE" not in c._capture.snapshot()["text"] and "OLD ATTEMPT LINE" not in " ".join(c.stderr)
    assert c.state == "ready" and c.last_failure is None
    c.stop()


def test_display_assets_client_behaviour_is_unchanged_for_ready_requests_and_mid_request_death(procs):
    ended = threading.Event()
    p = Proc(out=[J({"event": "ready", "device": "cuda:0"})], err=["boom CUDA out of memory"], rc=9, hold=ended)
    p.stdin = DiesOnRequest(ended)
    procs.append(p)
    c = client()
    assert c.start()["event"] == "ready" and c.state == "ready" and c.session == 1
    with pytest.raises(RuntimeError) as e:
        c.ping(2.0)
    assert "closed its output" in str(e.value) or "exited" in str(e.value)
    assert c.state == "unavailable" and "CUDA out of memory" in " ".join(c.stderr)
    h = c.health()
    assert {"state", "reason", "pid", "session", "requests", "failures", "rtt_p95_s", "stderr_tail", "receipt"} <= set(h)
    assert h["last_failure"]["category"] == "worker_eof" and c.last_failure["before_ready"] is False


def test_the_yoloe_client_restarts_cleanly_after_a_failed_attempt(procs):
    from echo_watch.yoloe.client import YoloeWorkerClient, yoloe_spec
    procs.append(Proc(out=[J(worker_fatal())], rc=2))
    procs.append(Proc(out=[J({"event": "ready", "device": "cuda:0", "dtype": "torch.float32"})], hold=threading.Event()))
    s = Settings(data_dir=work_dir("lab06r1-yclient"))
    c = YoloeWorkerClient(yoloe_spec(s, cmd_override=["scripted"]))
    r1 = c.start()
    assert r1["event"] == "fatal" and c.last_failure["stage"] == "warmup"
    r2 = c.start()
    assert r2["event"] == "ready" and c.last_failure is None and c.attempts == 2 and c.session == 1
    c.release(0.1)
    assert c.state == "not started"
