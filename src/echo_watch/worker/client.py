# SPDX-License-Identifier: AGPL-3.0-only
"""Supervised client for ONE inference worker process (echo_watch.worker.main).

Adapted from the preserved ECHO Watch reference runtime/echowatch/inference.py (SHA-256
192e35c541dcbe14bb1eb426dd80248e5be07366beab6ff1f08a75959fa2f33b; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md): one request in flight,
one absolute deadline per request, a bounded reply, a shared-memory frame slot, a ``session`` that increments on
every successful start so a reply can never be attributed to a later process, and a visible unavailable state.

Adaptations for the paired-result path:
* a dedicated reader thread parses replies (the reference spawned a thread per read);
* no priority scheduler / periodic supersede machinery: the foundation has exactly one in-flight inference, taken by
  the analysis loop, and operator scans run only while no run is active;
* no automatic restart loop: a failed request marks the worker unavailable and the session fails or rolls back
  visibly. "Worker death" is decided from the OS process (``poll``) and reply liveness, never only a state label;
* every reply carries controller-side wall stamps next to the worker's, so IPC out, inference, post-processing and
  IPC back are attributed separately;
* the child environment is minimal and every profile/cache/temp directory is pointed INSIDE the project data root.

The backend is explicit and never substituted: device ``cuda:0`` requires CUDA in the worker interpreter or the worker
exits with a fatal receipt; there is no CPU fallback.

EW-LAB-06 R1 diagnostics (backward compatible; messages keep their old prefixes): every start ATTEMPT gets its own reply
queue, stderr line deque and bounded ``StderrCapture`` (first 16 KiB + last 48 KiB, explicit truncation count), so a
previous child's late output can never reach the next attempt. A worker ``fatal`` keeps its structured ``failure``;
an unexpected EOF/exit/deadline carries the exit code and the captured stderr, and the short message quotes the useful
exception lines instead of only the last three (often advisory) lines. Failures are exposed as ``last_failure`` and in
the start result.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

from ..config import ROOT, Limits
from ..guards import popen_windowless


class WorkerUnavailable(RuntimeError):
    pass


class WorkerError(RuntimeError):
    """A worker failure with structured detail (``failure``); still a RuntimeError for existing callers."""

    def __init__(self, message: str, failure: dict | None = None):
        super().__init__(message)
        self.failure = failure


class WorkerDeadline(TimeoutError):
    def __init__(self, message: str, failure: dict | None = None):
        super().__init__(message)
        self.failure = failure


STDERR_CAPTURE_CHARS = 64 * 1024
_USEFUL = ("Error", "Exception", "Traceback", "error:", "FATAL", "Fatal", "out of memory", "Killed", "Aborted")


class StderrCapture:
    """Bounded stderr of ONE worker attempt: the first quarter and the last three quarters of ``limit`` characters,
    with an explicit count of what was dropped in between."""

    def __init__(self, limit: int = STDERR_CAPTURE_CHARS):
        self.head_max, self.tail_max = limit // 4, limit - limit // 4
        self.head: list[str] = []
        self.head_n = 0
        self.tail: deque[str] = deque()
        self.tail_n = 0
        self.dropped_chars = 0
        self.lines = 0
        self.finished = False
        self._lock = threading.Lock()

    def add(self, line: str) -> None:
        line = line.rstrip("\r\n")
        with self._lock:
            self.lines += 1
            if self.head_n + len(line) + 1 <= self.head_max and not self.tail:
                self.head.append(line)
                self.head_n += len(line) + 1
                return
            if len(line) > self.tail_max:
                self.dropped_chars += len(line) - self.tail_max
                line = line[-self.tail_max:]
            self.tail.append(line)
            self.tail_n += len(line) + 1
            while self.tail_n > self.tail_max and self.tail:
                x = self.tail.popleft()
                self.tail_n -= len(x) + 1
                self.dropped_chars += len(x) + 1

    def all_lines(self) -> list[str]:
        with self._lock:
            return list(self.head) + list(self.tail)

    def snapshot(self) -> dict:
        with self._lock:
            parts = list(self.head)
            if self.dropped_chars:
                parts.append(f"[... {self.dropped_chars} characters of worker stderr truncated ...]")
            parts += list(self.tail)
            return {"text": "\n".join(parts), "truncated": self.dropped_chars > 0, "dropped_chars": self.dropped_chars,
                    "lines": self.lines, "limit_chars": self.head_max + self.tail_max, "complete": self.finished}

    def useful(self, n: int = 3) -> list[str]:
        """The last n exception-like lines; an exception header whose message starts on the next line (Python
        prints ``AttributeError:`` then the message) is joined with that line."""
        lines = [ln.strip() for ln in self.all_lines() if ln.strip()]
        hits = []
        for i, ln in enumerate(lines):
            if any(k in ln for k in _USEFUL):
                if ln.endswith(":") and i + 1 < len(lines):
                    ln = f"{ln} {lines[i + 1]}"
                hits.append(ln)
        return [ln[:300] for ln in (hits or lines)[-n:]]


@dataclass
class WorkerSpec:
    interpreter: Path
    model: Path
    model_sha256: str
    vp_model: Path | None
    vp_sha256: str | None
    config_dir: Path
    device: str = "cuda:0"
    precision: str = "fp32"
    threads: int = 4
    home_dir: Path | None = None         # profile/cache/temp root (inside the project data directory)
    cmd_override: list[str] | None = None   # tests only (scripted/fault-injecting stand-ins)
    startup_timeout_s: float = 90.0


def child_env(home: Path | None, extra: dict | None = None) -> dict:
    """A minimal environment for an owned child. No inherited PYTHON*/ECHOWATCH_*/PIP_* and no user-profile
    locations: caches and temp files land under ``home`` (inside the project)."""
    sysroot = os.environ.get("SYSTEMROOT", r"C:\Windows")
    env = {"SYSTEMROOT": sysroot, "WINDIR": sysroot, "PATH": sysroot + r"\System32",
           "NUMBER_OF_PROCESSORS": os.environ.get("NUMBER_OF_PROCESSORS", "1"),
           "PROCESSOR_ARCHITECTURE": os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
           # torch's CUDA build (inductor/triton cache location) calls getpass.getuser(), which needs one of these; without it
           # the GPU worker dies at import ("No username set in the environment") - found by the first real GPU start
           "USERNAME": os.environ.get("USERNAME") or "echo-native"}
    if home is not None:
        home.mkdir(parents=True, exist_ok=True)
        tmp = home / "tmp"
        tmp.mkdir(exist_ok=True)
        env.update(USERPROFILE=str(home), HOME=str(home), APPDATA=str(home / "AppData"),
                   LOCALAPPDATA=str(home / "AppData"), TEMP=str(tmp), TMP=str(tmp), TORCH_HOME=str(home / "torch"),
                   MPLCONFIGDIR=str(home / "mpl"), XDG_CACHE_HOME=str(home / "cache"))
    env.update(extra or {})
    return env


class WorkerClient:
    def __init__(self, spec: WorkerSpec, limits: Limits | None = None):
        self.spec = spec
        self.limits = limits or Limits()
        self.proc: subprocess.Popen | None = None
        self.receipt: dict | None = None
        self.state = "not started"        # not started | starting | ready | unavailable | stopped
        self.reason: str | None = None
        self.session = 0                  # increments on every successful start (a new process)
        self.stderr: deque[str] = deque(maxlen=60)
        self._capture = StderrCapture()
        self._drain_thread: threading.Thread | None = None
        self.attempts = 0
        self.last_failure: dict | None = None
        self.refs_classes: list | None = None
        self.refs_digest: str | None = None
        self.latency: deque[float] = deque(maxlen=2000)
        self.requests = 0
        self.failures = 0
        self.last_vram: tuple | None = None
        self._req_lock = threading.Lock()
        self._replies: "queue.Queue[dict]" = queue.Queue()
        self._job = 0
        self._shm: shared_memory.SharedMemory | None = None
        self._stopped = False
        self._reader: threading.Thread | None = None

    # ---- process -----------------------------------------------------------------------------------------
    def _cmd(self) -> list[str]:
        s = self.spec
        if s.cmd_override:
            return list(s.cmd_override)
        cmd = [str(s.interpreter), "-I", "-m", "echo_watch.worker.main", "--model", str(s.model),
               "--sha256", s.model_sha256, "--config-dir", str(s.config_dir), "--threads", str(s.threads),
               "--device", s.device, "--precision", s.precision]
        if s.vp_model is not None:
            cmd += ["--vp-model", str(s.vp_model), "--vp-sha256", s.vp_sha256 or ""]
        return cmd

    def start(self) -> dict:
        """Start the worker under its own startup deadline. Blocks until the ready receipt (or fails)."""
        if self._stopped:
            raise WorkerUnavailable("worker stopped")
        if self.state == "ready" and self.pid() is not None:
            return self.receipt or {}
        self._kill()
        self.state, self.reason = "starting", None
        self.attempts += 1
        self.last_failure = None
        # per-attempt buffers: a previous child's late output can never reach this attempt
        self.stderr = deque(maxlen=60)
        self._capture = StderrCapture()
        self._replies = queue.Queue()
        s = self.spec
        if not s.cmd_override and not s.interpreter.is_file():
            return self._fail(f"worker interpreter missing: {s.interpreter}")
        env = child_env(s.home_dir)
        env["PATH"] = str(s.interpreter.parent) + ";" + env["PATH"]
        t0 = time.monotonic()
        self.proc = popen_windowless(self._cmd(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, bufsize=1, text=True, encoding="utf-8", env=env,
                                     cwd=str(ROOT))
        self._drain_thread = threading.Thread(target=self._drain, args=(self.proc, self.stderr, self._capture),
                                              name="worker-stderr", daemon=True)
        self._drain_thread.start()
        self._reader = threading.Thread(target=self._read_loop, args=(self.proc, self._replies), name="worker-replies",
                                        daemon=True)
        self._reader.start()
        try:
            msg = self._wait(None, t0 + s.startup_timeout_s)
        except Exception as e:  # noqa: BLE001
            return self._fail(f"worker did not start: {type(e).__name__}: {e}", getattr(e, "failure", None),
                              before_ready=True)
        if msg.get("event") != "ready":
            return self._fail(msg.get("error") or f"worker not ready: {msg}")
        if s.device.startswith("cuda") and not str(msg.get("device", "")).startswith("cuda"):
            return self._fail(f"worker reports device {msg.get('device')!r}, not the requested {s.device}; "
                              "no CPU substitute is accepted")
        msg["start_s"] = round(time.monotonic() - t0, 3)
        self.receipt = msg
        self.session += 1
        self.state = "ready"
        return msg

    def _fail(self, why: str, failure: dict | None = None, before_ready: bool = False) -> dict:
        self._kill()
        self.state, self.reason = "unavailable", why
        self.failures += 1
        self.last_failure = self._failure(failure, why, before_ready)
        return {"event": "fatal", "error": why, "failure": self.last_failure}

    def _settle_stderr(self, wait_s: float = 1.0) -> None:
        """Bounded wait for THIS attempt's stderr reader to finish (the child has ended or been killed)."""
        t = self._drain_thread
        if t is not None and t is not threading.current_thread():
            t.join(wait_s)

    def _failure(self, failure: dict | None, why: str, before_ready: bool) -> dict:
        out = dict(failure or {"category": "worker_start" if before_ready else "worker_request", "message": why})
        self._settle_stderr()
        out.setdefault("stderr", self._capture.snapshot())
        out.update(attempt=self.attempts, before_ready=before_ready, client_message=why[:2000])
        return out

    def _ended(self, kind: str, rc) -> dict:
        self._settle_stderr()
        return {"category": kind, "exit_code": rc, "stderr": self._capture.snapshot(),
                "stderr_useful": self._capture.useful(6)}

    def _drain(self, proc: subprocess.Popen, lines: deque | None = None, capture: StderrCapture | None = None) -> None:
        lines = self.stderr if lines is None else lines
        try:
            for line in proc.stderr:
                lines.append(line.rstrip()[:300])
                if capture is not None:
                    capture.add(line)
        except Exception:  # noqa: BLE001
            pass
        if capture is not None:
            capture.finished = True

    def _read_loop(self, proc: subprocess.Popen, replies: "queue.Queue | None" = None) -> None:
        replies = self._replies if replies is None else replies
        try:
            while True:
                line = proc.stdout.readline(self.limits.worker_reply_max_bytes + 1)
                t_read = time.time()
                if not line:
                    break
                if len(line) > self.limits.worker_reply_max_bytes:
                    replies.put({"event": "__oversize__", "_session": self.session})
                    break
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                obj["_read_wall"] = t_read
                replies.put(obj)
        except Exception:  # noqa: BLE001
            pass
        replies.put({"event": "__eof__"})

    def _kill(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:     # only the child this client created
            p.kill()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass

    def pid(self) -> int | None:
        p = self.proc
        return p.pid if p is not None and p.poll() is None else None

    def alive(self) -> bool:
        return self.state == "ready" and self.pid() is not None

    # ---- protocol ----------------------------------------------------------------------------------------
    def _wait(self, job_id: str | None, deadline: float) -> dict:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                rc = self.proc.poll() if self.proc is not None else None
                raise WorkerDeadline("inference worker deadline exceeded", {**self._ended("deadline", rc),
                                                                            "exit_code": rc})
            try:
                msg = self._replies.get(timeout=min(remaining, 0.5))
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    f = self._ended("worker_exit", self.proc.returncode)
                    tail = " | ".join(f["stderr_useful"][-3:])
                    raise WorkerError(f"inference worker exited (code {self.proc.returncode})"
                                      f"{': ' + tail if tail else ''}", f)
                continue
            ev = msg.get("event")
            if ev == "__eof__":
                rc = None
                if self.proc is not None:
                    try:
                        rc = self.proc.wait(1.0)
                    except subprocess.TimeoutExpired:
                        rc = None
                f = self._ended("worker_eof", rc)
                tail = " | ".join(f["stderr_useful"][-3:])
                raise WorkerError(f"inference worker closed its output{': ' + tail if tail else ''}", f)
            if ev == "__oversize__":
                raise WorkerError("inference worker reply exceeded the bound", {"category": "reply_oversize"})
            if ev == "fatal":
                raise WorkerError(msg.get("error", "worker fatal"),
                                  {"category": "worker_fatal", **(msg.get("failure") or {}),
                                   "worker_error": msg.get("error")})
            if job_id is not None and msg.get("job_id") != job_id:
                continue          # a reply to an earlier (timed-out) job is never attributed to this one
            return msg

    def _send(self, obj: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def request(self, payload: dict, timeout_s: float, prepare=None) -> dict:
        """One request in flight. ``prepare`` (optional) runs while this request owns the worker and returns extra
        payload fields (used to write the shared-memory frame slot only while no other request can read it)."""
        if self._stopped:
            raise WorkerUnavailable("worker stopped")
        if self.state != "ready" or self.pid() is None:
            raise WorkerUnavailable(self.reason or f"worker {self.state}")
        deadline = time.monotonic() + timeout_s
        if not self._req_lock.acquire(timeout=timeout_s):
            raise TimeoutError("request deadline expired while waiting for the worker")
        try:
            self._job += 1
            job_id = f"j{self._job}"
            session = self.session
            payload = {**payload, "job_id": job_id, **(prepare() if prepare is not None else {})}
            t_send_wall, t_send = time.time(), time.perf_counter()
            self.requests += 1
            try:
                self._send(payload)
                msg = self._wait(job_id, deadline)
            except Exception as e:  # noqa: BLE001
                self.failures += 1
                self._kill()
                self.state, self.reason = "unavailable", f"{type(e).__name__}: {e}"
                self.last_failure = self._failure(getattr(e, "failure", None), self.reason, False)
                raise
            if self.session != session:
                raise WorkerUnavailable("worker session changed during the request")
            t_done = time.perf_counter()
            self.latency.append(t_done - t_send)
            if msg.get("event") == "error":
                raise RuntimeError(msg.get("error"))
            msg["_session"] = session
            if msg.get("vram_free_mib") is not None:
                self.last_vram = (msg.get("vram_free_mib"), msg.get("vram_total_mib"))
            msg["_send_wall"] = t_send_wall
            msg["_rtt_s"] = t_done - t_send
            if "recv_wall" in msg and "done_wall" in msg:
                msg["_ipc_out_s"] = msg["recv_wall"] - t_send_wall
                msg["_worker_s"] = msg["done_wall"] - msg["recv_wall"]
                msg["_ipc_back_s"] = msg.get("_read_wall", msg["done_wall"]) - msg["done_wall"]
            return msg
        finally:
            self._req_lock.release()

    def put_frame(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        n = h * w * 3
        if n > self.limits.frame_max_bytes:
            raise ValueError("frame exceeds 1920x1080")
        if self._shm is None:
            self._shm = shared_memory.SharedMemory(create=True, size=self.limits.frame_max_bytes)
        np.ndarray((h, w, 3), np.uint8, buffer=self._shm.buf[:n])[:] = image
        return {"shm": self._shm.name, "shape": [h, w]}

    def scan(self, image_path: Path, imgsz: int, conf: float, iou: float, max_det: int, timeout_s: float,
             area: list | None = None) -> dict:
        return self.request({"cmd": "scan", "image_path": str(image_path), "imgsz": imgsz, "conf": conf, "iou": iou,
                             "max_det": max_det, "area": area}, timeout_s)

    def set_refs(self, refs: list[dict], digest: str, timeout_s: float = 60.0) -> dict:
        reply = self.request({"cmd": "set_refs", "refs": refs}, timeout_s)
        self.refs_classes, self.refs_digest = reply.get("classes"), digest     # only acknowledged references count
        return reply

    def verify(self, image: np.ndarray, frame_ref: str, timeout_s: float, imgsz: int, conf: float, iou: float,
               max_det: int) -> dict:
        return self.request({"cmd": "verify", "frame_ref": frame_ref, "imgsz": imgsz, "conf": conf, "iou": iou,
                             "max_det": max_det, "polygons": False}, timeout_s,
                            prepare=lambda: self.put_frame(image))

    def ping(self, timeout_s: float = 5.0) -> dict:
        return self.request({"cmd": "ping"}, timeout_s)

    def stop(self) -> None:
        """Ask the worker to exit, then terminate ONLY this client's child if it does not within a bound."""
        self._stopped = True
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                self._send({"cmd": "quit", "job_id": "quit"})
            except Exception:  # noqa: BLE001
                pass
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        self._kill()
        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except (OSError, FileNotFoundError):
                pass
            self._shm = None
        self.state = "stopped"

    def health(self) -> dict:
        lat = sorted(self.latency)
        return {"state": self.state, "reason": self.reason, "pid": self.pid(), "session": self.session,
                "requests": self.requests, "failures": self.failures,
                "rtt_p95_s": round(lat[int(0.95 * (len(lat) - 1))], 4) if lat else None,
                "stderr_tail": list(self.stderr)[-3:], "receipt": self.receipt, "attempts": self.attempts,
                "last_failure": None if self.last_failure is None else
                {k: self.last_failure.get(k) for k in ("category", "stage", "exc_type", "client_message")}}
