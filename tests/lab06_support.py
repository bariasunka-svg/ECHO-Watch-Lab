# SPDX-License-Identifier: AGPL-3.0-only
"""GUI-free, model-free support for the EW-LAB-06 tests (UI runtime).

* ``SpyModel``: a scripted stand-in detector that records EVERY ``predict`` call and its keyword arguments.
* ``ScriptTracker``: a deterministic stand-in with the BotSortAdapter interface (snapshot/restore, unsettled refusal,
  tentative-then-confirmed ids keyed by class). The REAL BoT-SORT adapter is verified separately in the GPU runtime
  (tools/lab06/tracker_synthetic_check.py); here the point is the session/worker-core protocol around it.
* ``InProcessClient``: the YoloeWorkerClient API around the PRODUCTION ``YoloeCore`` with JSON round-trips, plus fault
  hooks (slow inference, failure, wrong shape, restore failure).
* ``ScriptSource``: an ordered source over generated frames (in order, SourceEnded at the end, records every read).
No Tk, no model, no GPU, no FFmpeg, no worker process.
"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path

import numpy as np

from echo_watch.frames import Epoch, Frame
from echo_watch.sources.base import SourceEnded
from echo_watch.sources.fingerprint import fingerprint
from echo_watch.yoloe.session import SourceInfo
from echo_watch.yoloe.worker import YoloeCore
from support import work_dir as _work_dir


def work_dir(name: str) -> Path:
    """A NEW directory per call (support.work_dir is per-second; back-to-back runs would otherwise share it)."""
    import uuid
    return _work_dir(f"{name}-{uuid.uuid4().hex[:8]}")

NAMES = {0: "person", 1: "cup", 2: "bottle", 3: "laptop", 4: "mug", 5: "phone"}


class _Boxes:
    def __init__(self, rows):
        self.xyxy = np.array([r[0] for r in rows], np.float32).reshape(-1, 4)
        self.conf = np.array([r[1] for r in rows], np.float32)
        self.cls = np.array([r[2] for r in rows], np.float32)

    def __len__(self):
        return len(self.conf)


class _Result:
    def __init__(self, rows):
        self.boxes = _Boxes(rows) if rows else None
        self.speed = {"preprocess": 1.0, "inference": 5.0, "postprocess": 1.0}
        self.masks = object()


class SpyModel:
    """``script(n, img)`` -> [(xyxy, score, cls)] for the n-th predict call (0-based)."""

    def __init__(self, script=None):
        self.calls: list[dict] = []
        self.script = script or (lambda n, img: [((10, 10, 30, 40), 0.9, 1), ((40, 5, 60, 30), 0.15, 2)])

    def predict(self, img, **kw):
        n = len(self.calls)
        self.calls.append({**kw, "shape": tuple(img.shape)})
        return [_Result(self.script(n, img))]


class ScriptTracker:
    def __init__(self):
        self.ids: dict[int, int] = {}
        self.seen: dict[int, int] = {}
        self.frame_id = 0
        self.next = 0
        self._pending = None
        self.updates = 0
        self.fail_restore = False

    def identity(self):
        return {"class": "ScriptTracker", "encoder": None, "gmc_method": "sparseOptFlow", "max_frames_lost": 30}

    @property
    def unsettled(self):
        return self._pending is not None

    def update(self, dets, img):
        if self._pending is not None:
            raise RuntimeError("the previous tracker update is not settled")
        snap = copy.deepcopy((self.ids, self.seen, self.frame_id, self.next))
        self.frame_id += 1
        rows, tentative = [], []
        for k, d in enumerate(np.asarray(dets).reshape(-1, 6)):
            x1, y1, x2, y2, s, c = (float(v) for v in d)
            if s < 0.25:
                continue
            c = int(c)
            if c not in self.ids:
                self.next += 1
                self.ids[c] = self.next
                self.seen[c] = 0
            self.seen[c] += 1
            if self.seen[c] == 1 and self.frame_id != 1:
                tentative.append((self.ids[c], float(k), [x1, y1, x2, y2], s, float(c)))
            else:
                rows.append([x1, y1, x2, y2, float(self.ids[c]), s, float(c), float(k)])
        self._pending = snap
        self.updates += 1
        return {"rows": rows, "tentative": tentative, "assoc": {}, "errors": [], "frame_id": self.frame_id,
                "lost": 0, "removed_this_frame": 0, "warnings": [], "tracker_s": 0.0001}

    def settle(self, accept):
        if self._pending is None:
            return False
        if not accept:
            if self.fail_restore:
                raise RuntimeError("restore failed (injected)")
            self.ids, self.seen, self.frame_id, self.next = self._pending
            self.updates -= 1
        self._pending = None
        return True


class InProcessClient:
    """The YoloeWorkerClient surface over the production YoloeCore (JSON round-trips, no process)."""

    def __init__(self, model=None, *, delay_s=0.0, ready=True):
        self.model = model or SpyModel()
        self.trackers: list[ScriptTracker] = []
        self.core = YoloeCore(self.model, NAMES, "cuda:0", None, self._new_tracker)
        self.state, self.reason, self.session, self.receipt = "not started", None, 0, None
        self.delay_s, self.ready = delay_s, ready
        self.calls: list[tuple] = []
        self.fail_at: str | None = None
        self.shape_at: str | None = None
        self.gate: threading.Event | None = None
        self.released = 0

    def _new_tracker(self):
        t = ScriptTracker()
        self.trackers.append(t)
        return t

    def alive(self):
        return self.state == "ready"

    def start(self):
        self.calls.append(("start",))
        if not self.ready:
            self.state, self.reason = "unavailable", "injected start failure"
            return {"event": "fatal", "error": "injected start failure"}
        self.state, self.session = "ready", self.session + 1
        self.receipt = {"event": "ready", "device": "cuda:0", "dtype": "torch.float32", "vocabulary": len(NAMES),
                        "end2end": True, "iou_applies": False, "worker_pid": 0}
        return self.receipt

    def _rt(self, msg):
        out = json.loads(json.dumps(msg))
        out.update(_session=self.session, _rtt_s=0.001)
        return out

    def reset_epoch(self, epoch, timeout_s=15.0):
        self.calls.append(("reset", epoch))
        return self._rt(self.core.reset(epoch))

    def detect_track(self, image, epoch, frame, accept, timeout_s):
        self.calls.append(("detect_track", epoch, frame, accept))
        if self.gate is not None:
            self.gate.wait(5)
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail_at == frame:
            self.state = "unavailable"
            raise TimeoutError("inference worker deadline exceeded (injected)")
        rep = self.core.detect_track({"epoch": epoch, "frame": frame, "accept": accept}, image)
        if self.shape_at == frame:
            rep["input_shape"] = [1, 1]
        return self._rt(rep)

    def settle(self, epoch, frame, decision, timeout_s=5.0):
        self.calls.append(("settle", epoch, frame, decision))
        return self._rt(self.core.settle({"epoch": epoch, "frame": frame, "decision": decision}))

    def release(self, wait_s=5.0):
        self.released += 1
        self.state = "not started"

    def stop(self):
        self.state = "stopped"

    def detect_calls(self):
        return [c for c in self.calls if c[0] == "detect_track"]


def media_frames(n: int, fps: float = 10.0, size=(64, 48), pts=None, gaps: dict | None = None) -> list[tuple]:
    """Generated frames (value-coded pixels) with verified media facts; ``pts`` overrides timestamps (VFR);
    ``gaps`` {output index: missing source frames before it}."""
    out, src = [], -1
    for k in range(n):
        img = np.full((size[1], size[0], 3), (k * 7) % 256, np.uint8)
        img[0, 0] = (k % 256, k // 256, 77)
        src += 1 + (gaps or {}).get(k, 0)
        t = pts[k] if pts is not None else k / fps
        md = {"out_n": k, "pts": round(t, 6), "media_pts": round(t, 6), "src_n": src, "verified": True}
        if (gaps or {}).get(k):
            md["gap_before"] = gaps[k]
        out.append((img, md))
    return out


class ScriptSource:
    def __init__(self, frames, *, error_after: int | None = None):
        self.frames = list(frames)
        self.i = 0
        self.reads = 0
        self.stopped = False
        self.started = False
        self.error_after = error_after
        self.epoch = None

    def start(self, epoch):
        self.started, self.epoch = True, epoch

    def read(self, timeout):
        self.reads += 1
        if self.stopped:
            return None
        if self.error_after is not None and self.i >= self.error_after:
            from echo_watch.sources.base import SourceError
            raise SourceError("decoder exited (code 1): injected")
        if self.i >= len(self.frames):
            raise SourceEnded("video file ended")
        img, md = self.frames[self.i]
        self.i += 1
        return Frame(self.epoch or Epoch.begin("yoloe", 0), self.i, time.monotonic(), time.time(), img, media=dict(md))

    def stop(self):
        self.stopped = True

    def diagnostics(self):
        return {"decoder": "script", "size": [64, 48], "native_size": [128, 96], "stats": {"max_queue": 0},
                "media": {"frames": len(self.frames)}, "queue_frames": 4}


def source_file(d: Path, name: str = "clip.mp4", payload: bytes = b"not really a video") -> SourceInfo:
    p = d / name
    p.write_bytes(payload)
    fp = fingerprint(p)
    return SourceInfo(str(p), name, fp["bytes"], fp["sha256"], tuple(fp["state"]))


def wait_for(cond, timeout: float = 10.0, step: float = 0.005) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


def read_trace(run_dir) -> list[dict]:
    return [json.loads(x) for x in (Path(run_dir) / "trace.jsonl").read_text("utf-8").splitlines() if x.strip()]
