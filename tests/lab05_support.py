# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-05 test support (GUI-free, codec-free, process-free).

* synthetic stores under data/tests (never the operator store);
* ``codec_stubs`` replaces the PNG encoder/decoder used by Library with np.save/np.load (no media codec call);
* ``FakeWorker`` answers Scan with configured candidates (no model); ``FakeSession`` records what the REAL engine Start
  boundary (``ItemEngine.start``, used unbound) hands to a session and settles with a configurable outcome;
* frames are numpy arrays; "videos" are small byte files whose CONTENT defines the replay scope.
"""
from __future__ import annotations

import io
import time
import uuid
from pathlib import Path

import numpy as np

from echo_watch.app.controller import WorkbenchController
from echo_watch.app.tasks import TaskRunner
from echo_watch.config import Settings
from echo_watch.engine import library as libmod
from echo_watch.engine.api import ItemEngine
from echo_watch.engine.contracts import (PHASE_ACTIVE, PHASE_IDLE, PHASE_PAUSED, CaptureOptions, OutputOutcome,
                                         SessionOutcome)
from echo_watch.engine.library import Library
from echo_watch.engine.store import Store
from echo_watch.frames import Epoch, Frame
from support import work_dir

SIZE = (640, 360)
CANDIDATES = [("phone", [100.0, 100.0, 60.0, 120.0]), ("mug", [300.0, 200.0, 80.0, 80.0]),
              ("table", [40.0, 60.0, 560.0, 280.0])]


def _enc(img) -> bytes:
    b = io.BytesIO()
    np.save(b, np.ascontiguousarray(img), allow_pickle=False)
    return b.getvalue()


def _dec(data: bytes):
    return np.load(io.BytesIO(data), allow_pickle=False)


def codec_stubs(monkeypatch) -> None:
    monkeypatch.setattr(libmod, "encode_png", _enc)
    monkeypatch.setattr(libmod, "decode_image", _dec)


def image(seed: int = 0) -> np.ndarray:
    img = np.full((SIZE[1], SIZE[0], 3), 40 + seed % 50, np.uint8)
    img[100:220, 100:160] = 200
    img[200:280, 300:380] = 120
    return img


def frame(scope: str, seed: int = 0) -> Frame:
    return Frame(Epoch("ep_setup", scope, 0, 0.0, 0.0), 1, time.monotonic(), time.time(), image(seed))


class FakeWorker:
    session = 1

    def __init__(self, candidates=None):
        self.receipt = {"model_sha256": "pf-test-model", "device": "cuda:0"}
        self.candidates = list(candidates or CANDIDATES)
        self.started = True
        self.delay = 0.0
        self.fail = None
        self.reason = None
        self.last_vram = None

    def alive(self):
        return self.started

    def scan(self, image_path, imgsz, conf, iou, max_det, timeout_s, area=None):
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError(self.fail)
        return {"candidates": [{"box": list(b), "label": n, "confidence": 0.9 - 0.1 * i, "polygon": None}
                               for i, (n, b) in enumerate(self.candidates)], "infer_s": 0.01, "device": "cuda:0",
                "_session": 1}


class FakeSession:
    def __init__(self):
        self.phase = PHASE_IDLE
        self.last_outcome = None
        self.discards = {}
        self.started: list[dict] = []
        self.outcome_kind = ("finalized", "operator")
        self.statuses = ("finalized", "finalized", "finalized")

    @property
    def busy(self):
        return self.phase != PHASE_IDLE

    def live_events(self, item_id, limit=8):
        return []

    def snapshot(self):
        return None

    def start_driver(self):
        pass

    def start(self, options, refs, source, *, allowance_s=60.0, setup=None):
        self.started.append({"options": options, "refs": [(r.item_id, r.ref_version) for r in refs], "source": source,
                             "allowance_s": allowance_s, "setup": setup})
        self.phase = PHASE_ACTIVE

    def pause(self):
        self.phase = PHASE_PAUSED

    def resume(self):
        self.phase = PHASE_ACTIVE

    def stop(self, **kw):
        self.phase = PHASE_IDLE
        kind, by = self.outcome_kind
        c, a, t = (OutputOutcome(True, s) for s in self.statuses)
        self.last_outcome = SessionOutcome(kind, by, f"run_{uuid.uuid4().hex[:8]}", None, CaptureOptions(), None, c, a, t)


class FakeEngine:
    """Real Store + Library + the REAL ItemEngine.start boundary; fake worker/session; numpy setup frames."""

    start = ItemEngine.start
    model_identity = ItemEngine.model_identity

    def __init__(self, d: Path, candidates=None):
        self.s = Settings(data_dir=d)
        self.s.ensure_dirs()
        self.store = Store(self.s.db_path)
        self.library = Library(self.s, self.store)
        self.worker = FakeWorker(candidates)
        self.session = FakeSession()

    def start_worker(self):
        self.worker.started = True
        return {"event": "ready", "device": "cuda:0"}

    def open_setup_source(self, path, fps, source_id="setup", timeout_s=30.0):
        return frame(source_id, seed=sum(Path(path).read_bytes()[:8]))

    def scan(self, f, source_id):
        return self.library.scan(f, source_id, self.worker)

    def latest_result(self):
        return None

    def shutdown(self, wait_s=30.0):
        return {}


def settle(ctl, cond, timeout=10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        ctl.pump()
        if cond():
            return True
        time.sleep(0.005)
    ctl.pump()
    return bool(cond())


class Rig:
    """A controller on a synthetic store with three local 'videos': A, B (different bytes, same frame size) and a
    byte-identical copy of A at another path."""

    def __init__(self, name: str, monkeypatch, candidates=None):
        codec_stubs(monkeypatch)
        self.d = work_dir(f"{name}-{uuid.uuid4().hex[:6]}")
        self.eng = FakeEngine(self.d, candidates)
        self.ctl = WorkbenchController(self.eng, TaskRunner(3), probe=lambda p: {"effective_fps": 15.0})
        self.a, self.b, self.a_copy = self.d / "video_a.mp4", self.d / "video_b.mp4", self.d / "copy" / "renamed_a.mp4"
        self.a.write_bytes(b"recording A " * 64)
        self.b.write_bytes(b"recording B " * 64)
        self.a_copy.parent.mkdir()
        self.a_copy.write_bytes(self.a.read_bytes())

    def close(self) -> None:
        self.ctl.tasks.shutdown(1.0)
        self.eng.store.close()

    def prepare(self, video: Path) -> None:
        c = self.ctl
        assert c.choose_source(str(video))
        assert settle(c, lambda: c.source_status == "ready")

    def discover(self, video: Path | None = None) -> None:
        c = self.ctl
        if video is not None:
            self.prepare(video)
        assert c.scan()
        assert settle(c, lambda: c.scan_status == "done")
        assert settle(c, lambda: not c.tasks.busy("saved"))
        assert c.go("select") and c.view_step == "select"

    def cid(self, label: str) -> str:
        return next(x.candidate_id for x in self.ctl.candidates if x.label == label)

    def remember(self) -> None:
        c = self.ctl
        assert c.remember(), [m.text for m in c.messages[-2:]]
        assert settle(c, lambda: not c.remember_pending)

    def start(self) -> dict:
        c = self.ctl
        assert c.start(), [m.text for m in c.messages[-2:]]
        assert settle(c, lambda: not c.start_pending)
        assert self.eng.session.started, [m.text for m in c.messages[-2:]]
        return self.eng.session.started[-1]

    def stop(self, kind=("finalized", "operator"), statuses=("finalized", "finalized", "finalized")) -> None:
        c = self.ctl
        self.eng.session.outcome_kind, self.eng.session.statuses = kind, statuses
        assert c.stop()
        assert settle(c, lambda: c.finished is not None)
