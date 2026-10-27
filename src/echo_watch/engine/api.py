# SPDX-License-Identifier: AGPL-3.0-only
"""The engine facade: the command / snapshot / result / outcome / review interface the later native view will call.

Commands (all non-UI, all returning immutable values):
  start_worker()                      load the pinned models in the isolated worker (explicit device, no CPU fallback)
  open_setup_source(path, fps)        replay a file once and RETAIN its first frame for Scan (setup view)
  scan(frame, source_id)              real prompt-free discovery on exactly that retained frame -> Proposals
  remember(op_id, scan_id, picks)     operator selection -> versioned item references (fresh library, idempotent)
                                      (the workbench uses library.enroll: new + explicit reuse as one command)
  update_reference(...)               explicit new reference version (never silent)
  start(options, items, source, allowance_s=, setup=)
                                      managed Prepare/Start -> poll() until phase 'active' (or a rolled-back outcome);
                                      items are exact (item_id, ref_version) pairs validated at this boundary with
                                      engine.compat (content scope, size, model) and their crop hashes
  pause() / resume() / stop()         advance the control generation; stop finalizes every owned output
  poll()                              advance the phase machine one tick (the native UI calls it from its timer)
Reads:
  snapshot() -> EngineSnapshot        phase/step, per-item rows, rates (capture / analysed / encoded; displayed is the
                                      view's own number), output states, source and worker health
  latest_result() -> ResultPacket     the newest COMMITTED paired image + results (immutable)
  list_runs() / run_detail() / item_timeline() / trace_window() / frame_map()    saved-run review queries
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..config import (FFMPEG, FFMPEG_SHA256, GPU_PYTHON, MODEL_PF, MODEL_PF_SHA256, MODEL_VP, MODEL_VP_SHA256,
                      Settings)
from ..frames import Epoch, Frame
from ..sources.base import SourceError
from ..sources.ffmpeg_source import FFmpegFileSource
from ..worker.client import WorkerClient, WorkerSpec
from . import compat
from .contracts import CaptureOptions, EngineSnapshot, ItemRef, ResultPacket
from .files import sha256_file
from .library import Library, LibraryError, ScanResult
from .session import ManagedItemSession, PollResult, SourceSpec
from .store import Store, loads


class ItemEngine:
    def __init__(self, settings: Settings, *, worker_cmd: list[str] | None = None):
        settings.ensure_dirs()
        self.s = settings
        self.store = Store(settings.db_path)
        self.library = Library(settings, self.store)
        # inference runs only in the independent GPU environment; there is no CPU interpreter or fallback
        interpreter = GPU_PYTHON
        spec = WorkerSpec(interpreter, MODEL_PF, MODEL_PF_SHA256, MODEL_VP, MODEL_VP_SHA256, settings.dirs()["ultralytics"],
                          settings.device, settings.precision, settings.threads,
                          settings.data_dir / "runtime" / "home", worker_cmd or settings.worker_cmd,
                          settings.limits.worker_ready_timeout_s)
        self.worker = WorkerClient(spec, settings.limits)
        self.session = ManagedItemSession(settings=settings, store=self.store, worker=self.worker)

    # ---- worker ----------------------------------------------------------------------------------------------
    def verify_pinned_artifacts(self) -> dict:
        """Hash the exact FFmpeg executable and both model files before any use (a mismatch refuses to run)."""
        out = {"ffmpeg": {"path": str(FFMPEG), "sha256": sha256_file(FFMPEG), "expected": FFMPEG_SHA256},
               "model_pf": {"path": str(MODEL_PF), "sha256": sha256_file(MODEL_PF), "expected": MODEL_PF_SHA256},
               "model_vp": {"path": str(MODEL_VP), "sha256": sha256_file(MODEL_VP), "expected": MODEL_VP_SHA256}}
        for k, v in out.items():
            v["ok"] = v["sha256"] == v["expected"]
        return out

    def start_worker(self) -> dict:
        return self.worker.start()

    # ---- setup ----------------------------------------------------------------------------------------------------
    def open_setup_source(self, path: str, fps: float, source_id: str = "setup", timeout_s: float = 30.0) -> Frame:
        """Decode the file's first frame in its own setup epoch and retain it; the source is then closed. Monitoring later
        restarts the file into a NEW epoch."""
        src = FFmpegFileSource(source_id, FFMPEG, path, fps, loop_file=False)
        src.start(Epoch.begin(source_id, 0, 0, 0))
        try:
            end = time.monotonic() + timeout_s
            while time.monotonic() < end:
                f = src.read(0.5)
                if f is not None:
                    return Frame(f.epoch, f.seq, f.t_mono, f.t_wall, f.image.copy(), f.synthetic, f.media)
            raise SourceError("the setup source produced no frame in time")
        finally:
            src.stop()

    def scan(self, frame: Frame, source_id: str) -> ScanResult:
        return self.library.scan(frame, source_id, self.worker)

    def remember(self, op_id: str, scan_id: str, picks: list[dict]) -> list[ItemRef]:
        return self.library.remember(op_id, scan_id, picks)

    def model_identity(self) -> str:
        """The discovery model identity references are compatible with: the running worker's verified model, else the
        pinned prompt-free model."""
        return (self.worker.receipt or {}).get("model_sha256") or MODEL_PF_SHA256

    # ---- managed session --------------------------------------------------------------------------------------------
    def start(self, options: CaptureOptions, items: list, source: SourceSpec, *, allowance_s: float = 60.0,
              setup: dict | None = None) -> None:
        """The engine-facing Start boundary (EW-LAB-05). ``items``: exact (item_id, ref_version) pairs (item ids alone
        mean their current versions, for engine harnesses). Every reference is re-validated here with the ONE
        compatibility predicate against this source's content scope, size and model identity, and its saved crop hash
        is re-checked, BEFORE worker preparation. ``setup`` (a SavedSetup.identity()) must name the same recording and
        exactly these references; the session binds and records it."""
        scope = compat.Scope(source.source_id, tuple(source.size) if source.size else None, self.model_identity())
        refs = self.library.resolve_refs(items, scope)
        if setup is not None:
            if setup.get("scope") not in (None, source.source_id):
                raise LibraryError("the saved setup belongs to a different recording; discover items for this one")
            if [list(x) for x in setup.get("items", [])] != [[r.item_id, r.ref_version] for r in refs]:
                raise LibraryError("the saved setup does not list exactly these reference versions")
        self.session.start(options, refs, source, allowance_s=allowance_s, setup=setup)

    def poll(self) -> PollResult:
        return self.session.poll()

    def pause(self) -> None:
        self.session.pause()

    def resume(self) -> None:
        self.session.resume()

    def stop(self) -> None:
        self.session.stop()

    def snapshot(self) -> EngineSnapshot:
        return self.session.snapshot()

    def latest_result(self) -> ResultPacket | None:
        return self.session.latest_packet()

    def shutdown(self, wait_s: float = 30.0) -> dict:
        """Application-owned shutdown: an active session is stopped and given ``wait_s`` to finalize every owned output
        BEFORE the worker is stopped; the receipt says whether closure was complete (never claims a clean close that did
        not happen)."""
        rec: dict = {"session_was_busy": self.session.busy}
        if self.session.busy:
            self.session.stop(ended_by="operator")
            if self.session._driver is None or not self.session._driver.is_alive():
                self.session.start_driver()
            out = self.session.wait_outcome(wait_s)
            rec["session_settled"] = out is not None
            rec["outcome_complete"] = bool(out and out.complete)
            rec["outcome"] = None if out is None else out.summary_text()
        self.session.stop_driver()
        self.worker.stop()
        self.store.close()
        rec["worker_state"] = self.worker.state
        return rec

    # ---- review queries (saved runs; no inference, no device, no media decode) -------------------------------------------
    def list_runs(self) -> list[dict]:
        return [{"run_id": r["run_id"], "phase": r["phase"], "started_wall": r["started_wall"], "ended_wall": r["ended_wall"],
                 "outcome": loads(r["outcome"]), "dir": r["dir"]}
                for r in self.store.query("SELECT * FROM runs ORDER BY created_wall DESC")]

    def run_detail(self, run_id: str) -> dict | None:
        r = self.store.one("SELECT * FROM runs WHERE run_id=?", (run_id,))
        if r is None:
            return None
        mp = Path(r["dir"]) / "manifest.json"
        return {"run": dict(r), "items": [dict(x) for x in self.store.query("SELECT * FROM run_items WHERE run_id=?", (run_id,))],
                "manifest": json.loads(mp.read_text("utf-8")) if mp.is_file() else None}

    def item_timeline(self, run_id: str, item_id: str) -> dict:
        iv = [dict(x) for x in self.store.query("SELECT * FROM run_intervals WHERE run_id=? AND item_id=? ORDER BY start_mono",
                                                (run_id, item_id))]
        ev = [loads(x["payload"]) for x in self.store.query("SELECT payload FROM run_events WHERE run_id=? AND item_id=? "
                                                            "ORDER BY event_id", (run_id, item_id))]
        return {"intervals": iv, "events": ev}

    def trace_window(self, run_id: str, kinds: tuple[str, ...] | None = None, t_from: float | None = None,
                     t_to: float | None = None, limit: int = 1000) -> list[dict]:
        r = self.store.one("SELECT dir FROM runs WHERE run_id=?", (run_id,))
        if r is None:
            return []
        out = []
        with open(Path(r["dir"]) / "trace.jsonl", "r", encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                if kinds and rec["k"] not in kinds:
                    continue
                if t_from is not None and rec["t"] < t_from:
                    continue
                if t_to is not None and rec["t"] > t_to:
                    continue
                out.append(rec)
                if len(out) >= limit:
                    break
        return out

    def frame_map(self, run_id: str, which: str = "annotated") -> dict | None:
        r = self.store.one("SELECT dir FROM runs WHERE run_id=?", (run_id,))
        p = Path(r["dir"]) / f"{which}.framemap.json" if r else None
        return json.loads(p.read_text("utf-8")) if p is not None and p.is_file() else None
