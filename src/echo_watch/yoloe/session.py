# SPDX-License-Identifier: AGPL-3.0-only
"""One YOLOE detection + tracking test of one local video (EW-LAB-06). Tk-free; separate from the Display Assets
session, policy, Library and store.

Phases: idle -> preparing -> running <-> paused -> finalizing -> idle (with an outcome). ONE driver thread per run:

  prepare   free-space floor, run directory, manifest + trace, source unchanged since chosen, worker ready (PF model,
            pinned tracker), tracking epoch acknowledged by the worker, ordered source started
  loop      take the NEXT source frame in order (bounded queue; a slow loop back-pressures the decoder), wait for its
            media time (source speed when possible; late frames slow playback instead of bursting), one worker
            ``detect_track`` request, then the AUTHORITATIVE COMMIT: run / epoch / frame key / image shape are checked,
            every detection is kept, tracker output is joined by returned index (association.join), the packet is
            published as the newest result and its trace record queued. Only then is the update accepted (with the
            next request); a rejected update is restored exactly in the worker, or the epoch is reset with a recorded
            continuity break.
  pause     no further frames are taken (the decoder blocks), the tracker does not advance, the media clock stops; a
            request already in flight completes in order and is committed with that disposition. Resume continues
            at the next source frame with a re-anchored clock (no catch-up burst).
  stop      bounded: the in-flight request (if any) finishes or times out, its result is NOT published and its update
            is rejected; the decoder is killed (the rest of the file is not consumed); evidence is finalized.
  EOF       the last frame is processed and published before normal finalization.

Packets bind run / source content / profile / worker session / tracking epoch / control generation / frame ordinal /
PTS; the UI paints only a packet's own detections on that packet's own image. Track ids are local to (run, epoch).

EW-LAB-06 R1: every run tracks its current STAGE; any failure becomes ONE structured ``failure`` record (stage, category,
exception type, bounded message/traceback, the worker's own structured startup failure and bounded stderr when the
worker reported or died) that is carried into the outcome, a ``failure`` trace record and the manifest outcome, so a
zero-frame failure is readable later without running anything. ``Outcome.headline`` says in plain words what failed.
"""
from __future__ import annotations

import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import FFMPEG, Limits
from ..engine.recording import signature
from ..frames import Epoch, new_id
from ..metrics import percentile
from ..sources.base import SourceEnded, SourceError
from ..sources.fingerprint import verify_unchanged
from . import association as A
from .bootstrap import bounded, failure_record, first_line
from .evidence import SCHEMA, TraceWriter, atomic_json
from .profile import DETECTOR, profile_identity
from .source import MediaClock, OrderedFileSource

IDLE, PREPARING, RUNNING, PAUSED, FINALIZING = "idle", "preparing", "running", "paused", "finalizing"
STAGE_LABEL = {"evidence": "opening the saved evidence", "disk_space": "the free disk space check",
               "source_check": "checking the chosen video", "worker_start": "starting the YOLOE GPU worker",
               "tracking_epoch": "starting the tracker", "decoder_start": "starting the video decoder",
               "analysis": "analysing frames", "decoding": "reading video frames", "evidence_write": "saving evidence",
               "finalize": "finalizing"}
PHASE_TEXT = {IDLE: "Idle", PREPARING: "Preparing", RUNNING: "Running", PAUSED: "Paused", FINALIZING: "Finalizing"}
APP_VERSION = "0.3.0"


class YoloeError(RuntimeError):
    pass


class StageError(YoloeError):
    """A failed stage with a structured ``failure`` (e.g. the worker's own startup failure)."""

    def __init__(self, message: str, worker_failure: dict | None = None):
        super().__init__(message)
        self.worker_failure = worker_failure


def make_failure(stage: str, exc: BaseException, worker: dict | None = None) -> dict:
    """One bounded, structured failure: where (session stage + label), what (exception type, message, traceback) and,
    when the worker was involved, its own structured detail (worker stage, exception, traceback, stderr)."""
    rec = failure_record(stage, exc)
    rec["stage_label"] = STAGE_LABEL.get(stage, stage)
    rec["category"] = (worker or {}).get("category") or rec["category"]
    rec["worker"] = worker
    rec["cause"] = failure_cause(rec)
    return rec


def failure_cause(f: dict | None) -> str:
    """A short plain cause for screens and summaries (the full detail stays in the record)."""
    if not f:
        return ""
    w = f.get("worker") or {}
    if w.get("exc_type"):
        where = w.get("stage_label") or w.get("stage") or "worker"
        text = f"{where}: {w['exc_type']}: {first_line(w.get('message', ''))}"
    elif w.get("stderr_useful"):
        code = w.get("exit_code")
        text = (f"the worker ended (exit code {code})" if code is not None else "the worker ended") + \
            ": " + " | ".join(w["stderr_useful"][-2:])
    elif w.get("category") == "deadline":
        text = "the worker did not answer before its deadline"
    else:
        text = f"{f.get('exc_type')}: {first_line(f.get('message', ''))}"
    return bounded(text, 240)[0]

@dataclass(frozen=True)
class SourceInfo:
    path: str
    name: str
    bytes: int
    sha256: str
    state: tuple                      # (size, mtime_ns) when the content hash was taken


@dataclass(frozen=True)
class YoloePacket:
    run_id: str
    epoch: str
    control_gen: int
    seq: int                          # processed-frame ordinal inside the run (commit order)
    media: dict = field(repr=False)   # out_n / src_n / pts / media_pts / verified / gap_before
    image: np.ndarray = field(repr=False, compare=False)
    size: tuple[int, int] = (0, 0)
    detections: tuple = field(default=(), repr=False)
    assoc: dict = field(default_factory=dict, repr=False)
    assoc_errors: tuple = ()
    counts: dict = field(default_factory=dict)
    max_det_reached: bool = False
    frame_t_mono: float = 0.0         # when the frame was taken from the source
    commit_mono: float = 0.0
    worker_session: int = 0
    source_sha256: str = ""
    profile_sha256: str = ""

    @property
    def key(self) -> tuple:
        return (self.run_id, self.epoch, self.seq)


@dataclass
class Outcome:
    run_id: str
    status: str                       # completed | stopped | failed
    reason: str
    complete: bool                    # every required evidence record was saved (NOT run success)
    counts: dict
    run_dir: str
    failure: dict | None = None       # structured failure (stage, exception, worker detail) when status is failed

    @property
    def frames(self) -> int:
        return int(self.counts.get("processed", 0))

    def headline(self) -> str:
        n, f = self.frames, self.failure or {}
        if self.status == "failed":
            label, cause = f.get("stage_label") or "the test", f.get("cause") or self.reason
            if n == 0:
                return f"Analysis could not start: {label} failed - {cause}. No video frame was analysed."
            return f"Analysis failed while {label} after {n} analysed frames - {cause}."
        if self.status == "stopped":
            return "Stopped before any video frame was analysed." if n == 0 else f"Test stopped after {n} analysed frames."
        return f"Test finished at the end of the video: {n} frames analysed."

    def summary_text(self) -> str:
        return self.headline() + (" Evidence saved." if self.complete else " Evidence INCOMPLETE - see Details.")


class YoloeSession:
    def __init__(self, *, runs_dir: Path, client, ffmpeg: Path = FFMPEG, source_factory=None, limits: Limits | None = None,
                 queue_frames: int = 4, request_timeout_s: float | None = None, clock=time.monotonic,
                 trace_queue: int = 256, disk_usage=shutil.disk_usage):
        self.runs_dir = Path(runs_dir)
        self.client = client
        self.ffmpeg = ffmpeg
        self.limits = limits or Limits()
        self._source_factory = source_factory or (lambda path: OrderedFileSource("yoloe", ffmpeg, path,
                                                                                  queue_frames=queue_frames))
        self.request_timeout_s = request_timeout_s or self.limits.request_timeout_s
        self.clock = clock
        self.trace_queue = trace_queue
        self._disk_usage = disk_usage
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self.phase = IDLE
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self.control_gen = 0
        self.epoch: str | None = None
        self.epoch_no = 0
        self.last_outcome: Outcome | None = None
        self._latest: YoloePacket | None = None
        self._stop_req = False
        self._stop_by: str | None = None
        self._pause_req = False
        self._driver: threading.Thread | None = None
        self._writer: TraceWriter | None = None
        self._reset_counters()

    # ---- state ---------------------------------------------------------------------------------------------------
    def _reset_counters(self) -> None:
        self.counts = {"processed": 0, "published": 0, "not_published": 0, "displayed": 0, "rejected": 0,
                       "epoch_resets": 0, "gaps": 0, "gap_frames": 0, "unverified_mapping": 0, "assoc_errors": 0,
                       "max_det_frames": 0, "tracker_warnings": 0, "detections": 0, "associated": 0}
        self.errors: list[str] = []
        self.resets: list[dict] = []
        self._painted: set = set()
        self._delays: list[float] = []
        self._analysed: list[float] = []          # active-wall commit times (sliding window)
        self._clock = MediaClock()
        self._source = None
        self._receipt: dict | None = None
        self._profile: dict | None = None
        self._info: SourceInfo | None = None
        self._started_wall = None
        self._inflight_since: float | None = None
        self.stage: str | None = None
        self.failure: dict | None = None

    @property
    def busy(self) -> bool:
        return self.phase != IDLE

    def latest_packet(self) -> YoloePacket | None:
        return self._latest

    # ---- commands (non-blocking; the driver does the work) -----------------------------------------------------
    def start(self, info: SourceInfo) -> str:
        with self._lock:
            if self.busy:
                raise YoloeError(f"a YOLOE test is already {PHASE_TEXT[self.phase].lower()}")
            self._reset_counters()
            self._latest = None
            self._stop_req = self._pause_req = False
            self._stop_by = None
            self.run_id = new_id("yrun")
            self.run_dir = self.runs_dir / self.run_id
            self.control_gen += 1
            self.epoch, self.epoch_no = None, 0
            self.phase = PREPARING
            self._info = info
            self._driver = threading.Thread(target=self._drive, args=(self.run_id,), name="yoloe-driver", daemon=True)
            self._driver.start()
            return self.run_id

    def pause(self) -> bool:
        with self._lock:
            if self.phase != RUNNING or self._stop_req:
                return False
            self._pause_req = True
            self.phase = PAUSED
            self.control_gen += 1
            gen = self.control_gen
            self._clock.suspend(self.clock())
            self._wake.notify_all()
        self._trace({"k": "pause", "control_gen": gen}, block_s=0.5)
        return True

    def resume(self) -> bool:
        with self._lock:
            if self.phase != PAUSED or self._stop_req:
                return False
            self._pause_req = False
            self.phase = RUNNING
            self.control_gen += 1
            gen = self.control_gen
            self._clock.resume(self.clock())
            self._wake.notify_all()
        self._trace({"k": "resume", "control_gen": gen}, block_s=0.5)
        return True

    def stop(self, by: str = "operator") -> bool:
        """Bounded stop: the rest of the file is not consumed. ``by`` is recorded (operator / application close)."""
        with self._lock:
            if self.phase in (IDLE, FINALIZING) or self._stop_req:
                return False
            self._stop_req, self._stop_by = True, by
            self.control_gen += 1
            gen = self.control_gen
            self._wake.notify_all()
            src = self._source
        self._trace({"k": "stop_requested", "by": by, "control_gen": gen}, block_s=0.5)
        if src is not None:
            threading.Thread(target=src.stop, name="yoloe-source-stop", daemon=True).start()
        if getattr(self.client, "state", None) == "starting":
            # Stop while the worker is still loading: end that (owned) worker instead of waiting out its startup
            threading.Thread(target=self.client.release, name="yoloe-worker-release", daemon=True).start()
        return True

    def wait_idle(self, timeout: float) -> bool:
        t = self._driver
        if t is not None:
            t.join(timeout)
        return not self.busy

    def note_view(self, view: str, shown_conf: float) -> None:
        """Presentation changes are saved SEPARATELY from detector records (they never alter them)."""
        self._trace({"k": "view", "view": view, "shown_conf": shown_conf, "control_gen": self.control_gen},
                    optional=True)

    def note_painted(self, pk: YoloePacket, paint_mono: float) -> bool:
        """The UI painted this packet's image and boxes (distinct packets only)."""
        if pk.run_id != self.run_id or pk.key in self._painted:
            return False
        self._painted.add(pk.key)
        self.counts["displayed"] += 1
        self._trace({"k": "paint", "seq": pk.seq, "epoch": pk.epoch, "age_ms": round((paint_mono - pk.frame_t_mono) * 1000, 1)},
                    optional=True)
        return True

    # ---- read model ------------------------------------------------------------------------------------------------
    def state(self) -> dict:
        """The cheap per-tick authority state (no copies of counters or diagnostics)."""
        t = self._inflight_since
        return {"phase": self.phase, "run_id": self.run_id, "epoch": self.epoch, "control_gen": self.control_gen,
                "outcome": self.last_outcome, "inflight_s": None if t is None else round(self.clock() - t, 3),
                "stage": self.stage, "stage_label": STAGE_LABEL.get(self.stage or "", self.stage)}

    def snapshot(self) -> dict:
        now = self.clock()
        act = self._clock.active(now)
        recent = [t for t in self._analysed if act - t <= 3.0]
        rate = None
        if len(recent) >= 2 and recent[-1] > recent[0]:
            rate = round((len(recent) - 1) / (recent[-1] - recent[0]), 2)
        d = sorted(self._delays[-300:])
        src = self._source
        return {"phase": self.phase, "phase_text": PHASE_TEXT[self.phase], "run_id": self.run_id, "epoch": self.epoch,
                "control_gen": self.control_gen, "counts": dict(self.counts), "analysed_fps": rate,
                "playback_speed": self._clock.speed(), "processing_ms_p50": round(percentile(d, 50), 1) if d else None,
                "processing_ms_p95": round(percentile(d, 95), 1) if d else None,
                "source": src.diagnostics() if src is not None else None, "receipt": self._receipt,
                "profile": self._profile, "resets": list(self.resets[-10:]), "errors": list(self.errors[-10:]),
                "stop_requested": self._stop_req, "outcome": self.last_outcome, "stage": self.stage,
                "stage_label": STAGE_LABEL.get(self.stage or "", self.stage), "failure": self.failure,
                "inflight_s": None if self._inflight_since is None else round(now - self._inflight_since, 3)}

    # ---- evidence helpers -------------------------------------------------------------------------------------------
    def _trace(self, rec: dict, optional: bool = False, block_s: float = 5.0) -> bool:
        w = self._writer
        if w is None:
            return False
        rec = {"t": round(time.time(), 4), "mono": round(self.clock(), 4), **rec}
        if optional:
            return w.note(rec)
        if not w.put(rec, block_s):
            with self._lock:
                if not self._stop_req:
                    self.errors.append(f"evidence: {w.error}")
                    self._stop_req, self._stop_by = True, "evidence"
                    if self.failure is None:
                        self.failure = make_failure("evidence_write", YoloeError(w.error or "evidence write failed"))
                    self._wake.notify_all()
            return False
        return True

    def _manifest(self, status: str, extra: dict | None = None) -> None:
        info, src = self._info, self._source
        diag = src.diagnostics() if src is not None else {}
        size, native = diag.get("size"), diag.get("native_size")
        transform = None
        if size and native:
            transform = {"processing_size": list(size), "native_size": list(native),
                         "processing_to_native": [round(native[0] / size[0], 6), round(native[1] / size[1], 6)],
                         "note": "boxes are in processing-size pixels; multiply by processing_to_native for native"}
        doc = {"schema": SCHEMA, "run_id": self.run_id, "app_version": APP_VERSION, "status": status,
               "written": time.time(), "started_wall": self._started_wall,
               "source": {"path": info.path, "name": info.name, "bytes": info.bytes, "sha256": info.sha256,
                          "file_state": list(info.state), "scope": f"file-sha256:{info.sha256}"},
               "profile": self._profile, "worker": self._receipt,
               "decoder": {k: diag.get(k) for k in ("decoder", "size", "native_size", "media", "stats", "queue_frames")},
               "coordinate_transform": transform,
               "tracking": {"track_buffer_unit": "tracker updates (one per processed source frame)",
                            "ids": "local to run_id + tracking epoch; not persistent physical identity",
                            "epochs": self.epoch_no, "resets": self.resets},
               "counts": dict(self.counts), "errors": list(self.errors),
               "rates": {"playback_speed_last": self._clock.speed(), "clock_reanchors": dict(self._clock.reanchors),
                         "paused_s": round(self._clock.paused_total, 3),
                         "processing_ms_p50": round(percentile(sorted(self._delays), 50), 1) if self._delays else None,
                         "processing_ms_p95": round(percentile(sorted(self._delays), 95), 1) if self._delays else None},
               "trace": {"path": "trace.jsonl", "complete": None}, **(extra or {})}
        try:
            atomic_json(self.run_dir / "manifest.json", doc)
        except OSError as e:
            self.errors.append(f"manifest write failed: {e}")

    # ---- driver ------------------------------------------------------------------------------------------------------
    def _drive(self, run_id: str) -> None:
        status, reason = "failed", "not started"
        eof = False
        try:
            self._prepare()
            with self._lock:
                if self._stop_req:
                    status, reason = "stopped", "stopped while preparing"
                    return
                self.phase = RUNNING
            self._trace({"k": "running", "control_gen": self.control_gen})
            eof = self._loop(run_id)
            with self._lock:
                if eof:
                    status, reason = "completed", "end of the video"
                elif self._stop_by in ("operator", "application close"):
                    status, reason = "stopped", f"stopped ({self._stop_by})"
                else:
                    if self.failure is None:
                        self.failure = make_failure(self.stage or "analysis",
                                                    YoloeError((self.errors or ["the test ended unexpectedly"])[-1]))
                    status, reason = "failed", self.failure["cause"]
        except Exception as e:  # noqa: BLE001 - every failure becomes a visible outcome
            self.errors.append(f"{type(e).__name__}: {e}")
            if self._stop_by in ("operator", "application close"):
                status, reason = "stopped", f"stopped ({self._stop_by}) while preparing: {self.errors[-1]}"
            else:
                self.failure = self.failure or make_failure(self.stage or "evidence", e,
                                                            getattr(e, "worker_failure", None))
                status, reason = "failed", self.failure["cause"]
        finally:
            self._finalize(run_id, status, reason)

    def _prepare(self) -> None:
        info = self._info
        self.stage = "disk_space"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        free = self._disk_usage(self.runs_dir).free
        if free < self.limits.min_free_disk_bytes:
            raise YoloeError(f"less than {self.limits.min_free_disk_bytes / 2 ** 30:.0f} GiB free disk space")
        self.stage = "evidence"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._started_wall = time.time()
        self._profile = profile_identity()
        self._writer = TraceWriter(self.run_dir / "trace.jsonl", queue_max=self.trace_queue,
                                   min_free_bytes=self.limits.min_free_disk_bytes,
                                   cap_bytes=self.limits.run_evidence_cap_bytes, disk_usage=self._disk_usage)
        self._manifest("preparing")
        self._trace({"k": "start", "run_id": self.run_id, "source": info.name, "sha256": info.sha256,
                     "profile_sha256": self._profile["profile_sha256"], "control_gen": self.control_gen})
        self.stage = "source_check"
        verify_unchanged(info.path, info.state)
        self.stage = "worker_start"
        if not self.client.alive():
            self._trace({"k": "worker_start", "attempt": getattr(self.client, "attempts", 0) + 1})
            r = self.client.start()
            if r.get("event") != "ready":
                raise StageError(f"the YOLOE GPU worker could not start: {r.get('error') or self.client.reason}",
                                 r.get("failure") or getattr(self.client, "last_failure", None))
        self._receipt = {k: v for k, v in (self.client.receipt or {}).items() if not k.startswith("_")}
        self._receipt["worker_session"] = self.client.session
        self.stage = "tracking_epoch"
        self._new_epoch("run_start")
        self.stage = "decoder_start"
        src = self._source_factory(info.path)
        with self._lock:
            self._source = src
            if self._stop_req:
                return
        src.start(Epoch.begin("yoloe", 0))
        self.stage = "analysis"
        self._manifest("running")

    def _new_epoch(self, reason: str) -> None:
        self.epoch_no += 1
        epoch = f"{self.run_id}/e{self.epoch_no}"
        ack = self.client.reset_epoch(epoch)
        with self._lock:
            self.epoch = epoch
            if self.epoch_no > 1:
                self.counts["epoch_resets"] += 1
            self.resets.append({"epoch": epoch, "reason": reason, "acknowledged": ack.get("epoch") == epoch,
                                "after_seq": self.counts["processed"]})
        self._trace({"k": "epoch", "epoch": epoch, "reason": reason, "tracker": ack.get("tracker"),
                     "worker_session": self.client.session})

    def _wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; False as soon as Stop or Pause is requested."""
        end = self.clock() + seconds
        with self._lock:
            while True:
                if self._stop_req or self._pause_req:
                    return False
                left = end - self.clock()
                if left <= 0:
                    return True
                self._wake.wait(min(left, 0.05))

    def _loop(self, run_id: str) -> bool:
        src = self._source
        pending: str | None = None
        held = None
        seq = 0
        while True:
            with self._lock:
                if self._stop_req:
                    break
                if self._pause_req:
                    self._wake.wait(0.1)
                    continue
            frame = held
            held = None
            if frame is None:
                try:
                    frame = src.read(0.1)
                except SourceEnded:
                    if pending is not None:
                        self._settle(pending, "accept")
                    return True
                except SourceError as e:
                    self.errors.append(f"source: {e}")
                    self.failure = self.failure or make_failure("decoding", e)
                    break
                if frame is None:
                    continue
            media = frame.media or {}
            mpts = media.get("media_pts", media.get("pts"))
            if mpts is not None:
                if not self._wait(self._clock.plan(float(mpts), self.clock())):
                    held = frame
                    continue
            t_take = self.clock()
            gen = self.control_gen
            seq += 1
            key = f"{seq}"
            self._inflight_since = t_take
            try:
                reply = self.client.detect_track(frame.image, self.epoch, key, pending, self.request_timeout_s)
            except Exception as e:  # noqa: BLE001 - the worker failed or timed out: its tracker state is unknown
                self.errors.append(f"worker: {type(e).__name__}: {e}")
                self.failure = self.failure or make_failure(
                    "analysis", e, getattr(e, "failure", None) or getattr(self.client, "last_failure", None))
                pending = None
                break
            finally:
                self._inflight_since = None
            pending = key
            committed = self._commit(run_id, frame, media, reply, key, seq, gen, t_take)
            if committed == "stop":
                self._settle(key, "reject")
                pending = None
                break
            if committed == "rejected":
                if not self._settle(key, "reject"):
                    self._new_epoch("restore_failed")
                pending = None
        if pending is not None:
            self._settle(pending, "accept")
        return False

    def _settle(self, key: str, decision: str) -> bool:
        try:
            r = self.client.settle(self.epoch, key, decision)
            ok = r.get("event") == "settled" and (decision == "accept" or r.get("restored") is True)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"settle {decision} of frame {key} failed: {type(e).__name__}: {e}")
            ok = False
        self._trace({"k": "settle", "seq": int(key), "epoch": self.epoch, "decision": decision, "ok": ok})
        return ok

    def _commit(self, run_id, frame, media, reply, key, seq, gen, t_take) -> str:
        now = self.clock()
        dets = reply.get("detections") or []
        n = len(dets)
        valid_dets = all(d.get("i") == k for k, d in enumerate(dets))
        shape_ok = list(reply.get("input_shape") or []) == list(frame.image.shape[:2])
        tr = reply.get("tracker") or {}
        with self._lock:
            stop = self._stop_req or run_id != self.run_id
            epoch_ok = reply.get("epoch") == self.epoch and reply.get("frame") == key
        base = {"k": "frame", "seq": seq, "epoch": reply.get("epoch"), "control_gen": gen, "media": media,
                "sig": signature(frame.image), "size": [frame.width, frame.height], "detections": dets,
                "max_det_reached": bool(reply.get("max_det_reached")),
                "tracker": tr, "worker_assoc_errors": reply.get("worker_assoc_errors") or [],
                "timing": {"rtt_ms": round(reply.get("_rtt_s", 0.0) * 1000, 2), "infer_ms": round(reply.get("infer_s", 0.0) * 1000, 2),
                           "speed_ms": reply.get("speed_ms"), "tracker_ms": round((tr.get("tracker_s") or 0) * 1000, 3),
                           "processing_ms": round((now - t_take) * 1000, 2)},
                "worker_session": reply.get("_session")}
        self.counts["processed"] += 1
        if stop:
            self.counts["not_published"] += 1
            self._trace({**base, "disposition": "not_published_stop_requested"})
            return "stop"
        if not (epoch_ok and valid_dets and shape_ok):
            why = ("epoch/frame mismatch" if not epoch_ok else "detection indices out of order" if not valid_dets
                   else "image shape mismatch")
            self.counts["rejected"] += 1
            self.counts["not_published"] += 1
            self.errors.append(f"frame {seq} rejected at commit: {why}")
            self._trace({**base, "disposition": f"rejected: {why}"})
            return "rejected"
        assoc, errors = A.join(n, tr.get("rows"), [tuple(t) for t in tr.get("tentative") or []])
        counts = A.summarize(n, assoc)
        if media.get("gap_before"):
            self.counts["gaps"] += 1
            self.counts["gap_frames"] += media["gap_before"]
            self._trace({"k": "gap", "seq": seq, "missing_source_frames": media["gap_before"], "before_src_n": media.get("src_n"),
                         "tracker_handling": "not compensated: the tracker advanced ONE update across the gap"})
        if not media.get("verified"):
            self.counts["unverified_mapping"] += 1
        self.counts["assoc_errors"] += len(errors)
        self.counts["max_det_frames"] += int(bool(reply.get("max_det_reached")))
        self.counts["tracker_warnings"] += len(tr.get("warnings") or [])
        self.counts["detections"] += n
        self.counts["associated"] += counts["associated"]
        pk = YoloePacket(run_id, reply["epoch"], gen, seq, dict(media), frame.image, (frame.width, frame.height),
                         tuple(dets), assoc, tuple(errors), counts, bool(reply.get("max_det_reached")),
                         t_take, now, int(reply.get("_session") or 0), self._info.sha256,
                         (self._profile or {}).get("profile_sha256", ""))
        with self._lock:
            disposition = "published" if gen == self.control_gen else "published_after_pause_request"
            self._latest = pk
            self.counts["published"] += 1
            self._delays.append((now - t_take) * 1000)
            del self._delays[:-2000]
            self._analysed.append(self._clock.active(now))
            del self._analysed[:-200]
        if media.get("media_pts", media.get("pts")) is not None:
            self._clock.shown(float(media.get("media_pts", media.get("pts"))), now)
        self._trace({**base, "assoc": {str(i): a for i, a in assoc.items()}, "assoc_errors": errors, "counts": counts,
                     "disposition": disposition})
        return "published"

    def _finalize(self, run_id: str, status: str, reason: str) -> None:
        with self._lock:
            self.phase = FINALIZING
            src = self._source
        if src is not None:
            try:
                src.stop()
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"source stop: {e}")
        failure = self.failure if status == "failed" else None
        if self._writer is not None:
            if failure is not None:
                self._trace({"k": "failure", **failure}, block_s=2.0)
            self._trace({"k": "end", "status": status, "reason": reason, "counts": dict(self.counts)})
            tr = self._writer.close()
        else:
            tr = {"complete": False, "error": "the trace was never opened"}
        complete = bool(tr.get("complete"))
        if not tr.get("complete") and status == "completed":
            status, reason = "failed", f"evidence incomplete: {tr.get('error')}"
            failure = self.failure = make_failure("evidence_write", YoloeError(reason))
        if self.run_dir is not None and self.run_dir.is_dir():
            self._manifest(status, {"outcome": {"status": status, "reason": reason, "complete": complete,
                                                "frames_analysed": self.counts["processed"], "failure": failure,
                                                "details_captured": True},
                                    "ended_wall": time.time(), "trace": tr})
        elif failure is not None:
            complete = False                     # nothing could be saved; the failure lives only in this outcome
        out = Outcome(run_id, status, reason, complete, dict(self.counts), str(self.run_dir), failure)
        with self._lock:
            self.last_outcome = out
            self.phase = IDLE
            self._stop_req = self._pause_req = False
