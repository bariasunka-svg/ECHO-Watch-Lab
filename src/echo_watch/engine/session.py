# SPDX-License-Identifier: AGPL-3.0-only
"""One managed session: Prepare / Start / Pause / Resume / Stop / finalization / rollback, and the SINGLE commit authority.

The phase machine is adapted from the frozen PatronSight reference src/patronsight/live_session.py (SHA-256
f1fe5c1d611b4876dd2a4f0512692a8420e11df20e632590f8696839ffd68118; LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only):
  idle -> start() -> starting -> active -> stop() -> stopping -> idle (with a SessionOutcome); any startup failure ->
  rolling_back -> idle (outcome ``rolled_back``); any required output failing while active -> stopping with the error.
Kept from the reference: every output defaults ON (clean source, annotated result recording, decision trace) and a
required output failing prevents (or ends) a healthy ``active`` indication; the session owns exactly the components it
started; rollback and stop REQUEST stops and never block; the phase returns to idle only when every owned component has
settled (recorders terminal, analysis threads exited, trace terminal) - no orphan worker, half-open recorder or stale
"active" indicator; whatever the operator deliberately disabled is recorded as ``not_requested``; progress is advanced by
``poll()`` (the native UI calls it from its timer; ``start_driver`` runs it on a thread for headless use).
Different (item) behaviour:
* the camera/detection/identity components are the FFmpeg file capture, the supervised worker and the item evaluator;
* references are prepared and acknowledged by the worker BEFORE the (finite) source starts, so a replayed file is not
  consumed while references load;
* a ``ControlBasis`` (run, control generation, running segment, capture-time lower boundary, source epoch/loop, reference
  digest, worker session, policy) is created at Start and replaced at Pause, Resume and Stop. THIS CLASS IS THE ONLY PLACE
  A RESULT BECOMES CURRENT: ``evaluate_and_commit`` re-checks the packet's basis before evaluation and again, under the
  commit lock, immediately before publication; any mismatch (old generation, paused, before the boundary, foreign epoch,
  stopping) discards the attempt - it stays in the trace as a diagnostic and never touches the item board, the
  last-supported memory, the timers, the latest-result slot or the recorders;
* the item list / profile rows, the annotated recorder and the trace all read the same committed packet.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import FFMPEG, Settings
from ..frames import Epoch, Frame, new_id
from ..item_policy import POLICY_VERSION
from ..metrics import RollingRate, ResourceSampler
from ..sources.ffmpeg_source import FFmpegFileSource
from ..worker.client import WorkerClient
from .board import ItemBoard
from .capture import CaptureSession
from .contracts import (OUTCOME_FINALIZED, OUTCOME_ROLLED_BACK, OUTCOME_STOPPED_AFTER_FAILURE, OUTPUT_NOT_REQUESTED,
                        OUTPUT_NOT_STARTED, PHASE_ACTIVE, PHASE_IDLE, PHASE_PAUSED, PHASE_ROLLING_BACK,
                        PHASE_STARTING, PHASE_STOPPING, STATE_OFFLINE, STATE_PAUSED, STATE_STOPPED, CaptureOptions,
                        ControlBasis, DetectionRecord, EngineSnapshot, ItemObservation, ItemRef, OutputOutcome,
                        ResultPacket, SessionOutcome)
from .evaluator import ItemEvaluator, StaleFrame, prepare_member
from .latest import LatestSlot
from .pipeline import STATE_FAILED as PIPE_FAILED
from .pipeline import AnalysisPipeline, Attempt
from .recording import AnnotatedPump, VideoItem, VideoRecorder, signature
from .render import render_result
from .store import Store, dumps
from .trace import TERMINAL as TRACE_TERMINAL
from .trace import TraceWriter

logger = logging.getLogger(__name__)

STEP_LOADING = "loading_references"
STEP_WAITING_FOR_FRAME = "waiting_for_source_frame"
STEP_WAITING_FOR_RESULT = "waiting_for_first_result"
STEP_STARTING_OUTPUTS = "starting_annotated_recording"
DEFAULT_STARTUP_TIMEOUT_SECONDS = 180.0


class SessionError(RuntimeError):
    """A managed session could not be started (nothing was left running)."""


@dataclass(frozen=True)
class SourceSpec:
    """The one selected source: an existing local video file replayed once through the controlled decoder."""

    source_id: str
    path: str
    fps: float                        # declared decode rate (the file's native rate - never invented)
    repeat: bool = False              # Repeat OFF: a finished file ends the source
    size: tuple[int, int] | None = None


@dataclass(frozen=True)
class PollResult:
    outcome: SessionOutcome | None = None


class ManagedItemSession:
    def __init__(self, *, settings: Settings, store: Store, worker: WorkerClient, ffmpeg: Path = FFMPEG,
                 clock: Callable[[], float] = time.monotonic,
                 startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
                 clean_launch=None, annotated_launch=None):
        self.s, self.store, self.worker, self.ffmpeg = settings, store, worker, Path(ffmpeg)
        self._clock = clock
        self._startup_timeout = float(startup_timeout_seconds)
        self._clean_launch, self._annotated_launch = clean_launch, annotated_launch
        self._lock = threading.RLock()                   # the commit lock
        self.last_outcome: SessionOutcome | None = None
        self.sessions_started = 0
        self.sampler: ResourceSampler | None = None
        self._driver: threading.Thread | None = None
        self._driver_stop = threading.Event()
        self._outcome_event = threading.Event()
        #: callables invoked (outside the commit lock) with every COMMITTED ResultPacket; exceptions are swallowed.
        #: Qualification harnesses use this to score each packet; a native view may poll latest_packet() instead.
        self.commit_observers: list[Callable[[ResultPacket], None]] = []
        self._reset()

    # ---- state ----------------------------------------------------------------------------------------------
    def _reset(self) -> None:
        self.phase = PHASE_IDLE
        self.step: str | None = None
        self.error: str | None = None
        self.options = CaptureOptions()
        self._deadline: float | None = None
        self.run_id: str | None = None
        self.run_dir: Path | None = None
        self._refs: list[ItemRef] = []
        self._source: SourceSpec | None = None
        self._members: dict = {}
        self._class_map: dict = {}
        self._refs_digest: str | None = None
        self._prep: dict | None = None
        self._prep_thread: threading.Thread | None = None
        self._capture: CaptureSession | None = None
        self._slot_epoch: Epoch | None = None
        self._pipeline: AnalysisPipeline | None = None
        self._evaluator: ItemEvaluator | None = None
        self._board: ItemBoard | None = None
        self._basis: ControlBasis | None = None
        self._basis_history: list[dict] = []
        self._gen = 0
        self._segment = 0
        self._latest: LatestSlot[ResultPacket] = LatestSlot()
        self._seq = 0
        self._last_committed: dict[str, int] = {}
        self.trace: TraceWriter | None = None
        self.clean: VideoRecorder | None = None
        self.annotated: VideoRecorder | None = None
        self.pump: AnnotatedPump | None = None
        self._rate_commit = RollingRate()
        self.discards: dict[str, int] = {}
        self.committed = 0
        self.commit_log: list[dict] = []                 # compact per-commit record (bounded) for headless evaluation
        self.start_requested_mono: float | None = None
        self.active_ack_mono: float | None = None
        self.stopped_mono: float | None = None
        self._ended_by = "operator"
        self._banner: str | None = None
        self._stop_called = False
        self._pause_mono: float | None = None
        self._first_frame: Frame | None = None
        self._pipeline_failed: str | None = None
        self._outputs_started = False
        self._facts_pending: list[tuple[str, int, dict]] = []
        self._active_wall: float | None = None
        self._carried_ambiguity: dict = {}
        self._allowance_s = 60.0                         # EW-LAB-05: this run's immutable review allowance
        self._setup: dict | None = None                  # EW-LAB-05: the saved setup identity this run is bound to

    @property
    def idle(self) -> bool:
        return self.phase == PHASE_IDLE

    @property
    def busy(self) -> bool:
        return self.phase != PHASE_IDLE

    # ---- trace helper -----------------------------------------------------------------------------------------
    def _t(self, kind: str, **f) -> None:
        tr = self.trace
        if tr is not None:
            tr.emit(kind, **f)

    # ---- basis ------------------------------------------------------------------------------------------------
    def _new_basis(self, reason: str, *, active: bool, new_segment: bool = False, lower_bound: float | None = None) -> ControlBasis:
        self._gen += 1
        if new_segment:
            self._segment += 1
        ep = self._slot_epoch
        b = ControlBasis(new_id("basis"), self.run_id or "", self._gen, self._segment,
                         lower_bound if lower_bound is not None else self._clock(),
                         ep.epoch_id if ep else "", 0, self._refs_digest or "", self.worker.session, POLICY_VERSION, active)
        self._basis = b
        self._basis_history.append({"basis_id": b.basis_id, "control_generation": b.control_generation,
                                    "segment_id": b.segment_id, "lower_bound_mono": b.lower_bound_mono, "active": active,
                                    "reason": reason, "t_mono": self._clock(), "t_wall": time.time()})
        self._t("basis", **self._basis_history[-1])
        return b

    # ---- start ------------------------------------------------------------------------------------------------
    def start(self, options: CaptureOptions, items: list[ItemRef], source: SourceSpec, *, allowance_s: float = 60.0,
              setup: dict | None = None) -> None:
        """Validate, create the run and begin starting. Raises ``SessionError`` (nothing left running) when it cannot
        begin; later failures are reported by ``poll`` as a rollback.

        EW-LAB-05: ``allowance_s`` (1-3600 s, default 60) is passed unchanged to the existing ItemBoard/ItemTimeline
        as this run's immutable supported-Not-visible review allowance; ``setup`` is the saved setup identity this run
        binds to - its exact [item, reference version] list must equal ``items``. Both are recorded in the run row,
        trace and manifest. Neither changes identity thresholds, admission or stale/unknown rules."""
        with self._lock:
            if self.busy:
                raise SessionError(f"a session is already {self.phase.replace('_', ' ')}")
            if not items:
                raise SessionError("select at least one remembered item for this run")
            allowance = float(allowance_s)
            if not 1.0 <= allowance <= 3600.0:
                raise SessionError("the review allowance must be between 1 and 3600 seconds")
            if setup is not None and [list(x) for x in setup.get("items", [])] != [[r.item_id, r.ref_version] for r in items]:
                raise SessionError("the saved setup does not match the exact references being started")
            if not self.worker.alive():
                raise SessionError("the inference worker is not ready (no CPU substitute is accepted)")
            sizes = {tuple(r.frame_size) for r in items}
            if len(sizes) != 1:
                raise SessionError("remembered items come from different frame sizes; one run needs one size")
            import shutil
            free = shutil.disk_usage(self.s.data_dir).free
            if free < self.s.limits.min_free_disk_bytes:
                raise SessionError(f"less than {self.s.limits.min_free_disk_bytes // 2**30} GiB free disk space")
            self._reset()
            self._outcome_event.clear()
            self.options, self._refs, self._source = options, list(items), source
            self._allowance_s, self._setup = allowance, (dict(setup) if setup is not None else None)
            self.start_requested_mono = self._clock()
            self.run_id = new_id("run")
            self.run_dir = self.s.dirs()["runs"] / self.run_id
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._refs_digest = hashlib.sha256(json.dumps(sorted((r.ref_key, r.crop_sha256) for r in items)).encode()).hexdigest()
            now = time.time()
            with self.store.tx("create run") as c:
                c.execute("""INSERT INTO runs(run_id,created_wall,started_wall,phase,options,source,policy_version,refs_digest,dir)
                             VALUES(?,?,?,'starting',?,?,?,?,?)""",
                          (self.run_id, now, now, dumps({**options.to_dict(), "review_allowance_s": allowance,
                                                         "setup": self._setup}),
                           dumps({"source_id": source.source_id, "path": source.path, "fps": source.fps,
                                  "repeat": source.repeat}), POLICY_VERSION, self._refs_digest, str(self.run_dir)))
                for r in items:
                    c.execute("INSERT INTO run_items(run_id,item_id,ref_version) VALUES(?,?,?)", (self.run_id, r.item_id, r.ref_version))
            if options.decision_trace:
                try:
                    self.trace = TraceWriter(self.run_dir / "trace.jsonl", self.s.limits.trace_queue_max)
                    self.trace.start()
                except OSError as e:
                    self._reset_after_failed_start()
                    raise SessionError(f"the decision trace could not start: {e}") from e
            self._board = ItemBoard(self.run_id, list(items), allowance_s=allowance)
            self.phase, self.step = PHASE_STARTING, STEP_LOADING
            self._deadline = self._clock() + self._startup_timeout
            self.sessions_started += 1
            self._t("session", event="start_requested", options=options.to_dict(),
                    items=[{"item_id": r.item_id, "ref": r.ref_key, "crop_sha256": r.crop_sha256} for r in items],
                    source={"id": source.source_id, "path": source.path, "fps": source.fps, "repeat": source.repeat},
                    refs_digest=self._refs_digest, policy=POLICY_VERSION, worker_session=self.worker.session,
                    review_allowance_s=allowance, setup=self._setup,
                    admission_gate=self.gate_info(),
                    device=(self.worker.receipt or {}).get("device"))
            self._prep_thread = threading.Thread(target=self._prepare_references, name="prepare-references", daemon=True)
            self._prep_thread.start()

    def _reset_after_failed_start(self) -> None:
        if self.trace is not None:
            self.trace.finish(5)
        self.phase = PHASE_IDLE

    def _prepare_references(self) -> None:
        """Off the caller's thread: read reference images, build appearance keys, and have the WORKER compute and
        acknowledge the visual-prompt embeddings for exactly this membership."""
        t0 = time.monotonic()
        try:
            ambiguity = {}
            for r in self._refs:
                rows = self.store.query("SELECT detail FROM item_ambiguity WHERE item_id=? AND ref_version=? AND "
                                        "resolved_wall IS NULL ORDER BY created_wall DESC LIMIT 1", (r.item_id, r.ref_version))
                ambiguity[r.item_id] = json.loads(rows[0]["detail"]) if rows else None
            members = {r.item_id: prepare_member(r, ambiguity.get(r.item_id)) for r in self._refs}
            t1 = time.monotonic()
            payload = [{"key": r.ref_key, "image_path": r.context_path,
                        "boxes": [[r.box[0], r.box[1], r.box[0] + r.box[2], r.box[1] + r.box[3]]]} for r in self._refs]
            reply = self.worker.set_refs(payload, self._refs_digest or "")
            classes = reply.get("classes") or []
            class_map = {r.item_id: tuple(i for i, (key, _k) in enumerate(classes) if key == r.ref_key) for r in self._refs}
            missing = [i for i, c in class_map.items() if not c]
            if missing:
                raise RuntimeError(f"the worker did not acknowledge a reference for {missing}")
            self._prep = {"ok": True, "members": members, "class_map": class_map, "prep_s": round(t1 - t0, 3),
                          "refs_s": reply.get("seconds"), "warm_s": reply.get("warm_s"), "computed": reply.get("computed"),
                          "worker_session": reply.get("_session")}
        except Exception as e:  # noqa: BLE001
            self._prep = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ---- stop -------------------------------------------------------------------------------------------------
    def stop(self, *, error: str | None = None, ended_by: str = "operator") -> None:
        """Stop and finalize (idempotent). While starting, this cancels the start (reported as a rollback)."""
        with self._lock:
            if self.phase == PHASE_IDLE or self._stop_called:
                return
            if self.phase == PHASE_STARTING:
                self._fail(error or "the start was cancelled", ended_by="cancelled" if not error else "failure")
                return
            if self.phase in (PHASE_ACTIVE, PHASE_PAUSED):
                self.phase, self.step = PHASE_STOPPING, None
                self._ended_by = ended_by
                if error and not self.error:
                    self.error = error
            self._request_stops()

    def _fail(self, error: str, ended_by: str = "failure") -> None:
        logger.warning("managed session start failed: %s", error)
        self.error = error
        self._ended_by = ended_by
        self.phase, self.step = PHASE_ROLLING_BACK, None
        self._t("session", event="rollback", error=error)
        self._request_stops()

    def _request_stops(self) -> None:
        """Signal every owned component to stop - never blocks. After this nothing can be committed."""
        with self._lock:
            self._stop_called = True
            self.stopped_mono = self._clock()
            # a closed basis: any in-flight attempt is discarded at commit
            if self._basis is not None and self._basis.active:
                self._new_basis("stop requested", active=False)
            if self._board is not None and self._slot_epoch is not None:
                if self._ended_by == "source_ended":
                    self._banner, state = "SOURCE ENDED", STATE_OFFLINE
                    why = "the video file ended (not repeating); unknown, not absence"
                else:
                    self._banner, state = "STOPPED", STATE_STOPPED
                    why = "the session stopped; this run no longer observes the item"
                if any(st.obs is not None for st in self._board.items.values()):
                    self._board.apply_not_observing(state, why, self._clock(), time.time(), self._slot_epoch.epoch_id)
        self._t("session", event="stop_requested", ended_by=self._ended_by, error=self.error)
        try:
            if self.pump is not None:
                self.pump.stop()                          # offer barrier first: the recorder then settles
            if self.annotated is not None and self.annotated.is_recording:
                self.annotated.stop()
            if self.clean is not None and self.clean.is_recording:
                self.clean.stop()
            if self._pipeline is not None:
                self._pipeline.stop()
            if self._capture is not None:
                self._capture.stop(timeout=0.0)
        except Exception:  # noqa: BLE001 - a stop request must never raise into the caller
            logger.exception("stopping the managed session raised")

    # ---- pause / resume --------------------------------------------------------------------------------------------
    def pause(self) -> None:
        with self._lock:
            if self.phase != PHASE_ACTIVE:
                raise SessionError("only an active session can be paused")
            self._new_basis("operator pause", active=False)
            # established ambiguity is never erased by a control change: it is carried into the post-Resume evaluator
            if self._evaluator is not None:
                self._carried_ambiguity = {i: dict(it.ambiguity) for i, it in self._evaluator.policy.items.items()
                                           if it.ambiguity}
            self._evaluator = None
            assert self._board is not None and self._slot_epoch is not None
            self._board.apply_not_observing(STATE_PAUSED, "paused by the operator; unknown, not absence", self._clock(),
                                            time.time(), self._slot_epoch.epoch_id)
            self.phase = PHASE_PAUSED
            self._banner = "PAUSED"
            self._pause_mono = self._clock()
            self._t("session", event="paused")

    def resume(self) -> None:
        with self._lock:
            if self.phase != PHASE_PAUSED:
                raise SessionError("only a paused session can be resumed")
            assert self._board is not None and self._slot_epoch is not None
            now = self._clock()
            b = self._new_basis("operator resume", active=True, new_segment=True, lower_bound=now)
            seed = {i: not self._board.confirmed_ever(i) for i in self._members}
            carried = self._carried_ambiguity or {}
            members = {i: (dataclasses.replace(m, ambiguity=carried[i]) if i in carried else m) for i, m in self._members.items()}
            self._evaluator = ItemEvaluator(members, self._size(), self._class_map, self._slot_epoch.epoch_id, seed)
            if carried:
                self._t("session", event="ambiguity_carried_over_resume", items=sorted(carried))
            self._board.apply_resumed("resumed: waiting for the first check captured after Resume", now, time.time(),
                                      self._slot_epoch.epoch_id)
            self._latest.clear()
            self.phase = PHASE_ACTIVE
            self._banner = None
            self._t("session", event="resumed", lower_bound_mono=b.lower_bound_mono)

    def _size(self) -> tuple[int, int]:
        return tuple(self._refs[0].frame_size)

    # ---- poll -------------------------------------------------------------------------------------------------------
    def poll(self) -> PollResult:
        outcome = None
        with self._lock:
            ph = self.phase
        if ph == PHASE_STARTING:
            self._poll_starting()
        elif ph in (PHASE_ACTIVE, PHASE_PAUSED):
            self._poll_active()
        if self.phase in (PHASE_STOPPING, PHASE_ROLLING_BACK):
            outcome = self._poll_settling()
        return PollResult(outcome)

    def _check_timeout(self) -> None:
        if self._deadline is not None and self._clock() > self._deadline:
            self._fail(f"timed out after {self._startup_timeout:g}s while {(self.step or 'starting').replace('_', ' ')}")

    def _poll_starting(self) -> None:
        if self._pipeline_failed:
            self._fail(self._pipeline_failed)
            return
        if self.step == STEP_LOADING:
            prep = self._prep
            if prep is None:
                return self._check_timeout()
            if not prep["ok"]:
                self._fail(f"references could not be prepared: {prep['error']}")
                return
            self._begin_capture(prep)
            return
        cap = self._capture
        assert cap is not None
        if cap.error:
            self._fail(f"the source stopped during startup: {cap.error}")
            return
        if self.step == STEP_WAITING_FOR_FRAME:
            f = cap.peek()
            if f is None:
                if cap.state != "running":
                    self._fail("the source ended before its first frame")
                    return
                return self._check_timeout()
            if (f.width, f.height) != self._size():
                self._fail(f"the source frame is {f.width}x{f.height} but the remembered references are "
                           f"{self._size()[0]}x{self._size()[1]}; update the references from a new scan")
                return
            self._first_frame = f
            self._start_analysis(f)
            return
        if self.step == STEP_WAITING_FOR_RESULT:
            if self._pipeline is not None and self._pipeline.state == PIPE_FAILED:
                self._fail(self._pipeline.error or "analysis failed while starting")
                return
            if self._latest.peek() is None:
                return self._check_timeout()
            if self.options.record_annotated:
                self.step = STEP_STARTING_OUTPUTS
                try:
                    self._start_annotated()
                except Exception as e:  # noqa: BLE001
                    self._fail(f"the annotated recording could not start: {e}")
                    return
            self._activate()

    def _begin_capture(self, prep: dict) -> None:
        assert self._source is not None and self.run_dir is not None
        sp = self._source
        self._members, self._class_map = prep["members"], prep["class_map"]
        self._t("session", event="references_acknowledged", prep_s=prep["prep_s"], refs_s=prep["refs_s"], warm_s=prep["warm_s"],
                computed=prep["computed"], class_map=self._class_map, worker_session=prep["worker_session"])
        try:
            src = FFmpegFileSource(sp.source_id, self.ffmpeg, sp.path, sp.fps, loop_file=sp.repeat)
        except Exception as e:  # noqa: BLE001
            self._fail(f"the source is not usable: {e}")
            return
        self._slot_epoch = Epoch.begin(sp.source_id, 1, 0, self._gen + 1)
        cap = CaptureSession(src, self._slot_epoch)
        self._capture = cap
        # the clean source is recorded from the first captured frame of this epoch
        if self.options.clean_source:
            try:
                self.clean = VideoRecorder("clean", self.run_dir / "clean.mp4", self._size(), fps=self.s.record_fps,
                                           ffmpeg=self.ffmpeg, queue_maxsize=self.s.limits.clean_queue_max,
                                           launch=self._clean_launch)
                self.clean.start()
            except Exception as e:  # noqa: BLE001
                self._fail(f"the clean-source recording could not start: {e}")
                return
            cap.attach_clean_sink(self._offer_clean)
        cap.attach_ledger(self._ledger)
        self._t("session", event="capture_start", epoch=self._slot_epoch.epoch_id,
                clean_recorder_started=self.clean is not None)
        cap.start()
        self.step = STEP_WAITING_FOR_FRAME

    def _offer_clean(self, f: Frame) -> bool:
        rec = self.clean
        if rec is None or not rec.is_recording:
            return False
        m = f.media or {}
        return rec.offer(VideoItem(f.image, {"kind": "capture", "ident": [f.epoch.epoch_id, f.seq], "t_mono": f.t_mono,
                                             "t_wall": f.t_wall, "media_pts": m.get("media_pts"), "src_n": m.get("src_n"),
                                             "dup_of_prev": bool(m.get("dup_of_prev"))}))

    def _ledger(self, f: Frame, dup: bool) -> None:
        m = f.media or {}
        self._t("frame", ident=[f.epoch.epoch_id, f.seq], t_mono=f.t_mono, media_pts=m.get("media_pts"),
                src_n=m.get("src_n"), out_n=m.get("out_n"), dup=dup)

    def _start_analysis(self, f: Frame) -> None:
        assert self._slot_epoch is not None and self._capture is not None
        with self._lock:
            b = self._new_basis("start", active=True, new_segment=True, lower_bound=self.start_requested_mono)
            seed = {i: True for i in self._members}
            self._evaluator = ItemEvaluator(self._members, self._size(), self._class_map, self._slot_epoch.epoch_id, seed)
        self._pipeline = AnalysisPipeline(controller=self, worker=self.worker, slot=self._capture.inference_slot,
                                          settings=self.s, trace_emit=self._t)
        self._pipeline.start()
        self.step = STEP_WAITING_FOR_RESULT
        self._t("session", event="analysis_started", first_frame=[f.epoch.epoch_id, f.seq])

    def _provider(self) -> tuple:
        return self._latest.peek(), self._banner

    def _start_annotated(self) -> None:
        assert self.run_dir is not None
        self.annotated = VideoRecorder("annotated", self.run_dir / "annotated.mp4", self._size(), fps=self.s.record_fps,
                                       ffmpeg=self.ffmpeg, queue_maxsize=self.s.limits.annotated_queue_max,
                                       launch=self._annotated_launch)
        self.annotated.start()
        self.pump = AnnotatedPump(provider=self._provider, render=render_result, sink=self.annotated, fps=self.s.record_fps)
        self.pump.start()
        self._outputs_started = True
        self._t("session", event="annotated_recording_started")

    def _activate(self) -> None:
        with self._lock:
            self.phase, self.step = PHASE_ACTIVE, None
            self._deadline = None
            self.active_ack_mono = self._clock()
            self._active_wall = time.time()
        self._t("session", event="active", active_ack_mono=self.active_ack_mono,
                requested_to_active_s=round(self.active_ack_mono - (self.start_requested_mono or 0), 3))

    def _emit_metrics(self) -> None:
        """~1 Hz: the four distinct rates and every bounded-queue / skip counter, into the decision trace."""
        now = self._clock()
        if now - getattr(self, "_last_metrics", 0.0) < 1.0:
            return
        self._last_metrics = now
        cap, pipe = self._capture, self._pipeline
        self._t("metrics", capture_fps=None if cap is None else round(cap.capture_fps, 2),
                analyzed_fps=round(self._rate_commit.rate(now=now), 2),
                clean_encode_fps=None if self.clean is None else round(self.clean.recorder_fps(), 2),
                annotated_encode_fps=None if self.annotated is None else round(self.annotated.recorder_fps(), 2),
                frames_captured=None if cap is None else cap.frames_captured,
                replaced_before_analysis=None if cap is None else cap.inference_slot.dropped,
                duplicate_source_frames=None if cap is None else cap.frames_duplicate_source,
                attempts=None if pipe is None else pipe.attempts, replies=None if pipe is None else pipe.replies,
                committed=self.committed, discards=dict(self.discards),
                clean_dropped=None if self.clean is None else self.clean.frames_dropped,
                annotated_dropped=None if self.annotated is None else self.annotated.frames_dropped,
                trace_dropped=None if self.trace is None else self.trace.dropped,
                pump=None if self.pump is None else self.pump.summary(), phase=self.phase)

    def _poll_active(self) -> None:
        self._emit_metrics()
        if self._pipeline_failed:
            self.stop(error=self._pipeline_failed, ended_by="failure")
            return
        cap = self._capture
        if cap is not None:
            if cap.ended and self.phase == PHASE_ACTIVE:
                self.stop(ended_by="source_ended")
                return
            if cap.ended and self.phase == PHASE_PAUSED:
                self.stop(ended_by="source_ended")
                return
            if cap.error:
                self.stop(error=f"the source stopped: {cap.error}", ended_by="failure")
                return
        if self.options.record_annotated and (self.annotated is None or not self.annotated.is_recording):
            self.stop(error="the annotated recording stopped unexpectedly: "
                      f"{getattr(self.annotated, 'error', None) or 'no longer recording'}", ended_by="failure")
            return
        if self.options.clean_source and (self.clean is None or not self.clean.is_recording):
            self.stop(error=f"the clean-source recording stopped unexpectedly ({getattr(self.clean, 'error', None) or 'no longer recording'})",
                      ended_by="failure")
            return
        if self.trace is not None and self.trace.state in TRACE_TERMINAL:
            self.stop(error=f"the decision trace stopped unexpectedly ({self.trace.error or self.trace.state})", ended_by="failure")
            return
        sp = self.sampler
        if sp is not None and sp.aborted:
            self.stop(error=f"aborted by the resource guard: {sp.aborted}", ended_by="abort")

    def _poll_settling(self) -> SessionOutcome | None:
        cap, pipe = self._capture, self._pipeline
        if self._prep_thread is not None and self._prep_thread.is_alive():
            return None                      # reference preparation (a worker round trip) has not returned yet
        if cap is not None and cap.state != "idle":
            cap.stop(timeout=0.0)
            return None
        if pipe is not None and pipe.alive:
            return None
        if self.pump is not None and self.pump.alive:
            return None
        for rec in (self.annotated, self.clean):
            if rec is not None and rec.state in ("recording", "finalizing"):
                if rec.state == "recording":
                    rec.stop()
                return None
        out = self._build_outcome()
        self.last_outcome = out
        self._persist_run(out)
        self._reset()
        self._outcome_event.set()
        return out

    # ---- the single commit authority -----------------------------------------------------------------------------------
    def dispatch_basis(self) -> ControlBasis | None:
        b = self._basis
        return b if (b is not None and b.active and self.phase in (PHASE_STARTING, PHASE_ACTIVE)) else None

    def note_ineligible(self, frame: Frame, reason: str) -> None:
        self.discards["ineligible:" + reason] = self.discards.get("ineligible:" + reason, 0) + 1
        self._t("ineligible", ident=list(frame.ident), reason=reason, t_mono=frame.t_mono)

    def pipeline_failed(self, why: str) -> None:
        self._pipeline_failed = why
        self._t("session", event="analysis_failed", error=why)

    def _discard(self, att: Attempt, reason: str) -> bool:
        self.discards[reason] = self.discards.get(reason, 0) + 1
        w = att.reply
        self._t("discard", attempt=att.attempt_id, reason=reason, ident=list(att.frame.ident), frame_t_mono=att.frame.t_mono,
                age_s=round(att.frame.age(self._clock()), 4),
                basis_id=att.basis.basis_id, control_generation=att.basis.control_generation,
                detections=len(w.get("detections", [])), t_reply=att.t_reply)
        return False

    def evaluate_and_commit(self, att: Attempt) -> bool:
        """Evaluate one real worker reply and COMMIT it once - only if its control basis is still the current one.

        EW-LAB-01 defect A: evaluation runs on an isolated FORK of the evaluator's mutable policy state (tracks,
        supports, confirmations, exclusions, ambiguity, reference context, timestamps; immutable features and image
        buffers are shared, never copied). The fork is adopted only under the final commit lock, after basis,
        evaluator identity/version, capture boundary, epoch, ordering AND frame age (``max_frame_age_s``) are rechecked
        on the session clock. A discarded evaluation leaves the current state exactly as it was, so it can never seed
        later ownership or support and never erases established ambiguity."""
        t_eval0 = time.monotonic()
        frame = att.frame
        max_age = self.s.limits.max_frame_age_s
        with self._lock:
            cur, ev_obj, phase = self._basis, self._evaluator, self.phase
            ok = (cur is not None and att.basis.basis_id == cur.basis_id and cur.active and ev_obj is not None
                  and phase in (PHASE_STARTING, PHASE_ACTIVE))
            ev_version = getattr(ev_obj, "version", None)
        if not ok:
            return self._discard(att, "control_boundary_before_evaluation")
        if frame.age(self._clock()) > max_age:
            return self._discard(att, "expired_before_evaluation")
        dets = [{"cls": d["cls"], "confidence": d["confidence"], "box": d["box"], "ref": d["ref"][0]}
                for d in att.reply.get("detections", [])]
        fork = ev_obj.fork()                     # speculative state: adopted only at a successful commit
        try:
            ev = fork.evaluate(frame, dets)
        except StaleFrame as e:
            return self._discard(att, f"stale_frame: {e}")
        t_eval1 = time.monotonic()
        image_sig = signature(frame.image)       # 16x9 signature of the exact analysed pixels (pairing evidence in the trace)
        facts = []
        with self._lock:
            cur, phase = self._basis, self.phase
            if (cur is None or att.basis.basis_id != cur.basis_id or not cur.active or phase not in (PHASE_STARTING, PHASE_ACTIVE)
                    or self._evaluator is not ev_obj or getattr(ev_obj, "version", None) != ev_version):
                return self._discard(att, "control_boundary_at_commit")
            if frame.age(self._clock()) > max_age:
                return self._discard(att, "expired_at_commit")
            if frame.t_mono < cur.lower_bound_mono or frame.epoch.epoch_id != cur.source_epoch_id:
                return self._discard(att, "captured_before_boundary_at_commit")
            last = self._last_committed.get(frame.epoch.epoch_id, 0)
            if frame.seq <= last:
                return self._discard(att, "duplicate_or_out_of_order_frame")
            assert self._board is not None
            ev_obj.adopt(fork)                   # the single authoritative policy-state transition
            self._seq += 1
            packet_id = f"res_{self._seq:06d}"
            prior_states = {i: (st.obs.state if st.obs else None) for i, st in self._board.items.items()}
            obs: tuple[ItemObservation, ...] = self._board.apply_evaluation(frame, ev, packet_id)
            self._last_committed[frame.epoch.epoch_id] = frame.seq
            commit_mono = time.monotonic()
            rp = att.reply
            timing = {"capture_mono": frame.t_mono, "taken_mono": att.t_taken, "dispatch_mono": att.t_dispatch,
                      "reply_mono": att.t_reply, "eval_start_mono": t_eval0, "eval_end_mono": t_eval1,
                      "commit_mono": commit_mono,
                      "stage_ms": {"queue_wait": round((att.t_dispatch - frame.t_mono) * 1000, 2),
                                   "worker_round_trip": round((att.t_reply - att.t_dispatch) * 1000, 2),
                                   "handoff_wait": round((t_eval0 - att.t_reply) * 1000, 2),
                                   "evaluate": round((t_eval1 - t_eval0) * 1000, 2),
                                   "commit": round((commit_mono - t_eval1) * 1000, 2), **{f"eval_{k}": v for k, v in ev.stage_ms.items()}},
                      "capture_to_commit_ms": round((commit_mono - frame.t_mono) * 1000, 2)}
            wr = self.worker.receipt or {}
            worker = {"session": rp.get("_session"), "device": rp.get("device"), "precision": rp.get("precision"),
                      "vp_model_sha256": wr.get("vp_model_sha256"), "pf_model_sha256": wr.get("model_sha256"),
                      "infer_ms": round(1000 * rp.get("infer_s", 0), 2),
                      "post_ms": round(1000 * rp.get("post_s", 0), 2),
                      "ipc_out_ms": None if "_ipc_out_s" not in rp else round(1000 * rp["_ipc_out_s"], 2),
                      "worker_ms": None if "_worker_s" not in rp else round(1000 * rp["_worker_s"], 2),
                      "ipc_back_ms": None if "_ipc_back_s" not in rp else round(1000 * rp["_ipc_back_s"], 2),
                      "rtt_ms": round(1000 * rp.get("_rtt_s", 0), 2), "vram_free_mib": rp.get("vram_free_mib")}
            detections = tuple(DetectionRecord(d["cls"], d["ref"][0], d["confidence"], tuple(d["box"]))
                               for d in rp.get("detections", []))
            packet = ResultPacket(packet_id, self._seq, self.run_id or "", cur, frame.ident, frame.t_mono, frame.t_wall,
                                  (frame.width, frame.height), frame.image, frame.media, detections, obs, timing, worker,
                                  {"camera_moved": ev.camera_moved, "whole_view_mean": ev.whole_view_mean,
                                   "attempt": att.attempt_id, "image_sig": image_sig,
                                   "candidates_refused": sum(1 for c in ev.candidate_quality if c["refused"])})
            self._latest.publish(packet)
            self.committed += 1
            self._rate_commit.record(commit_mono)
            self.commit_log.append({"seq": self._seq, "frame_t_mono": frame.t_mono, "commit_mono": commit_mono,
                                    "ident": list(frame.ident), "media_pts": (frame.media or {}).get("media_pts"),
                                    "src_n": (frame.media or {}).get("src_n")})
            facts = ev.facts
        # outside the lock: observers, trace, durable facts
        for obs_fn in list(self.commit_observers):
            try:
                obs_fn(packet)
            except Exception:  # noqa: BLE001 - an observer never disturbs the run
                logger.exception("commit observer raised")
        proj = packet.projection()
        proj["attempt"] = att.attempt_id
        proj["geometry"] = {k: {x: v.get(x) for x in ("state", "usable", "still", "shifted", "changed", "why")} for k, v in ev.geometry.items()}
        proj["candidate_quality_refused"] = [c for c in ev.candidate_quality if c["refused"]]
        proj["candidates"] = ev.candidates
        proj["negative"] = {k: list(v) for k, v in ev.negative.items()}
        proj["prior_states"] = prior_states
        self._t("result", **proj)
        for f in facts:
            self._persist_fact(f)
        return True

    def _persist_fact(self, f: dict) -> None:
        try:
            if f.get("kind") == "distractor_ambiguity":
                it = next((r for r in self._refs if r.item_id == f.get("item_id")), None)
                if it is not None:
                    with self.store.tx("ambiguity") as c:
                        c.execute("INSERT INTO item_ambiguity(item_id,ref_version,created_wall,detail) VALUES(?,?,?,?)",
                                  (it.item_id, it.ref_version, time.time(),
                                   dumps({"track_id": f.get("track_id"), "app": f.get("app"),
                                          "reason": "a similar unremembered object was seen together with this item"})))
            self._t("fact", fact=f)
        except Exception:  # noqa: BLE001 - never disturbs the run
            logger.exception("persisting a policy fact failed")

    # ---- outcome ------------------------------------------------------------------------------------------------------
    def _out(self, rec: VideoRecorder | None, requested: bool) -> OutputOutcome:
        if not requested:
            return OutputOutcome(False, OUTPUT_NOT_REQUESTED)
        if rec is None or rec.state == "idle":          # constructed but its encoder never started
            return OutputOutcome(True, OUTPUT_NOT_STARTED)
        s = rec.summary()
        status = "finalized" if rec.state == "finalized" else ("failed" if rec.state == "failed" else rec.state)
        if rec.state == "finalized" and rec.frames_dropped:
            status = "partial"
        return OutputOutcome(True, status, s["path"], s["frames_written"], s["frames_dropped"], s)

    def _build_outcome(self) -> SessionOutcome:
        tr = None
        if self.trace is not None and self.options.decision_trace:
            self._t("session", event="settled", ended_by=self._ended_by, error=self.error, committed=self.committed,
                    discards=dict(self.discards))
            st = self.trace.finish(30)
            tr = OutputOutcome(True, st["state"], st["path"], st["records"], st["dropped"], st)
        elif not self.options.decision_trace:
            tr = OutputOutcome(False, OUTPUT_NOT_REQUESTED)
        else:
            tr = OutputOutcome(True, OUTPUT_NOT_STARTED)
        kind = OUTCOME_ROLLED_BACK if self.phase == PHASE_ROLLING_BACK else (
            OUTCOME_STOPPED_AFTER_FAILURE if self.error else OUTCOME_FINALIZED)
        outcome = SessionOutcome(kind, self._ended_by, self.run_id, str(self.run_dir) if self.run_dir else None, self.options,
                                 self.error, self._out(self.clean, self.options.clean_source),
                                 self._out(self.annotated, self.options.record_annotated), tr,
                                 self.start_requested_mono, self.active_ack_mono, self.stopped_mono, self.committed,
                                 dict(self.discards))
        return outcome

    def _persist_run(self, out: SessionOutcome) -> None:
        board = self._board
        try:
            if board is not None:
                board.finish()
                with self.store.tx("persist run") as c:
                    for iv in board.intervals:
                        c.execute("""INSERT OR REPLACE INTO run_intervals(interval_id,run_id,item_id,state,start_mono,end_mono,
                                     start_wall,end_wall,start_ident,end_ident,samples,box,track_id,reason,closed)
                                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
                                  (iv["interval_id"], iv["run_id"], iv["item_id"], iv["state"], iv["start_mono"], iv["end_mono"],
                                   iv["start_wall"], iv["end_wall"], dumps(iv["start_ident"]), dumps(iv["end_ident"]),
                                   iv["samples"], dumps(iv["box"]), iv["track_id"], iv["reason"]))
                    started = self._active_wall or time.time()
                    for r in self._refs:
                        c.execute("INSERT INTO run_events(run_id,item_id,wall,kind,payload) VALUES(?,?,?,?,?)",
                                  (self.run_id, r.item_id, started, "monitoring_started",
                                   dumps({"kind": "monitoring_started", "item_id": r.item_id, "wall": started,
                                          "label": "Monitoring started", "ref_version": r.ref_version,
                                          "active": self.active_ack_mono is not None})))
                    for ev in board.events:
                        c.execute("INSERT INTO run_events(run_id,item_id,wall,kind,payload) VALUES(?,?,?,?,?)",
                                  (self.run_id, ev["item_id"], ev["wall"], ev["kind"], dumps(ev)))
                    for r in self._refs:
                        c.execute("INSERT INTO run_events(run_id,item_id,wall,kind,payload) VALUES(?,?,?,?,?)",
                                  (self.run_id, r.item_id, time.time(), "monitoring_ended",
                                   dumps({"kind": "monitoring_ended", "item_id": r.item_id, "label": "Monitoring ended",
                                          "outcome": {"kind": out.kind, "ended_by": out.ended_by, "error": out.error}})))
            manifest = self._manifest(out)
            if self.run_dir is not None:
                mp = self.run_dir / "manifest.json"
                mp.write_text(json.dumps(manifest, indent=1, default=str), "utf-8")
            with self.store.tx("finish run") as c:
                c.execute("UPDATE runs SET ended_wall=?, phase=?, outcome=? WHERE run_id=?",
                          (time.time(), out.kind, dumps({"kind": out.kind, "ended_by": out.ended_by, "error": out.error,
                                                         "clean": out.clean.status, "annotated": out.annotated.status,
                                                         "trace": out.trace.status, "committed": out.results_committed}), self.run_id))
            object.__setattr__(out, "manifest_path", str(self.run_dir / "manifest.json") if self.run_dir else None)
        except Exception:  # noqa: BLE001
            logger.exception("persisting the run failed")

    def _manifest(self, out: SessionOutcome) -> dict:
        cap = self._capture
        pipe = self._pipeline
        sp = self._source
        m = {"version": 1, "run_id": self.run_id, "outcome": {"kind": out.kind, "ended_by": out.ended_by, "error": out.error,
                                                              "complete": out.complete},
             "options": self.options.to_dict(), "policy_version": POLICY_VERSION, "refs_digest": self._refs_digest,
             "review_allowance_s": self._allowance_s, "setup": self._setup,
             "admission_gate": self.gate_info(),
             "source": {"id": sp.source_id, "path": sp.path, "fps": sp.fps, "repeat": sp.repeat} if sp else None,
             "decoder": cap.source.diagnostics() if cap else None,
             "items": [{"item_id": r.item_id, "name": r.name, "ref": r.ref_key, "box": list(r.box),
                        "crop_sha256": r.crop_sha256} for r in self._refs],
             "times": {"start_requested_mono": self.start_requested_mono, "active_ack_mono": self.active_ack_mono,
                       "stop_requested_mono": self.stopped_mono,
                       "requested_to_active_s": None if self.active_ack_mono is None else round(self.active_ack_mono - (self.start_requested_mono or 0), 3),
                       "clean_recorder_started_mono": self.clean.started_mono if self.clean else None,
                       "annotated_recorder_started_mono": self.annotated.started_mono if self.annotated else None},
             "counts": {"results_committed": self.committed, "discards": dict(self.discards),
                        "attempts": pipe.attempts if pipe else None, "replies": pipe.replies if pipe else None,
                        "frames_captured": cap.frames_captured if cap else None,
                        "frames_replaced_before_analysis": cap.inference_slot.dropped if cap else None,
                        "duplicate_source_frames": cap.frames_duplicate_source if cap else None,
                        "ineligible": dict(pipe.ineligible) if pipe else None},
             "basis_history": self._basis_history,
             "outputs": {"clean": out.clean.detail, "annotated": out.annotated.detail, "trace": out.trace.detail,
                         "pump": self.pump.summary() if self.pump else None},
             "worker": self.worker.health(), "board": self._board.summary() if self._board else None,
             "resource": self.sampler.summary() if self.sampler else None}
        return m

    # ---- reads -------------------------------------------------------------------------------------------------------
    def latest_packet(self) -> ResultPacket | None:
        return self._latest.peek()

    def live_events(self, item_id: str, limit: int = 8) -> list[dict]:
        """This run's committed but NOT YET SAVED activity for one item (newest first); saved at finalization."""
        with self._lock:
            b = self._board
            if b is None:
                return []
            return [dict(e) for e in reversed(b.events) if e["item_id"] == item_id][:limit]

    def gate_info(self) -> dict:
        from ..matching import default_gate
        return default_gate().to_dict()

    def snapshot(self) -> EngineSnapshot:
        with self._lock:
            now = self._clock()
            lp = self._latest.peek()
            live = self.phase == PHASE_ACTIVE and lp is not None
            rows = self._board.rows(now, live, self.s.limits.stale_result_s) if self._board else ()
            age = None if lp is None else now - lp.frame_t_mono
            cap = self._capture
            return EngineSnapshot(
                self.phase, self.step, self.run_id, self._basis.basis_id if self._basis else None,
                self._basis.control_generation if self._basis else None, rows, lp.packet_id if lp else None, age,
                {"capture_fps": None if cap is None else round(cap.capture_fps, 2),
                 "analyzed_fps": None if self._pipeline is None else round(self._rate_commit.rate(now=now), 2),
                 "encoded_clean_fps": None if self.clean is None else round(self.clean.recorder_fps(), 2),
                 "encoded_annotated_fps": None if self.annotated is None else round(self.annotated.recorder_fps(), 2),
                 "displayed_fps": None},
                {"clean": None if self.clean is None else self.clean.state,
                 "annotated": None if self.annotated is None else self.annotated.state,
                 "trace": None if self.trace is None else self.trace.state},
                {} if cap is None else cap.stats(), self.worker.health(), self.error)

    # ---- headless driver -------------------------------------------------------------------------------------------------
    def start_driver(self, period_s: float = 0.02) -> None:
        if self._driver is not None and self._driver.is_alive():
            return
        self._driver_stop.clear()

        def run():
            while not self._driver_stop.is_set():
                try:
                    self.poll()
                except Exception:  # noqa: BLE001
                    logger.exception("session poll raised")
                self._driver_stop.wait(period_s)
        self._driver = threading.Thread(target=run, name="session-driver", daemon=True)
        self._driver.start()

    def stop_driver(self) -> None:
        self._driver_stop.set()
        if self._driver is not None:
            self._driver.join(5)

    def wait_phase(self, phases: tuple[str, ...], timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.phase in phases:
                return True
            time.sleep(0.02)
        return self.phase in phases

    def wait_outcome(self, timeout: float) -> SessionOutcome | None:
        self._outcome_event.wait(timeout)
        return self.last_outcome if self._outcome_event.is_set() else None
