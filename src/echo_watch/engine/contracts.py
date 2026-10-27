# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable contracts between the engine and any future view: result packets, control basis, snapshots, outcomes.

Everything a renderer, item list, trace or annotated recorder shows is a PROJECTION of one committed ``ResultPacket``.
A packet binds image / frame / source / control / reference / model / worker identity to detections, item association,
assessability and states, and is committed exactly once by the controller (see engine/session.py).

Item states are distinct evidence classes, never one boolean:
  visible        a confirmed, soundly owned temporary track matched this item in this analysed frame
  checking       a compatible candidate is accumulating independent supports (tentative; never Visible)
  not_seen       an ELIGIBLE check found no compatible evidence (UI label "Not visible"); never proof of physical absence
  uncertain      compatible evidence exists but identity cannot be assigned (look-alike, overlap, suspect chain, margin)
  cannot_assess  the relevant view could not be assessed (dark, glared, obstructed, camera unproven, bound, resumed)
  paused / offline / stopped   the session is not observing: unknown, never absence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

STATE_VISIBLE = "visible"
STATE_CHECKING = "checking"
STATE_NOT_SEEN = "not_seen"
STATE_UNCERTAIN = "uncertain"
STATE_CANNOT_ASSESS = "cannot_assess"
STATE_PAUSED = "paused"
STATE_OFFLINE = "offline"
STATE_STOPPED = "stopped"
STATES = (STATE_VISIBLE, STATE_CHECKING, STATE_NOT_SEEN, STATE_UNCERTAIN, STATE_CANNOT_ASSESS, STATE_PAUSED,
          STATE_OFFLINE, STATE_STOPPED)
LABEL = {STATE_VISIBLE: "Visible", STATE_CHECKING: "Checking", STATE_NOT_SEEN: "Not visible",
         STATE_UNCERTAIN: "Match uncertain", STATE_CANNOT_ASSESS: "Cannot assess", STATE_PAUSED: "Paused",
         STATE_OFFLINE: "Offline", STATE_STOPPED: "Stopped"}
#: states that carry a live box on the analysed image of the packet that reported them
BOXED_STATES = (STATE_VISIBLE, STATE_CHECKING)

PHASE_IDLE = "idle"
PHASE_STARTING = "starting"
PHASE_ACTIVE = "active"
PHASE_PAUSED = "paused"
PHASE_STOPPING = "stopping"
PHASE_ROLLING_BACK = "rolling_back"

OUTCOME_FINALIZED = "finalized"
OUTCOME_STOPPED_AFTER_FAILURE = "stopped_after_failure"
OUTCOME_ROLLED_BACK = "rolled_back"
OUTPUT_NOT_REQUESTED = "not_requested"
OUTPUT_NOT_STARTED = "not_started"


@dataclass(frozen=True)
class ControlBasis:
    """The identity under which a result may become CURRENT. Any change below creates a new basis; a packet whose
    basis is not the controller's current one is discarded (kept only as diagnostic trace), never applied."""

    basis_id: str
    run_id: str
    control_generation: int       # advances on Start, Pause, Resume, source change, reference change, stop
    segment_id: int               # running segment (Start = 1; every Resume opens the next one)
    lower_bound_mono: float       # a frame captured before this instant is ineligible for this basis
    source_epoch_id: str
    source_loop: int              # replay loop id (0 while Repeat is OFF)
    refs_digest: str              # digest of the acknowledged item references (keys + versions + crop hashes)
    worker_session: int
    policy_version: str
    active: bool                  # False while paused / stopping: nothing may be committed


@dataclass(frozen=True)
class ItemRef:
    """A remembered item's reference as used by one run (immutable snapshot of the library row)."""

    item_id: str
    name: str
    category: str
    ref_version: int
    ref_key: str                  # '<item_id>@v<version>': the worker's reference key
    box: tuple[float, float, float, float]
    frame_size: tuple[int, int]
    crop_path: str
    context_path: str
    crop_sha256: str


@dataclass(frozen=True)
class DetectionRecord:
    cls: int
    ref_key: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ItemObservation:
    """One member item's committed result for one analysed image."""

    item_id: str
    name: str
    ref_version: int
    state: str
    reason: str | None
    box: tuple[float, float, float, float] | None      # on THIS packet's image, only for boxed states
    track_id: str | None = None
    score: float | None = None
    app: float | None = None
    conf: float | None = None
    supports: tuple[float, ...] = ()
    confirmed_now: bool = False
    negative_ok: bool | None = None                     # may a missing observation count as Not visible?
    negative_why: str | None = None
    alternatives: tuple[dict, ...] = ()
    last_supported: dict | None = None                  # {ident, t_mono, t_wall, box}: a PREVIOUS observation
    gap: dict | None = None                             # open not-seen episode counters
    events: tuple[dict, ...] = ()                       # timeline events emitted by this commit
    admission: str | None = None                        # support | unresolved | reject of the decided candidate

    @property
    def label(self) -> str:
        return LABEL.get(self.state, self.state)


@dataclass(frozen=True)
class ResultPacket:
    packet_id: str
    seq: int                                            # commit ordinal within the run
    run_id: str
    basis: ControlBasis
    frame_ident: tuple[str, int]                        # (epoch_id, sequence)
    frame_t_mono: float
    frame_t_wall: float
    size: tuple[int, int]
    image: np.ndarray = field(repr=False, compare=False)    # the exact analysed pixels (read-only)
    media: dict | None = field(default=None, repr=False, compare=False)
    detections: tuple[DetectionRecord, ...] = ()
    items: tuple[ItemObservation, ...] = ()
    timing: dict = field(default_factory=dict)          # monotonic stamps and per-stage milliseconds
    worker: dict = field(default_factory=dict)          # session, device, infer_s, ipc stamps
    flags: dict = field(default_factory=dict)

    @property
    def capture_to_commit_s(self) -> float | None:
        c = self.timing.get("commit_mono")
        return None if c is None else c - self.frame_t_mono

    def item(self, item_id: str) -> ItemObservation | None:
        return next((o for o in self.items if o.item_id == item_id), None)

    def projection(self) -> dict:
        """A JSON-safe projection (no pixels) shared by list, profile, trace and annotated output."""
        return {"packet_id": self.packet_id, "seq": self.seq, "run_id": self.run_id,
                "basis_id": self.basis.basis_id, "control_generation": self.basis.control_generation,
                "segment_id": self.basis.segment_id, "frame": list(self.frame_ident),
                "frame_t_mono": self.frame_t_mono, "frame_t_wall": self.frame_t_wall, "size": list(self.size),
                "media": self.media,
                "detections": [{"cls": d.cls, "ref": d.ref_key, "conf": d.confidence, "box": list(d.box)}
                               for d in self.detections],
                "items": [{"item_id": o.item_id, "state": o.state, "label": o.label, "reason": o.reason,
                           "box": None if o.box is None else [round(v, 1) for v in o.box], "track_id": o.track_id,
                           "score": o.score, "app": o.app, "conf": o.conf, "supports": list(o.supports),
                           "confirmed_now": o.confirmed_now, "negative_ok": o.negative_ok,
                           "negative_why": o.negative_why, "last_supported": o.last_supported, "gap": o.gap,
                           "admission": o.admission, "alternatives": list(o.alternatives),
                           "events": [e.get("kind") for e in o.events]} for o in self.items],
                "timing": self.timing, "worker": self.worker, "flags": self.flags}


@dataclass(frozen=True)
class CaptureOptions:
    """What a session saves. All three default ON; the operator may deliberately switch any off before starting."""

    record_annotated: bool = True
    clean_source: bool = True
    decision_trace: bool = True

    @property
    def effective_clean_source(self) -> bool:
        return self.clean_source

    def to_dict(self) -> dict[str, bool]:
        return {"record_annotated": self.record_annotated, "clean_source": self.clean_source,
                "decision_trace": self.decision_trace}


@dataclass(frozen=True)
class OutputOutcome:
    requested: bool
    status: str                    # finalized | failed | not_requested | not_started | partial
    path: str | None = None
    frames_written: int | None = None
    frames_dropped: int | None = None
    detail: dict | None = None


@dataclass(frozen=True)
class SessionOutcome:
    """How one managed session ended - every enabled output's terminal state, with requested time, actual start and
    dropped intervals reported separately."""

    kind: str                      # finalized | stopped_after_failure | rolled_back
    ended_by: str                  # operator | source_ended | failure | cancelled | abort
    run_id: str | None
    run_dir: str | None
    options: CaptureOptions
    error: str | None
    clean: OutputOutcome
    annotated: OutputOutcome
    trace: OutputOutcome
    started_requested_mono: float | None = None
    active_ack_mono: float | None = None
    stopped_mono: float | None = None
    results_committed: int = 0
    discarded: dict | None = None
    manifest_path: str | None = None

    @property
    def complete(self) -> bool:
        if self.kind != OUTCOME_FINALIZED:
            return False
        return all(o.status in ("finalized", OUTPUT_NOT_REQUESTED) for o in (self.clean, self.annotated, self.trace))

    def summary_text(self) -> str:
        if self.kind == OUTCOME_ROLLED_BACK:
            head = f"Session NOT started - everything it started was stopped: {self.error}"
        elif self.kind == OUTCOME_STOPPED_AFTER_FAILURE:
            head = f"Session stopped after a failure: {self.error}"
        else:
            head = "Session stopped and finalized." if self.complete else "Session stopped - NOT every output finalized cleanly."
        return (f"{head} Clean: {self.clean.status} | Annotated: {self.annotated.status} | "
                f"Decision trace: {self.trace.status}.")


@dataclass(frozen=True)
class ItemRow:
    """One row of the live item list / profile header (a projection of the latest committed packet)."""

    item_id: str
    name: str
    ref_version: int
    state: str
    label: str
    reason: str | None
    box: tuple | None
    last_supported: dict | None
    gap: dict | None
    packet_id: str | None
    age_s: float | None
    live: bool                    # False once the owning result is stale / paused / offline: no live authority


@dataclass(frozen=True)
class EngineSnapshot:
    phase: str
    step: str | None
    run_id: str | None
    basis_id: str | None
    control_generation: int | None
    items: tuple[ItemRow, ...]
    latest_packet_id: str | None
    latest_age_s: float | None
    rates: dict[str, Any]
    outputs: dict[str, Any]
    source: dict[str, Any]
    worker: dict[str, Any]
    error: str | None = None
