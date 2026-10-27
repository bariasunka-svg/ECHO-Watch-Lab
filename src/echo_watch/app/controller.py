# SPDX-License-Identifier: AGPL-3.0-only
"""The workbench's GUI-free command and view-model state (one owner for the Tk view; testable without Tcl/Tk).

EW-LAB-05 guided workflow (PatronSight's remembered-Visitor experience adapted to display items):
  launch STOPPED -> Discover: choose a local video (stopped-only; never starts monitoring) and run the real Scan of the
  retained setup frame -> Select: toggle suggestions, name them, decide NEW / USE SAVED / (separately) REPLACE REFERENCE,
  include compatible saved items -> "Remember & configure N items" = ONE captured, idempotent enrollment command ->
  Configure: saved reference cards, display name, review allowance, outputs, factual readiness, Save configuration ->
  Start monitoring (saves/confirms the exact setup version, then managed preparation) -> Monitor (latest committed packet
  + its own boxes) -> Pause / Resume -> Stop (non-blocking finalization) -> current membership clears; Library, saved
  setups and runs remain -> Review saved run / Reuse this setup / New setup.

Separate state that this class never mixes:
* the SETUP step (discover / select / configure) - navigation only, it never starts or stops anything;
* the ENGINE lifecycle (idle / starting / active / paused / stopping) owned by the session;
* the DRAFT selection (candidate checks, dispositions, saved-item picks), the saved LIBRARY, the SAVED SETUP versions
  and the current MEMBERSHIP (``tray``: exact item ids + reference versions confirmed by an enrollment or a loaded
  setup). Stop/finalization clears membership; nothing refills it until the operator explicitly reuses a setup.

Rules this class enforces (not the widgets):
* every slow command runs as a bounded background task with a GENERATION token per concern (source, scan, enroll,
  saved, setup, run, library, ...); a stale completion is discarded and recorded - except that a stale enrollment which
  DID commit is reported as the durable Library outcome it is, without attaching it to the new setup;
* one compatibility predicate (engine.compat: content scope + reference size + model identity) decides which saved items
  can be reused, hinted, replaced, loaded from a setup or started; resolution alone never suffices;
* a source change invalidates drafts, membership, saved-setup selection and in-flight setup work; Library/history stay;
* repeated Remember / Start / Stop clicks are idempotent or refused while one is pending, with an explicit failure path;
  an uncertain enrollment is reconciled with the SAME operation id before anything else is saved;
* settings distinguish Pending (on screen), Applied (saved configuration used by the next Start) and Running (what the
  active run actually acknowledged); pending edits never change a running configuration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from ..engine.compat import Scope
from ..engine.contracts import (LABEL, PHASE_ACTIVE, PHASE_IDLE, PHASE_PAUSED, PHASE_ROLLING_BACK, PHASE_STARTING,
                                PHASE_STOPPING, CaptureOptions)
from ..engine.setups import ALLOWANCE_DEFAULT, SetupError, SetupStore, build, content_digest, validate_allowance
from ..frames import new_id
from ..geometry import iou
from .tasks import Completion, TaskRunner

PHASE_TEXT = {PHASE_IDLE: "Stopped", PHASE_STARTING: "Preparing", PHASE_ACTIVE: "Monitoring", PHASE_PAUSED: "Paused",
              PHASE_STOPPING: "Finalizing", PHASE_ROLLING_BACK: "Rolling back"}
STATE_ORDER = {"visible": 0, "checking": 1, "uncertain": 2, "not_seen": 3, "cannot_assess": 4, "paused": 5, "offline": 6,
               "stopped": 7}
STEPS = ("discover", "select", "configure", "monitor")
STEP_LABEL = {"discover": "Discover", "select": "Select", "configure": "Configure", "monitor": "Monitor"}
CATEGORIES = ("Phone", "Tablet", "Laptop", "Headphones", "Speaker", "Watch", "Camera", "Accessory", "Sign", "Other")
_CATEGORY_WORDS = {"phone": "Phone", "smartphone": "Phone", "mobile": "Phone", "tablet": "Tablet", "ipad": "Tablet",
                   "laptop": "Laptop", "notebook": "Laptop", "computer": "Laptop", "headphone": "Headphones",
                   "headset": "Headphones", "earphone": "Headphones", "speaker": "Speaker", "watch": "Watch",
                   "clock": "Watch", "camera": "Camera", "sign": "Sign", "poster": "Sign", "card": "Sign"}


def suggest_category(label: str | None) -> str:
    low = (label or "").lower()
    return next((cat for word, cat in _CATEGORY_WORDS.items() if word in low), "Other")


@dataclass
class Candidate:
    candidate_id: str
    box: tuple[float, float, float, float]
    label: str
    confidence: float | None
    name: str
    manual: bool = False
    checked: bool = False
    corrected: bool = False
    duplicate_of: str | None = None
    overlaps_item: str | None = None
    category: str = "Other"
    disposition: str = "new"                 # new | reuse | choose (meaningful while checked)
    explicit: bool = False                   # the operator chose the disposition (a refresh never overrides it)
    reuse_item: str | None = None            # the saved item used for "reuse"
    matches: list = field(default_factory=list)   # compatible saved items overlapping this outline (suggestions only)
    enrolled: dict | None = None             # an item created from THIS candidate of THIS scan


@dataclass
class OutputSettings:
    record_annotated: bool = True
    clean_source: bool = True
    decision_trace: bool = True
    repeat: bool = False

    def as_options(self) -> CaptureOptions:
        return CaptureOptions(self.record_annotated, self.clean_source, self.decision_trace)

    def to_dict(self) -> dict:
        return {"record_annotated": self.record_annotated, "clean_source": self.clean_source,
                "decision_trace": self.decision_trace, "repeat": self.repeat}


@dataclass(frozen=True)
class EnrollCommand:
    """One captured enrollment request; its operation id is bound to the request digest."""

    op_id: str
    scan_id: str
    scope: str
    digest: str
    new: tuple
    reuse: tuple

    def new_list(self) -> list[dict]:
        return [dict(x) for x in self.new]

    def reuse_list(self) -> list[dict]:
        return [dict(x) for x in self.reuse]


@dataclass
class Message:
    text: str
    tone: str = "neutral"          # neutral | success | warning | danger
    wall: float = field(default_factory=time.time)


def _freeze(d: dict) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in d.items()))


class WorkbenchController:
    def __init__(self, engine, tasks: TaskRunner | None = None, probe=None, corrections=None):
        self.engine = engine
        self.tasks = tasks or TaskRunner()
        self._probe = probe
        if corrections is None:
            from ..review.corrections import Corrections
            corrections = Corrections(engine.store)
        self.corrections = corrections
        self.setups = SetupStore(Path(engine.s.data_dir) / "setups")
        self.gen = {"source": 0, "scan": 0, "run": 0, "library": 0, "runs": 0, "review": 0, "activity": 0, "worker": 0,
                    "enroll": 0, "saved": 0, "setup": 0, "setups": 0}
        # source / setup frame
        self.source_path: str | None = None
        self.source_info: dict | None = None
        self.setup_frame = None
        self.source_status = "none"           # none | preparing | ready | failed
        # setup step (navigation only; never the engine lifecycle)
        self.step = "discover"
        # scan
        self.scan_status = "idle"             # idle | waiting_worker | running | done | failed
        self.scan_id: str | None = None
        self.scan_frame_sha: str | None = None
        self.scan_model: str | None = None
        self.scan_error: str | None = None
        self.candidates: list[Candidate] = []
        # saved items for THIS recording (compatible) and the ones that cannot be reused here (with reasons)
        self.saved_items: list[dict] = []
        self.saved_blocked: list[dict] = []
        self.saved_picks: dict[str, int] = {}        # "Use saved items": item_id -> exact reference version
        # current membership (the tray) - never the Library; exact versions
        self.tray: list[str] = []
        self.tray_profiles: dict[str, dict] = {}
        self.member_versions: dict[str, int] = {}
        # enrollment command
        self.remember_pending = False
        self.enroll_cmd: EnrollCommand | None = None
        self.enroll_state = "idle"            # idle | pending | confirmed | rejected | not_saved | uncertain | conflict
        self.enroll_message: str | None = None
        self.replace_pending = False
        # saved setup / configuration draft
        self.setup = None                     # SavedSetup matching the membership/configuration last saved or loaded
        self.last_setup = None                # the setup the last run used (Reuse this setup)
        self.setup_name = "Display 01"
        self.setup_status = "unsaved"         # unsaved | saving | saved | failed | loading
        self.setup_error: str | None = None
        self.saved_setups: list[dict] = []
        self.all_setups: list[dict] = []
        self.pending_allowance: int = ALLOWANCE_DEFAULT
        self.applied_allowance: int = ALLOWANCE_DEFAULT
        self.start_pending = False
        self.stop_pending = False
        self.selected: str | None = None      # "cand:<id>" | "item:<id>" | "saved:<id>"
        # settings
        self.applied = OutputSettings()
        self.pending = OutputSettings()
        self.running_settings: dict | None = None
        self.running_setup: dict | None = None
        # worker
        self.worker_status = "not started"
        self.worker_receipt: dict | None = None
        self._after_worker: list[str] = []
        # outcome / history
        self.last_outcome = None
        self.last_outcome_text: str | None = None
        self.finished: dict | None = None     # the finish panel after settlement (Review / Reuse / New setup)
        self.review_request: str | None = None
        self.library: list[dict] = []
        self.runs: list[dict] = []
        self.messages: list[Message] = []
        self.discarded_completions: list[dict] = []
        self.command_acks: list[dict] = []
        self.ui_failures: list[str] = []
        self.closing = False
        self.closed = False
        self._seen_outcome = None
        self._window_completions: list[Completion] = []
        # EW-LAB-06: the application's single GPU-analysis owner (None in engine harnesses = no other mode exists)
        self.resources = None

    # ---- helpers ------------------------------------------------------------------------------------------------
    def say(self, text: str, tone: str = "neutral") -> None:
        self.messages.append(Message(text, tone))
        del self.messages[:-40]

    @property
    def phase(self) -> str:
        return self.engine.session.phase

    @property
    def stopped(self) -> bool:
        return self.engine.session.phase == PHASE_IDLE and not self.start_pending

    def _submit(self, kind: str, concern: str, fn) -> int:
        self.gen[concern] += 1
        tok = (concern, self.gen[concern])
        self.tasks.submit(kind, tok, fn)
        return self.gen[concern]

    def _current(self, c: Completion) -> bool:
        concern, g = c.token
        ok = self.gen.get(concern) == g
        if not ok:
            self.discarded_completions.append({"kind": c.kind, "token": list(c.token), "current": self.gen.get(concern),
                                               "wall": time.time()})
            del self.discarded_completions[:-100]
        return ok

    def gate(self) -> dict:
        from ..matching import default_gate
        return default_gate().to_dict()

    def recognition_qualified(self) -> bool:
        return bool(self.gate().get("qualified"))

    def model_identity(self) -> str | None:
        if self.scan_model:
            return self.scan_model
        f = getattr(self.engine, "model_identity", None)
        return f() if callable(f) else None

    def current_scope(self) -> Scope | None:
        info, f = self.source_info, self.setup_frame
        if info is None or f is None:
            return None
        return Scope(info.get("scope"), (f.width, f.height), self.model_identity())

    # ---- setup steps (navigation never starts or stops monitoring) ---------------------------------------------
    @property
    def view_step(self) -> str:
        if self.phase != PHASE_IDLE or self.start_pending or self.finished is not None:
            return "monitor"
        return self.step

    def step_allowed(self, step: str) -> tuple[bool, str]:
        if step == "monitor":
            return (self.view_step == "monitor", "Start monitoring from Configure")
        if self.view_step == "monitor":
            if self.finished is not None:
                return False, "Choose Review saved run, Reuse this setup or New setup"
            return False, "Monitoring is in progress; Stop it before changing the setup"
        if step == "discover":
            return True, ""
        if step == "select":
            if self.source_status != "ready":
                return False, "Choose a video first"
            if self.scan_status != "done":
                return False, "Available when discovery finishes"
            return True, ""
        if step == "configure":
            if not self.tray:
                return False, "Remember items in Select first"
            return True, ""
        return False, ""

    def go(self, step: str) -> bool:
        ok, why = self.step_allowed(step)
        if not ok:
            if why:
                self.say(why + ".", "warning")
            return False
        if step != "monitor":
            self.step = step
        return True

    def back(self) -> bool:
        vs = self.view_step
        if vs == "configure":
            return self.go("select" if self.scan_status == "done" else "discover")
        if vs == "select":
            return self.go("discover")
        return False

    def steps(self) -> list[dict]:
        cur = self.view_step
        idx = STEPS.index(cur)
        out = []
        for i, s in enumerate(STEPS):
            state = "complete" if i < idx else ("current" if i == idx else "upcoming")
            ok, _ = self.step_allowed(s)
            out.append({"n": i + 1, "key": s, "label": STEP_LABEL[s], "state": state, "enabled": ok and s != cur})
        return out

    # ---- source --------------------------------------------------------------------------------------------------
    def choose_source(self, path: str) -> bool:
        if not self.stopped:
            self.say("Stop monitoring before choosing another video.", "warning")
            return False
        p = Path(path)
        if not p.is_file():
            self.say("That video file does not exist.", "danger")
            return False
        self.source_path, self.source_info, self.setup_frame = str(p), None, None
        self.source_status = "preparing"
        self._clear_scan()
        self._reset_setup("source changed")
        eng = self.engine
        probe = self._probe
        mine = self.gen["source"] + 1                     # the generation this preparation will belong to

        def work():
            from ..config import FFMPEG
            from ..sources.fingerprint import fingerprint, verify_unchanged
            # content scope: bounded chunked hashing, cancelled as soon as a newer selection exists
            fp = fingerprint(p, cancelled=lambda: self.gen["source"] != mine)
            if probe is not None:
                info = probe(str(p))
            else:
                from ..sources.video_probe import probe_video
                info = probe_video(str(p), FFMPEG).as_dict()
            fps = float(info.get("effective_fps") or info.get("reported_fps") or 15.0)
            frame = eng.open_setup_source(str(p), fps, source_id=fp["scope"])
            verify_unchanged(p, fp["state"])              # stable around the whole preparation
            return {"info": {**info, "fps": fps, "path": str(p), "name": p.name, "scope": fp["scope"],
                             "sha256": fp["sha256"], "bytes": fp["bytes"], "file_state": fp["state"]}, "frame": frame}
        self._submit("source", "source", work)
        return True

    def _reset_setup(self, why: str) -> None:
        """Drafts, membership, saved-setup selection and in-flight setup work belong to ONE recording. Library profiles,
        saved setup files and run history are untouched."""
        for k in ("enroll", "saved", "setup", "setups"):
            self.gen[k] += 1
        self.tray, self.tray_profiles, self.member_versions = [], {}, {}
        self.saved_items, self.saved_blocked, self.saved_picks = [], [], {}
        self.enroll_cmd, self.enroll_state, self.enroll_message = None, "idle", None
        self.remember_pending = self.replace_pending = False
        self.setup, self.setup_status, self.setup_error = None, "unsaved", None
        self.saved_setups = []
        self.finished = None
        self.step = "discover"
        if self.selected and not self.selected.startswith("cand:"):
            self.selected = None

    def _invalidate_source(self, why: str) -> None:
        """The selected file changed: the setup frame, Scan, drafts and membership no longer describe it."""
        self.gen["source"] += 1
        self.source_info, self.setup_frame, self.source_status = None, None, "failed"
        self._clear_scan()
        self._reset_setup(why)
        self.start_pending = False
        self.say(f"{why}. Choose the video again and Discover items.", "danger")

    def _clear_scan(self) -> None:
        self.gen["scan"] += 1
        self.scan_status, self.scan_id, self.scan_frame_sha, self.candidates = "idle", None, None, []
        self.scan_model, self.scan_error = None, None
        if self.selected and self.selected.startswith("cand:"):
            self.selected = None

    # ---- worker ----------------------------------------------------------------------------------------------------
    def ensure_worker(self, then: str) -> bool:
        """True when the worker is ready now; otherwise starts it (once) in the background and runs ``then`` after."""
        w = self.engine.worker
        if w.alive():
            return True
        if then not in self._after_worker:
            self._after_worker.append(then)
        if self.worker_status != "starting":
            self.worker_status = "starting"
            self._submit("worker", "worker", self._claim_and_start_worker)
        return False

    def _claim_and_start_worker(self) -> dict:
        """Background task: the other mode's idle worker is released first (EW-LAB-06), then ours starts."""
        if self.resources is not None:
            self.resources.handover("display_assets")
        return self.engine.start_worker()

    def gpu_blocked(self) -> str | None:
        """Why Display Assets may not use the GPU now (another analysis mode is active), or None."""
        return self.resources.blocked_by("display_assets") if self.resources is not None else None

    def analysis_busy(self) -> bool:
        """Display Assets holds GPU analysis: a run in any phase, a pending Start, a discovery or a worker start."""
        return (self.phase != PHASE_IDLE or self.start_pending or self.scan_status in ("running", "waiting_worker")
                or self.worker_status == "starting")

    # ---- discover (scan) ----------------------------------------------------------------------------------------------
    def scan(self) -> bool:
        if not self.stopped:
            self.say("Discovery is available while monitoring is stopped.", "warning")
            return False
        if self.setup_frame is None:
            self.say("Choose a video first; discovery analyses its retained setup frame.", "warning")
            return False
        if self.scan_status in ("running", "waiting_worker"):
            return False
        why = self.gpu_blocked()
        if why:
            self.say(why, "warning")
            return False
        self._clear_scan()
        # a new discovery replaces the draft selection of the old one (membership already confirmed stays)
        self.saved_picks = {}
        if not self.ensure_worker("scan"):
            self.scan_status = "waiting_worker"
            return True
        self.scan_status = "running"
        frame, eng, info = self.setup_frame, self.engine, dict(self.source_info or {})

        def work():
            from ..sources.fingerprint import verify_unchanged
            verify_unchanged(info["path"], info["file_state"])
            return eng.scan(frame, info["scope"])          # scan + remembered items carry the content scope
        self._submit("scan", "scan", work)
        return True

    def select(self, key: str | None) -> None:
        self.selected = key

    def candidate(self, cid: str) -> Candidate | None:
        return next((c for c in self.candidates if c.candidate_id == cid), None)

    def saved_item(self, item_id: str) -> dict | None:
        return next((p for p in self.saved_items if p["item_id"] == item_id), None)

    # ---- select -------------------------------------------------------------------------------------------------------
    def _default_disposition(self, c: Candidate) -> None:
        if c.explicit:
            return
        e = c.enrolled
        if e is not None and (s := self.saved_item(e["item_id"])) is not None and \
                s["current_ref_version"] == e["current_ref_version"]:
            c.disposition, c.reuse_item = "reuse", e["item_id"]      # the exact same suggestion was remembered before
        elif c.matches:
            c.disposition, c.reuse_item = "choose", None              # overlap is a suggestion: the operator decides
        else:
            c.disposition, c.reuse_item = "new", None

    def toggle_candidate(self, cid: str, checked: bool | None = None) -> None:
        c = self.candidate(cid)
        if c is not None:
            c.checked = (not c.checked) if checked is None else checked
            if c.checked:
                self._default_disposition(c)

    def set_disposition(self, cid: str, disposition: str, item_id: str | None = None) -> bool:
        """Explicit operator decision for one suggestion: "new" (Create separate item) or "reuse" (Use saved item)."""
        c = self.candidate(cid)
        if c is None or not self.stopped:
            return False
        if disposition == "reuse":
            s = self.saved_item(item_id or "")
            if s is None:
                self.say("That saved item cannot be reused for this recording.", "warning")
                return False
            other = next((x for x in self.candidates if x is not c and x.checked and x.disposition == "reuse"
                          and x.reuse_item == item_id), None)
            if other is not None:
                self.say(f"{s['name']} is already used for {other.name}; one saved item is one identity.", "warning")
                return False
            c.disposition, c.reuse_item = "reuse", item_id
            self.saved_picks.pop(item_id, None)
        elif disposition == "new":
            c.disposition, c.reuse_item = "new", None
        else:
            return False
        c.checked, c.explicit = True, True
        return True

    def toggle_saved(self, item_id: str, on: bool | None = None) -> bool:
        """"Use saved items": include a compatible saved profile at its exact current reference version."""
        s = self.saved_item(item_id)
        if s is None or not self.stopped:
            return False
        want = (item_id not in self.saved_picks) if on is None else on
        if want:
            if any(x.checked and x.disposition == "reuse" and x.reuse_item == item_id for x in self.candidates):
                return False
            self.saved_picks[item_id] = int(s["current_ref_version"])
        else:
            self.saved_picks.pop(item_id, None)
        return True

    def set_category(self, cid: str, category: str) -> None:
        c = self.candidate(cid)
        if c is not None:
            c.category = (category or "Other").strip()[:40] or "Other"

    def rename_candidate(self, cid: str, name: str) -> bool:
        c = self.candidate(cid)
        name = (name or "").strip()
        if c is None or not name or len(name) > 80:
            self.say("A name needs 1-80 characters.", "warning")
            return False
        c.name = name
        return True

    def correct_region(self, cid: str, box) -> bool:
        c = self.candidate(cid)
        if c is None or self.setup_frame is None:
            return False
        w, h = self.setup_frame.width, self.setup_frame.height
        x, y, bw, bh = box
        if bw < 8 or bh < 8 or x < 0 or y < 0 or x + bw > w or y + bh > h:
            self.say("The adjusted outline must be inside the frame and at least 8 px.", "warning")
            return False
        c.box, c.corrected = tuple(round(v, 1) for v in box), True
        self._recompute_matches()
        return True

    def add_manual_region(self, box, name: str | None = None) -> bool:
        if self.scan_status != "done" or self.scan_id is None:
            self.say("Discover items first: Add missed item only recovers an item discovery missed.", "warning")
            return False
        scan_id, lib = self.scan_id, self.engine.library
        tok = self.gen["scan"]
        name = name or f"Item {len(self.candidates) + 1:02d}"

        def work():
            return lib.add_manual_candidate(scan_id, list(box))
        self.tasks.submit("manual", ("scan", tok), lambda: (work(), tuple(box), name))
        return True

    def selection_request(self) -> tuple[list[dict], list[dict], list[str]]:
        """(new, reuse, problems) of the CURRENT draft selection, in a stable order."""
        new, reuse, problems, used = [], [], [], set()
        for c in self.candidates:
            if not c.checked:
                continue
            if c.disposition == "choose":
                problems.append(f"{c.name}: overlaps saved item(s) - choose Use saved item or Create separate item")
            elif c.disposition == "reuse":
                s = self.saved_item(c.reuse_item or "")
                if s is None:
                    problems.append(f"{c.name}: the chosen saved item is not available for this recording")
                    continue
                used.add(s["item_id"])
                reuse.append({"item_id": s["item_id"], "ref_version": int(s["current_ref_version"]),
                              "candidate_id": c.candidate_id})
            else:
                n = {"candidate_id": c.candidate_id, "name": c.name, "category": c.category}
                if c.corrected or c.manual:
                    n["box"] = list(c.box)
                new.append(n)
        for item_id, ver in self.saved_picks.items():
            if item_id in used:
                continue
            reuse.append({"item_id": item_id, "ref_version": int(ver), "candidate_id": None})
        return new, reuse, problems

    def selection_counts(self) -> dict:
        """``selected`` counts every chosen suggestion and saved item, INCLUDING those still waiting for the
        new-or-saved decision (``choose``); ``new``/``reused`` count only decided ones."""
        new, reuse, problems = self.selection_request()
        choose = len(self.pending_choices())
        return {"selected": len(new) + len(reuse) + choose, "suggestions": len(self.candidates), "new": len(new),
                "reused": len(reuse), "choose": choose, "problems": problems}

    def pending_choices(self) -> list[Candidate]:
        """Selected suggestions that overlap saved items and still need the operator's decision."""
        return [c for c in self.candidates if c.checked and c.disposition == "choose"]

    def resolve_choices(self, disposition: str) -> int:
        """One explicit operator action for every pending overlap: "reuse" = Use saved item (each suggestion takes its
        best-overlapping saved item that no other selection already uses; a suggestion whose matches are all taken stays
        pending) or "new" = Create separate item. Returns how many were decided."""
        if not self.stopped or disposition not in ("reuse", "new"):
            return 0
        done = 0
        for c in self.pending_choices():
            if disposition == "new":
                done += self.set_disposition(c.candidate_id, "new")
                continue
            used = {x.reuse_item for x in self.candidates if x.checked and x.disposition == "reuse"}
            m = next((m for m in c.matches if m["item_id"] not in used and self.saved_item(m["item_id"])), None)
            if m is not None:
                done += self.set_disposition(c.candidate_id, "reuse", m["item_id"])
        left = len(self.pending_choices())
        verb = "Use saved item" if disposition == "reuse" else "Create separate item"
        self.say(f"{verb} chosen for {done} item(s)." + (f" {left} still need a choice (their saved items are already "
                                                         "used)." if left else ""), "neutral" if not left else "warning")
        return done

    def _selection_is_membership(self, new, reuse) -> bool:
        return (not new and self.enroll_state == "confirmed" and bool(self.tray)
                and {(r["item_id"], int(r["ref_version"])) for r in reuse} == set(self.member_versions.items()))

    # ---- remember & configure (one idempotent enrollment command) -----------------------------------------------
    def remember(self) -> bool:
        if self.remember_pending:
            return False
        if not self.stopped:
            self.say("Remember items while monitoring is stopped.", "warning")
            return False
        if self.enroll_state == "uncertain" and self.enroll_cmd is not None:
            self.say("The previous save's outcome is unknown; checking it first (same operation)...", "warning")
            return self.reconcile_enrollment()
        if self.scan_id is None:
            self.say("Discover items first.", "warning")
            return False
        new, reuse, problems = self.selection_request()
        if problems:
            self.say(problems[0] + ".", "warning")
            return False
        if not new and not reuse:
            self.say("Select at least one suggestion or saved item.", "warning")
            return False
        if self._selection_is_membership(new, reuse):
            self.step = "configure"                       # Back/Continue with an unchanged, confirmed selection
            return True
        from ..engine.library import request_digest
        digest = request_digest(self.scan_id, new, reuse)
        cmd = EnrollCommand(f"enroll-{digest[:24]}", self.scan_id, (self.source_info or {}).get("scope") or "", digest,
                            tuple(_freeze(x) for x in new), tuple(_freeze(x) for x in reuse))
        self._run_enroll(cmd, "remember")
        return True

    def _run_enroll(self, cmd: EnrollCommand, kind: str) -> None:
        self.enroll_cmd, self.enroll_state, self.enroll_message = cmd, "pending", None
        self.remember_pending = True
        lib = self.engine.library
        if kind == "remember":
            fn = lambda: lib.enroll(cmd.op_id, cmd.scan_id, cmd.new_list(), cmd.reuse_list())  # noqa: E731
        else:
            fn = lambda: lib.enroll_outcome(cmd.op_id, cmd.digest)  # noqa: E731
        self._submit(kind, "enroll", fn)

    def reconcile_enrollment(self) -> bool:
        """Read the durable outcome of the SAME operation (never a new id)."""
        if self.enroll_cmd is None or self.remember_pending:
            return False
        self._run_enroll(self.enroll_cmd, "reconcile")
        return True

    def _on_enroll(self, c: Completion) -> None:
        self.remember_pending = False
        if not c.ok:
            self.enroll_state = "uncertain"
            self.enroll_message = (f"Outcome unknown: {c.error}. Use Check saved outcome - it reads the same operation; "
                                   "nothing is saved twice.")
            self.say(self.enroll_message, "warning")
            return
        out = c.value
        self.enroll_state, self.enroll_message = out.status, out.message
        if out.status == "confirmed":
            try:
                self._adopt_enrollment(out)
            except Exception as e:  # noqa: BLE001 - the durable save is NOT undone by a screen failure
                self.ui_failures.append(f"{type(e).__name__}: {e}")
                self.say(f"{out.message} The items are saved in the Library, but this screen could not be updated "
                         f"({type(e).__name__}); open Select again to continue.", "warning")
                self.refresh_library()
                return
            self.say(out.message, "success")
            self.refresh_library()
            self.refresh_saved()
        elif out.status == "uncertain":
            self.say(out.message, "warning")
        else:
            self.say(out.message, "danger" if out.status != "not_saved" else "warning")

    def _adopt_enrollment(self, out) -> None:
        """All-or-nothing: membership becomes exactly the confirmed command's items at their exact versions."""
        refs = {r.item_id: r for r in out.refs}
        tray, profiles, versions = [], {}, {}
        for it in out.items:
            r = refs[it.item_id]
            tray.append(it.item_id)
            versions[it.item_id] = it.ref_version
            profiles[it.item_id] = {"item_id": it.item_id, "name": r.name, "crop_path": r.crop_path,
                                    "frame_width": r.frame_size[0], "frame_height": r.frame_size[1],
                                    "current_ref_version": r.ref_version, "ref_version": r.ref_version,
                                    "crop_sha256": r.crop_sha256, "box": list(r.box), "category": r.category,
                                    "disposition": it.disposition}
        for it in out.items:
            if it.disposition == "new" and it.candidate_id:
                cnd = self.candidate(it.candidate_id)
                if cnd is not None:
                    cnd.enrolled = {"item_id": it.item_id, "ref_version": it.ref_version, "name": it.name,
                                    "current_ref_version": it.ref_version}
                    cnd.disposition, cnd.reuse_item, cnd.explicit = "reuse", it.item_id, False
        self.tray, self.tray_profiles, self.member_versions = tray, profiles, versions
        known = {s["item_id"] for s in self.saved_items}
        for it in out.items:                          # new profiles are saved items of THIS recording right away
            if it.item_id not in known:
                p = profiles[it.item_id]
                self.saved_items.append({**p, "current_ref_version": it.ref_version, "compat_code": "ok",
                                         "compat_reason": "saved from this recording"})
        for it in out.items:
            if it.disposition == "reused" and it.candidate_id is None:
                self.saved_picks[it.item_id] = it.ref_version
        self.setup, self.setup_status, self.setup_error = None, "unsaved", None
        if tray:
            self.selected = f"item:{tray[0]}"
        self.step = "configure"

    def replace_reference(self, cid: str, item_id: str) -> bool:
        """Explicit, versioned, stopped-only reference replacement of a COMPATIBLE saved item from this suggestion.
        Not needed to rerun; earlier versions and history are kept."""
        c = self.candidate(cid)
        s = self.saved_item(item_id)
        if self.replace_pending or not self.stopped or c is None or s is None or self.scan_id is None:
            if s is None:
                self.say("Only a saved item of this recording can have its reference replaced here.", "warning")
            return False
        self.replace_pending = True
        lib, scan_id, op = self.engine.library, self.scan_id, new_id("refupd")
        box = list(c.box) if (c.corrected or c.manual) else None
        self._submit("replace_ref", "saved", lambda: (lib.update_reference(op, item_id, scan_id, cid, box=box), cid))
        self.say(f"Saving a NEW reference version for {s['name']}; earlier versions and their history are kept.")
        return True

    # ---- saved items / library ------------------------------------------------------------------------------------
    def refresh_saved(self) -> None:
        scope, lib, scan_id = self.current_scope(), self.engine.library, self.scan_id
        if scope is None:
            return

        def work():
            ok, bad = lib.saved_for_scope(scope)
            return ok, bad, (lib.enrolled_from_scan(scan_id) if scan_id else {})
        self._submit("saved", "saved", work)

    def _recompute_matches(self) -> None:
        for c in self.candidates:
            c.matches = sorted(({"item_id": s["item_id"], "name": s["name"], "iou": round(iou(c.box, s["box"]), 3),
                                 "ref_version": s["current_ref_version"]} for s in self.saved_items
                                if iou(c.box, s["box"]) > 0.5), key=lambda m: -m["iou"])
            if c.checked:
                if c.disposition == "reuse" and self.saved_item(c.reuse_item or "") is None:
                    c.explicit = False
                self._default_disposition(c)

    def refresh_library(self) -> None:
        lib = self.engine.library
        self._submit("library", "library", lib.profiles)

    def refresh_setups(self) -> None:
        st = self.setups
        self._submit("setups", "setups", st.list_latest)

    def add_to_run(self, item_id: str) -> bool:
        """Explicit reuse of one saved profile (from Library) - only when compatible with the prepared recording."""
        if not self.stopped:
            self.say("Run membership changes while stopped.", "warning")
            return False
        if item_id in self.tray:
            return False
        prof = next((p for p in self.library if p["item_id"] == item_id), None)
        if prof is None:
            return False
        scope = self.current_scope()
        if scope is None:
            self.say("Choose the recording first; saved items are reused only with the recording they came from.",
                     "warning")
            return False
        from ..engine.compat import check_row
        v = check_row(prof, scope)
        if not v.ok:
            self.say(f"{prof['name']} cannot be used here: {v.reason}.", "warning")
            return False
        ver = int(prof["current_ref_version"])
        self.tray.insert(0, item_id)
        self.member_versions[item_id] = ver
        self.tray_profiles[item_id] = {**prof, "ref_version": ver}
        self.saved_picks[item_id] = ver
        self.setup, self.setup_status = None, "unsaved"
        self.selected = f"item:{item_id}"
        self.say(f"{prof['name']} added to this setup (Saved, not monitoring).", "neutral")
        return True

    def remove_from_run(self, item_id: str) -> bool:
        """Removes MEMBERSHIP only (never the Library profile or its history)."""
        if not self.stopped or item_id not in self.tray:
            return False
        self.tray.remove(item_id)
        self.tray_profiles.pop(item_id, None)
        self.member_versions.pop(item_id, None)
        self.saved_picks.pop(item_id, None)
        for c in self.candidates:
            if c.checked and c.disposition == "reuse" and c.reuse_item == item_id:
                c.checked = False
        self.setup, self.setup_status = None, "unsaved"
        if self.selected == f"item:{item_id}":
            self.selected = None
        if self.enroll_state == "confirmed":
            self.enroll_state = "idle"
        return True

    def rename_item(self, item_id: str, name: str) -> None:
        lib = self.engine.library
        op = new_id("rename")
        self._submit("rename", "library", lambda: (lib.rename(op, item_id, name), lib.profiles())[1])

    # ---- configure ------------------------------------------------------------------------------------------------
    def set_pending(self, **kw) -> None:
        for k, v in kw.items():
            if hasattr(self.pending, k):
                setattr(self.pending, k, bool(v))

    def set_allowance(self, value) -> bool:
        try:
            self.pending_allowance = validate_allowance(value)
            return True
        except SetupError as e:
            self.say(str(e) + ".", "warning")
            return False

    def set_setup_name(self, name: str) -> bool:
        name = (name or "").strip()
        if not name or len(name) > 80:
            self.say("The display name needs 1-80 characters.", "warning")
            return False
        self.setup_name = name
        return True

    def apply_settings(self) -> bool:
        self.applied = OutputSettings(**self.pending.to_dict())
        self.applied_allowance = self.pending_allowance
        self.say("Settings applied: they take effect at the next Start." if not self.stopped else "Settings applied.",
                 "neutral")
        return True

    def settings_status(self) -> dict:
        p, a = self.pending.to_dict(), self.applied.to_dict()
        return {k: {"pending": p[k], "applied": a[k],
                    "running": None if self.running_settings is None else self.running_settings.get(k),
                    "dirty": p[k] != a[k]} for k in p}

    def allowance_status(self) -> dict:
        run = (self.running_settings or {}).get("review_allowance_s")
        return {"pending": self.pending_allowance, "applied": self.applied_allowance, "running": run,
                "dirty": self.pending_allowance != self.applied_allowance}

    def setup_document(self) -> dict:
        """The setup document of what is on screen now (raises SetupError when it is not savable)."""
        info = self.source_info
        if info is None or self.setup_frame is None:
            raise SetupError("Choose the recording first")
        if not self.tray:
            raise SetupError("Remember at least one item first")
        from ..config import MODEL_PF_SHA256, MODEL_VP_SHA256
        from ..item_policy import POLICY_VERSION
        items = []
        for i in self.tray:
            p = self.tray_profiles.get(i) or {}
            if not p.get("crop_sha256"):
                raise SetupError(f"{p.get('name', i)}: its saved reference is not available")
            items.append({"item_id": i, "name": p.get("name", i), "ref_version": int(self.member_versions[i]),
                          "crop_sha256": p["crop_sha256"]})
        g = self.gate()
        return build(self.setup_name, {"scope": info["scope"], "name": info.get("name"), "path": info.get("path"),
                                       "bytes": info.get("bytes"), "width": self.setup_frame.width,
                                       "height": self.setup_frame.height, "fps": info.get("fps")},
                     items, self.pending.to_dict(), self.pending_allowance,
                     {"discovery_model_sha256": self.model_identity(), "pf_sha256": MODEL_PF_SHA256,
                      "vp_sha256": MODEL_VP_SHA256},
                     {"calibration_id": g.get("calibration_id"), "descriptor": g.get("descriptor"),
                      "qualified": g.get("qualified")}, POLICY_VERSION)

    def setup_dirty(self) -> bool:
        if self.setup is None:
            return True
        try:
            return content_digest(self.setup_document()) != self.setup.digest
        except SetupError:
            return True

    def save_configuration(self) -> bool:
        if not self.stopped or self.setup_status == "saving":
            return False
        try:
            doc = self.setup_document()
        except SetupError as e:
            self.setup_status, self.setup_error = "failed", str(e)
            self.say(f"Configuration not saved: {e}.", "warning")
            return False
        st, sid = self.setups, (self.setup.setup_id if self.setup is not None else None)
        self.setup_status, self.setup_error = "saving", None
        self._submit("setup_save", "setup", lambda: st.save(doc, sid))
        return True

    def load_setup(self, setup_id: str, version: int | None = None) -> bool:
        """Validate a saved setup against the Library and THIS recording, then make it the membership (Configure).
        Never opens a device or starts a run; a failing link changes nothing."""
        if not self.stopped:
            return False
        scope = self.current_scope()
        if scope is None:
            self.say("Choose the recording this setup belongs to first.", "warning")
            return False
        st, lib = self.setups, self.engine.library

        def work():
            s = st.load(setup_id, version)
            if s.scope != scope.content:
                raise SetupError(f"'{s.name}' belongs to {s.doc['source'].get('name') or 'another recording'} "
                                 "(different file content); choose that recording, or discover items for this one")
            problems, refs = [], []
            for entry in s.doc["items"]:
                try:
                    r = lib.resolve_refs([(entry["item_id"], entry["ref_version"])], scope)[0]
                    if r.crop_sha256 != entry["crop_sha256"]:
                        raise SetupError("its reference image differs from the saved setup")
                    refs.append(r)
                except Exception as e:  # noqa: BLE001 - every broken link is listed, none is repaired silently
                    problems.append(f"{entry.get('name', entry['item_id'])} v{entry['ref_version']}: {e}")
            if problems:
                raise SetupError("This saved setup cannot be used as it is - " + "; ".join(problems))
            return s, refs
        self.setup_status, self.setup_error = "loading", None
        self._submit("setup_load", "setup", work)
        return True

    # ---- lifecycle ---------------------------------------------------------------------------------------------------
    def start(self) -> bool:
        if self.start_pending or not self.stopped:
            return False
        if not self.tray:
            self.say("Remember an item or add a saved one before starting.", "warning")
            return False
        if self.source_info is None:
            self.say("Choose a video first.", "warning")
            return False
        try:
            doc = self.setup_document()
        except SetupError as e:
            self.say(f"Monitoring did not start: {e}.", "warning")
            return False
        why = self.gpu_blocked()
        if why:
            self.say(why, "warning")
            return False
        if not self.ensure_worker("start"):
            self.start_pending = True
            return True
        self.start_pending = True
        eng, info, st = self.engine, dict(self.source_info), self.setups
        sid = self.setup.setup_id if self.setup is not None else None
        size = (self.setup_frame.width, self.setup_frame.height)
        progress: dict = {}

        def work():
            from ..engine.session import SourceSpec
            from ..sources.fingerprint import verify_unchanged
            saved = st.save(doc, sid)                     # the exact configuration this Start uses (confirmed first)
            progress["saved"] = saved
            verify_unchanged(info["path"], info["file_state"])
            t0 = time.monotonic()
            out = saved.outputs
            opts = CaptureOptions(out["record_annotated"], out["clean_source"], out["decision_trace"])
            eng.start(opts, [list(x) for x in saved.items],
                      SourceSpec(info["scope"], info["path"], info["fps"], out["repeat"], size),
                      allowance_s=saved.allowance_s, setup=saved.identity())
            eng.session.start_driver()
            return {"ack_s": round(time.monotonic() - t0, 3), "settings": {**out, "review_allowance_s": saved.allowance_s},
                    "setup": saved}
        self._start_progress = progress
        self._submit("start", "run", work)
        self.command_acks.append({"command": "start", "wall": time.time()})
        return True

    def pause(self) -> bool:
        if self.phase != PHASE_ACTIVE:
            return False
        s = self.engine.session
        self._submit("pause", "run", s.pause)
        return True

    def resume(self) -> bool:
        if self.phase != PHASE_PAUSED:
            return False
        s = self.engine.session
        self._submit("resume", "run", s.resume)
        return True

    def stop(self) -> bool:
        if self.stop_pending or self.phase not in (PHASE_ACTIVE, PHASE_PAUSED, PHASE_STARTING):
            return False
        self.stop_pending = True
        s = self.engine.session
        self._submit("stop", "run", s.stop)
        self.command_acks.append({"command": "stop", "wall": time.time()})
        return True

    # ---- finish actions --------------------------------------------------------------------------------------------
    def review_saved_run(self) -> str | None:
        run_id = (self.finished or {}).get("run_id")
        if run_id:
            self.review_request = run_id
            self.refresh_runs()
        return run_id

    def reuse_setup(self) -> bool:
        s = self.last_setup
        if s is None:
            self.say("There is no saved setup from the last run to reuse.", "warning")
            return False
        scope = self.current_scope()
        if scope is None or scope.content != s.scope:
            self.say(f"'{s.name}' belongs to {s.doc['source'].get('name') or 'another recording'}; choose that recording "
                     "(same file content) in Discover to reuse it.", "warning")
            return False
        self.finished = None
        return self.load_setup(s.setup_id, s.version)

    def new_setup(self) -> bool:
        if not self.stopped:
            return False
        self.finished = None
        for k in ("enroll", "setup"):
            self.gen[k] += 1
        for c in self.candidates:
            c.checked, c.explicit = False, False
        self.saved_picks = {}
        self.tray, self.tray_profiles, self.member_versions = [], {}, {}
        self.enroll_cmd, self.enroll_state, self.enroll_message = None, "idle", None
        self.setup, self.setup_status, self.setup_error = None, "unsaved", None
        self.setup_name = "Display 01"
        self.selected = None
        self.step = "discover"
        self.refresh_setups()
        return True

    # ---- runs ---------------------------------------------------------------------------------------------------------
    def refresh_runs(self) -> None:
        from ..review.runs import list_runs
        store = self.engine.store
        self._submit("runs", "runs", lambda: list_runs(store))

    # ---- the UI thread's periodic pump ------------------------------------------------------------------------------
    def pump(self) -> list[Completion]:
        """Apply finished background work (current tokens only) and observe session settlement. Cheap; no I/O."""
        done = self.tasks.drain()
        for c in done:
            self._apply(c)
        out = self.engine.session.last_outcome
        if out is not None and out is not self._seen_outcome and self.engine.session.phase == PHASE_IDLE:
            self._seen_outcome = out
            self._on_outcome(out)
        return done

    CONTROLLER_KINDS = ("worker", "source", "scan", "manual", "remember", "reconcile", "library", "rename", "start",
                        "pause", "resume", "stop", "runs", "saved", "replace_ref", "setup_save", "setup_load", "setups")

    def _apply(self, c: Completion) -> None:
        k = c.kind
        if k not in self.CONTROLLER_KINDS:
            self._window_completions.append(c)          # the view's own tasks (thumbnails, review, export, ...)
            return
        if k == "worker":
            ok = c.ok and isinstance(c.value, dict) and c.value.get("event") == "ready"
            self.worker_status = "ready" if ok else "failed"
            self.worker_receipt = c.value if c.ok else None
            if not ok:
                why = c.error or (c.value or {}).get("error") or self.engine.worker.reason
                self.say(f"The GPU inference worker could not start: {why}. No CPU substitute is used.", "danger")
                if self.scan_status == "waiting_worker":
                    self.scan_status, self.scan_error = "failed", f"the GPU worker could not start: {why}"
                self.start_pending = False
                self._after_worker.clear()
                return
            todo, self._after_worker = self._after_worker, []
            for t in todo:
                if t == "scan" and self.scan_status == "waiting_worker":
                    self.scan_status = "idle"
                    self.scan()
                elif t == "start" and self.start_pending:
                    self.start_pending = False
                    self.start()
            return
        if not self._current(c):
            if k in ("remember", "reconcile"):
                self.remember_pending = False
                self._stale_enrollment(c)
            elif k == "replace_ref":
                self.replace_pending = False
                if c.ok:
                    self.say("An earlier reference replacement completed for the previous recording; it is in the "
                             "Library (not part of this setup).", "neutral")
                    self.refresh_library()
            return
        if k in ("source", "scan", "start") and not c.ok and (c.error or "").startswith("SourceChanged"):
            if k == "start":
                self.start_pending = False
            self._invalidate_source("The selected video file changed" if k != "source" else
                                    "The video file changed while it was being prepared")
            return
        if k == "source":
            if not c.ok:
                self.source_status = "failed"
                self.say(f"The video could not be prepared: {c.error}", "danger")
                return
            self.source_info, self.setup_frame = c.value["info"], c.value["frame"]
            self.source_status = "ready"
            self.say(f"{self.source_info['name']} ready. Discover items analyses this setup frame; monitoring has not "
                     "started.", "neutral")
            self.refresh_setups()
        elif k == "scan":
            r = c.value if c.ok else None
            if r is None or r.status != "succeeded":
                self.scan_status = "failed"
                self.scan_error = c.error or (r.error if r else "") or "unknown error"
                self.say(f"Discovery failed: {self.scan_error}. Retry is available.", "danger")
                return
            self.scan_status, self.scan_id, self.scan_frame_sha = "done", r.scan_id, r.frame_sha256
            self.scan_model = getattr(r, "model_sha256", None)
            self.candidates = [Candidate(p.candidate_id, tuple(p.box), p.label, p.confidence,
                                         self._suggest_name(p.label, i), duplicate_of=p.duplicate_of,
                                         overlaps_item=p.overlaps_item, category=suggest_category(p.label))
                               for i, p in enumerate(r.proposals)]
            n = len(self.candidates)
            self.say(f"Discovery found {n} candidate{'s' if n != 1 else ''} on the setup frame ({r.infer_s or 0:.2f} s on "
                     f"{r.device}). Continue to selection." if n else
                     "Discovery finished but found no candidates. Discover again, or continue and Add missed item.",
                     "success" if n else "warning")
            self.refresh_saved()
        elif k == "manual":
            if not c.ok:
                self.say(f"The missed item was not added: {c.error}", "danger")
                return
            cid, box, name = c.value
            cnd = Candidate(cid, tuple(box), "manual outline", None, name, manual=True, checked=True)
            self.candidates.append(cnd)
            self._recompute_matches()
            self._default_disposition(cnd)
            self.selected = f"cand:{cid}"
        elif k in ("remember", "reconcile"):
            self._on_enroll(c)
        elif k == "saved":
            if not c.ok:
                self.say(f"Saved items could not be read: {c.error}", "danger")
                return
            ok, bad, enrolled = c.value
            self.saved_items, self.saved_blocked = ok, bad
            for cnd in self.candidates:
                e = enrolled.get(cnd.candidate_id)
                cnd.enrolled = e
            dropped = [i for i, v in self.saved_picks.items()
                       if (s := self.saved_item(i)) is None or int(s["current_ref_version"]) != v]
            for i in dropped:
                self.saved_picks.pop(i, None)
            if dropped:
                self.say(f"{len(dropped)} chosen saved item(s) changed or are no longer available; review the selection.",
                         "warning")
            self._recompute_matches()
        elif k == "replace_ref":
            self.replace_pending = False
            if not c.ok:
                self.say(f"Reference not replaced: {c.error}", "danger")
                return
            ref, cid = c.value
            self.say(f"Saved reference v{ref.ref_version} for {ref.name}. Earlier versions and their history are kept; "
                     "the suggestion now uses this saved item.", "success")
            cnd = self.candidate(cid)
            if cnd is not None:
                cnd.checked, cnd.disposition, cnd.reuse_item, cnd.explicit = True, "reuse", ref.item_id, True
            if ref.item_id in self.member_versions:
                self.remove_from_run(ref.item_id)        # membership pinned the old version; it must be reconfirmed
            self.refresh_saved()
            self.refresh_library()
        elif k == "setup_save":
            if not c.ok:
                self.setup_status, self.setup_error = "failed", c.error
                self.say(f"Configuration not saved: {c.error}. Your selections and settings are kept; nothing started.",
                         "danger")
                return
            self.setup, self.setup_status = c.value, "saved"
            self.applied = OutputSettings(**{k2: c.value.outputs[k2] for k2 in ("record_annotated", "clean_source",
                                                                                 "decision_trace", "repeat")})
            self.applied_allowance = c.value.allowance_s
            self.say(f"Configuration saved: {c.value.name} v{c.value.version}.", "success")
            self.refresh_setups()
        elif k == "setup_load":
            if not c.ok:
                self.setup_status, self.setup_error = "failed", c.error
                self.say(f"Setup not loaded: {(c.error or '').removeprefix('SetupError: ')}", "danger")
                return
            s, refs = c.value
            rm = {r.item_id: r for r in refs}
            self.tray = [i for i, _v in s.items]
            self.member_versions = {i: v for i, v in s.items}
            self.tray_profiles = {i: {"item_id": i, "name": rm[i].name, "crop_path": rm[i].crop_path,
                                      "frame_width": rm[i].frame_size[0], "frame_height": rm[i].frame_size[1],
                                      "current_ref_version": v, "ref_version": v, "crop_sha256": rm[i].crop_sha256,
                                      "box": list(rm[i].box), "category": rm[i].category, "disposition": "reused"}
                                  for i, v in s.items}
            o = s.outputs
            self.pending = OutputSettings(o["record_annotated"], o["clean_source"], o["decision_trace"], o["repeat"])
            self.applied = OutputSettings(**self.pending.to_dict())
            self.pending_allowance = self.applied_allowance = s.allowance_s
            self.setup_name = s.name
            self.setup, self.setup_status, self.setup_error = s, "saved", None
            self.enroll_state = "idle"
            self.finished = None
            self.selected = f"item:{self.tray[0]}" if self.tray else None
            self.step = "configure"
            self.say(f"Loaded {s.name} v{s.version}: {len(self.tray)} saved item(s), same ids and reference versions. "
                     "Saved, not monitoring.", "success")
        elif k == "setups":
            if c.ok:
                self.all_setups = c.value
                scope = (self.source_info or {}).get("scope")
                self.saved_setups = [x for x in c.value if scope and x.get("scope") == scope and not x.get("error")]
        elif k == "library" or k == "rename":
            if c.ok:
                self.library = c.value
                for p in self.library:
                    if p["item_id"] in self.tray_profiles:
                        self.tray_profiles[p["item_id"]].update(name=p["name"])
            else:
                self.say(f"Library could not be read: {c.error}", "danger")
        elif k == "start":
            self.start_pending = False
            prog = getattr(self, "_start_progress", {}) or {}
            if not c.ok:
                saved = prog.get("saved")
                if saved is not None:
                    self.setup, self.setup_status = saved, "saved"
                    self.say(f"Configuration saved as {saved.name} v{saved.version}, but monitoring did not start: "
                             f"{c.error}", "danger")
                else:
                    self.setup_status, self.setup_error = "failed", c.error
                    self.say(f"Configuration could not be saved, so monitoring did not start: {c.error}. Your selections "
                             "are kept.", "danger")
                return
            self.running_settings = c.value["settings"]
            saved = c.value["setup"]
            self.setup = self.last_setup = saved
            self.setup_status = "saved"
            self.running_setup = saved.identity()
            self.applied = OutputSettings(**{k2: saved.outputs[k2] for k2 in ("record_annotated", "clean_source",
                                                                               "decision_trace", "repeat")})
            self.applied_allowance = saved.allowance_s
        elif k in ("pause", "resume"):
            if not c.ok:
                self.say(f"{k.title()} was refused: {c.error}", "warning")
        elif k == "stop":
            if not c.ok:
                self.stop_pending = False
                self.say(f"Stop could not be requested: {c.error}", "danger")
            else:
                self.say("Stopping: finalizing recordings and the decision trace...", "neutral")
        elif k == "runs":
            if c.ok:
                self.runs = c.value
            else:
                self.say(f"Runs could not be read: {c.error}", "danger")

    def _stale_enrollment(self, c: Completion) -> None:
        """An enrollment of a PREVIOUS recording/setup finished late: never attach it here, but report what is durable."""
        out = c.value if c.ok else None
        if out is not None and getattr(out, "status", None) == "confirmed":
            self.say(f"An earlier save for the previous recording completed: {len(out.items)} item(s) are saved in the "
                     "Library. They were not added to this setup.", "neutral")
            self.refresh_library()
        elif out is None or getattr(out, "status", None) == "uncertain":
            self.say("An earlier save for the previous recording has an unknown outcome; the Library shows what was "
                     "saved. Nothing was added to this setup.", "warning")
            self.refresh_library()

    def _on_outcome(self, out) -> None:
        self.last_outcome = out
        self.last_outcome_text = out.summary_text()
        self.stop_pending = False
        self.running_settings = None
        self.running_setup = None
        # Stop / finalization clears THIS run's membership; the Library, saved setups and the saved run remain
        self.tray, self.tray_profiles, self.member_versions = [], {}, {}
        self.enroll_state = "idle"
        if self.selected and self.selected.startswith("item:"):
            self.selected = None
        self.gen["run"] += 1
        self.finished = {"run_id": out.run_id, "kind": out.kind, "ended_by": out.ended_by, "complete": out.complete,
                         "error": out.error, "summary": self.last_outcome_text,
                         "outputs": {"Clean recording": out.clean.status, "Annotated recording": out.annotated.status,
                                     "Decision trace": out.trace.status},
                         "setup": None if self.last_setup is None else
                         {"name": self.last_setup.name, "version": self.last_setup.version}}
        tone = "success" if out.complete else ("warning" if out.kind == "finalized" else "danger")
        self.say(self.last_outcome_text + " Current membership is cleared; the run is saved in Runs & Review.", tone)
        self.refresh_runs()
        self.refresh_library()

    @staticmethod
    def _suggest_name(label: str, i: int) -> str:
        return f"Item {i + 1:02d}"

    # ---- read model for the view -----------------------------------------------------------------------------------
    def primary_status(self) -> tuple[str, str]:
        ph = self.phase
        if self.start_pending and ph == PHASE_IDLE:
            return "Preparing", "accent"
        txt = PHASE_TEXT.get(ph, ph)
        tone = {"Monitoring": "success", "Preparing": "accent", "Paused": "warning", "Finalizing": "warning",
                "Rolling back": "danger"}.get(txt, "neutral")
        return txt, tone

    def tray_rows(self, snapshot) -> list[dict]:
        rows = {r.item_id: r for r in (snapshot.items if snapshot else ())}
        out = []
        for item_id in self.tray:
            p = self.tray_profiles.get(item_id, {})
            r = rows.get(item_id)
            if r is None:
                state, label, reason, live = "saved", "Saved, not monitoring", None, False
            else:
                state, label, reason, live = r.state, LABEL.get(r.state, r.state), r.reason, r.live
            ls = r.last_supported if r is not None else None
            out.append({"item_id": item_id, "name": (r.name if r else p.get("name", item_id)), "state": state,
                        "label": label, "reason": reason, "live": live, "crop_path": p.get("crop_path"),
                        "last_supported_wall": (ls or {}).get("t_wall"), "box": r.box if r is not None else None,
                        "packet_id": r.packet_id if r is not None else None})
        return out

    def can(self) -> dict[str, bool]:
        st = self.stopped
        ph = self.phase
        new, reuse, problems = self.selection_request()
        return {"choose": st, "scan": st and self.setup_frame is not None and self.scan_status not in ("running", "waiting_worker"),
                "remember": st and not self.remember_pending and self.scan_id is not None and bool(new or reuse)
                and not problems,
                "reconcile": st and not self.remember_pending and self.enroll_state == "uncertain",
                "start": st and bool(self.tray) and self.source_info is not None and not self.start_pending,
                "save_config": st and bool(self.tray) and self.setup_status not in ("saving", "loading"),
                "pause": ph == PHASE_ACTIVE, "resume": ph == PHASE_PAUSED,
                "stop": ph in (PHASE_ACTIVE, PHASE_PAUSED, PHASE_STARTING) and not self.stop_pending,
                "manual": st and self.scan_status == "done", "edit_membership": st,
                "back": self.view_step in ("select", "configure") and st,
                "continue": self.step_allowed("select")[0] if self.view_step == "discover" else True}

    # ---- exit --------------------------------------------------------------------------------------------------------
    def request_exit(self) -> str:
        """``closing`` when finalization must settle first, ``ready`` when the window may close now."""
        if self.engine.session.busy:
            if not self.closing:
                self.closing = True
                self.engine.session.stop(ended_by="operator")
            return "closing"
        return "ready"

    def shutdown(self) -> dict:
        rec = self.engine.shutdown(wait_s=30.0)
        self.tasks.shutdown(2.0)
        self.closed = True
        return rec
