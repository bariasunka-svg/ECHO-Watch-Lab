# SPDX-License-Identifier: AGPL-3.0-only
"""Per-basis item evaluator: one analysed frame + the real visual-prompt detections -> a decision per remembered item.

This is the pure computation of the preserved ECHO Watch reference's basic Item Memory watch (runtime/echowatch/
watch.py, SHA-256 bd655c787fa5292ec5880ed3460fa34938289ed457552d1e7fc70d7662e15815; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md),
re-expressed WITHOUT its database, mailbox, presenter, tracer and scheduling. Stages kept (same thresholds):
  1. direct-positive pixel quality of every raw candidate (``_candidate_quality``): a dark/overexposed/glared candidate
     is withheld from the policy and recorded - it can never create or keep a direct positive;
  2. whole-view usability (uniformly dark/overexposed view blocks every positive);
  3. camera / background geometry from FOREGROUND-EXCLUDED, spatially spread background patches (``_assess``): movement
     or unproven geometry freezes NEGATIVE evidence only; it never vetoes a sound direct positive;
  4. a negative-assessability gate per item (area brightness, glare, ring obstruction) deciding whether "no compatible
     observation" may count as Not visible;
  5. the operator's own enrollment selection ("seed") may resolve look-alike references ONLY during the initial setup of
     a fresh reference, while its continuity holds; it expires at the setup deadline, on an unusable view, or when nothing
     compatible overlaps it for longer than the track TTL (never re-armed on resume/restart);
  6. ``item_policy.MemoryPolicy.step`` (item-memory-v2).
Differences: the item search area is the WHOLE view (an item may be found anywhere - the home-only adapter is not used),
time is the frame's capture clock (never wall time at processing), and the evaluator is an object per control basis:
any control change discards it (and its temporary tracks) - a result evaluated by an old evaluator can never update the
current state because the controller commits only packets whose basis is the current one.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..frames import Frame
from ..geometry import clamp_box, crop, iou
from ..item_policy import Decision, MemoryParams, MemoryPolicy, WatchedItem
from ..matching import reference_appearance
from ..vision import (Features, anchor_image, background_geometry, features, ring_crops, ring_similarity)
from .contracts import ItemRef

CAND_DARK, CAND_BRIGHT, CAND_GLARE = 18.0, 242.0, 0.5
SETUP_DEADLINE_S = 15.0


class StaleFrame(RuntimeError):
    """The frame does not advance this evaluator's source time (duplicate / out of order / foreign epoch)."""


@dataclass
class Member:
    ref: ItemRef
    feats: Features
    ref_size: tuple[float, float]
    anchor: np.ndarray
    anchor_key: str
    ring_ref: list
    ring_box: list[float]
    ref_glare: float
    ambiguity: dict | None = None


def prepare_member(ref: ItemRef, ambiguity: dict | None = None) -> Member:
    crop_img = cv2.imread(ref.crop_path, cv2.IMREAD_COLOR)
    ctx = cv2.imread(ref.context_path, cv2.IMREAD_COLOR)
    if crop_img is None or ctx is None:
        raise FileNotFoundError(f"reference images for {ref.ref_key} could not be read")
    box = list(ref.box)
    anchor = anchor_image(ctx)
    g = cv2.cvtColor(crop(ctx, box), cv2.COLOR_BGR2GRAY)
    return Member(ref, reference_appearance(crop_img), (box[2], box[3]), anchor,
                  hashlib.sha256(anchor.tobytes() + repr(anchor.shape).encode()).hexdigest()[:24],
                  [features(x) if x is not None else None for x in ring_crops(ctx, box)], box,
                  float((g >= 250).mean()), ambiguity)


@dataclass
class Evaluation:
    decisions: dict[str, Decision]
    candidate_quality: list[dict]
    negative: dict[str, tuple[bool, str | None]]
    blocked: dict[str, str]
    geometry: dict
    whole_view_mean: float
    camera_moved: bool
    facts: list[dict]
    stage_ms: dict[str, float]
    events: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)   # every evaluated candidate: compatibility, admission, ranking


class ItemEvaluator:
    def __init__(self, members: dict[str, Member], size: tuple[int, int], class_map: dict[str, tuple[int, ...]],
                 epoch_id: str, seed_enabled: dict[str, bool] | None = None, params: MemoryParams | None = None):
        self.members = members
        self.size = size
        self.epoch_id = epoch_id
        self.params = params or MemoryParams()
        W, H = size
        seed_enabled = seed_enabled or {}
        items = [WatchedItem(i, tuple(class_map.get(i, ())), m.feats, m.ref_size, (0.0, 0.0, float(W), float(H)),
                             tuple(m.ref.box) if seed_enabled.get(i, False) else None, m.ambiguity, [])
                 for i, m in members.items()]
        self.policy = MemoryPolicy(items, self.params, W)
        self.ring = {i: (list(m.ring_box), m.ring_ref) for i, m in members.items()}
        self.geom_box: dict[str, list[float] | None] = {i: None for i in members}
        self.seed_since: dict[str, float | None] = {i: None for i in members}
        self.seed_seen: dict[str, float | None] = {i: None for i in members}
        self.last_t: float | None = None
        self.evaluations = 0
        self.version = 0                 # advances only when a committed fork is adopted

    # ---- speculative evaluation (EW-LAB-01 defect A) ---------------------------------------------------------
    def fork(self) -> "ItemEvaluator":
        """An isolated copy of every MUTABLE field (policy tracks/edges/supports/ambiguity/seed state, ring context,
        timestamps). Immutable inputs (members, reference features, image buffers, ring reference features) are shared."""
        f = object.__new__(ItemEvaluator)
        f.members, f.size, f.epoch_id, f.params = self.members, self.size, self.epoch_id, self.params
        f.policy = self.policy.clone()
        f.ring = dict(self.ring)                     # values are replaced, never mutated in place
        f.geom_box = {k: (None if v is None else list(v)) for k, v in self.geom_box.items()}
        f.seed_since, f.seed_seen = dict(self.seed_since), dict(self.seed_seen)
        f.last_t, f.evaluations, f.version = self.last_t, self.evaluations, self.version
        return f

    def adopt(self, f: "ItemEvaluator") -> None:
        """Make a successfully committed fork's state current (called only under the session's commit lock)."""
        self.policy = f.policy
        self.ring, self.geom_box = f.ring, f.geom_box
        self.seed_since, self.seed_seen = f.seed_since, f.seed_seen
        self.last_t, self.evaluations = f.last_t, f.evaluations
        self.version += 1

    # ---- stages -------------------------------------------------------------------------------------------
    @staticmethod
    def candidate_quality(img: np.ndarray, dets: list[dict]) -> tuple[list[dict], list[dict]]:
        h, w = img.shape[:2]
        keep, rows = [], []
        for k, d in enumerate(dets):
            box = list(clamp_box(d["box"], w, h))
            why, m, gl = None, None, None
            if box[2] >= 2 and box[3] >= 2:
                g = cv2.cvtColor(crop(img, box), cv2.COLOR_BGR2GRAY)
                m, gl = float(g.mean()), float((g >= 250).mean())
                why = ("candidate too dark" if m < CAND_DARK else "candidate overexposed" if m > CAND_BRIGHT else
                       "glare over the candidate" if gl > CAND_GLARE else None)
            rows.append({"idx": k, "box": [round(v, 1) for v in box], "mean": None if m is None else round(m, 1),
                         "glare": None if gl is None else round(gl, 3), "refused": why})
            if why is None:
                keep.append(d)
        return keep, rows

    def _assess(self, mem: Member, img: np.ndarray, geo_cache: dict, exclude: list) -> tuple[bool, str | None, dict]:
        h, w = img.shape[:2]
        key = mem.anchor_key
        if key not in geo_cache:
            geo_cache[key] = background_geometry(mem.anchor, img, exclude)
        geo = geo_cache[key]
        q: dict = {"geometry_key": key}
        if geo["state"] == "moved":
            return False, geo["why"], q
        if geo["state"] != "stable":
            return False, "Not-visible checks paused: " + (geo.get("why") or "camera position unproven"), q
        box, ring_ref = self.ring[mem.ref.item_id]
        box = list(clamp_box(box, w, h))
        g = cv2.cvtColor(crop(img, box), cv2.COLOR_BGR2GRAY)
        m = float(g.mean())
        gl = float((g >= 250).mean())
        rs = ring_similarity(ring_crops(img, box), ring_ref) if ring_ref else None
        q["area"] = {"box": [round(v, 1) for v in box], "mean": round(m, 1), "glare": round(gl, 3),
                     "ring": None if rs is None else round(rs, 4), "ref_glare": round(mem.ref_glare, 3)}
        if m < 18:
            return False, "the last known area is too dark to assess", q
        if m > 242:
            return False, "the last known area is overexposed", q
        if gl > max(0.25, mem.ref_glare + 0.20):
            return False, "glare over the last known area", q
        if rs is not None and rs < 0.25:
            return False, "the view around the last known position is obstructed", q
        return True, None, q

    # ---- one analysed frame ---------------------------------------------------------------------------------
    def evaluate(self, frame: Frame, dets: list[dict]) -> Evaluation:
        if frame.epoch.epoch_id != self.epoch_id:
            raise StaleFrame("frame belongs to another source epoch")
        if self.last_t is not None and frame.t_mono <= self.last_t:
            raise StaleFrame("frame does not advance source time (duplicate or out of order)")
        img = frame.image
        if (frame.width, frame.height) != self.size:
            raise StaleFrame("frame size differs from the references' frame size")
        sm: dict[str, float] = {}
        ts = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal ts
            now_ = time.perf_counter()
            sm[name] = round((now_ - ts) * 1000, 3)
            ts = now_
        pol = self.policy
        kept, cand_q = self.candidate_quality(img, dets)
        gm = float(cv2.cvtColor(cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).mean())
        blocked: dict[str, str] = {}
        if gm < CAND_DARK or gm > CAND_BRIGHT:
            blocked = {i: ("the whole view is too dark to assess" if gm < CAND_DARK else "the whole view is overexposed")
                       for i in pol.items}
        lap("quality")
        exclude = [list(d["box"]) for d in dets]
        for i, mem in self.members.items():
            exclude.append(list(mem.ref.box))
            exclude.append(list(self.ring[i][0]))
            if self.geom_box[i]:
                exclude.append(list(self.geom_box[i]))
        negative: dict[str, tuple[bool, str | None]] = {}
        geo_cache: dict = {}
        for i, mem in self.members.items():
            ok, why, _q = self._assess(mem, img, geo_cache, exclude)
            bad = [c for c in cand_q if c["refused"]]       # the search area is the whole view
            if ok and bad:
                ok, why = False, f"{bad[0]['refused']} in the search area; Not-visible checks paused"
            negative[i] = (ok, why)
        lap("assessability_geometry")
        moved = any(g.get("state") == "moved" for g in geo_cache.values())
        for i, it in pol.items.items():
            if it.seed_box is None:
                continue
            if self.seed_since[i] is None:
                self.seed_since[i] = self.seed_seen[i] = frame.t_mono
            if any(iou(list(d["box"]), list(it.seed_box)) >= 0.5 for d in kept):
                self.seed_seen[i] = frame.t_mono
            why = ("setup deadline passed" if frame.t_mono - self.seed_since[i] > SETUP_DEADLINE_S else
                   "view not assessable" if i in blocked or moved else
                   "selection not observed for longer than the track TTL"
                   if frame.t_mono - self.seed_seen[i] > self.params.track_ttl_s else None)
            if why:
                it.seed_box = None
        lap("seed")
        dec = pol.step(frame.t_mono, img, kept, negative, blocked=blocked, camera_moved=moved)
        lap("matching")
        for i, d in dec.items():
            if d.status == "visible" and d.box is not None:
                self.ring[i] = (list(d.box), [features(x) if x is not None else None for x in ring_crops(img, d.box)])
            self.geom_box[i] = list(d.box) if d.status in ("visible", "checking") and d.box is not None else None
        lap("context_features")
        self.last_t = frame.t_mono
        self.evaluations += 1
        return Evaluation(dec, cand_q, negative, blocked, {str(k): v for k, v in geo_cache.items()}, round(gm, 1),
                          moved, pol.drain_facts(), sm, [], pol.last_candidates)
