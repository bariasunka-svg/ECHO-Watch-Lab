# SPDX-License-Identifier: AGPL-3.0-only
# ADOPTED from the ECHO Watch Native foundation archive (tree 3d14d650, src/echo_watch/item_policy.py, SHA-256
# 5700dc29cec75c506af3d1d2ade277bf1f6757117ccb85e61c0a8c5c5d1db0a9), itself adopted from the preserved ECHO Watch
# reference runtime/echowatch/item_memory.py (SHA-256 f09a5288d827f2a1735ebdbdb11725343e72fd459ba596a5de636193b5cf73d4;
# AGPL-3.0-only). Native adaptations retained: `from .matching import appearance`; rate-independent evidence windows.
#
# EW-LAB-01 changes (item-memory-v3, "admitted evidence"):
# (1) RANKING IS NOT ADMISSION. The former single crop-compatibility floor (app_min 0.35) let a weak, repeated proposal
#     (the tablet at appearance 0.398) accumulate confirmation and own a remembered phone. Every candidate/item pair is
#     now classified by a frozen, independently calibrated absolute gate (matching.AdmissionGate): ``support`` (strong),
#     ``unresolved`` (between thresholds) or ``reject``. Ranking/score only orders candidates that already passed.
# (2) The gate applies to BOTH acquisition and continued ownership: only ``support`` frames enter a confirmation; a
#     ``reject`` ends an owned chain; an ``unresolved`` frame may continue an owned chain only within
#     ``hold_unresolved_s`` of its last strong frame (continuity is not identity), otherwise the chain ends.
# (3) Weak (unresolved) evidence is never charged as absence: it keeps the item Match uncertain, not Not visible.
# (4) Operator seed selection still only picks among STRONG candidates; it never bypasses quality.
# (5) ``clone()`` gives an isolated copy of all mutable state for the session's fork/commit evaluation (defect A).
# (6) a below-conf_min proposal that the gate does not reject for an item blocks Not visible for it (never identity);
# (7) two strong look-alikes at once record a run-scoped ambiguity: when one disappears, the other is not given the
#     identity (resolved only by observed, excluded look-alikes); track ids are unique per incarnation;
# (8) every evaluated candidate (compatibility, admission, ranking, track, ownership, withholding) is kept per step
#     in ``last_candidates`` for the decision trace and the Review investigation view.
# The original docstring follows.
"""Item Memory basic-visibility association policy (pure, deterministic).

Inputs per step: the frame pixels, capture monotonic time, the shared visual-prompt detections (one class per approved
reference), and per-item assessability decided by the caller. Outputs per remembered item a decision about APPEARANCE
evidence -- never physical serial identity:

  visible        a confirmed, soundly owned temporary track matched this item in this frame with admitted evidence
  checking       an admitted candidate is accumulating confirmation supports (tentative; never Visible)
  uncertain      compatible or weak evidence exists but identity cannot be assigned (look-alike, overlap,
                 suspect chain, margin, persisted distractor ambiguity, weak appearance, bounds)
  not_seen       an eligible check found no compatible evidence (NOT proof of physical absence)
  cannot_assess  the relevant view could not be assessed (decided by the caller) or a bound overflowed

Behaviour adapted (no code imported) from the frozen Patsight reference PATSIGHT-20261020-PS27D-C3,
tracking_v2/identity/graph.py: one-to-one assignment, co-visibility cannot-link edges, withholding identity on suspect
ownership. Attribution: Bari Asunka, AGPL-3.0-only. Detector and crop agreement are correlated evidence, never counted
as two independent proofs. Track ids are opaque incarnation labels: they never decide identity on their own.
"""
from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field

import numpy as np

from .geometry import clamp_box, crop, iou
from .matching import REJECT, SUPPORT, UNRESOLVED, AdmissionGate, appearance, default_gate
from .vision import Features, features, similarity

POLICY_VERSION = "item-memory-v3-admitted"


@dataclass(frozen=True)
class MemoryParams:
    conf_min: float = 0.25
    margin: float = 0.12           # per-item AND per-observation runner-up margin (ranking among ADMITTED candidates)
    confirm_n: int = 3             # consistent ADMITTED supports for initial/recovered identity
    confirm_span_s: float = 1.0
    hold_unresolved_s: float = 1.0  # an owned chain may continue through unresolved frames only this long after strong
    track_ttl_s: float = 2.0
    v_max_px_s: float = 240.0      # at 1920 px width; scaled by source width / 1920
    slack_px: float = 60.0
    cont_min: float = 0.35         # crop continuity between successive observations of one track
    dup_iou: float = 0.8
    dup_contain: float = 0.9
    overlap_iou: float = 0.3       # partially overlapping non-duplicate instances: identity withheld
    covis_iou: float = 0.1
    covis_contain: float = 0.2
    covis_n: int = 3
    covis_span_s: float = 1.0
    lookalike_app: float = 0.80    # remembered-reference pairs this similar are look-alike rivals
    seed_iou: float = 0.5          # operator enrollment attribution (initial setup only)
    max_obs: int = 64
    max_tracks: int = 128
    max_edges: int = 4096
    max_alts: int = 8
    support_window_s: float = 2.0
    max_supports: int = 64
    version: str = POLICY_VERSION


@dataclass
class WatchedItem:
    item_id: str
    classes: tuple[int, ...]
    ref: Features
    ref_size: tuple[float, float]
    region: tuple[float, float, float, float]
    seed_box: tuple[float, float, float, float] | None = None   # sound operator selection at enrollment
    ambiguity: dict | None = None                               # persisted reference-basis distractor record
    extra_refs: list = field(default_factory=list)               # (Features, (w, h)) of the same item only


@dataclass
class Candidate:
    idx: int
    box: list[float]
    conf: float
    support: dict[int, float]
    members: list[int]
    feats: Features | None = None
    withheld: str | None = None
    app: dict[str, float] = field(default_factory=dict)
    adm: dict[str, str] = field(default_factory=dict)       # admission class per item: support | unresolved | reject
    score: dict[str, float] = field(default_factory=dict)   # RANKING only (never admission)
    track: "TTrack | None" = None
    best: int | None = None


@dataclass
class TTrack:
    tid: str
    box: list[float]
    t_last: float
    born: float
    feats: Features | None
    owner: str | None = None
    sound: bool = False
    suspect: bool = False
    pending: tuple[str, list[float]] | None = None
    cannot: set[str] = field(default_factory=set)
    observed_now: bool = False
    last_strong: float | None = None


@dataclass
class Decision:
    item_id: str
    status: str
    box: list[float] | None = None
    track_id: str | None = None
    score: float | None = None
    app: float | None = None
    conf: float | None = None
    reason: str | None = None
    alternatives: list[dict] = field(default_factory=list)
    supports: list[float] = field(default_factory=list)
    confirmed_now: bool = False
    det: int | None = None
    admission: str | None = None

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        for k in ("score", "app", "conf"):
            if d[k] is not None:
                d[k] = round(float(d[k]), 4)
        if d["box"] is not None:
            d["box"] = [round(float(v), 1) for v in d["box"]]
        return d


def _centre(b) -> np.ndarray:
    return np.array([b[0] + b[2] / 2, b[1] + b[3] / 2], np.float64)


def _inside(b, region) -> bool:
    c = _centre(b)
    return region[0] <= c[0] <= region[0] + region[2] and region[1] <= c[1] <= region[1] + region[3]


def contain(a, b) -> float:
    """Intersection over the smaller area."""
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    m = min(a[2] * a[3], b[2] * b[3])
    return (ix * iy) / m if m > 0 else 0.0


class MemoryPolicy:
    """One camera epoch's temporary continuity plus the enduring item exclusions it may propagate.
    Create a new instance (or call reset) on any source/epoch/reference/model/settings/correction cutover."""

    def __init__(self, items: list[WatchedItem], params: MemoryParams | None = None, width: int = 1920,
                 gate: AdmissionGate | None = None):
        self.p = params or MemoryParams()
        self.gate = gate or default_gate()
        self.items = {it.item_id: it for it in items}
        self.scale = width / 1920.0
        self.tracks: dict[str, TTrack] = {}
        self.edges: dict[frozenset, dict] = {}       # track pair -> {"t": [...], "established": bool}
        self.edge_overflow = False
        self.track_overflow = False
        self._n = 0
        self._inc = uuid.uuid4().hex[:4]             # track ids are unique across evaluator incarnations (Resume, restart)
        self._last_t: float | None = None
        self.facts: list[dict] = []                  # new durable facts since the last drain
        self.last_candidates: list[dict] = []
        self.rivals = self._rivals()

    def clone(self) -> "MemoryPolicy":
        """Isolated copy of every mutable field; reference features, params, gate and rival relations are shared."""
        c = object.__new__(MemoryPolicy)
        c.p, c.gate, c.scale, c.rivals = self.p, self.gate, self.scale, self.rivals
        c.pair_similarity = self.pair_similarity
        c.items = {k: copy.copy(v) for k, v in self.items.items()}
        for v in c.items.values():
            v.ambiguity = None if v.ambiguity is None else dict(v.ambiguity)
            v.extra_refs = list(v.extra_refs)
        c.tracks = {}
        for k, t in self.tracks.items():
            n = copy.copy(t)
            n.box = list(t.box)
            n.cannot = set(t.cannot)
            n.pending = None if t.pending is None else (t.pending[0], list(t.pending[1]))
            c.tracks[k] = n
        c.edges = {k: {"t": list(e["t"]), "established": e["established"]} for k, e in self.edges.items()}
        c.edge_overflow, c.track_overflow, c._n, c._last_t = self.edge_overflow, self.track_overflow, self._n, self._last_t
        c._inc = self._inc
        c.facts = []
        c.last_candidates = []
        c._low_rows = []
        return c

    # ---- static item relations --------------------------------------------------------------------
    def _rivals(self) -> dict[str, set[str]]:
        """Remembered items whose approved references are mutually compatible (look-alike group)."""
        out = {i: set() for i in self.items}
        ids = sorted(self.items)
        self.pair_similarity = {}
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                A, B = self.items[ids[a]], self.items[ids[b]]
                s = min(similarity(A.ref, B.ref), similarity(B.ref, A.ref))
                self.pair_similarity[f"{ids[a]}|{ids[b]}"] = round(s, 4)
                if s >= self.p.lookalike_app:
                    out[ids[a]].add(ids[b])
                    out[ids[b]].add(ids[a])
        return out

    def reset(self) -> None:
        self.tracks.clear()
        self.edges.clear()
        self.edge_overflow = False

    def drain_facts(self) -> list[dict]:
        f, self.facts = self.facts, []
        return f

    def owned(self, item_id: str) -> TTrack | None:
        return next((t for t in self.tracks.values() if t.owner == item_id and t.sound), None)

    def attribute(self, track_id: str, item_id: str) -> bool:
        """Explicit operator attribution of a CURRENT track (new evidence, not silent learning)."""
        tr = self.tracks.get(track_id)
        if tr is None or item_id not in self.items:
            return False
        for t in self.tracks.values():
            if t.owner == item_id:
                t.owner, t.sound, t.pending = None, False, None
        tr.owner, tr.sound, tr.suspect, tr.pending = item_id, True, False, None
        tr.cannot.discard(item_id)
        self.items[item_id].ambiguity = None
        self._propagate(tr)
        self.facts.append({"kind": "operator_attribution", "item_id": item_id, "track_id": track_id})
        return True

    def adopt(self, item_id: str, box, feats, t_last: float) -> str:
        """Apply an operator's attribution, carried across a correction cutover, to the corresponding track in
        THIS revision: a soundly owned track at the attributed object's last observed place/time. Continuity
        must then hold (unique, within TTL) for it to stay current; nothing else is exempted."""
        self._n += 1
        tr = TTrack(f"tk{self._inc}.{self._n}", list(box), t_last, t_last, feats, owner=item_id, sound=True)
        for t in self.tracks.values():
            if t.owner == item_id:
                t.owner, t.sound = None, False
        self.tracks[tr.tid] = tr
        if item_id in self.items:   # the persisted look-alike ambiguity stays: if this chain breaks, identity
            self.items[item_id].seed_box = None   # is again uncertain until new evidence
        self.facts.append({"kind": "operator_attribution", "item_id": item_id, "track_id": tr.tid, "carried": True})
        return tr.tid

    # ---- step ------------------------------------------------------------------------------------
    def step(self, t: float, frame: np.ndarray, detections: list[dict],
             assessable: dict[str, tuple[bool, str | None]] | None = None,
             blocked: dict[str, str] | None = None, camera_moved: bool = False) -> dict[str, Decision]:
        """``assessable``: per item, may a missing compatible observation count as Not seen (NEGATIVE evidence)?
        It never vetoes a sound direct positive. ``blocked``: items whose direct positive is impossible this step.
        ``camera_moved``: a confirmed camera move ends every track not supported by a fresh observation."""
        p = self.p
        assessable = assessable or {}
        blocked = blocked or {}
        h, w = frame.shape[:2]
        for k in [k for k, tr in self.tracks.items() if t - tr.t_last > p.track_ttl_s]:
            self._end(k, "expired")
        for tr in self.tracks.values():
            tr.observed_now = False
        cands, overflow = self._candidates(frame, detections, w, h)
        if overflow:
            self.last_candidates = []
            return {i: Decision(i, "cannot_assess", reason=overflow) for i in self.items}
        self.track_overflow = False
        self._continuity(t, frame, cands)
        if camera_moved:
            for k in [k for k, tr in self.tracks.items() if not tr.observed_now]:
                self._end(k, "camera moved; unsupported continuity reset")
        if self.edge_overflow or self.track_overflow:
            why = ("exclusion bound exceeded; identity cannot be assessed safely" if self.edge_overflow else
                   "temporary track bound exceeded; identity cannot be assessed safely")
            self.last_candidates = self._candidate_rows(cands)
            return {i: Decision(i, "cannot_assess", reason=why) for i in self.items}
        refused = {i: why for i, why in blocked.items() if i in self.items}
        for tr in self.tracks.values():
            if tr.owner in refused:
                if tr.sound:
                    self.facts.append({"kind": "chain_ended", "item_id": tr.owner, "track_id": tr.tid,
                                       "why": "view not assessable"})
                tr.owner, tr.sound = None, False
            if tr.pending and tr.pending[0] in refused:
                tr.pending = None
        self._covisibility(t, cands)
        out = self._assign(t, cands, {i: v for i, v in assessable.items() if i not in refused}, skip=set(refused))
        for i, why in refused.items():
            out[i] = Decision(i, "cannot_assess", reason=why or "view could not be assessed")
        self.last_candidates = self._candidate_rows(cands) + list(getattr(self, "_low_rows", []))
        return out

    def _candidate_rows(self, cands: list[Candidate]) -> list[dict]:
        rows = []
        for c in cands:
            rows.append({"idx": c.idx, "det": c.best, "box": [round(v, 1) for v in c.box], "conf": round(c.conf, 4),
                         "withheld": c.withheld, "track": c.track.tid if c.track else None,
                         "track_owner": c.track.owner if c.track and c.track.sound else None,
                         "track_pending": c.track.pending[0] if c.track and c.track.pending else None,
                         "cannot": sorted(c.track.cannot) if c.track else [],
                         "app": {k: round(v, 4) for k, v in c.app.items()}, "admission": dict(c.adm),
                         "rank_score": {k: round(v, 4) for k, v in c.score.items()}})
        return rows

    def _candidates(self, frame, detections, w, h) -> tuple[list[Candidate], str | None]:
        p = self.p
        regions = [it.region for it in self.items.values()]
        dets = []
        low = []
        for k, d in enumerate(detections):
            box = list(clamp_box(d["box"], w, h))
            if box[2] < 4 or box[3] < 4 or not any(_inside(box, r) for r in regions):
                continue
            if d["confidence"] < p.conf_min:
                low.append((k, box, float(d["confidence"])))
                continue
            dets.append((k, box, float(d["confidence"]), int(d["cls"])))
        self._low_rows = self._low_confidence(frame, low[:p.max_obs])
        if len(dets) > p.max_obs:
            return [], f"more than {p.max_obs} observations in one frame; bound exceeded"
        parent = list(range(len(dets)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a in range(len(dets)):
            for b in range(a + 1, len(dets)):
                if iou(dets[a][1], dets[b][1]) >= p.dup_iou or contain(dets[a][1], dets[b][1]) >= p.dup_contain:
                    parent[find(b)] = find(a)
        groups: dict[int, list] = {}
        for i, d in enumerate(dets):
            groups.setdefault(find(i), []).append(d)
        cands = []
        for n, g in enumerate(sorted(groups.values(), key=lambda g: -max(x[2] for x in g))):
            best = max(g, key=lambda x: x[2])
            sup: dict[int, float] = {}
            for _, _, c, cls in g:
                sup[cls] = max(sup.get(cls, 0.0), c)
            cands.append(Candidate(n, best[1], best[2], sup, [x[0] for x in g], best=best[0]))
        for a in range(len(cands)):
            for b in range(a + 1, len(cands)):
                A, B = cands[a].box, cands[b].box
                if iou(A, B) >= p.overlap_iou:
                    why = "overlapping instances cannot be separated"
                    cands[a].withheld = cands[a].withheld or why
                    cands[b].withheld = cands[b].withheld or why
        for c in cands:
            try:
                c.feats = features(crop(frame, c.box))
            except ValueError:
                c.feats = None
            for i, it in self.items.items():
                if not _inside(c.box, it.region):
                    continue
                try:
                    a = max(appearance(frame, c.box, f, sz) for f, sz in [(it.ref, it.ref_size)] + it.extra_refs)
                except ValueError:
                    continue
                c.app[i] = a
                c.adm[i] = self.gate.classify(a)
                s = max((c.support.get(k, 0.0) for k in it.classes), default=0.0)
                # ranking only: the prompt model's class label never decides between look-alike references
                c.score[i] = a if self.rivals[i] else 0.5 * s + 0.5 * a
        return cands, None

    def _low_confidence(self, frame, low) -> list[dict]:
        """EW-LAB-01: proposals below ``conf_min`` never become candidates, tracks or supports, but one that the frozen
        gate does not REJECT for an item is evidence that something like the item may be there: it blocks charging
        that item as Not visible (a failed detection is not absence)."""
        rows = []
        for k, box, conf in low:
            app, adm = {}, {}
            for i, it in self.items.items():
                if not _inside(box, it.region):
                    continue
                try:
                    a = max(appearance(frame, box, f, sz) for f, sz in [(it.ref, it.ref_size)] + it.extra_refs)
                except ValueError:
                    continue
                app[i], adm[i] = round(a, 4), self.gate.classify(a)
            rows.append({"det": k, "box": [round(v, 1) for v in box], "conf": round(conf, 4), "low_confidence": True,
                         "app": app, "admission": adm})
        return rows

    def _new_track(self, t: float, c: Candidate, suspect: bool) -> TTrack:
        self._n += 1
        tr = TTrack(f"tk{self._inc}.{self._n}", list(c.box), t, t, c.feats, suspect=suspect, observed_now=True)
        self.tracks[tr.tid] = tr
        c.track = tr
        return tr

    def _end(self, tid: str, why: str) -> None:
        tr = self.tracks.pop(tid, None)
        if tr is None:
            return
        if tr.owner and tr.sound:
            self.facts.append({"kind": "chain_ended", "item_id": tr.owner, "track_id": tid, "why": why})
        for k in [k for k in self.edges if tid in k]:
            del self.edges[k]

    def _continuity(self, t: float, frame, cands: list[Candidate]) -> None:
        """Unique bidirectional continuity (unchanged from the adopted policy)."""
        p = self.p
        prev_t = self._last_t
        fresh = {k for k, tr in self.tracks.items() if prev_t is not None and tr.t_last >= prev_t - 1e-6}
        byidx = {c.idx: c for c in cands}
        broken: set[str] = set()
        suspect_c: set[int] = set()
        involved: set[int] = set()
        for tier in (fresh, set(self.tracks) - fresh):
            gate_t: dict[str, list[int]] = {k: [] for k in tier}
            gate_c: dict[int, list[str]] = {c.idx: [] for c in cands if c.track is None and c.idx not in involved}
            for k in tier:
                tr = self.tracks[k]
                dt = max(0.0, t - tr.t_last)
                lim = (p.v_max_px_s * dt + p.slack_px) * self.scale
                for ci in gate_c:
                    c = byidx[ci]
                    if float(np.linalg.norm(_centre(c.box) - _centre(tr.box))) > lim:
                        continue
                    if tr.feats is None or c.feats is None or similarity(c.feats, tr.feats) < p.cont_min:
                        continue
                    gate_t[k].append(ci)
                    gate_c[ci].append(k)
            for k, cs in gate_t.items():
                if len(cs) == 1 and len(gate_c[cs[0]]) == 1:
                    c = byidx[cs[0]]
                    tr = self.tracks[k]
                    tr.box, tr.t_last, tr.feats, tr.observed_now = list(c.box), t, c.feats, True
                    c.track = tr
                elif len(cs) >= 2:      # split / competing successors
                    broken.add(k)
                    suspect_c.update(cs)
            for ci, ks in gate_c.items():
                if len(ks) >= 2:        # crossing / merge
                    broken.update(ks)
                    suspect_c.add(ci)
            involved |= suspect_c
        self._last_t = t
        for k in broken:
            self._end(k, "split, crossing or competing successor")
        for c in cands:
            if c.track is not None and c.track.tid in broken:
                c.track = None
            if c.track is None:
                if len(self.tracks) >= p.max_tracks:
                    self.track_overflow = True  # never evict a track (and its constraints) to force a match
                    continue
                self._new_track(t, c, c.idx in suspect_c)

    def _covisibility(self, t: float, cands: list[Candidate]) -> None:
        p = self.p
        live = [c for c in cands if c.track is not None]
        for a in range(len(live)):
            for b in range(a + 1, len(live)):
                A, B = live[a], live[b]
                if iou(A.box, B.box) > p.covis_iou or contain(A.box, B.box) > p.covis_contain:
                    continue  # overlapping duplicates/reflections are never invented as distinct edges
                key = frozenset((A.track.tid, B.track.tid))
                e = self.edges.get(key)
                if e is None:
                    if len(self.edges) >= p.max_edges:
                        self.edge_overflow = True
                        return
                    e = self.edges[key] = {"t": [], "established": False}
                e["t"].append(t)
                e["t"] = [x for x in e["t"] if t - x <= p.support_window_s][-p.max_supports:]
                if not e["established"] and len(e["t"]) >= p.covis_n and e["t"][-1] - e["t"][0] >= p.covis_span_s:
                    e["established"] = True
                    for x, y in ((A.track, B.track), (B.track, A.track)):
                        if x.owner and x.sound:
                            y.cannot.add(x.owner)
                    self.facts.append({"kind": "covisible_distinct", "tracks": sorted(key)})
                # a sound co-visible distractor that STRONGLY matches a remembered target: reference-basis ambiguity
                for x, y, yc in ((A.track, B.track, B), (B.track, A.track, A)):
                    if x.owner and x.sound and y.owner is None and not y.pending:
                        it = self.items[x.owner]
                        if yc.adm.get(x.owner) == SUPPORT and (
                                it.ambiguity is None or it.ambiguity.get("track_id") != y.tid):
                            it.ambiguity = {"track_id": y.tid, "app": round(yc.app[x.owner], 4),
                                            "reason": "a similar unremembered object was seen together with this item"}
                            self.facts.append({"kind": "distractor_ambiguity", "item_id": x.owner,
                                               "track_id": y.tid, "app": it.ambiguity["app"]})

    def _propagate(self, tr: TTrack) -> None:
        for key, e in self.edges.items():
            if tr.tid in key and e["established"]:
                other = next(iter(key - {tr.tid}))
                if other in self.tracks:
                    self.tracks[other].cannot.add(tr.owner)

    def _alts(self, i: str, cands: list[Candidate], skip: Candidate | None = None) -> list[dict]:
        out = []
        for c in sorted(cands, key=lambda c: -c.score.get(i, -1)):
            adm = c.adm.get(i)
            if c is skip or adm in (None, REJECT):
                continue
            why = ("withheld: " + c.withheld if c.withheld else
                   "already this item's own track" if c.track and c.track.owner == i else
                   "explained by another item" if c.track and c.track.owner and c.track.sound else
                   "excluded: seen together with this item" if c.track and i in c.track.cannot else
                   "weak: appearance between the qualified thresholds" if adm == UNRESOLVED else "compatible")
            out.append({"track_id": c.track.tid if c.track else None, "box": [round(v, 1) for v in c.box],
                        "score": round(c.score.get(i, 0.0), 4), "app": round(c.app.get(i, 0.0), 4),
                        "admission": adm, "why": why})
            if len(out) >= self.p.max_alts:
                break
        return out

    def _assign(self, t: float, cands: list[Candidate], assessable, skip: set | None = None) -> dict[str, Decision]:
        p = self.p
        skip = skip or set()
        out: dict[str, Decision] = {i: Decision(i, "cannot_assess") for i in skip}
        taken: set[int] = set()
        # 1. continuing soundly owned tracks keep their item only while admitted evidence still supports it
        for i in sorted(self.items):
            tr = self.owned(i)
            if tr is None or not tr.observed_now:
                continue
            c = next((c for c in cands if c.track is tr), None)
            if c is None:
                continue
            taken.add(c.idx)
            adm = c.adm.get(i)
            stale_strong = tr.last_strong is None or t - tr.last_strong > p.hold_unresolved_s
            ends = c.withheld or adm in (None, REJECT) or (adm == UNRESOLVED and stale_strong)
            if ends:
                why = (c.withheld or ("appearance no longer compatible" if adm in (None, REJECT) else
                                      "appearance evidence stayed below the qualified support level"))
                tr.owner, tr.sound = None, False
                self.facts.append({"kind": "chain_ended", "item_id": i, "track_id": tr.tid, "why": why})
                ok, vwhy = assessable.get(i, (True, None))
                if not c.withheld and not ok:
                    out[i] = Decision(i, "cannot_assess", reason=vwhy or "view could not be assessed",
                                      alternatives=self._alts(i, cands, c), admission=adm)
                    continue
                out[i] = Decision(i, "uncertain", c.box, tr.tid, c.score.get(i), c.app.get(i), c.conf,
                                  c.withheld or ("the tracked object no longer matches this item's reference"
                                                 if adm in (None, REJECT) else
                                                 "continuity alone cannot keep the identity: appearance evidence is weak"),
                                  self._alts(i, cands, c), admission=adm)
                continue
            if adm == SUPPORT:
                tr.last_strong = t
            out[i] = Decision(i, "visible", c.box, tr.tid, c.score.get(i), c.app.get(i), c.conf,
                              None if adm == SUPPORT else "held by continuity within 1 s of strong evidence",
                              supports=[t], det=c.best, admission=adm)
        # 2. acquisition / recovery for the remaining items: STRONG evidence only, best first, one-to-one
        wants: list[tuple[float, str, Candidate | None, str | None]] = []
        for i, it in self.items.items():
            if i in out:
                continue
            pool = [c for c in cands if c.idx not in taken and c.adm.get(i) == SUPPORT
                    and not (c.track and c.track.owner and c.track.owner != i and c.track.sound)
                    and not (c.track and i in c.track.cannot)]
            if not pool:
                wants.append((-1.0, i, None, None))
                continue
            pool.sort(key=lambda c: (-c.score[i], c.idx))
            best = pool[0]
            why = None
            if it.seed_box is not None:   # the operator's own selection picks ITS candidate among strong ones only
                on_seed = [c for c in pool if iou(c.box, it.seed_box) >= p.seed_iou]
                if on_seed:
                    best = max(on_seed, key=lambda c: (iou(c.box, it.seed_box), -c.idx))
            seeded = it.seed_box is not None and iou(best.box, it.seed_box) >= p.seed_iou
            if best.withheld:
                why = best.withheld
            elif len(pool) > 1 and max(c.score[i] for c in pool if c is not best) > best.score[i] - p.margin \
                    and not seeded:
                why = "two observations fit this item similarly"
                # EW-LAB-01: two similar objects at once leave the identity unresolved; remember that, so the item
                # is not later given to whichever look-alike remains when the other disappears
                top = max(x.score[i] for x in pool)
                tids = sorted(c.track.tid for c in pool if c.track is not None and top - c.score[i] < p.margin)
                prev = set((it.ambiguity or {}).get("tracks") or [])
                if tids and set(tids) - prev:
                    it.ambiguity = {"tracks": sorted(prev | set(tids)), "kind": "lookalike",
                                    "reason": "two similar objects were seen at once"}
                    self.facts.append({"kind": "lookalike_ambiguity", "item_id": i, "tracks": it.ambiguity["tracks"]})
            if why is None:
                rivals = [j for j, s in best.score.items() if j != i and best.adm.get(j) == SUPPORT
                          and s > best.score[i] - p.margin]
                if rivals and not (seeded and not any(self.items[j].seed_box is not None and
                                                       iou(best.box, self.items[j].seed_box) >= p.seed_iou
                                                       for j in rivals)):
                    why = "another remembered item matches this observation equally"
            if why is None and it.ambiguity is not None:
                # every recorded look-alike other than this candidate must be observed now AND excluded from this item
                tids = it.ambiguity.get("tracks") or [it.ambiguity.get("track_id")]
                others = [self.tracks.get(x) for x in tids if x is not None and not (best.track and best.track.tid == x)]
                distinguished = bool(others) and all(o is not None and o.observed_now and i in o.cannot for o in others)
                if not distinguished:
                    why = it.ambiguity.get("reason") + "; identity cannot be re-established automatically"
            if why is None and best.track is not None and best.track.suspect and len(pool) > 1:
                why = "a split or crossing broke continuity and another compatible object is present"
            wants.append((best.score[i], i, best, why))
        wants.sort(key=lambda x: (-x[0], x[1]))
        claimed: dict[int, str] = {}
        for score, i, c, why in wants:
            if c is None:
                continue
            if why is None and c.idx in claimed:
                why = "another remembered item is acquiring this observation"
            if why is not None:
                out[i] = Decision(i, "uncertain", c.box, c.track.tid if c.track else None, c.score.get(i),
                                  c.app.get(i), c.conf, why, self._alts(i, cands), admission=c.adm.get(i))
                if c.track and c.track.pending and c.track.pending[0] == i:
                    c.track.pending = None
                claimed.setdefault(c.idx, i)
                continue
            claimed[c.idx] = i
            tr = c.track
            if tr is None:  # track bound reached: never a forced match
                out[i] = Decision(i, "cannot_assess", reason="temporary track bound exceeded")
                continue
            for other in self.tracks.values():  # one pending acquisition per item
                if other is not tr and other.pending and other.pending[0] == i:
                    other.pending = None
            sup = tr.pending[1] if tr.pending and tr.pending[0] == i else []
            sup = [x for x in sup + [t] if t - x <= p.support_window_s][-p.max_supports:]
            tr.pending = (i, sup)
            if len(sup) >= p.confirm_n and sup[-1] - sup[0] >= p.confirm_span_s:
                tr.owner, tr.sound, tr.pending, tr.suspect = i, True, None, False
                tr.last_strong = t
                self._propagate(tr)
                self.items[i].seed_box = None   # enrollment attribution is used for initial setup only
                self.facts.append({"kind": "confirmed", "item_id": i, "track_id": tr.tid,
                                   "supports": [round(x, 3) for x in sup]})
                out[i] = Decision(i, "visible", c.box, tr.tid, c.score[i], c.app[i], c.conf, None,
                                  supports=sup, confirmed_now=True, det=c.best, admission=SUPPORT)
            else:
                out[i] = Decision(i, "checking", c.box, tr.tid, c.score[i], c.app[i], c.conf,
                                  f"confirming ({len(sup)}/{p.confirm_n} strong supports)", supports=sup,
                                  admission=SUPPORT)
        # 3. nothing assignable: weak or unassignable compatible evidence is uncertainty, never absence
        for i in self.items:
            if i in out:
                continue
            alts = self._alts(i, cands)
            blocked = [a for a in alts if a["why"] not in ("explained by another item",
                                                           "excluded: seen together with this item")]
            ok, why = assessable.get(i, (True, None))
            if blocked:
                weak_only = all(a["admission"] == UNRESOLVED for a in blocked)
                out[i] = Decision(i, "uncertain", reason=("only weak appearance evidence; it cannot confirm or rule out "
                                                          "this item" if weak_only else
                                                          "compatible evidence could not be assigned"),
                                  alternatives=alts)
            elif not ok:
                out[i] = Decision(i, "cannot_assess", reason=why or "view could not be assessed", alternatives=alts)
            elif any(r["admission"].get(i) in (SUPPORT, UNRESOLVED) for r in getattr(self, "_low_rows", ())):
                out[i] = Decision(i, "uncertain", reason="a low-confidence proposal resembling this item is present; "
                                  "Not visible cannot be established", alternatives=alts)
            else:
                out[i] = Decision(i, "not_seen", reason="no compatible observation in the search area"
                                  + ("; similar objects are accounted for" if alts else ""), alternatives=alts)
        return out


# ---- review timer / episodes (pure) ------------------------------------------------------------------
CHARGE_MAX_DT = 2.0
RECOVER_N = 3
RECOVER_SPAN_S = 1.0
SUPPORT_WINDOW_S = 2.0           # rate-independent recovery window (was: last 8 by count)
MAX_SUPPORTS = 64
BUCKET = {"not_seen": "charged_s", "uncertain": "uncertain_s", "checking": "uncertain_s",
          "cannot_assess": "unknown_s", "offline": "unknown_s", "paused": "paused_s", "visible": "visible_s"}


@dataclass
class Sample:
    t_mono: float
    t_wall: float
    epoch: str
    category: str      # visible | checking | uncertain | not_seen | cannot_assess | offline | paused
    frame_seq: int | None = None
    box: list[float] | None = None
    supports: list[float] = field(default_factory=list)
    reason: str | None = None


class ItemTimeline:
    """Per-item review episode coordinator. Time belongs to capture clocks: only ADJACENT samples of one
    epoch with delta <= 2 s are charged/allocated; nothing is extrapolated past the last observation.

    After first confirmed visibility, two consecutive eligible Not seen samples start a gap at the first
    sample. Inside an unrecovered gap only eligible Not seen time increments the review counter; Match
    uncertain / Cannot assess / offline / paused freeze it (never reset, never recover). At charged >=
    allowance exactly one review is requested. Three supported samples spanning >= 1 s recover the
    condition; review acknowledgement is separate and never recovers."""

    def __init__(self, item_id: str, allowance_s: float = 60.0, state: dict | None = None):
        self.item_id, self.allowance_s = item_id, float(allowance_s)
        st = state or {}
        self.confirmed_ever: bool = bool(st.get("confirmed_ever"))
        self.last_confirmed: dict | None = st.get("last_confirmed")
        self.gap: dict | None = st.get("gap")
        self.prev: Sample | None = None
        self.pending_ns: Sample | None = None
        self.returns: list[float] = []
        self.return_epoch: str | None = None
        self.needs_setup = not self.confirmed_ever

    def snapshot(self) -> dict:
        return {"confirmed_ever": self.confirmed_ever, "last_confirmed": self.last_confirmed, "gap": self.gap}

    def restored(self, downtime_note: str) -> None:
        """After a restart: counters kept, downtime unknown and never charged."""
        if self.gap is not None:
            self.gap["interruptions"] = self.gap.get("interruptions", 0) + 1
            self.gap.setdefault("notes", []).append(downtime_note)
        self.prev = None

    def basis_changed(self, why: str, wall: float) -> list[dict]:
        """Reference/search/model change: the open episode is segmented (its review preserved) and the
        new basis needs fresh setup before any time can be charged."""
        ev = []
        if self.gap is not None:
            g = self.gap
            g["ended_wall"], g["end_reason"] = wall, f"monitoring basis changed: {why}"
            ev.append({"kind": "episode_segmented", "gap": dict(g)})
            self.gap = None
        self.confirmed_ever, self.needs_setup = False, True
        self.prev, self.pending_ns, self.returns = None, None, []
        return ev

    def _alloc(self, s: Sample) -> None:
        pv = self.prev
        if self.gap is None or pv is None:
            return
        g = self.gap
        if pv.epoch != s.epoch:
            g["unobserved_wall_s"] = g.get("unobserved_wall_s", 0.0) + max(0.0, s.t_wall - pv.t_wall)
            g["interruptions"] = g.get("interruptions", 0) + 1
            return
        dt = s.t_mono - pv.t_mono
        if dt <= 0:
            return
        if dt > CHARGE_MAX_DT:
            g["unobserved_wall_s"] = g.get("unobserved_wall_s", 0.0) + dt
            g["interruptions"] = g.get("interruptions", 0) + 1
            return
        if pv.category == s.category or (BUCKET.get(pv.category) == BUCKET.get(s.category)):
            b = BUCKET.get(s.category, "unknown_s")
            g[b] = g.get(b, 0.0) + dt
        else:
            g["boundary_s"] = g.get("boundary_s", 0.0) + dt
            if s.category != "not_seen" and pv.category == "not_seen":
                g["interruptions"] = g.get("interruptions", 0) + 1

    def sample(self, s: Sample) -> list[dict]:
        ev: list[dict] = []
        cat = s.category
        if cat == "visible":
            if self.return_epoch != s.epoch:
                self.returns = []
                self.return_epoch = s.epoch
            r_all = sorted(set(self.returns + [x for x in (s.supports or [s.t_mono])]))
            self.returns = [x for x in r_all if r_all[-1] - x <= SUPPORT_WINDOW_S][-MAX_SUPPORTS:]
        elif cat != "checking":
            self.returns, self.return_epoch = [], None
        self._alloc(s)
        if cat == "visible":
            lc = {"t_wall": s.t_wall, "epoch": s.epoch, "frame_seq": s.frame_seq, "box": s.box}
            if not self.confirmed_ever:
                self.confirmed_ever, self.needs_setup = True, False
                ev.append({"kind": "first_confirmed", "sample": lc})
            self.last_confirmed = lc
            self.pending_ns = None
            if self.gap is not None:
                r = self.returns
                if len(r) >= RECOVER_N and r[-1] - r[0] >= RECOVER_SPAN_S:
                    g = self.gap
                    g["recovered_wall"], g["recovered_frame"] = s.t_wall, [s.epoch, s.frame_seq]
                    ev.append({"kind": "recovered", "gap": dict(g)})
                    self.gap = None
        elif cat == "not_seen" and self.confirmed_ever:
            if self.gap is None:
                pv = self.pending_ns
                if pv is not None and pv.epoch == s.epoch and 0 < s.t_mono - pv.t_mono <= CHARGE_MAX_DT:
                    self.gap = {"onset_wall": pv.t_wall, "onset_mono": pv.t_mono, "onset_epoch": pv.epoch,
                                "onset_frame": pv.frame_seq, "charged_s": s.t_mono - pv.t_mono,
                                "uncertain_s": 0.0, "unknown_s": 0.0, "paused_s": 0.0, "visible_s": 0.0,
                                "boundary_s": 0.0, "unobserved_wall_s": 0.0, "interruptions": 0,
                                "allowance_s": self.allowance_s, "review": None,
                                "last_confirmed": self.last_confirmed}
                    self.pending_ns = None
                    ev.append({"kind": "gap_started", "gap": dict(self.gap)})
                else:
                    self.pending_ns = s
            if self.gap is not None and self.gap.get("review") is None and \
                    self.gap["charged_s"] >= self.allowance_s:
                self.gap["review"] = {"requested_wall": s.t_wall, "charged_s": round(self.gap["charged_s"], 3),
                                      "frame": [s.epoch, s.frame_seq], "allowance_s": self.allowance_s,
                                      "interrupted": bool(self.gap.get("interruptions"))}
                ev.append({"kind": "review_requested", "gap": dict(self.gap)})
        else:
            if cat != "checking":
                self.pending_ns = None
        self.prev = s
        return ev
