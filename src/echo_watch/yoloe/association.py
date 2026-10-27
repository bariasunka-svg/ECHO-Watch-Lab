# SPDX-License-Identifier: AGPL-3.0-only
"""Detection / association separation for the YOLOE workspace (NumPy only; used by the GPU worker AND re-checked by the
UI process).

Why this exists: in the installed Ultralytics 8.4.160 ``trackers/track.py`` the post-process callback REPLACES each
result with ``result[idx]`` of the tracked subset, so ``model.track``'s returned boxes hide every detection the tracker
did not (yet) output - first-frame/unconfirmed, low-score and unmatched detections alike. The workspace therefore keeps
ONE immutable list of every post-model detection per frame (stable index ``i`` = its row in the detector result) and
joins tracker output back to it ONLY through the tracker's own returned detection index, with bounds, integrality and
uniqueness checks. Nothing is matched by nearest box or list position; an entry that fails a check is reported as an
association error and attached to nothing.

``DetArray`` is the minimal Results-like view BoT-SORT's ``update`` needs (``conf``, ``cls``, ``xywh``, ``xyxy``,
boolean indexing, ``len``). It deliberately has no ``xywhr`` (axis-aligned boxes only).
"""
from __future__ import annotations

import math

import numpy as np

TRACKED, TENTATIVE, UNTRACKED = "tracked", "tentative", "untracked"


class DetArray:
    """Rows ``[x1, y1, x2, y2, conf, cls]`` (float32) of one frame's detector result."""

    def __init__(self, data: np.ndarray):
        a = np.asarray(data, dtype=np.float32)
        if a.size == 0:
            a = np.zeros((0, 6), np.float32)
        if a.ndim != 2 or a.shape[1] != 6:
            raise ValueError(f"detections must be N x 6, got {a.shape}")
        self.data = a

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx) -> "DetArray":
        return DetArray(self.data[idx])

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]

    @property
    def xywh(self) -> np.ndarray:
        b = self.data[:, :4]
        out = np.empty_like(b)
        out[:, 0] = (b[:, 0] + b[:, 2]) / 2
        out[:, 1] = (b[:, 1] + b[:, 3]) / 2
        out[:, 2] = b[:, 2] - b[:, 0]
        out[:, 3] = b[:, 3] - b[:, 1]
        return out


def detections_array(dets: list[dict]) -> np.ndarray:
    """Immutable detection records -> the N x 6 array handed to the tracker (same order = same indices)."""
    if not dets:
        return np.zeros((0, 6), np.float32)
    for k, d in enumerate(dets):
        if d["i"] != k:
            raise ValueError(f"detection {k} carries index {d['i']}: records must be in detector order")
    return np.asarray([[*d["xyxy"], d["score"], d["cls"]] for d in dets], np.float32)


def _index(v, n: int) -> tuple[int | None, str | None]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None, f"detection index {v!r} is not a number"
    if not math.isfinite(f) or abs(f - round(f)) > 1e-6:
        return None, f"detection index {v!r} is not an integer"
    i = int(round(f))
    if not 0 <= i < n:
        return None, f"detection index {i} is outside 0..{n - 1}" if n else f"detection index {i} with no detections"
    return i, None


def join(n_dets: int, tracked_rows, tentative: list | None = None) -> tuple[dict[int, dict], list[str]]:
    """Join tracker output to the frame's detections BY RETURNED INDEX.

    ``tracked_rows``: the tracker's output rows ``[x1, y1, x2, y2, track_id, score, cls, idx]`` (confirmed tracks
    updated this frame). ``tentative``: ``[(track_id, idx, [x1, y1, x2, y2], score, cls)]`` for tracks created this
    frame but not yet confirmed. Returns ({detection index: association}, errors). A row that fails any check - bad or
    duplicate index, bad or duplicate track id - is listed in ``errors`` and attached to NOTHING; when two rows claim
    the same detection, neither is attached."""
    errors: list[str] = []
    claims: dict[int, list[dict]] = {}
    seen_ids: dict[int, int] = {}
    rows = np.asarray(tracked_rows if tracked_rows is not None else [], dtype=np.float64)
    if rows.size and (rows.ndim != 2 or rows.shape[1] < 8):
        return {}, [f"tracker output has shape {rows.shape}, expected M x 8"]
    entries = [(TRACKED, r[4], r[7], r[:4], r[5], r[6]) for r in (rows if rows.size else [])]
    entries += [(TENTATIVE, t[0], t[1], t[2], t[3], t[4]) for t in (tentative or [])]
    for state, tid, idx, box, score, cls in entries:
        i, why = _index(idx, n_dets)
        try:
            tf = float(tid)
            t = int(round(tf)) if math.isfinite(tf) and abs(tf - round(tf)) < 1e-6 and tf > 0 else None
        except (TypeError, ValueError):
            t = None
        if t is None:
            errors.append(f"{state} entry has an invalid track id {tid!r}")
            continue
        if why:
            errors.append(f"{state} track #{t}: {why}")
            continue
        seen_ids[t] = seen_ids.get(t, 0) + 1
        claims.setdefault(i, []).append({"track_id": t, "state": state,
                                         "tracker_xyxy": [round(float(x), 2) for x in box],
                                         "tracker_score": round(float(score), 4), "tracker_cls": int(round(float(cls)))})
    dup_ids = {t for t, c in seen_ids.items() if c > 1}
    for t in sorted(dup_ids):
        errors.append(f"track #{t} is reported for more than one detection")
    out: dict[int, dict] = {}
    for i, cl in sorted(claims.items()):
        if len(cl) > 1:
            errors.append(f"detection {i} is claimed by tracks {[c['track_id'] for c in cl]}")
            continue
        if cl[0]["track_id"] in dup_ids:
            continue
        out[i] = cl[0]
    return out, errors


def summarize(n_dets: int, assoc: dict[int, dict]) -> dict:
    tracked = sum(1 for a in assoc.values() if a["state"] == TRACKED)
    return {"detections": n_dets, "associated": tracked, "tentative": len(assoc) - tracked,
            "untracked": n_dets - tracked}
