# SPDX-License-Identifier: AGPL-3.0-only
"""Saved runs and their decisions, read back from the canonical decision trace (never recomputed).

Patterns adapted from the frozen PatronSight reference (app/session_review.py / session_review_state.py: saved review
reads saved facts, opening a run starts no inference, partial/unavailable evidence is shown as such). Item adaptation:
a run's decisions are the committed ``result`` records of its ``trace.jsonl``. A light per-run index (one row per
committed result: byte offset, identity, media time, per-item state/box) is built once in the background and cached
beside the trace as ``decisions.index.jsonl`` with the trace's SHA-256; the full record is read by seeking to its offset.
A trace that is partial, missing or changed is reported, never silently reconstructed.
"""
from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ..engine.store import Store, loads

INDEX_NAME = "decisions.index.jsonl"
INDEX_VERSION = 1


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class DecisionIndex:
    run_id: str
    trace_path: Path
    trace_sha256: str | None
    rows: list[dict] = field(default_factory=list)       # one per committed result, commit order
    other_counts: dict = field(default_factory=dict)      # frames, discards, ineligible, metrics, ...
    status: str = "complete"                              # complete | partial | missing | unreadable
    problems: list[str] = field(default_factory=list)
    sessions: list[dict] = field(default_factory=list)

    def by_seq(self, seq: int) -> dict | None:
        lo, hi = 0, len(self.rows) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.rows[mid]["seq"]
            if s == seq:
                return self.rows[mid]
            if s < seq:
                lo = mid + 1
            else:
                hi = mid - 1
        return None


def _row(rec: dict, offset: int) -> dict:
    return {"off": offset, "seq": rec.get("seq"), "packet_id": rec.get("packet_id"), "frame": rec.get("frame"),
            "t_mono": rec.get("frame_t_mono"), "t_wall": rec.get("frame_t_wall"),
            "media_pts": (rec.get("media") or {}).get("media_pts"), "media_n": (rec.get("media") or {}).get("media_n"),
            "items": {o["item_id"]: {"state": o["state"], "box": o.get("box"), "reason": o.get("reason")}
                      for o in rec.get("items", [])}}


def build_index(run_id: str, run_dir: Path, cancel: threading.Event | None = None) -> DecisionIndex:
    trace = Path(run_dir) / "trace.jsonl"
    if not trace.is_file():
        return DecisionIndex(run_id, trace, None, status="missing", problems=["no decision trace was saved for this run"])
    sha = _sha(trace)
    cached = Path(run_dir) / INDEX_NAME
    if cached.is_file():
        try:
            with open(cached, encoding="utf-8") as fh:
                head = json.loads(fh.readline())
                if head.get("trace_sha256") == sha and head.get("version") == INDEX_VERSION:
                    rows = [json.loads(line) for line in fh]
                    return DecisionIndex(run_id, trace, sha, rows, head.get("other_counts", {}), head.get("status", "complete"),
                                         head.get("problems", []), head.get("sessions", []))
        except (OSError, ValueError):
            pass
    rows, counts, problems, sessions = [], {}, [], []
    status = "complete"
    last_n = 0
    with open(trace, "rb") as fh:
        off = 0
        for raw in fh:
            if cancel is not None and cancel.is_set():
                return DecisionIndex(run_id, trace, sha, status="cancelled")
            try:
                rec = json.loads(raw)
            except ValueError:
                status = "partial"
                problems.append(f"unreadable trace line at byte {off}")
                off += len(raw)
                continue
            n = rec.get("n")
            if isinstance(n, int):
                if n != last_n + 1:
                    status = "partial"
                    problems.append(f"trace sequence gap {last_n} -> {n} (records dropped or lost)")
                last_n = n
            k = rec.get("k")
            if k == "result":
                rows.append(_row(rec, off))
            else:
                counts[k] = counts.get(k, 0) + 1
                if k == "session":
                    sessions.append({x: rec.get(x) for x in ("event", "t", "w", "error", "ended_by")})
            off += len(raw)
    rows.sort(key=lambda r: r["seq"])
    idx = DecisionIndex(run_id, trace, sha, rows, counts, status, problems[:50], sessions)
    try:
        tmp = cached.with_name(cached.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"version": INDEX_VERSION, "trace_sha256": sha, "other_counts": counts, "status": status,
                                 "problems": problems[:50], "sessions": sessions,
                                 "note": "projection of trace.jsonl; never an independent record"}) + "\n")
            for r in rows:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
        tmp.replace(cached)
    except OSError:
        pass
    return idx


def read_decision(idx: DecisionIndex, row: dict) -> dict:
    """The exact canonical record of one committed decision (seek by offset; verified by sequence and packet id)."""
    with open(idx.trace_path, "rb") as fh:
        fh.seek(row["off"])
        rec = json.loads(fh.readline())
    if rec.get("seq") != row["seq"] or rec.get("packet_id") != row["packet_id"]:
        raise ValueError("the decision trace changed since its index was built")
    return rec


def list_runs(store: Store) -> list[dict]:
    out = []
    for r in store.query("SELECT * FROM runs ORDER BY created_wall DESC"):
        items = [dict(x) for x in store.query(
            "SELECT ri.item_id, ri.ref_version, i.name FROM run_items ri LEFT JOIN items i ON i.item_id=ri.item_id "
            "WHERE ri.run_id=?", (r["run_id"],))]
        out.append({"run_id": r["run_id"], "phase": r["phase"], "created_wall": r["created_wall"],
                    "started_wall": r["started_wall"], "ended_wall": r["ended_wall"], "outcome": loads(r["outcome"]),
                    "source": loads(r["source"]), "options": loads(r["options"]), "dir": r["dir"], "items": items,
                    "policy_version": r["policy_version"]})
    return out


def run_manifest(run: dict) -> dict | None:
    p = Path(run["dir"]) / "manifest.json"
    try:
        return json.loads(p.read_text("utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


def item_intervals(store: Store, run_id: str, item_id: str) -> list[dict]:
    return [dict(x) for x in store.query(
        "SELECT state,start_mono,end_mono,start_wall,end_wall,samples,reason,box FROM run_intervals "
        "WHERE run_id=? AND item_id=? ORDER BY start_mono", (run_id, item_id))]
