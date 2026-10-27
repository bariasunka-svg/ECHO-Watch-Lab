# SPDX-License-Identifier: AGPL-3.0-only
"""Paged, filtered item activity over SAVED facts, with original and effective attribution.

Pattern adapted (no code imported) from the frozen PatronSight reference src/patronsight/visitor_activity.py and
app/visitor_history_page.py: View all activity reads recorded events for the selected EFFECTIVE profile within explicit
filters, one bounded page at a time (indexed SQL, never a full-history scan per frame); loading / empty / partial /
error are distinct; original attribution stays visible beside the effective (corrected) one; live facts not yet saved
are marked by the caller, never mixed in silently.
"""
from __future__ import annotations

from ..engine.store import Store, loads
from .corrections import Corrections

EVENT_LABEL = {
    "monitoring_started": "Monitoring started (item added to this run)",
    "monitoring_ended": "Monitoring ended",
    "first_confirmed": "First confirmed live sighting",
    "gap_started": "Not visible (two consecutive eligible checks)",
    "recovered": "Visible again (three supported checks over at least 1 s)",
    "review_requested": "Review requested after the Not-visible allowance (not proof of removal)",
    "episode_segmented": "Episode segmented (monitoring basis changed)",
}


def page(store: Store, corrections: Corrections, item_id: str | None = None, run_id: str | None = None,
         kinds: list[str] | None = None, t_from: float | None = None, t_to: float | None = None,
         offset: int = 0, limit: int = 50) -> dict:
    limit = max(1, min(200, int(limit)))
    g = corrections.fold()
    members = g.members(item_id) if item_id else None
    where, args = [], []
    if members:
        where.append(f"e.item_id IN ({','.join('?' * len(members))})")
        args += members
    if run_id:
        where.append("e.run_id=?")
        args.append(run_id)
    if kinds:
        where.append(f"e.kind IN ({','.join('?' * len(kinds))})")
        args += kinds
    if t_from is not None:
        where.append("e.wall>=?")
        args.append(t_from)
    if t_to is not None:
        where.append("e.wall<?")
        args.append(t_to)
    w = (" WHERE " + " AND ".join(where)) if where else ""
    total = store.one(f"SELECT COUNT(*) AS n FROM run_events e{w}", tuple(args))["n"]
    rows = store.query(f"SELECT e.event_id, e.run_id, e.item_id, e.wall, e.kind, e.payload, i.name FROM run_events e "
                       f"LEFT JOIN items i ON i.item_id=e.item_id{w} ORDER BY e.wall DESC, e.event_id DESC LIMIT ? OFFSET ?",
                       tuple(args) + (limit, int(offset)))
    out = []
    names = {r["item_id"]: r["name"] for r in store.query("SELECT item_id, name FROM items")}
    for r in rows:
        p = loads(r["payload"]) or {}
        eff = g.effective(r["item_id"])
        out.append({"event_id": r["event_id"], "run_id": r["run_id"], "wall": r["wall"], "kind": r["kind"],
                    "label": EVENT_LABEL.get(r["kind"], p.get("label") or r["kind"]),
                    "original_item": r["item_id"], "original_name": r["name"], "effective_item": eff,
                    "effective_name": names.get(eff, eff), "corrected": eff != r["item_id"],
                    "frame": p.get("frame"), "packet_id": p.get("packet_id"), "detail": p.get("gap") or p.get("outcome")})
    return {"total": total, "offset": int(offset), "limit": limit, "rows": out, "members": members or [],
            "status": "empty" if total == 0 else "ok",
            "note": "saved facts only; a run's live events are saved when it finalizes"}
