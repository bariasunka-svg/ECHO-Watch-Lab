# SPDX-License-Identifier: AGPL-3.0-only
"""One paired presentation for the monitor image, its boxes, the tray rows and the selected-item fields
(Tk-free; EW-LAB-03 R3).

Defect (Lab-02 review R3): after a control cutover the canvas correctly kept the previous-basis image without live
boxes, but the tray fell back to the newer session snapshot and showed a live Visible row from a packet whose pixels
were not on screen; tray/inspector also refreshed on a slower timer than the image.

Rules:
* Observation states (Visible / Checking / Not visible / Match uncertain / Cannot assess) come ONLY from the painted
  packet, and only when that packet belongs to the current run and control basis. Live authority additionally needs
  the active phase and an image at most ``stale_s`` old (the same rule as the canvas boxes).
* Monitoring with no eligible painted image (first Start, after Resume, new run): every row says it is awaiting the
  current image - no live state, no box. Lifecycle phases (Preparing, Paused, Finalizing/Stopped, Offline) may be shown
  from the session because they describe the session, not an observation; they never carry live authority.
* A last-supported time is a historical fact and is marked historical; it never makes a row live.
* ``display`` (EW-LAB-04) is the single formatter of badge text/tone and last-supported wording for tray AND inspector:
  without live authority the tone is neutral and the wording says stale/awaiting/lifecycle; times are historical.
* ``key`` identifies what is presented, so the view republishes tray + inspector in the SAME callback whenever the
  painted packet, its authority (expiry/cutover) or any row state changes.
"""
from __future__ import annotations

LIFECYCLE = ("paused", "offline", "stopped")
AWAIT = "awaiting"
LIFECYCLE_LABEL = {"starting": "Preparing", "paused": "Paused", "stopping": "Finalizing", "rolling_back": "Stopping"}


def packet_identity(packet) -> tuple | None:
    """Full identity: result numbers restart per run, so the run id is part of it."""
    return None if packet is None else (packet.run_id, packet.packet_id)


def current(packet, snap) -> bool:
    return (packet is not None and snap is not None and packet.run_id == snap.run_id
            and packet.basis.basis_id == snap.basis_id)


def project(base_rows: list[dict], painted_packet, snap, phase: str, now_mono: float, label_of: dict,
            stale_s: float = 1.0) -> tuple[list[dict], str, tuple]:
    """(rows, authority, key). ``base_rows``: the run's tray rows as "Saved, not monitoring" (names, thumbnails).
    authority: idle | painted | awaiting | lifecycle."""
    rows = [dict(r) for r in base_rows]
    pk = painted_packet
    if phase == "idle":
        authority = "idle"
    elif phase == "active" and current(pk, snap):
        authority = "painted"
        obs = {o.item_id: o for o in pk.items}
        live = now_mono - pk.frame_t_mono <= stale_s
        for r in rows:
            o = obs.get(r["item_id"])
            if o is None:
                continue
            r.update(state=o.state, label=label_of.get(o.state, o.state), reason=o.reason, name=o.name,
                     last_supported_wall=(o.last_supported or {}).get("t_wall"), live=live, box=o.box if live else None,
                     packet=packet_identity(pk), historical_last_supported=o.state != "visible")
    elif phase == "active":
        authority = "awaiting"
        prior = {o.item_id: o for o in pk.items} if (pk is not None and snap is not None and pk.run_id == snap.run_id) else {}
        for r in rows:
            o = prior.get(r["item_id"])
            r.update(state=AWAIT, label="Cannot assess - awaiting the current image", live=False, box=None, packet=None,
                     reason="monitoring changed (start, resume or new basis); no image of the current state is shown yet",
                     last_supported_wall=(o.last_supported or {}).get("t_wall") if o is not None else None,
                     historical_last_supported=True)
    else:
        authority = "lifecycle"
        srows = {x.item_id: x for x in (snap.items if snap is not None else ())}
        for r in rows:
            s = srows.get(r["item_id"])
            if s is not None and s.state in LIFECYCLE:
                state, label, reason = s.state, label_of.get(s.state, s.state), s.reason
            else:
                state, label, reason = phase, LIFECYCLE_LABEL.get(phase, phase.title()), "monitoring is not live"
            r.update(state=state, label=label, reason=reason, live=False, box=None, packet=None,
                     last_supported_wall=(s.last_supported or {}).get("t_wall") if s is not None else None,
                     historical_last_supported=True)
    age = None if (pk is None or authority != "painted") else now_mono - pk.frame_t_mono
    for r in rows:
        r.update(display(r, authority, age))
    key = (authority, packet_identity(pk) if authority == "painted" else None,
           tuple((r["item_id"], r.get("state"), bool(r.get("live")), r.get("name"), r["display_label"]) for r in rows))
    return rows, authority, key


# ---- operator-facing wording (EW-LAB-04 C2) ------------------------------------------------------------------------
#: badge tone of a LIVE observation; anything without live authority is "neutral" (never success/danger styling)
LIVE_TONE = {"visible": "success", "checking": "warning", "not_seen": "danger", "uncertain": "pending",
             "cannot_assess": "neutral"}


def display(row: dict, authority: str, age_s: float | None) -> dict:
    """The ONE place that turns a projected row into badge text, badge tone and last-supported wording, from the
    presentation authority - not from the historical observation's state alone. The original state/reason/time stay
    in the row unchanged; a stale observation is not reinterpreted as absence."""
    label = row.get("label") or ""
    if authority == "idle" or row.get("state") == "saved":
        return {"display_label": label, "display_tone": "accent", "last_supported_note": ""}
    if authority == "painted" and row.get("packet") is not None:
        if row.get("live"):
            hist = "" if row.get("state") == "visible" else " (historical position; not drawn on newer images)"
            return {"display_label": label, "display_tone": LIVE_TONE.get(row.get("state"), "neutral"),
                    "last_supported_note": hist}
        old = f" {age_s:.1f} s old" if age_s is not None else ""
        return {"display_label": f"Stale - last image{old} showed {label}", "display_tone": "neutral",
                "last_supported_note": " (historical; the shown image is stale and has no live authority)"}
    # awaiting / lifecycle (and painted rows without an observation for this item): never live
    return {"display_label": label, "display_tone": "neutral",
            "last_supported_note": " (historical; no current image has live authority)"}

