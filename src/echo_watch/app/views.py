# SPDX-License-Identifier: AGPL-3.0-only
"""Tk-free view projections of the guided Display Assets workbench (EW-LAB-05).

Every screen of the native workbench renders from these projections of the controller state - the step indicator,
header, Discover, Select, Configure, the monitoring summary and inspector evidence, the finish panel, the bottom action
bar and the responsive layout policy. They hold no widgets, so the production wiring can be exercised GUI-free with
non-widget sinks. Wording rules kept here: setup is "Saved, not monitoring" (never Ready/Visible); this delivery is
"Monitoring - video replay" (never camera status); motion/removal/return are not tested by setup; uncertainty, paused
and stale states are never counted as Visible or Not visible.
"""
from __future__ import annotations

from datetime import datetime

from ..engine.contracts import PHASE_ACTIVE, PHASE_IDLE, PHASE_PAUSED, PHASE_STARTING, PHASE_STOPPING

SUBTITLE = {"discover": "Discover items", "select": "Choose items to monitor", "configure": "Saved references and rules",
            "monitor": "Monitoring - video replay"}
OUTPUT_TEXT = {"finalized": "saved", "failed": "FAILED", "not_requested": "off (not requested)",
               "not_started": "not started", "partial": "partial - see Runs & Review"}


def wall(t: float | None) -> str:
    return "-" if not t else datetime.fromtimestamp(t).strftime("%H:%M:%S")


def short_id(item_id: str | None) -> str:
    if not item_id:
        return "-"
    return item_id.split("_", 1)[-1][:6].upper()


def ellipsize(text: str | None, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[:max(1, n - 1)].rstrip() + "…"


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# ---- shell --------------------------------------------------------------------------------------------------------
def header(ctl) -> dict:
    """Compact top context: display/setup name, file, honest replay-source status and one session state."""
    info = ctl.source_info or {}
    name = info.get("name") or (ctl.source_path and ctl.source_path.replace("\\", "/").rsplit("/", 1)[-1]) or "No video chosen"
    ss = ctl.source_status
    if ctl.phase != PHASE_IDLE:
        src = ("Replay source in use", "accent")
    else:
        src = {"none": ("No replay source", "neutral"), "preparing": ("Preparing replay source", "accent"),
               "ready": ("Replay source ready", "accent"), "failed": ("Replay source unavailable", "danger")}[ss]
    ph = ctl.phase
    if ctl.start_pending and ph == PHASE_IDLE:
        sess = ("Preparing", "accent")
    elif ph == PHASE_ACTIVE:
        sess = ("Monitoring - video replay", "success")
    elif ph == PHASE_PAUSED:
        sess = ("Paused", "warning")
    elif ph == PHASE_STOPPING:
        sess = ("Finalizing", "warning")
    elif ph == PHASE_STARTING:
        sess = ("Preparing", "accent")
    elif ph != PHASE_IDLE:
        sess = ("Rolling back", "danger")
    else:
        sess = ("Monitoring stopped" if ctl.last_outcome is not None else "Monitoring not started", "neutral")
    return {"setup_name": ctl.setup_name, "source_name": name, "source": src, "session": sess,
            "show_controls": ph in (PHASE_ACTIVE, PHASE_PAUSED, PHASE_STARTING, PHASE_STOPPING) or ctl.start_pending}


def title(ctl) -> dict:
    vs = ctl.view_step
    if vs == "monitor":
        return {"title": "Display Assets", "subtitle": SUBTITLE["monitor"] if ctl.finished is None else "Run finished"}
    return {"title": "Set up display monitoring", "subtitle": SUBTITLE[vs]}


def stepper(ctl) -> list[dict]:
    return ctl.steps()


# ---- Discover -----------------------------------------------------------------------------------------------------
def discover(ctl) -> dict:
    info = ctl.source_info or {}
    ss, sc = ctl.source_status, ctl.scan_status
    n = len(ctl.candidates)
    progress = None
    primary = ("Discover items", False)
    retry = False
    if ss == "none":
        status, tone = "Choose a local video to begin. Choosing a video never starts monitoring.", "neutral"
    elif ss == "preparing":
        status, tone, progress = "Preparing the video: reading the file and keeping its first frame for setup...", "accent", \
            "indeterminate"
    elif ss == "failed":
        status, tone = "The video could not be prepared. Choose it again (see the message below).", "danger"
    elif sc == "waiting_worker":
        status, tone, progress = "Starting the GPU worker for discovery (first use only)...", "accent", "indeterminate"
    elif sc == "running":
        status, tone, progress = "Scanning visible items on the setup frame...", "accent", "indeterminate"
    elif sc == "failed":
        status, tone = f"Discovery failed: {ctl.scan_error or 'unknown error'}. The setup frame is kept; retry below.", "danger"
        primary, retry = ("Discover again", True), True
    elif sc == "done":
        if n:
            status, tone = f"Discovery finished: {plural(n, 'candidate')} on this frame. Next, choose which items to remember.", \
                "success"
        else:
            status, tone = ("Discovery finished but found no candidates. Discover again, or continue and use Add missed "
                            "item."), "warning"
        primary = ("Discover again", True)
    else:
        status, tone = "Ready. Discover items runs the real discovery on this setup frame.", "neutral"
        primary = ("Discover items", True)
    if ss != "ready" or not ctl.stopped or sc in ("running", "waiting_worker"):
        primary = (primary[0], False)
    sel = ctl.selection_counts() if sc == "done" else {"selected": 0}
    saved_n = len(ctl.saved_items)
    count = (f"{sel['selected']} selected · " + (f"{plural(saved_n, 'saved item')} for this recording" if saved_n else
                                                       "no saved items for this recording")) if sc == "done" else ""
    return {"filename": info.get("name") or "No video chosen",
            "frame_caption": "Setup frame - monitoring has not started" if ctl.setup_frame is not None else None,
            "status": status, "tone": tone, "progress": progress, "primary": primary, "retry": retry,
            "count_title": f"Scan candidates ({n})" if sc == "done" else "Scan candidates",
            "count": count,
            "cards": [{"key": f"cand:{c.candidate_id}", "name": c.name, "label": c.label, "box": c.box,
                       "checked": c.checked} for c in ctl.candidates],
            "empty": sc == "done" and n == 0,
            "can_continue": ctl.step_allowed("select")[0],
            "continue_hint": "" if ctl.step_allowed("select")[0] else "Available when discovery finishes",
            "saved_setups": [{"setup_id": s["setup_id"], "text": f"{s['name']} v{s['version']} · {plural(s['items'], 'item')}"}
                             for s in ctl.saved_setups]}


# ---- Select -------------------------------------------------------------------------------------------------------
def disposition_text(c) -> str:
    if not c.checked:
        return "Not selected"
    if c.disposition == "reuse":
        return "Use saved item" + (" (remembered from this suggestion)" if c.enrolled and c.enrolled.get("item_id") == c.reuse_item
                                   else "")
    if c.disposition == "choose":
        return "Overlaps a saved item - choose"
    return "New item"


def select(ctl) -> dict:
    counts = ctl.selection_counts()
    focus = ctl.selected or ""
    rows = []
    for c in ctl.candidates:
        rows.append({"key": f"cand:{c.candidate_id}", "name": c.name, "checked": c.checked,
                     "status": disposition_text(c), "warn": c.checked and c.disposition == "choose",
                     "label": c.label, "manual": c.manual, "focused": focus == f"cand:{c.candidate_id}"})
    used = {c.reuse_item: c.name for c in ctl.candidates if c.checked and c.disposition == "reuse"}
    saved = []
    for s in ctl.saved_items:
        picked = s["item_id"] in ctl.saved_picks
        via = used.get(s["item_id"])
        saved.append({"key": f"saved:{s['item_id']}", "item_id": s["item_id"], "name": s["name"],
                      "checked": picked or via is not None,
                      "status": (f"Used for {via}" if via else ("Use saved item" if picked else "Not used")) +
                                f" · v{s['current_ref_version']} · ID {short_id(s['item_id'])}",
                      "focused": focus == f"saved:{s['item_id']}"})
    blocked = None
    if ctl.saved_blocked:
        b = ctl.saved_blocked
        codes = {}
        for x in b:
            codes.setdefault(x["compat_code"], x)
        first = next(iter(codes.values()))
        blocked = (f"{plural(len(b), 'saved item')} cannot be reused for this recording - for example "
                   f"{ellipsize(first['name'], 24)}: {first['compat_reason']}.")
    detail = None
    if focus.startswith("cand:"):
        c = ctl.candidate(focus[5:])
        if c is not None:
            detail = {"kind": "candidate", "key": focus, "name": c.name, "category": c.category,
                      "detected": (f"Detected as \"{c.label}\" - a suggestion, not an identity" if not c.manual else
                                   "Outline added by you (recovers an item discovery missed)")
                                  + (" · outline adjusted" if c.corrected else ""),
                      "checked": c.checked, "disposition": c.disposition if c.checked else None,
                      "status": disposition_text(c),
                      "matches": [{"item_id": m["item_id"], "name": m["name"], "ref_version": m["ref_version"],
                                   "text": f"{m['name']} (reference v{m['ref_version']}, {round(m['iou'] * 100)}% overlap)"}
                                  for m in c.matches],
                      "reuse_item": c.reuse_item,
                      "explain": ("Overlap is only a suggestion. Use saved item keeps the same identity and reference "
                                  "version; Create separate item remembers a different item at this place (both may then "
                                  "look alike - similar items can be uncertain). Replace reference saves a new version "
                                  "of the saved item and is not needed to rerun.") if c.matches else
                                 "Selected items are saved as remembered identities. The outline is the reference image, "
                                 "not a fixed monitoring zone: monitoring follows the item."}
    elif focus.startswith("saved:"):
        s = ctl.saved_item(focus[6:])
        if s is not None:
            detail = {"kind": "saved", "key": focus, "name": s["name"], "item_id": s["item_id"],
                      "ref_version": s["current_ref_version"], "crop_path": s.get("crop_path"),
                      "checked": s["item_id"] in ctl.saved_picks or s["item_id"] in used,
                      "status": f"Saved item · ID {short_id(s['item_id'])} · reference v{s['current_ref_version']}",
                      "explain": "Using a saved item keeps its identity and reference version. It is your decision, not "
                                 "a recognition result, and it does not mean the item is visible."}
    st = ctl.enroll_state
    banner = None
    if st == "pending":
        banner = ("Saving the selection...", "accent")
    elif st in ("uncertain", "not_saved", "rejected", "conflict") and ctl.enroll_message:
        banner = (ctl.enroll_message, "warning" if st in ("uncertain", "not_saved") else "danger")
    n, k = counts["selected"], counts["choose"]
    if st == "uncertain":
        primary = ("Check saved outcome", ctl.can()["reconcile"])
    else:
        primary = (f"Remember & configure {plural(n, 'item')}" if n else "Remember & configure items",
                   ctl.can()["remember"])
    choices = None
    pend = ctl.pending_choices()
    if pend:
        first = pend[0]
        m = first.matches[0] if first.matches else None
        lines = [f"{c.name} overlaps saved {c.matches[0]['name']}" + (f" (+{len(c.matches) - 1} more)"
                                                                     if len(c.matches) > 1 else "")
                 for c in pend[:4] if c.matches] + ([f"... and {len(pend) - 4} more"] if len(pend) > 4 else [])
        choices = {"title": f"{plural(len(pend), 'selected item')} need{'s' if len(pend) == 1 else ''} your choice",
                   "text": "\n".join(lines) + "\nSame physical item as before? Use saved item keeps its ID and history. "
                           "A different item at the same place? Create separate item.",
                   "next_key": f"cand:{first.candidate_id}", "next_name": first.name,
                   "next_saved": None if m is None else m["item_id"],
                   "use_text": f"{ellipsize(first.name, 14)}: use saved {ellipsize(m['name'], 14)}" if m else "",
                   "new_text": f"{ellipsize(first.name, 14)}: separate item"}
    if counts["problems"]:
        hint = (f"{plural(k, 'selected item')} overlap{'s' if k == 1 else ''} saved items - choose Use saved item or "
                "Create separate item" if k else counts["problems"][0])
    else:
        hint = f"{counts['new']} new · {counts['reused']} reused" if n else "Click outlines or check suggestions to select"
    count_text = f"{n} selected · " + (f"{k} to decide · " if k else "") + plural(counts["suggestions"], "suggestion")
    return {"count_text": count_text,
            "summary": f"{counts['new']} new · {counts['reused']} reused" + (f" · {k} to decide" if k else ""),
            "rows": rows, "saved": saved, "saved_title": f"Use saved items ({len(saved)})", "blocked": blocked,
            "detail": detail, "banner": banner, "primary": primary, "hint": hint, "problems": counts["problems"],
            "choices": choices}


# ---- Configure ----------------------------------------------------------------------------------------------------
def configure(ctl) -> dict:
    cards = []
    sel = ctl.selected or ""
    for i in ctl.tray:
        p = ctl.tray_profiles.get(i, {})
        v = ctl.member_versions.get(i)
        cards.append({"key": f"item:{i}", "item_id": i, "name": p.get("name", i), "short_id": short_id(i),
                      "version": v, "crop_path": p.get("crop_path"), "badge": "Saved reference",
                      "status": "Saved", "sub": f"ID {short_id(i)} · reference v{v}",
                      "disposition": "reused" if p.get("disposition") == "reused" else "new",
                      "selected": sel == f"item:{i}"})
    detail = None
    if sel.startswith("item:") and sel[5:] in ctl.tray_profiles:
        i = sel[5:]
        p = ctl.tray_profiles[i]
        detail = {"item_id": i, "name": p.get("name", i), "short_id": short_id(i), "version": ctl.member_versions.get(i),
                  "crop_path": p.get("crop_path"), "category": p.get("category") or "Other",
                  "rows": [("Item ID", i), ("Reference", f"v{ctl.member_versions.get(i)} (saved)"),
                           ("Recording", (ctl.source_info or {}).get("name") or "-"),
                           ("Monitoring", "Follows the item where it is supported (not a fixed position)")]}
    info = ctl.source_info or {}
    try:
        dirty = ctl.setup_dirty()
    except Exception:  # noqa: BLE001
        dirty = True
    s = ctl.setup
    if ctl.setup_status == "saving":
        setup_line = ("Saving configuration...", "accent")
    elif ctl.setup_status == "loading":
        setup_line = ("Checking the saved setup...", "accent")
    elif ctl.setup_status == "failed" and ctl.setup_error:
        setup_line = (f"Not saved: {ctl.setup_error}", "danger")
    elif s is not None and not dirty:
        setup_line = (f"Saved as {s.name} v{s.version}", "success")
    elif s is not None:
        setup_line = (f"Changed since {s.name} v{s.version} - Start saves a new version first", "warning")
    else:
        setup_line = ("Not saved yet - Save configuration, or Start (Start saves this configuration first)", "neutral")
    outs = ctl.pending.to_dict()
    readiness = [
        {"ok": ctl.source_status == "ready", "label": "Recording prepared",
         "detail": (f"{info.get('name')} - re-checked for changes at Start" if ctl.source_status == "ready" else
                    "Choose the recording again")},
        {"ok": bool(ctl.tray) and all(ctl.tray_profiles.get(i, {}).get("crop_sha256") for i in ctl.tray),
         "label": "Saved references available",
         "detail": f"{plural(len(ctl.tray), 'reference')}; hashes re-verified at Start"},
        {"ok": True, "label": "Output configuration valid",
         "detail": ("Nothing will be recorded (every output is off)" if not any(outs[k] for k in
                    ("clean_source", "record_annotated", "decision_trace")) else
                    "Turned-off outputs are recorded as not requested")},
        {"ok": None, "label": "Worker preparation at Start",
         "detail": "References are prepared and acknowledged by the GPU worker when you start (shown as Preparing)"},
        {"ok": None, "label": "Motion/removal/return not tested by setup",
         "detail": "Setup saves references only; it does not test movement, lifting or return"},
    ]
    a = ctl.pending_allowance
    return {"count_text": f"{plural(len(cards), 'saved reference')}", "context": info.get("name") or "-",
            "cards": cards, "detail": detail, "setup_name": ctl.setup_name, "allowance": a,
            "allowance_text": f"Request review after {plural(a, 'second')} of supported Not visible",
            "allowance_note": ("Only supported Not-visible time counts. Unknown, uncertain, obscured, stale and paused "
                               "time never counts. This requests a review; it is not proof of theft or removal."),
            "outputs": outs, "setup_line": setup_line, "readiness": readiness,
            "save": ("Save configuration", ctl.can()["save_config"]),
            "primary": ("Preparing..." if ctl.start_pending else "Start monitoring", ctl.can()["start"]),
            "running_note": None if ctl.running_setup is None else
            f"Running: {ctl.running_setup['name']} v{ctl.running_setup['version']} (pending edits apply to the next run)"}


# ---- Monitor ------------------------------------------------------------------------------------------------------
def monitor_summary(rows: list[dict]) -> dict:
    """Counts derived from the SAME projected rows that the tray, inspector and boxes use. Only live rows count as
    Visible / Not visible; everything else (checking, uncertain, cannot assess, awaiting, stale, paused) is other."""
    vis = sum(1 for r in rows if r.get("live") and r.get("state") == "visible")
    nv = sum(1 for r in rows if r.get("live") and r.get("state") == "not_seen")
    other = len(rows) - vis - nv
    parts: dict[str, int] = {}
    for r in rows:
        if r.get("live") and r.get("state") in ("visible", "not_seen"):
            continue
        k = r.get("display_label") or r.get("label") or r.get("state")
        k = "Stale" if str(k).startswith("Stale") else k
        parts[k] = parts.get(k, 0) + 1
    return {"monitored": len(rows), "visible": vis, "not_visible": nv, "other": other,
            "other_text": ", ".join(f"{n} {k.lower()}" for k, n in parts.items()) or "none"}


def evidence(row: dict | None, prof: dict, painted_time: float | None) -> dict:
    """Inspector image captions: the saved reference vs the current supported view (or an honest absence of one)."""
    ver = prof.get("ref_version") or prof.get("current_ref_version") or 1
    if row is None:
        return {"saved": f"Saved reference v{ver}", "current": "No monitoring data", "current_live": False}
    if row.get("live") and row.get("box") is not None and row.get("state") in ("visible", "checking"):
        cur = f"Current view · {wall(painted_time)}" + (" (tentative)" if row.get("state") == "checking" else "")
        return {"saved": f"Saved reference v{ver}", "current": cur, "current_live": True}
    ls = row.get("last_supported_wall")
    cur = (f"No current supported view. Last supported {wall(ls)} (historical)" if ls else
           "No supported view of this item in this run yet")
    return {"saved": f"Saved reference v{ver}", "current": cur, "current_live": False}


def timeline(events: list[dict], limit: int = 5) -> list[dict]:
    ev = sorted(events, key=lambda e: e.get("wall") or 0)[-limit:]
    return [{"time": wall(e.get("wall")), "label": e.get("label") or e.get("kind") or "", "kind": e.get("kind")} for e in ev]


def finish(ctl) -> dict | None:
    f = ctl.finished
    if f is None:
        return None
    kind, by = f["kind"], f["ended_by"]
    if kind == "rolled_back":
        head = "Monitoring did not start"
    elif kind == "stopped_after_failure":
        head = "Monitoring stopped after a failure"
    elif by == "source_ended":
        head = "Video ended"
    else:
        head = "Monitoring stopped"
    lines = [f"{k}: {OUTPUT_TEXT.get(v, v)}" for k, v in f["outputs"].items()]
    notes = []
    if by == "source_ended":
        notes.append("The video ended. That is not evidence that any item is absent.")
    if f.get("error"):
        notes.append(f"Reason: {f['error']}")
    notes.append("Current membership is cleared. Library profiles, saved setups and this run are kept.")
    return {"title": head, "complete": f["complete"], "outputs": lines, "notes": notes,
            "setup": None if not f.get("setup") else f"Setup used: {f['setup']['name']} v{f['setup']['version']}",
            "actions": [("review", "Review saved run", bool(f.get("run_id")) and kind != "rolled_back"),
                        ("reuse", "Reuse this setup", ctl.last_setup is not None),
                        ("new", "New setup", True)]}


# ---- bottom action bar ----------------------------------------------------------------------------------------------
def action_bar(ctl) -> dict:
    """One primary next action and a secondary Back; monitoring keeps Pause/Resume and Stop obvious."""
    vs = ctl.view_step
    can = ctl.can()
    if vs == "discover":
        d = discover(ctl)
        return {"back": None, "secondary": [], "primary": ("continue", "Continue to selection", d["can_continue"]),
                "hint": d["continue_hint"]}
    if vs == "select":
        s = select(ctl)
        sec = [("use_all", "Use saved items for all", ctl.stopped)] if s["choices"] else []
        return {"back": ("back", "Back", can["back"]), "secondary": sec,
                "primary": ("reconcile" if ctl.enroll_state == "uncertain" else "remember", s["primary"][0], s["primary"][1]),
                "hint": s["hint"]}
    if vs == "configure":
        c = configure(ctl)
        return {"back": ("back", "Back", can["back"]), "secondary": [("save", c["save"][0], c["save"][1])],
                "primary": ("start", c["primary"][0], c["primary"][1]), "hint": c["setup_line"][0]}
    fin = finish(ctl)
    if fin is not None:
        acts = fin["actions"]
        return {"back": None, "secondary": [(k, t, e) for k, t, e in acts[:2]], "primary": (acts[2][0], acts[2][1], True),
                "hint": fin["title"]}
    pr = ("resume", "Resume", can["resume"]) if ctl.phase == PHASE_PAUSED else ("pause", "Pause", can["pause"])
    return {"back": None, "secondary": [pr], "primary": ("stop", "Finalizing..." if ctl.stop_pending else "Stop",
                                                          can["stop"]), "danger_primary": True,
            "hint": "Navigation never stops monitoring. Stop finalizes the recordings and the decision trace."}


# ---- overlap hit selection and responsive layout -----------------------------------------------------------------------
def pick_at(boxes, point) -> list[str]:
    """Keys of every outline containing ``point``, smallest area first (a table-sized outline never hides a phone)."""
    px, py = point
    hits = [(w * h, i, key) for i, (key, (x, y, w, h)) in enumerate(boxes) if x <= px <= x + w and y <= py <= y + h]
    return [k for _a, _i, k in sorted(hits)]


def layout(width_px: int, height_px: int, scale: float = 1.0) -> dict:
    """Logical-pixel layout policy (1280x720 at 100/125/150% and up). Reflows by available logical space: narrower
    rail, narrower side panel, collapsible details, compact header. Never shrinks text and never moves the primary
    action off screen (the action bar is a fixed row outside every scroll area)."""
    s = max(1.0, float(scale or 1.0))
    lw, lh = width_px / s, height_px / s
    if lw >= 1360:
        rail, side, tier = 196, 336, "wide"
    elif lw >= 1100:
        rail, side, tier = 172, 300, "standard"
    else:
        rail, side, tier = 132, 264, "compact"
    compact_header = lh < 760
    collapsible = lw < 1000
    return {"tier": tier, "logical": (round(lw), round(lh)), "rail": rail, "side": side, "rail_px": int(rail * s),
            "side_px": int(side * s), "gutter_px": int(16 * s if tier != "compact" else 12 * s),
            "compact_header": compact_header, "side_collapsible": collapsible,
            "grid_cols": 2 if side >= 280 else 1, "card_cols": 3 if lw >= 1360 else 2,
            "tray_card_px": int((180 if lw >= 1200 else 156) * s), "wrap_px": int((side - 40) * s),
            "nav_short": tier == "compact"}
