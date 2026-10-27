# SPDX-License-Identifier: AGPL-3.0-only
"""Tk-free projections for the YOLOE workspace (EW-LAB-06): what the boxes, list, counts, details, footer and banners
say for ONE painted packet. The Tk page only copies these values into widgets.

* Default view ``Detections + tracks``: ``cup · #17 · 0.82`` in the track's stable colour; a detection without a
  CONFIRMED track (first frame, tentative, low score, unmatched) keeps its box with a neutral outline and
  ``cup · Untracked · 0.31`` - never an invented id. ``Detections only`` removes id decorations, not detections.
* ``Shown confidence`` filters only what is drawn/listed; counts say ``shown of available``.
* Colours identify tracks of this run/epoch; they never mean Visible / Not visible.
* Labels are the model's own vocabulary names for THIS frame (a track whose class changes shows the new name).

EW-LAB-06 R1: ``page_state`` is the ONE read model for the canvas message, finish card, status pill and header on the
YOLOE page (no source / checking / ready / start refused / preparing + stage / running / paused / finalizing / failed
before any image / stopped before any image / ended / stopped / failed after frames). Counters show dashes and "not
analysed" before an image exists and "on this image" once one is analysed. ``details_text`` / ``result_details`` give
the bounded technical failure detail (stage, exception, traceback, worker stderr, truncation, evidence status) for
the current attempt and for a reopened saved test - including saved tests from before R1, whose details were not
captured and are said to be missing rather than reconstructed.
"""
from __future__ import annotations

import hashlib
import time

from ..yoloe.association import TRACKED
from ..yoloe.profile import VIEW_DETECTIONS, VIEW_TRACKS

TRACK_PALETTE = ("#22d3ee", "#f472b6", "#a3e635", "#fb923c", "#c084fc", "#facc15", "#38bdf8", "#f87171",
                 "#34d399", "#e879f9", "#fde047", "#60a5fa", "#fda4af", "#bef264", "#fdba74", "#67e8f9")
UNTRACKED_COLOR = "#9fb3c4"
DETECTION_COLOR = "#20d9e9"
STALE_S = 1.0
DETAILS_MAX = 48 * 1024
NOT_ANALYSED = "—"
VIEW_LABEL = {VIEW_TRACKS: "Detections + tracks", VIEW_DETECTIONS: "Detections only"}


def ellipsize(text: str, n: int) -> str:
    text = str(text)
    return text if len(text) <= n else text[: max(1, n - 1)] + "…"


def track_color(run_id: str, epoch: str, track_id: int) -> str:
    h = hashlib.sha256(f"{run_id}|{epoch}|{track_id}".encode()).digest()
    return TRACK_PALETTE[h[0] % len(TRACK_PALETTE)]


def det_entries(pk, view: str, shown_conf: float) -> list[dict]:
    """Every detection of the packet with its presentation; ``shown`` False when filtered by Shown confidence."""
    out = []
    for d in pk.detections:
        a = pk.assoc.get(d["i"])
        tracked = a is not None and a["state"] == TRACKED
        if view == VIEW_TRACKS and tracked:
            color, tag = track_color(pk.run_id, pk.epoch, a["track_id"]), f"#{a['track_id']}"
        elif view == VIEW_TRACKS:
            color, tag = UNTRACKED_COLOR, "Untracked"
        else:
            color, tag = DETECTION_COLOR, None
        name = ellipsize(d["name"], 28)
        label = f"{name} · {tag} · {d['score']:.2f}" if tag else f"{name} · {d['score']:.2f}"
        out.append({"i": d["i"], "xyxy": tuple(d["xyxy"]), "name": d["name"], "score": d["score"],
                    "track_id": a["track_id"] if tracked else None, "tentative": a is not None and not tracked,
                    "color": color, "dash": (6, 4) if (view == VIEW_TRACKS and not tracked) else (),
                    "label": label, "shown": d["score"] >= shown_conf - 1e-9,
                    "track_text": f"#{a['track_id']}" if tracked else "Untracked"})
    return out


def counts(pk, shown_conf: float) -> dict:
    if pk is None:
        return {"detections": 0, "associated": 0, "untracked": 0, "shown": 0}
    c = dict(pk.counts)
    c["shown"] = sum(1 for d in pk.detections if d["score"] >= shown_conf - 1e-9)
    return c


def count_texts(pk, shown_conf: float) -> dict:
    """Before any analysed image: dashes and "not analysed" (never a zero that looks like an empty result)."""
    if pk is None:
        return {"detections": (NOT_ANALYSED, "detections · not analysed"),
                "associated": (NOT_ANALYSED, "with track ID"), "untracked": (NOT_ANALYSED, "untracked")}
    c = counts(pk, shown_conf)
    return {"detections": (str(c["detections"]), f"on this image · {c['shown']} shown"),
            "associated": (str(c["associated"]), "with track ID"),
            "untracked": (str(c["untracked"]), "untracked")}


def empty_note(pk, shown_conf: float) -> str | None:
    """An ANALYSED image with nothing to draw says so (distinct from 'not analysed')."""
    if pk is None:
        return None
    if not pk.detections:
        return "Analysed: no detections on this image"
    if not any(d["score"] >= shown_conf - 1e-9 for d in pk.detections):
        return f"Analysed: {len(pk.detections)} detections, none at or above the shown confidence {shown_conf:.2f}"
    return None


def list_rows(pk, view: str, shown_conf: float, limit: int = 300) -> list[tuple[str, tuple]]:
    """(iid, (name, score, track)) for the side list - shown detections only, detector order, bounded."""
    if pk is None:
        return []
    rows = []
    for e in det_entries(pk, view, shown_conf):
        if not e["shown"]:
            continue
        track = e["track_text"] if view == VIEW_TRACKS else "-"
        rows.append((str(e["i"]), (ellipsize(e["name"], 30), f"{e['score']:.2f}", track)))
        if len(rows) >= limit:
            break
    return rows


def hit(entries: list[dict], x: float, y: float) -> int | None:
    """The SMALLEST shown box containing the source point (nested objects stay selectable)."""
    best = None
    for e in entries:
        if not e["shown"]:
            continue
        x1, y1, x2, y2 = e["xyxy"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best[0]:
                best = (area, e["i"])
    return None if best is None else best[1]


def follow_selection(sel: dict | None, pk) -> tuple[dict | None, str | None]:
    """Selection on a NEW packet: the same (run, epoch, track) when that track is associated on this image; otherwise
    the selection becomes historical text. Details never come from another frame's detection."""
    if sel is None or pk is None:
        return sel, None
    if sel["key"] == pk.key:
        return sel, None
    tid = sel.get("track_id")
    if tid is not None and sel["run_id"] == pk.run_id and sel["epoch"] == pk.epoch:
        for i, a in pk.assoc.items():
            if a["state"] == TRACKED and a["track_id"] == tid:
                d = next(d for d in pk.detections if d["i"] == i)
                return {**sel, "key": pk.key, "i": i, "seq": pk.seq, "name": d["name"]}, None
    what = f"#{tid} {sel['name']}" if tid is not None else f"untracked {sel['name']}"
    return None, f"Selected {what} is not in the current frame (last selected in frame {sel['seq']})."


def select(pk, i: int | None) -> dict | None:
    if pk is None or i is None:
        return None
    d = next((d for d in pk.detections if d["i"] == i), None)
    if d is None:
        return None
    a = pk.assoc.get(i)
    return {"key": pk.key, "run_id": pk.run_id, "epoch": pk.epoch, "seq": pk.seq, "i": i, "name": d["name"],
            "track_id": a["track_id"] if a is not None and a["state"] == TRACKED else None}


def detail_lines(pk, sel: dict | None, historical: str | None, view: str) -> list[str]:
    if sel is None or pk is None or sel["key"] != pk.key:
        return [historical] if historical else ["Click a box or a row to see its details. Selecting never remembers "
                                                "or renames anything."]
    d = next(d for d in pk.detections if d["i"] == sel["i"])
    a = pk.assoc.get(sel["i"])
    x1, y1, x2, y2 = d["xyxy"]
    lines = [f"{d['name']}  (vocabulary class {d['cls']})", f"Confidence {d['score']:.3f} · detection {d['i']} of frame "
             f"{pk.seq}", f"Box {x1:.0f},{y1:.0f} - {x2:.0f},{y2:.0f} ({x2 - x1:.0f}x{y2 - y1:.0f} px)"]
    if view == VIEW_TRACKS:
        if a is None:
            lines.append("Untracked: no track was associated with this detection on this frame.")
        elif a["state"] == TRACKED:
            lines.append(f"Track #{a['track_id']} (this test, epoch {pk.epoch.rsplit('/e', 1)[-1]}) - continuity in "
                         "this test, not a remembered identity.")
        else:
            lines.append(f"Untracked: a new track (#{a['track_id']}) started here but is not confirmed yet.")
    return lines


def authority(pk, snap: dict, now: float, stale_s: float = STALE_S) -> tuple[bool, str | None]:
    """(live, banner). Live only for the session's newest packet of the current run/epoch/control state while running
    and while analysis is not delayed; otherwise the image is visibly historical."""
    if pk is None:
        return False, None
    ph = snap.get("phase")
    if pk.run_id != snap.get("run_id"):
        return False, "PREVIOUS TEST: this image belongs to an earlier test."
    if ph == "paused":
        return False, f"PAUSED at frame {pk.seq}: playback and tracking are suspended."
    if ph == "finalizing":
        return False, "FINALIZING: last analysed image; no live result."
    if ph == "idle":
        oc = snap.get("outcome")
        st = getattr(oc, "status", None)
        if st == "failed":
            f = getattr(oc, "failure", None) or {}
            return False, ellipsize(f"FAILED while {f.get('stage_label') or 'analysing'}: last analysed image before the "
                                    f"failure (frame {pk.seq}).", 140)
        return False, {"completed": "ENDED: last analysed frame of the video.",
                       "stopped": "STOPPED: last analysed image."}.get(st, "Not running.")
    if pk.epoch != snap.get("epoch") or pk.control_gen != snap.get("control_gen"):
        return False, "PREVIOUS STATE: this image was analysed before the last control change."
    infl = snap.get("inflight_s")
    if infl is not None and infl > stale_s:
        return False, f"ANALYSIS DELAYED ({infl:.1f} s): showing the last analysed image."
    return True, None


def page_state(ctl, st: dict, has_image: bool = False) -> dict:
    """What the YOLOE page says now: kind, short status label, tone, a title and plain text for the canvas/finish card."""
    ph = st.get("phase")
    name = ctl.source_name or "the chosen video"
    fin = ctl.finished

    def out(kind, label, tone, title, text):
        return {"kind": kind, "label": label, "tone": tone, "title": title, "text": text}
    if ph == "preparing" or (ctl.start_pending and ph == "idle"):
        stage = st.get("stage_label") or "starting"
        return out("preparing", "Preparing", "accent", "Preparing the test",
                   f"{stage[:1].upper() + stage[1:]}... ({name}). The first analysed image appears when it is ready.")
    if ph == "running":
        if has_image:
            return out("running", "Running", "success", "Running", "Running. Track IDs belong to this test only.")
        return out("running_no_image", "Running", "success", "Running",
                   "Running: waiting for the first analysed image...")
    if ph == "paused":
        return out("paused", "Paused", "warning", "Paused", "Paused: playback and tracking are suspended.")
    if ph == "finalizing":
        return out("finalizing", "Finalizing", "warning", "Finalizing", "Finalizing the saved evidence...")
    if fin is not None:
        n = fin.frames
        ev = "" if fin.complete else " Evidence is INCOMPLETE (see Details)."
        if fin.status == "failed":
            f = fin.failure or {}
            label, cause = f.get("stage_label") or "the test", f.get("cause") or fin.reason
            if n == 0:
                return out("failed_before_image", "Failed", "danger", "Analysis could not start",
                           f"{label[:1].upper() + label[1:]} failed: {ellipsize(cause, 220)}\nNo video frame was "
                           f"analysed. {name} is still selected - Run again, or open Details.{ev}")
            return out("failed_after_frames", "Failed", "danger", "Analysis failed",
                       f"Failed while {label} after {n} analysed frames: {ellipsize(cause, 200)}{ev}")
        if fin.status == "stopped":
            if n == 0:
                return out("stopped_before_image", "Stopped", "neutral", "Stopped",
                           f"Stopped before any video frame was analysed ({name}).{ev}")
            return out("stopped", "Stopped", "neutral", "Stopped", f"Test stopped after {n} analysed frames.{ev}")
        return out("ended", "Ended", "neutral", "Ended", f"Test finished at the end of the video: {n} frames analysed.{ev}")
    if getattr(ctl, "start_error", None):
        return out("start_refused", "Not started", "warning", "The test did not start", ellipsize(ctl.start_error, 300))
    if ctl.source_status == "hashing":
        return out("checking", "Checking", "neutral", "Checking the video", f"Checking {name}... nothing is analysed yet.")
    if ctl.source_status == "failed":
        return out("source_failed", "Not usable", "danger", "This video cannot be used",
                   ellipsize(ctl.source_error or "unknown error", 300))
    if ctl.source_status == "ready":
        return out("ready", "Ready", "accent", "Ready", f"{name} is ready. Press Start to analyse it frame by frame.")
    return out("no_source", "Idle", "neutral", "No video", "Choose a video and press Start.\nNothing is analysed until then.")


def status(ctl, snap: dict) -> tuple[str, str]:
    """(label, tone) of the status pill - the same read model as the canvas and finish card."""
    ps = page_state(ctl, snap)
    return ps["label"], ps["tone"]


def footer(snap: dict, displayed_fps: float) -> str:
    if snap.get("phase") in (None, "idle"):
        return ""
    a = snap.get("analysed_fps")
    sp = snap.get("playback_speed")
    p50 = snap.get("processing_ms_p50")
    parts = [f"Analysed {a:.1f}/s" if a is not None else "Analysed -", f"Displayed {displayed_fps:.1f}/s",
             f"Processing {p50:.0f} ms" if p50 is not None else "Processing -",
             f"Playback {sp:.2f}x source speed" if sp is not None else "Playback -"]
    return " · ".join(parts)


def _bounded(text: str, limit: int = DETAILS_MAX) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters not shown here; the saved files keep them ...]"


def failure_lines(f: dict | None, details_captured: bool = True) -> list[str]:
    """Full bounded technical detail of one structured failure (never reconstructed when it was not captured)."""
    if not f:
        if not details_captured:
            return ["Failure details: NOT CAPTURED. This test was saved by a version before EW-LAB-06 R1, which kept only "
                    "the short reason above; the original exception and traceback were not recorded and are not "
                    "reconstructed here."]
        return []
    out = [f"Failed stage: {f.get('stage_label') or f.get('stage')} ({f.get('stage')}) · category {f.get('category')}",
           f"Cause: {f.get('cause') or ''}",
           f"Exception: {f.get('exc_type')}" + (f" at {f['where']}" if f.get("where") else ""),
           "Message:" + (" (truncated)" if f.get("message_truncated") else ""), f.get("message") or ""]
    w = f.get("worker") or {}
    if w:
        out.append("")
        out.append(f"Worker report: category {w.get('category')} · attempt {w.get('attempt', '-')} · before ready "
                   f"{w.get('before_ready', '-')} · exit code {w.get('exit_code', '-')}")
        if w.get("exc_type"):
            out.append(f"Worker stage: {w.get('stage_label') or w.get('stage')} · {w.get('exc_type')}"
                       + (f" at {w['where']}" if w.get("where") else ""))
            out.append("Worker message:" + (" (truncated)" if w.get("message_truncated") else ""))
            out.append(w.get("message") or "")
        if w.get("traceback"):
            out.append("Worker traceback:" + (" (truncated)" if w.get("traceback_truncated") else ""))
            out.append(w["traceback"].rstrip())
        se = w.get("stderr") or {}
        if se.get("text"):
            out.append(f"Worker stderr ({se.get('lines', '?')} lines"
                       + (f", {se.get('dropped_chars')} characters truncated" if se.get("truncated") else "")
                       + ("" if se.get("complete", True) else ", capture may be incomplete") + "):")
            out.append(se["text"].rstrip())
    if f.get("traceback"):
        out.append("")
        out.append("Application traceback:" + (" (truncated)" if f.get("traceback_truncated") else ""))
        out.append(f["traceback"].rstrip())
    return out


def details_text(snap: dict, ctl) -> str:
    """Technical detail of the CURRENT attempt (status, failure, evidence, model/tracker/source facts), bounded."""
    fin = ctl.finished
    head = [f"Status: {page_state(ctl, snap).get('title')}"]
    if fin is not None:
        head.append(f"Saved test: {fin.run_dir} · evidence {'complete' if fin.complete else 'INCOMPLETE'} · "
                    f"{fin.frames} frames analysed")
        head.append(f"Outcome: {fin.status} - {fin.reason}")
        fl = failure_lines(fin.failure)
        if fl:
            head += [""] + fl
        head.append("")
    elif snap.get("phase") not in (None, "idle"):
        head.append(f"Stage: {snap.get('stage_label') or '-'} · run {snap.get('run_id')}")
        head.append("")
    if getattr(ctl, "start_error", None):
        head += [f"Start refused: {ctl.start_error}", ""]
    return _bounded("\n".join(head + [_facts(snap, ctl)]))


def _facts(snap: dict, ctl) -> str:
    r = snap.get("receipt") or {}
    pr = snap.get("profile") or {}
    det = pr.get("detector") or {}
    tr = pr.get("tracker") or {}
    src = snap.get("source") or {}
    c = snap.get("counts") or {}
    st = src.get("stats") or {}
    lines = [f"Model: yoloe-26s-seg-pf (prompt-free) · {r.get('device', '-')} {r.get('dtype', '')} · vocabulary "
             f"{r.get('vocabulary', '-')} names",
             f"Detector: imgsz {det.get('imgsz', '-')} · conf {det.get('conf', '-')} · iou {det.get('iou', '-')}"
             f"{'' if r.get('iou_applies', True) else ' (not applied: end-to-end head)'} · max {det.get('max_det', '-')}",
             f"Tracker: BoT-SORT · ReID off · motion {tr.get('gmc_method', '-')} · buffer {tr.get('track_buffer', '-')} "
             "updates",
             f"Source: {ctl.source_name or '-'} · processing {src.get('size') or '-'} · native {src.get('native_size') or '-'}",
             f"Frames: analysed {c.get('processed', 0)} · published {c.get('published', 0)} · displayed "
             f"{c.get('displayed', 0)} · gaps {c.get('gaps', 0)} ({c.get('gap_frames', 0)} frames) · unverified "
             f"{c.get('unverified_mapping', 0)}",
             f"Tracking epochs: {len(snap.get('resets') or [])} · association errors {c.get('assoc_errors', 0)} · "
             f"frames at max detections {c.get('max_det_frames', 0)} · decoder queue max {st.get('max_queue', '-')}"]
    for e in (snap.get("errors") or [])[-5:]:
        lines.append("Error: " + e)
    return "\n".join(lines)


def result_details(m: dict) -> str:
    """Bounded technical detail of a SAVED test from its manifest only (no model, worker or decoding)."""
    oc = m.get("outcome") or {}
    c = m.get("counts") or {}
    src = m.get("source") or {}
    w = m.get("worker") or {}
    tr = m.get("trace") or {}
    frames = oc.get("frames_analysed", c.get("processed"))
    lines = [f"Saved test {m.get('run_id')} · {src.get('name')} · status {oc.get('status') or m.get('status')}",
             f"Reason: {oc.get('reason') or '-'}",
             f"Frames analysed: {frames}" + (" - no video frame was analysed, so there is no frame to show"
                                             if frames == 0 else ""),
             f"Evidence: {'complete' if oc.get('complete') else 'INCOMPLETE or partial'} · trace {tr.get('lines', '?')} "
             f"lines" + (f" · trace error: {tr.get('error')}" if tr.get("error") else "")]
    fl = failure_lines(oc.get("failure"), bool(oc.get("details_captured")))
    if fl:
        lines += [""] + fl
    lines += ["", f"Source: {src.get('path')} ({src.get('bytes')} bytes, sha256 {str(src.get('sha256'))[:16]}...)",
              f"Worker: {w.get('device', 'never ready') if w else 'never ready'} {w.get('dtype', '') if w else ''}",
              "Counts: " + ", ".join(f"{k} {v}" for k, v in c.items())]
    for e in (m.get("errors") or [])[-5:]:
        lines.append("Error: " + str(e))
    return _bounded("\n".join(lines))


def results_rows(runs: list[dict]) -> list[tuple[str, tuple]]:
    rows = []
    for r in runs:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started_wall"])) if r.get("started_wall") else "-"
        ev = "" if r.get("complete") else " (partial)" if r.get("status") not in ("unreadable",) else ""
        if r.get("status") == "failed" and r.get("processed") == 0:
            ev = " (not started)" + ev
        rows.append((r["run_id"], (ellipsize(r.get("source") or "-", 34), when, f"{r.get('status') or '-'}{ev}",
                                   str(r.get("processed") if r.get("processed") is not None else "-"))))
    return rows


def record_rows(page: list[dict]) -> list[tuple[str, tuple]]:
    out = []
    for r in page:
        if r.get("error"):
            out.append((str(r["n"]), (str(r["n"]), "-", r["error"], "")))
            continue
        pts = f"{r['pts']:.3f}s" if isinstance(r.get("pts"), (int, float)) else "-"
        src = str(r["src_n"]) if r.get("src_n") is not None else "unverified"
        out.append((str(r["n"]), (str(r.get("seq")), f"{src} @ {pts}",
                                  f"{r.get('detections')} det · {r.get('associated')} tracked",
                                  "" if r.get("disposition") == "published" else str(r.get("disposition")))))
    return out


def layout(w: int, h: int, scale: float) -> dict:
    """Side-panel width and optional-detail collapse for the YOLOE page; the video keeps priority at 1280x720."""
    lw, lh = w / max(scale, 1.0), h / max(scale, 1.0)
    side = 300 if lw < 1360 else 340
    compact = lw < 1360 or lh < 760
    return {"side_px": int(side * scale), "details_open": lh >= 760, "list_rows": 6 if lh < 760 else 10,
            "compact": compact, "file_chars": 22 if compact else 34, "fin_px": int(150 * scale)}
