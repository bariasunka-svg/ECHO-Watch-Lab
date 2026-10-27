# SPDX-License-Identifier: AGPL-3.0-only
"""CSV projections of ONE run's canonical decision trace (frames, decisions, candidates, timing) + manifest.

Pattern adapted (no code imported) from the frozen PatronSight reference app/review_export.py and the identity trace
export (README + manifest with record/table counts and file hashes). Exporting never re-runs policy or repairs a past
decision: every row is copied from a trace record; blank stays blank (never coerced to zero). Each captured frame gets
an explicit disposition: analysed (result seq), discarded (reason), ineligible (reason) or not analysed (replaced by a
newer frame before analysis / duplicate source picture). Hashes identify the files; they do not certify conclusions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import threading
import time
from pathlib import Path


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def export_run(run_dir: Path, cancel: threading.Event | None = None) -> dict:
    run_dir = Path(run_dir)
    trace = run_dir / "trace.jsonl"
    if not trace.is_file():
        raise FileNotFoundError("this run has no decision trace to export")
    out = run_dir / f"export-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=False)
    frames: dict[tuple, dict] = {}
    order: list[tuple] = []
    n_lines = 0
    fdec = open(out / "decisions.csv", "w", newline="", encoding="utf-8")
    fcan = open(out / "candidates.csv", "w", newline="", encoding="utf-8")
    ftim = open(out / "timing.csv", "w", newline="", encoding="utf-8")
    try:
        wd, wc, wt = csv.writer(fdec), csv.writer(fcan), csv.writer(ftim)
        wd.writerow(["seq", "packet_id", "epoch", "frame_seq", "media_pts", "item_id", "prior_state", "state", "label",
                     "reason", "box", "track_id", "app", "admission", "conf", "confirmed_now", "negative_ok", "negative_why",
                     "alternatives", "basis_id", "control_generation", "worker_device", "vp_model_sha256"])
        wc.writerow(["seq", "cand_idx", "det", "low_confidence", "box", "conf", "withheld", "track", "track_owner",
                     "track_pending", "cannot", "item_id", "app", "admission", "rank_score"])
        wt.writerow(["seq", "capture_to_commit_ms", "queue_wait", "worker_round_trip", "handoff_wait", "evaluate", "commit",
                     "infer_ms", "post_ms", "rtt_ms", "eval_quality", "eval_assessability_geometry", "eval_matching"])
        with open(trace, encoding="utf-8") as fh:
            for line in fh:
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("export cancelled")
                n_lines += 1
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                k = r.get("k")
                if k == "frame":
                    key = tuple(r["ident"])
                    frames[key] = {"epoch": key[0], "seq": key[1], "t_mono": r.get("t_mono"), "media_pts": r.get("media_pts"),
                                   "src_n": r.get("src_n"), "dup": r.get("dup"), "disposition": "not analysed",
                                   "detail": "duplicate source picture" if r.get("dup") else
                                   "replaced by a newer frame before analysis (latest-frame handoff)"}
                    order.append(key)
                elif k in ("discard", "ineligible"):
                    key = tuple(r.get("ident") or ())
                    if key in frames:
                        frames[key].update(disposition=k, detail=r.get("reason"))
                elif k == "result":
                    key = tuple(r["frame"])
                    if key in frames:
                        frames[key].update(disposition="analysed", detail=f"result {r['seq']}")
                    prior = r.get("prior_states") or {}
                    w = r.get("worker") or {}
                    for o in r.get("items", []):
                        wd.writerow([r["seq"], r["packet_id"], r["frame"][0], r["frame"][1], (r.get("media") or {}).get("media_pts"),
                                     o["item_id"], prior.get(o["item_id"]), o["state"], o.get("label"), o.get("reason"),
                                     json.dumps(o.get("box")) if o.get("box") else "", o.get("track_id"), o.get("app"),
                                     o.get("admission"), o.get("conf"), o.get("confirmed_now"), o.get("negative_ok"),
                                     o.get("negative_why"), json.dumps(o.get("alternatives") or []), r.get("basis_id"),
                                     r.get("control_generation"), w.get("device"), w.get("vp_model_sha256")])
                    for c in r.get("candidates", []):
                        for item_id, app in (c.get("app") or {}).items():
                            wc.writerow([r["seq"], c.get("idx"), c.get("det"), bool(c.get("low_confidence")), json.dumps(c.get("box")),
                                         c.get("conf"), c.get("withheld"), c.get("track"), c.get("track_owner"),
                                         c.get("track_pending"), ";".join(c.get("cannot") or []), item_id, app,
                                         (c.get("admission") or {}).get(item_id), (c.get("rank_score") or {}).get(item_id)])
                    t = r.get("timing") or {}
                    sm = t.get("stage_ms") or {}
                    wt.writerow([r["seq"], t.get("capture_to_commit_ms"), sm.get("queue_wait"), sm.get("worker_round_trip"),
                                 sm.get("handoff_wait"), sm.get("evaluate"), sm.get("commit"), w.get("infer_ms"),
                                 w.get("post_ms"), w.get("rtt_ms"), sm.get("eval_quality"),
                                 sm.get("eval_assessability_geometry"), sm.get("eval_matching")])
    finally:
        fdec.close()
        fcan.close()
        ftim.close()
    with open(out / "frames.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["epoch", "seq", "t_mono", "media_pts", "src_n", "duplicate", "disposition", "detail"])
        for key in order:
            f = frames[key]
            w.writerow([f["epoch"], f["seq"], f["t_mono"], f["media_pts"], f["src_n"], f["dup"], f["disposition"], f["detail"]])
    readme = ("ECHO Watch Lab run export: CSV projections of trace.jsonl (the canonical decision trace).\n"
              "frames.csv: every captured frame with its disposition. decisions.csv: one row per committed result and item.\n"
              "candidates.csv: every evaluated candidate x item with appearance score and admission (support/unresolved/"
              "reject). timing.csv: per-result stage timings. Blank means unavailable, never zero. Nothing here was\n"
              "recomputed; hashes identify files and do not certify conclusions.\n")
    (out / "README.txt").write_text(readme, "utf-8")
    files = {p.name: {"sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(out.glob("*.csv"))}
    man = {"source_trace": str(trace), "source_trace_sha256": _sha(trace), "trace_lines": n_lines,
           "frames": len(order), "files": files, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "recomputed": False}
    (out / "manifest.json").write_text(json.dumps(man, indent=1), "utf-8")
    return {"dir": str(out), **man}
