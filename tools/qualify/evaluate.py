# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-01 run evaluator (never imported by the runtime). Corrects the inherited Native scorer (preserved unchanged as
evaluate_native_v0.py) for defects B and C of docs/references/NATIVE_FOUNDATION_REVIEW.md:

B. THROUGHPUT is scored over the run's explicit endpoint (``stop_requested_mono``: operator Stop, EOF or failure), not
   its last successful result. The steady window is [active_ack + 5 s, endpoint] minus deliberate PAUSED intervals
   (from the basis history). Leading and trailing gaps (window start -> first result, last result -> endpoint) and empty
   1 s bins count. Every gap > 1 s is classified: ``processing_stall`` (source frames kept arriving), ``source_gap``
   (no frames arrived) - both fail the healthy-steady criterion and are reported, never hidden. A session whose outcome
   is not ``finalized`` cannot be an overall throughput PASS, however fast its successful prefix was.
C. PAIRING requires evidence for EVERY committed result: a non-negative integer media ordinal inside the decoded
   source, a well-formed 16x9 signature (288 hex characters), a successful decode, a clean frame-map entry for the
   result's frame identity, and signature agreement. Any missing/invalid item makes the status FAIL (never PASS). The
   16x9 lossy signature tolerance is supporting evidence, NOT full-image cryptographic equality.
Recognition is scored by scoring.py against evaluator-only truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import scoring  # noqa: E402

from echo_watch.config import FFMPEG  # noqa: E402
from echo_watch.engine.inspect import decode_signatures, inspect_recording  # noqa: E402
from echo_watch.metrics import percentile  # noqa: E402

STEADY_EXCLUDE_S = 5.0
SIG_HEX = 16 * 9 * 2
SIG_TOLERANCE = 12.0


def load_run(run_dir: Path) -> dict:
    kinds: dict[str, list] = {}
    with open(run_dir / "trace.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            kinds.setdefault(r["k"], []).append(r)
    results = sorted(kinds.get("result", []), key=lambda r: r["seq"])
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    return {"results": results, "frames": kinds.get("frame", []), "discards": kinds.get("discard", []),
            "sessions": kinds.get("session", []), "basis": kinds.get("basis", []), "metrics": kinds.get("metrics", []),
            "resources": kinds.get("resource", []), "manifest": manifest, "dir": run_dir}


def _commit(r: dict) -> float:
    return r["timing"]["commit_mono"]


def paused_intervals(basis_history: list[dict], end: float) -> list[tuple[float, float]]:
    out, start = [], None
    for b in basis_history:
        if b.get("reason") == "operator pause":
            start = b["t_mono"]
        elif b.get("reason") == "operator resume" and start is not None:
            out.append((start, b["t_mono"]))
            start = None
    if start is not None:
        out.append((start, end))
    return out


def throughput(manifest: dict, results: list[dict], ingress: list[float]) -> dict:
    t = manifest["times"]
    active, end = t.get("active_ack_mono"), t.get("stop_requested_mono")
    outcome = manifest.get("outcome") or {}
    out: dict = {"results_committed": len(results), "active_ack_mono": active, "endpoint_mono": end,
                 "endpoint": outcome.get("ended_by"), "outcome": outcome.get("kind"),
                 "requested_to_active_s": t.get("requested_to_active_s")}
    if active is None or end is None:
        out.update(status="UNPERFORMED", why="the session never became active or has no recorded endpoint")
        return out
    t0 = active + STEADY_EXCLUDE_S
    pauses = paused_intervals(manifest.get("basis_history") or [], end)
    out["paused_intervals"] = [[round(a - active, 3), round(b - active, 3)] for a, b in pauses]

    def paused(x: float) -> bool:
        return any(a <= x < b for a, b in pauses)
    if end - t0 < 2.0:
        out.update(status="UNPERFORMED", why=f"steady window shorter than 2 s ({end - t0:.2f} s)")
        return out
    steady = [r for r in results if t0 <= _commit(r) <= end and not paused(_commit(r))]
    paused_s = sum(max(0.0, min(b, end) - max(a, t0)) for a, b in pauses)
    healthy_s = (end - t0) - paused_s
    nbins = int(end - t0)
    counts = []
    for k in range(nbins):
        lo, hi = t0 + k, t0 + k + 1
        if any(a < hi and lo < b for a, b in pauses):
            continue                                       # a second touched by a deliberate pause is not scored
        counts.append(sum(1 for r in steady if lo <= _commit(r) < hi))
    # every gap including the boundaries; a gap explained by a deliberate pause is reported as paused, not a stall
    marks = [t0] + [_commit(r) for r in steady] + [end]
    gaps = []
    for a, b in zip(marks, marks[1:]):
        if b - a <= 1.0:
            continue
        overlap = sum(max(0.0, min(b, pe) - max(a, ps)) for ps, pe in pauses)
        if (b - a) - overlap <= 1.0:
            continue                                       # explained by a deliberate pause
        arrived = sum(1 for x in ingress if a < x < b)
        gaps.append({"from_s": round(a - active, 3), "to_s": round(b - active, 3), "gap_s": round(b - a, 3),
                     "kind": "processing_stall" if arrived else "source_gap", "source_frames_inside": arrived,
                     "boundary": "leading" if a == t0 else ("trailing" if b == end else "internal")})
    lat = [(_commit(r) - r["frame_t_mono"]) * 1000 for r in steady]
    mean = len(steady) / healthy_s if healthy_s > 0 else 0.0
    frac5 = (sum(1 for c in counts if c >= 5) / len(counts)) if counts else 0.0
    out.update(steady_window_s=round(end - t0, 3), paused_s=round(paused_s, 3), healthy_s=round(healthy_s, 3),
               steady_results=len(steady), mean_results_per_s=round(mean, 3), bins=len(counts),
               bins_with_ge5=sum(1 for c in counts if c >= 5), fraction_bins_ge5=round(frac5, 4),
               empty_bins=sum(1 for c in counts if c == 0),
               capture_to_commit_ms_p50=round(percentile(lat, 50), 1) if lat else None,
               capture_to_commit_ms_p95=round(percentile(lat, 95), 1) if lat else None,
               capture_to_commit_ms_max=round(max(lat), 1) if lat else None,
               max_unexplained_gap_s=max([g["gap_s"] for g in gaps], default=0.0), unhealthy_gaps=gaps,
               leading_gap_s=round(marks[1] - marks[0], 3), trailing_gap_s=round(marks[-1] - marks[-2], 3))
    out["criteria"] = {"mean_ge_8": mean >= 8.0, "bins_ge5_in_95pct": frac5 >= 0.95,
                       "p95_capture_to_commit_le_500ms": bool(lat) and out["capture_to_commit_ms_p95"] <= 500.0,
                       "no_unexplained_gap_gt_1s": not gaps,
                       "session_finalized_not_failed": outcome.get("kind") == "finalized"}
    out["status"] = "PASS" if all(out["criteria"].values()) else "FAIL"
    return out


def pairing(results: list[dict], source_sigs: list[np.ndarray] | None, clean_map: dict | None,
            decode_error: str | None = None) -> dict:
    """Complete pairing evidence for EVERY committed result, or the status is FAIL."""
    problems: dict[str, int] = {}
    examples = []
    agree = 0

    def bad(kind, r, **kw):
        problems[kind] = problems.get(kind, 0) + 1
        if len(examples) < 12:
            examples.append({"seq": r.get("seq"), "problem": kind, **kw})
    clean_by_ident = {}
    if clean_map is not None:
        for e in clean_map.get("entries", []):
            if e.get("ident") is not None:
                clean_by_ident[tuple(e["ident"])] = e
    for r in results:
        m = r.get("media") or {}
        n = m.get("media_n")
        sig = (r.get("flags") or {}).get("image_sig")
        if decode_error is not None or source_sigs is None:
            bad("source_decode_failed", r)
            continue
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            bad("missing_or_invalid_media_ordinal", r, media_n=n)
            continue
        if n >= len(source_sigs):
            bad("media_ordinal_outside_decoded_source", r, media_n=n)
            continue
        if not isinstance(sig, str) or len(sig) != SIG_HEX:
            bad("missing_or_malformed_signature", r)
            continue
        try:
            a = np.frombuffer(bytes.fromhex(sig), np.uint8).astype(np.int16)
        except ValueError:
            bad("missing_or_malformed_signature", r)
            continue
        d = float(np.abs(a - source_sigs[n].astype(np.int16)).mean())
        if d > SIG_TOLERANCE:
            bad("signature_mismatch_with_source_frame", r, media_n=n, mean_abs_diff=round(d, 2))
            continue
        if clean_map is not None:
            e = clean_by_ident.get(tuple(r["frame"]))
            if e is None:
                bad("no_clean_frame_map_entry", r, frame=r["frame"])
                continue
            try:
                cs = np.frombuffer(bytes.fromhex(e.get("sig") or ""), np.uint8).astype(np.int16)
                ok = cs.size == a.size and float(np.abs(cs - a).mean()) <= SIG_TOLERANCE
            except ValueError:
                ok = False
            if not ok:
                bad("clean_frame_map_signature_mismatch", r, frame=r["frame"])
                continue
        agree += 1
    total = len(results)
    status = "PASS" if total and agree == total else ("UNPERFORMED" if not total else "FAIL")
    return {"results": total, "fully_evidenced": agree, "problems": problems, "examples": examples, "status": status,
            "evidence_class": "16x9 lossy signature agreement within tolerance: supporting evidence, not full-image "
                              "cryptographic equality"}


def recordings(run: dict) -> dict:
    d = run["dir"]
    out = {}
    for which in ("clean", "annotated"):
        v, m = d / f"{which}.mp4", d / f"{which}.framemap.json"
        if v.is_file() and m.is_file():
            try:
                out[which] = inspect_recording(FFMPEG, v, m)
            except Exception as e:  # noqa: BLE001 - reported
                out[which] = {"status": "inspection_failed", "error": str(e)}
            doc = json.loads(m.read_text("utf-8"))
            out[which].update(frames_dropped_queue=doc["frames_dropped_queue"], gaps=len(doc["gaps"]),
                              largest_gap_s=max([g["seconds"] for g in doc["gaps"]], default=0.0))
            if which == "annotated":
                fresh = [e for e in doc["entries"] if e["kind"] == "fresh"]
                ids = {e["result"] for e in fresh}
                out[which].update(entries=len(doc["entries"]), fresh=len(fresh),
                                  held=sum(1 for e in doc["entries"] if e["kind"] == "held"),
                                  results_not_in_annotated_cfr=len({r["packet_id"] for r in run["results"]} - ids))
        else:
            out[which] = {"status": "missing"}
    out["trace"] = run["manifest"]["outputs"].get("trace")
    return out


def stage_costs(results: list[dict]) -> dict:
    keys: dict[str, list] = {}
    for r in results:
        for k, v in (r["timing"].get("stage_ms") or {}).items():
            if isinstance(v, (int, float)):
                keys.setdefault(k, []).append(v)
        for k in ("infer_ms", "post_ms", "rtt_ms"):
            v = (r.get("worker") or {}).get(k)
            if isinstance(v, (int, float)):
                keys.setdefault("worker_" + k, []).append(v)
    return {k: {"n": len(v), "p50": round(percentile(v, 50), 2), "p95": round(percentile(v, 95), 2), "max": round(max(v), 2)}
            for k, v in sorted(keys.items())}


def recognition_results(run: dict, name_of: dict[str, str]) -> list[dict]:
    out = []
    for r in run["results"]:
        mt = (r.get("media") or {}).get("media_pts")
        if mt is None:
            continue
        out.append({"t": mt, "items": {name_of.get(o["item_id"], o["item_id"]): {"state": o["state"], "box": o["box"]}
                                       for o in r["items"]}})
    return out


def evaluate(run_dir: Path, scenario: str, truth_path: Path | None, source_video: Path,
             name_of: dict[str, str] | None, names: list[str] | None) -> dict:
    run = load_run(run_dir)
    man = run["manifest"]
    ingress = [f["t_mono"] for f in run["frames"]]
    decode_error = None
    try:
        sigs = decode_signatures(FFMPEG, source_video)
    except Exception as e:  # noqa: BLE001
        sigs, decode_error = None, f"{type(e).__name__}: {e}"
    cm = run_dir / "clean.framemap.json"
    clean_map = json.loads(cm.read_text("utf-8")) if cm.is_file() else None
    rep: dict = {"run_dir": str(run_dir), "scenario": scenario, "outcome": man["outcome"], "counts": man["counts"],
                 "times": man["times"], "throughput": throughput(man, run["results"], ingress),
                 "stage_costs_ms": stage_costs(run["results"]),
                 "pairing": pairing(run["results"], sigs, clean_map if man["options"].get("clean_source") else None, decode_error),
                 "recordings": recordings(run), "resource": man.get("resource"), "device_counts": {}}
    for r in run["results"]:
        dv = (r.get("worker") or {}).get("device")
        rep["device_counts"][str(dv)] = rep["device_counts"].get(str(dv), 0) + 1
    if truth_path is not None and name_of:
        truth = json.loads(Path(truth_path).read_text("utf-8"))
        res = recognition_results(run, name_of)
        if scenario in ("D", "H1", "H2"):
            rep["recognition"] = scoring.score_phone_variant(res, truth)
        elif scenario == "controls":
            rep["recognition"] = scoring.score_controls(res, truth)
            rep["throughput_note"] = "controls contain a deliberate 2 s pause: excluded from steady time, reported"
        else:
            rep["recognition"] = scoring.score_main_scene(res, truth, names or [])
    return rep
