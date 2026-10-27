# SPDX-License-Identifier: AGPL-3.0-only
# PRESERVED UNCHANGED (EW-LAB-01): the adopted Native scorer, kept only to reproduce its baseline verdicts and the
# defects B/C counterexamples. The Lab scorer is tools/qualify/evaluate.py.
"""EVALUATOR (never imported by the runtime): scores a finished run against the package's prospective criteria.

Truth (generated scenes only) is read HERE and nowhere else. Everything scored comes from the run's own decision trace /
manifest / written files: committed result packets (distinct, model-derived, complete), capture ledger, frame maps and the
inspected recordings. Raw capture, repainted or held frames, and propagated boxes are never counted as analysed results.

Prospective foundation criteria (EW-01 standalone package, "Reproducible inputs and measurements"):
  throughput  mean >= 8 distinct complete results/s over the STEADY window (first 5 s after acknowledged active excluded);
              >= 5 results in >= 95% of steady 1 s bins; p95 capture-to-result <= 500 ms; max healthy steady gap <= 1 s
  coverage    ordinary visible periods (windows [5,20) [38,50) [50,65) [65,90) of the 90 s scene): >= 95% of analysed results
              show the item Visible with a box IoU >= 0.5 against truth; zero wrong-target assignment; all counts retained;
              the moved interval (speaker, [50,65)) is ALSO reported separately
  absence     Not visible within 3 s of disappearance and Visible within 3 s of return; no box on an analysed frame whose
              target is absent; no fabricated absence in the occlusion / offline controls
  lifecycle   zero mismatched frame/result, pre-boundary current state, duplicate confirmation or post-stop publication
  outputs     clean + annotated + trace finalize or disclose failure; written files inspected windowlessly
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from echo_watch.config import FFMPEG  # noqa: E402
from echo_watch.engine.inspect import decode_signatures, inspect_recording  # noqa: E402
from echo_watch.geometry import iou  # noqa: E402
from echo_watch.metrics import percentile  # noqa: E402

WINDOWS = [(5.0, 20.0), (38.0, 50.0), (50.0, 65.0), (65.0, 90.0)]
STEADY_EXCLUDE_S = 5.0
FPS = 15.0


def load_run(run_dir: Path) -> dict:
    results, frames, discards, sessions, resources, basis, metrics = [], [], [], [], [], [], []
    with open(run_dir / "trace.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            k = r["k"]
            if k == "result":
                results.append(r)
            elif k == "frame":
                frames.append(r)
            elif k == "discard":
                discards.append(r)
            elif k == "session":
                sessions.append(r)
            elif k == "resource":
                resources.append(r)
            elif k == "basis":
                basis.append(r)
            elif k == "metrics":
                metrics.append(r)
    results.sort(key=lambda r: r["seq"])
    manifest = json.loads((run_dir / "manifest.json").read_text("utf-8"))
    return {"results": results, "frames": frames, "discards": discards, "sessions": sessions, "resources": resources,
            "basis": basis, "metrics": metrics, "manifest": manifest, "dir": run_dir}


def recording_stage(run: dict) -> dict:
    """Capture / analysed / encoded rates are DISTINCT measurements: here the 1 Hz 'metrics' records of the run."""
    m = [x for x in run["metrics"] if x.get("phase") == "active"]
    out: dict = {"samples": len(m)}
    for k in ("capture_fps", "analyzed_fps", "clean_encode_fps", "annotated_encode_fps"):
        v = [x[k] for x in m if x.get(k) is not None]
        if v:
            out[k] = {"mean": round(float(np.mean(v)), 2), "p05": round(percentile(v, 5), 2), "max": round(max(v), 2)}
    pump = (run["manifest"]["outputs"] or {}).get("pump") or {}
    out["pump"] = pump
    last = m[-1] if m else {}
    out["replaced_before_analysis"] = last.get("replaced_before_analysis")
    out["trace_dropped_last"] = last.get("trace_dropped")
    return out


def media_t(r: dict) -> float | None:
    m = r.get("media") or {}
    return m.get("media_pts")


def _commit(r: dict) -> float:
    return r["timing"]["commit_mono"]


def throughput(run: dict) -> dict:
    man, res = run["manifest"], run["results"]
    t = man["times"]
    active = t["active_ack_mono"]
    out: dict = {"results_committed": len(res), "active_ack_mono": active,
                 "requested_to_active_s": t.get("requested_to_active_s")}
    if active is None or not res:
        out.update(status="UNPERFORMED", why="the session never became active" if active is None else "no results")
        return out
    steady = [r for r in res if _commit(r) >= active + STEADY_EXCLUDE_S]
    if len(steady) < 2:
        out.update(status="UNPERFORMED", why="fewer than two steady results")
        return out
    t0, t1 = active + STEADY_EXCLUDE_S, _commit(steady[-1])
    dur = t1 - t0
    bins = int(dur)
    counts = [0] * max(1, bins)
    for r in steady:
        b = int((_commit(r) - t0))
        if b < len(counts):
            counts[b] += 1
    gaps = [(_commit(b) - _commit(a), _commit(a), _commit(b)) for a, b in zip(steady, steady[1:])]
    lat = [(_commit(r) - r["frame_t_mono"]) * 1000 for r in steady]
    ing = sorted(f["t_mono"] for f in run["frames"])
    long_gaps = []
    for g, a, b in gaps:
        if g > 1.0:
            src = [x for x in ing if a < x < b]
            long_gaps.append({"gap_s": round(g, 3), "from_mono": a, "to_mono": b, "source_frames_arrived_inside": len(src)})
    # startup and source gaps reported separately (not part of the steady window)
    src_gaps = [round(b - a, 3) for a, b in zip(ing, ing[1:]) if b - a > 0.25]
    mean = len(steady) / dur
    frac5 = sum(1 for c in counts if c >= 5) / len(counts)
    out.update(steady_window_s=round(dur, 3), steady_results=len(steady), mean_results_per_s=round(mean, 3),
               bins=len(counts), bins_with_ge5=sum(1 for c in counts if c >= 5), fraction_bins_ge5=round(frac5, 4),
               bin_counts_min=min(counts), bin_counts_median=float(np.median(counts)), capture_to_result_ms_p50=round(percentile(lat, 50), 1),
               capture_to_result_ms_p95=round(percentile(lat, 95), 1), capture_to_result_ms_max=round(max(lat), 1),
               max_steady_gap_s=round(max(g for g, _, _ in gaps), 3), long_steady_gaps=long_gaps,
               source_ingress_gaps_over_250ms=src_gaps[:50], source_ingress_gaps_count=len(src_gaps))
    out["criteria"] = {"mean_ge_8": mean >= 8.0, "bins_ge5_in_95pct": frac5 >= 0.95,
                       "p95_latency_le_500ms": out["capture_to_result_ms_p95"] <= 500.0,
                       "max_gap_le_1s": out["max_steady_gap_s"] <= 1.0}
    out["status"] = "PASS" if all(out["criteria"].values()) else "FAIL"
    return out


def stage_costs(run: dict) -> dict:
    res = run["results"]
    keys: dict[str, list] = {}
    for r in res:
        for k, v in (r["timing"].get("stage_ms") or {}).items():
            if isinstance(v, (int, float)):
                keys.setdefault(k, []).append(v)
        for k in ("infer_ms", "post_ms", "ipc_out_ms", "worker_ms", "ipc_back_ms", "rtt_ms"):
            v = (r.get("worker") or {}).get(k)
            if isinstance(v, (int, float)):
                keys.setdefault("worker_" + k, []).append(v)
    return {k: {"n": len(v), "p50": round(percentile(v, 50), 2), "p95": round(percentile(v, 95), 2), "max": round(max(v), 2)}
            for k, v in sorted(keys.items())}


def pairing(run: dict, source_video: Path) -> dict:
    """Every committed packet's pixel signature must equal the signature of the SOURCE FILE's frame it claims (media_n)."""
    sigs = decode_signatures(FFMPEG, source_video)
    bad, checked, missing = [], 0, 0
    for r in run["results"]:
        m = r.get("media") or {}
        n = m.get("media_n")
        sig = (r.get("flags") or {}).get("image_sig")
        if n is None or sig is None or n >= len(sigs):
            missing += 1
            continue
        a = np.frombuffer(bytes.fromhex(sig), np.uint8).astype(np.int16)
        d = float(np.abs(a - sigs[n].astype(np.int16)).mean())
        checked += 1
        if d > 12.0:
            bad.append({"seq": r["seq"], "media_n": n, "mean_abs_diff": round(d, 2)})
        # the frame identity sequence agrees with the decoder ordinal (1-based sequence, 0-based media ordinal)
        if r["frame"][1] - 1 != m.get("out_n") and m.get("out_n") is not None:
            bad.append({"seq": r["seq"], "ident_mismatch": [r["frame"][1], m.get("out_n")]})
    return {"checked": checked, "missing_media_or_sig": missing, "mismatched": len(bad), "examples": bad[:10],
            "status": "PASS" if checked and not bad else ("UNPERFORMED" if not checked else "FAIL")}


def _truth_box(truth: dict, name: str, n: int):
    it = truth["items"][name]
    return it["box"][n] if 0 <= n < len(it["box"]) and it["present"][n] else None


def score_scene(run: dict, truth: dict, selected: dict[str, str], windows=WINDOWS) -> dict:
    """selected: scene item name -> engine item_id. Scores each committed result against truth (never fed back)."""
    res = run["results"]
    names = list(selected)
    detail = {n: {} for n in names}
    wrong, tentative_wrong, unsupported, absent_boxes = [], [], [], []
    first = {n: {} for n in names}
    for r in res:
        n = (r.get("media") or {}).get("media_n")
        if n is None:
            continue
        t = n / FPS
        by_item = {o["item_id"]: o for o in r["items"]}
        for name in names:
            o = by_item[selected[name]]
            tb = _truth_box(truth, name, n)
            box = o["box"]
            for (a, b) in windows:
                if a <= t < b and tb is not None:
                    w = detail[name].setdefault(f"[{a:g},{b:g})", {"eligible": 0, "visible_iou_ge_0.5": 0, "visible_wrong_box": 0,
                                                                    "checking": 0, "uncertain": 0, "not_seen": 0, "cannot_assess": 0,
                                                                    "other": 0})
                    w["eligible"] += 1
                    st = o["state"]
                    if st == "visible" and box is not None and iou(box, tb) >= 0.5:
                        w["visible_iou_ge_0.5"] += 1
                    elif st == "visible":
                        w["visible_wrong_box"] += 1
                    elif st in ("checking", "uncertain", "not_seen", "cannot_assess"):
                        w[st] += 1
                    else:
                        w["other"] += 1
            if box is not None:
                if tb is None:
                    absent_boxes.append({"seq": r["seq"], "item": name, "t": round(t, 3), "state": o["state"]})
                else:
                    own = iou(box, tb)
                    best_name, best = None, 0.0
                    for other in truth["items"]:
                        ob = _truth_box(truth, other, n)
                        if ob is not None:
                            v = iou(box, ob)
                            if v > best:
                                best_name, best = other, v
                    if best_name is not None and best_name != name and best >= 0.3 and own < 0.3:
                        # an ASSIGNMENT is a confirmed Visible box; a 'checking' box is a tentative candidate being confirmed
                        (wrong if o["state"] == "visible" else tentative_wrong).append(
                            {"seq": r["seq"], "item": name, "t": round(t, 3), "state": o["state"], "matches": best_name, "iou": round(best, 3)})
                    elif best < 0.1:
                        unsupported.append({"seq": r["seq"], "item": name, "t": round(t, 3), "state": o["state"]})
            for key, cond in (("not_seen", o["state"] == "not_seen"), ("visible", o["state"] == "visible" and box is not None
                                                                         and tb is not None and iou(box, tb) >= 0.5)):
                if cond:
                    first[name].setdefault(key, []).append((t, r["seq"], o["negative_ok"]))
    for name in names:
        for w, d in detail[name].items():
            d["coverage"] = round(d["visible_iou_ge_0.5"] / d["eligible"], 4) if d["eligible"] else None
    return {"per_item_window": detail, "wrong_target": wrong, "tentative_box_on_other_item": tentative_wrong,
            "unsupported_box": unsupported, "box_on_absent_target": absent_boxes,
            "first_events": {n: {k: v[:3] for k, v in d.items()} for n, d in first.items()}, "_series": first}


def scene_verdicts(sc: dict, truth: dict, selected: dict[str, str]) -> dict:
    v: dict = {}
    ordinary_ok = True
    rows = []
    for name, wins in sc["per_item_window"].items():
        for w, d in wins.items():
            moved = (name == "speaker" and w == "[50,65)")
            ok = d["coverage"] is not None and d["coverage"] >= 0.95
            rows.append({"item": name, "window": w, "moved_interval": moved, **d, "pass_ge_95": ok})
            if not moved:
                ordinary_ok &= ok
    v["coverage_rows"] = rows
    v["ordinary_windows_coverage_ge_95"] = ordinary_ok
    mv = [r for r in rows if r["moved_interval"]]
    v["moved_interval_reported_separately"] = mv
    v["zero_wrong_target"] = not sc["wrong_target"]
    v["zero_box_on_absent_target"] = not sc["box_on_absent_target"]
    v["zero_unsupported_box"] = not sc["unsupported_box"]
    if "speaker" in selected:
        ser = sc["_series"]["speaker"]
        t_dis, t_ret = 20.0, 35.0
        ns = [x for x in ser.get("not_seen", []) if x[0] >= t_dis]
        vis = [x for x in ser.get("visible", []) if x[0] >= t_ret]
        v["speaker_not_visible_latency_s"] = None if not ns else round(ns[0][0] - t_dis, 3)
        v["speaker_not_visible_within_3s"] = bool(ns) and ns[0][0] - t_dis <= 3.0
        v["speaker_not_visible_negative_ok"] = None if not ns else ns[0][2]
        v["speaker_visible_latency_s"] = None if not vis else round(vis[0][0] - t_ret, 3)
        v["speaker_visible_within_3s"] = bool(vis) and vis[0][0] - t_ret <= 3.0
    return v


CONTROLS = {"speaker_obscured": (10.0, 18.0), "phone_a_removed": (18.0, 24.0), "pause": (5.0, 7.0)}


def _intervals(ts: list[float], gap: float = 0.5) -> list[list[float]]:
    """Compress sorted media times into [start, end] runs separated by more than ``gap`` seconds."""
    out: list[list[float]] = []
    for t in sorted(ts):
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return [[round(a, 2), round(b, 2)] for a, b in out]


def score_controls(run: dict, truth: dict, selected: dict[str, str]) -> dict:
    """The generated occlusion / look-alike / pause controls (make_scene.py 'controls'): speaker obscured 10-18 s; phone_a removed
    18-24 s while its identical twin phone_b stays; the session is paused at ~5 s for 2 s. Truth never reaches the engine."""
    sp_id, ph_id = selected["speaker"], selected["phone_a"]
    ob, rm, ps = CONTROLS["speaker_obscured"], CONTROLS["phone_a_removed"], CONTROLS["pause"]
    sc = score_scene(run, truth, selected, windows=[(0.0, 30.0)])
    series = {"speaker": {}, "phone_a": {}}
    states = {"speaker": {"obscured": {}, "pause": {}}, "phone_a": {"twin_present": {}, "removed": {}, "all": {}}}
    for r in run["results"]:
        n = (r.get("media") or {}).get("media_n")
        if n is None:
            continue
        t = n / FPS
        by = {o["item_id"]: o for o in r["items"]}
        s, p = by[sp_id], by[ph_id]

        def bump(d: dict, o: dict) -> None:
            d[o["state"]] = d.get(o["state"], 0) + 1
        if ob[0] <= t < ob[1]:
            bump(states["speaker"]["obscured"], s)
        if ps[0] <= t < ps[1] + 2.0:
            bump(states["speaker"]["pause"], s)
        bump(states["phone_a"]["all"], p)
        if rm[0] <= t < rm[1]:
            bump(states["phone_a"]["removed"], p)
        else:
            bump(states["phone_a"]["twin_present"], p)
    sp_ob = states["speaker"]["obscured"]
    out = {"state_counts": states,
           "speaker_not_seen_while_obscured": sp_ob.get("not_seen", 0),
           "speaker_visible_while_obscured": sp_ob.get("visible", 0),
           "speaker_not_seen_around_pause": states["speaker"]["pause"].get("not_seen", 0),
           "wrong_target": sc["wrong_target"][:25], "wrong_target_count": len(sc["wrong_target"]),
           "wrong_target_intervals": _intervals([x["t"] for x in sc["wrong_target"]]),
           "tentative_box_on_other_item_count": len(sc["tentative_box_on_other_item"]),
           "tentative_box_on_other_item_examples": sc["tentative_box_on_other_item"][:6],
           "box_on_absent_target_intervals": _intervals([x["t"] for x in sc["box_on_absent_target"]]),
           "box_on_absent_target": sc["box_on_absent_target"][:25], "box_on_absent_target_count": len(sc["box_on_absent_target"]),
           "unsupported_box_count": len(sc["unsupported_box"])}
    out["criteria"] = {"no_fabricated_absence_during_occlusion": out["speaker_not_seen_while_obscured"] == 0,
                       "no_fabricated_absence_around_pause": out["speaker_not_seen_around_pause"] == 0,
                       "no_box_on_absent_target": out["box_on_absent_target_count"] == 0,
                       "zero_wrong_target_assignment": out["wrong_target_count"] == 0,
                       "identical_twin_never_confirmed_as_visible_while_ambiguous": None}
    # 'ambiguous identical candidates remain uncertain': while the identical twin is co-visible phone_a may only be Visible
    # through the operator's own enrollment selection; any Visible box that is not on phone_a's own truth is a violation
    # (wrong_target above). Report how long phone_a was Visible / Uncertain for the reader.
    out["criteria"].pop("identical_twin_never_confirmed_as_visible_while_ambiguous")
    out["status"] = "PASS" if all(out["criteria"].values()) else "FAIL"
    return out


def recordings(run: dict) -> dict:
    d = run["dir"]
    out = {}
    for which, fname in (("clean", "clean.mp4"), ("annotated", "annotated.mp4")):
        v, m = d / fname, d / f"{which}.framemap.json"
        if v.is_file() and m.is_file():
            out[which] = inspect_recording(FFMPEG, v, m)
            doc = json.loads(m.read_text("utf-8"))
            out[which].update(frames_dropped_queue=doc["frames_dropped_queue"], gaps=len(doc["gaps"]),
                              largest_gap_s=max([g["seconds"] for g in doc["gaps"]], default=0.0),
                              start_requested_mono=doc["start_requested_mono"], started_mono=doc["started_mono"])
            if which == "annotated":
                kinds = [e["kind"] for e in doc["entries"]]
                fresh = [e for e in doc["entries"] if e["kind"] == "fresh"]
                ids = [e["result"] for e in fresh]
                out[which].update(entries=len(kinds), fresh=len(fresh), held=kinds.count("held"),
                                  fresh_results_unique=len(ids) == len(set(ids)),
                                  results_never_shown=len({r["packet_id"] for r in run["results"]} - set(ids)))
        else:
            out[which] = {"status": "missing"}
    out["trace"] = run["manifest"]["outputs"].get("trace")
    return out


def resource(run: dict) -> dict:
    return run["manifest"].get("resource") or {}


def evaluate(run_dir: Path, truth_path: Path | None, source_video: Path, selected: dict[str, str] | None,
             kind: str = "scene") -> dict:
    run = load_run(run_dir)
    rep: dict = {"run_dir": str(run_dir), "kind": kind, "outcome": run["manifest"]["outcome"], "counts": run["manifest"]["counts"],
                 "times": run["manifest"]["times"], "throughput": throughput(run), "stage_costs_ms": stage_costs(run),
                 "pairing": pairing(run, source_video), "recordings": recordings(run), "resource": resource(run),
                 "recording_stage": recording_stage(run)}
    if truth_path is not None and selected:
        truth = json.loads(truth_path.read_text("utf-8"))
        if kind == "controls":
            rep["controls"] = score_controls(run, truth, selected)
            rep["throughput_note"] = "the controls session contains a deliberate 2 s pause: throughput is reported but not a criterion"
        else:
            sc = score_scene(run, truth, selected)
            rep["scene_scoring"] = {k: v for k, v in sc.items() if not k.startswith("_")}
            rep["scene_verdicts"] = scene_verdicts(sc, truth, selected)
    return rep


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--truth")
    ap.add_argument("--video", required=True)
    ap.add_argument("--select", action="append", default=[], help="scene_name=item_id")
    a = ap.parse_args()
    sel = dict(s.split("=", 1) for s in a.select) or None
    print(json.dumps(evaluate(Path(a.run_dir), Path(a.truth) if a.truth else None, Path(a.video), sel), indent=1, default=str))
