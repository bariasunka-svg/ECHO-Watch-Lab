# SPDX-License-Identifier: AGPL-3.0-only
"""Offline calibration of the absolute admission gate from REAL development detections (EW-LAB-01 section 5).

Inputs: a collect_dev.py session (real Scan selections, acknowledged references, real visual-prompt detections on D and
the old main/controls clips). Truth is evaluator input only: it labels each detection (IoU >= 0.5 with a present item's
true box) as GENUINE for a reference, a SAME-KIND LOOK-ALIKE (another phone unit for the phone reference: identical or
same-product twins are ambiguity controls, never separable negatives), a DISTINCT-OBJECT DISTRACTOR, or UNMATCHED.

The rule is frozen by the package: P = min over references of the 5th percentile of genuine scores; N = max
distinct-object distractor score; qualified only if P - N >= 0.05; M = (P + N) / 2. No per-object exceptions.
Missed detections and the full per-item/time denominators are reported. Unmatched proposals are reported separately.

  env\\ui\\python.exe -I tools\\qualify\\calibrate.py --session <dir> --out <json> [--freeze]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ROOT, SCENARIOS  # noqa: E402

from echo_watch.geometry import clamp_box, iou  # noqa: E402
from echo_watch.item_policy import MemoryParams  # noqa: E402
from echo_watch.matching import DESCRIPTOR_VERSION, MIN_SEPARATION, appearance, reference_appearance  # noqa: E402
from echo_watch.metrics import percentile  # noqa: E402

KIND = {"phone_a": "phone", "phone_b": "phone", "speaker": "speaker", "watch_box": "box", "headphones": "headphones",
        "tablet": "tablet"}


def stats(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "min": round(min(xs), 4), "p01": round(percentile(xs, 1), 4), "p05": round(percentile(xs, 5), 4),
            "p50": round(percentile(xs, 50), 4), "p95": round(percentile(xs, 95), 4), "max": round(max(xs), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()
    sdir = ROOT / a.session
    res = json.loads((sdir / "result.json").read_text("utf-8"))
    conf_min = MemoryParams().conf_min
    genuine: dict[str, list] = {}
    distract: dict[str, dict[str, list]] = {}
    lookalike: dict[str, list] = {}
    unmatched: dict[str, list] = {}
    denominators: dict = {}
    worst: dict = {}
    per_scn = {}
    for scn, r in res["scenarios"].items():
        if "refs" not in r:
            continue
        sc = SCENARIOS[scn]
        truth = common.load_truth(sc)
        refs = {}
        for x in r["refs"]:
            crop_img = cv2.imread(x["crop_path"], cv2.IMREAD_COLOR)
            refs[x["name"]] = (reference_appearance(crop_img), (x["box"][2], x["box"][3]), x["ref_key"])
        dets = {}
        with open(sdir / f"detections_{scn}.jsonl", encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                dets[d["i"]] = d["dets"]
        den = {n: {"present_frames": 0, "frames_with_matched_detection": 0} for n in truth["items"]}
        n_dets = 0
        for i, img in common.decode_frames(sc["video"], sc["fps"], r["every"]):
            if i not in dets:
                continue
            h, w = img.shape[:2]
            present = {n: truth["items"][n]["box"][i] for n in truth["items"] if truth["items"][n]["present"][i]}
            matched_names = set()
            for d in dets[i]:
                box = list(clamp_box(d["box"], w, h))
                if box[2] < 4 or box[3] < 4 or d["conf"] < conf_min:
                    continue
                n_dets += 1
                best, bo = None, 0.0
                for n, tb in present.items():
                    o = iou(box, tb)
                    if o > bo:
                        best, bo = n, o
                obj = best if bo >= 0.5 else None
                if obj:
                    matched_names.add(obj)
                for rn, (feat, size, _k) in refs.items():
                    s = float(appearance(img, box, feat, size))
                    rec = (s, scn, i, [round(v, 1) for v in box], d["conf"], obj)
                    if obj == rn:
                        genuine.setdefault(rn, []).append(s)
                        cur = worst.setdefault(rn, {}).get("genuine_min")
                        if cur is None or s < cur[0]:
                            worst[rn]["genuine_min"] = rec
                    elif obj is None:
                        unmatched.setdefault(rn, []).append(s)
                        cur = worst.setdefault(rn, {}).get("unmatched_max")
                        if cur is None or s > cur[0]:
                            worst[rn]["unmatched_max"] = rec
                    elif KIND[obj] == KIND[rn]:
                        lookalike.setdefault(rn, []).append(s)
                    else:
                        distract.setdefault(rn, {}).setdefault(obj, []).append(s)
                        cur = worst.setdefault(rn, {}).get("distractor_max")
                        if cur is None or s > cur[0]:
                            worst[rn]["distractor_max"] = rec
            for n in present:
                den[n]["present_frames"] += 1
                if n in matched_names:
                    den[n]["frames_with_matched_detection"] += 1
        denominators[scn] = den
        per_scn[scn] = {"detections_scored": n_dets, "every": r["every"]}
    P_by_ref = {rn: percentile(v, 5) for rn, v in genuine.items()}
    N_by_ref = {rn: max(max(v) for v in d.values()) for rn, d in distract.items()}
    P = min(P_by_ref.values())
    N = max(N_by_ref.values())
    sep = P - N
    qualified = sep >= MIN_SEPARATION
    doc = {"descriptor": DESCRIPTOR_VERSION, "rule": "P=min_ref p05(genuine); N=max distinct-object distractor; "
           "qualified iff P-N>=0.05; M=(P+N)/2; reject < M-0.025; support > M+0.025; else unresolved",
           "session": a.session, "session_src_tree": res.get("src_tree_sha256"), "conf_min": conf_min,
           "P": round(P, 6), "N": round(N, 6), "separation": round(sep, 6), "qualified": qualified,
           "M": round((P + N) / 2, 6), "P_by_reference": {k: round(v, 6) for k, v in P_by_ref.items()},
           "N_by_reference": {k: round(v, 6) for k, v in N_by_ref.items()},
           "genuine": {k: stats(v) for k, v in genuine.items()},
           "distractors": {k: {o: stats(v) for o, v in d.items()} for k, d in distract.items()},
           "same_kind_lookalikes_excluded_from_N": {k: stats(v) for k, v in lookalike.items()},
           "unmatched_proposals": {k: stats(v) for k, v in unmatched.items()},
           "worst_cases": worst, "denominators": denominators, "per_scenario": per_scn}
    out = ROOT / a.out
    out.write_text(json.dumps(doc, indent=1, default=str), "utf-8")
    print(json.dumps({k: doc[k] for k in ("P", "N", "separation", "qualified", "M", "P_by_reference", "N_by_reference")}, indent=1))
    for rn in genuine:
        print(rn, "genuine", doc["genuine"][rn])
        for o, st in doc["distractors"].get(rn, {}).items():
            print("   vs", o, st)
        if rn in lookalike:
            print("   lookalike", doc["same_kind_lookalikes_excluded_from_N"][rn])
        if rn in unmatched:
            print("   unmatched", doc["unmatched_proposals"][rn])
    if a.freeze:
        cal = {"descriptor": DESCRIPTOR_VERSION, "P": doc["P"], "N": doc["N"], "qualified": qualified,
               "M": doc["M"], "source": a.out, "source_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
               "note": "Frozen global calibration (EW-LAB-01). No per-object exceptions."}
        cp = ROOT / "src" / "echo_watch" / "calibration" / "admission.json"
        cp.parent.mkdir(parents=True, exist_ok=True)
        if cp.exists():
            print("REFUSED: a frozen calibration already exists; preserved")
            return 2
        cp.write_text(json.dumps(cal, indent=1), "utf-8")
        print("FROZE", cp, hashlib.sha256(cp.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
