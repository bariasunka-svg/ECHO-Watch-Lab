# SPDX-License-Identifier: AGPL-3.0-only
"""Recognition scoring against evaluator-only truth (EW-LAB-01 section 6). Never imported by the runtime.

Input: ``results`` = [{"t": media seconds, "items": {name: {"state", "box"}}}] in commit order (from a run's decision
trace or an offline replay) and the scenario truth. A result is CORRECT for an item when the item is Visible with a
box IoU >= 0.5 against that item's true box. A WRONG-OBJECT Visible is a Visible box whose IoU with the item's truth is
< 0.5 (or the item is absent). A LIVE BOX WHILE ABSENT is any boxed state (Visible/Checking) for an absent item.
Denominators are every analysed result inside a window (no result in a window -> FAIL, never vacuous PASS).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from echo_watch.geometry import iou  # noqa: E402

BOXED = ("visible", "checking")


def _truth_at(truth: dict, name: str, t: float):
    i = int(round(t * truth["fps"]))
    i = max(0, min(truth["frames"] - 1, i))
    it = truth["items"][name]
    return it["present"][i], it["box"][i]


def coverage(results, truth, name, lo, hi):
    n = ok = 0
    states: dict[str, int] = {}
    for r in results:
        if not (lo <= r["t"] < hi):
            continue
        o = r["items"].get(name)
        if o is None:
            continue
        n += 1
        states[o["state"]] = states.get(o["state"], 0) + 1
        present, tb = _truth_at(truth, name, r["t"])
        if present and o["state"] == "visible" and o.get("box") and iou(o["box"], tb) >= 0.5:
            ok += 1
    frac = ok / n if n else None
    return {"window": [lo, hi], "results": n, "correct_visible": ok, "fraction": None if frac is None else round(frac, 4),
            "pass": bool(n) and frac >= 0.95, "states": states}


def first_state_after(results, name, t0, want, within):
    for r in results:
        if r["t"] < t0:
            continue
        o = r["items"].get(name)
        if o and o["state"] == want:
            dt = r["t"] - t0
            return {"first_t": round(r["t"], 3), "latency_s": round(dt, 3), "pass": dt <= within}
    return {"first_t": None, "latency_s": None, "pass": False}


def first_correct_visible_after(results, truth, name, t0, within):
    for r in results:
        if r["t"] < t0:
            continue
        o = r["items"].get(name)
        if o and o["state"] == "visible" and o.get("box"):
            present, tb = _truth_at(truth, name, r["t"])
            if present and iou(o["box"], tb) >= 0.5:
                dt = r["t"] - t0
                return {"first_t": round(r["t"], 3), "latency_s": round(dt, 3), "pass": dt <= within}
    return {"first_t": None, "latency_s": None, "pass": False}


def violations(results, truth, names):
    wrong, absent_box, examples = 0, 0, []
    for r in results:
        for name in names:
            o = r["items"].get(name)
            if not o:
                continue
            present, tb = _truth_at(truth, name, r["t"])
            if o["state"] == "visible" and o.get("box") and (not present or iou(o["box"], tb) < 0.5):
                wrong += 1
                if len(examples) < 10:
                    examples.append({"t": r["t"], "item": name, "kind": "wrong_object_visible", "box": o["box"]})
            if o["state"] in BOXED and o.get("box") and not present:
                absent_box += 1
                if len(examples) < 10:
                    examples.append({"t": r["t"], "item": name, "kind": "live_box_while_absent", "state": o["state"]})
    return {"wrong_object_visible": wrong, "live_box_while_absent": absent_box, "examples": examples,
            "pass": wrong == 0 and absent_box == 0}


def state_distribution(results, name, lo, hi):
    out: dict[str, int] = {}
    for r in results:
        if lo <= r["t"] < hi and name in r["items"]:
            s = r["items"][name]["state"]
            out[s] = out.get(s, 0) + 1
    return out


def score_phone_variant(results, truth):
    """D / H1 / H2 (phone_a only)."""
    n = "phone_a"
    out = {"coverage": [coverage(results, truth, n, 3, 8), coverage(results, truth, n, 19, 24)],
           "not_visible_after_removal_8s": first_state_after(results, n, 8.0, "not_seen", 3.0),
           "visible_after_return_16s": first_correct_visible_after(results, truth, n, 16.0, 3.0),
           "violations": violations(results, truth, [n]),
           "twin_segments": {"[24,32) phone_a present with identical twin": state_distribution(results, n, 24, 32),
                             "[32,38) phone_a removed, twin remains": state_distribution(results, n, 32, 38),
                             "[38,45) both removed": state_distribution(results, n, 38, 45)},
           "absent_window_[8,16)": state_distribution(results, n, 8, 16)}
    twin = out["twin_segments"]["[32,38) phone_a removed, twin remains"]
    out["twin_removed_never_visible"] = twin.get("visible", 0) == 0
    out["pass"] = (all(c["pass"] for c in out["coverage"]) and out["not_visible_after_removal_8s"]["pass"]
                   and out["visible_after_return_16s"]["pass"] and out["violations"]["pass"] and out["twin_removed_never_visible"])
    return out


SCENE_WINDOWS = [(5.0, 20.0), (38.0, 50.0), (50.0, 65.0), (65.0, 90.0)]


def score_main_scene(results, truth, names):
    cov = {n: [coverage(results, truth, n, lo, hi) for lo, hi in SCENE_WINDOWS if not (n == "speaker" and lo < 35 <= hi)]
           for n in names}
    out = {"coverage": cov, "movement_window_speaker_[50,65)": coverage(results, truth, "speaker", 50, 65) if "speaker" in names else None,
           "violations": violations(results, truth, names)}
    if "speaker" in names:
        out["speaker_not_visible_after_removal_20s"] = first_state_after(results, "speaker", 20.0, "not_seen", 3.0)
        out["speaker_visible_after_return_35s"] = first_correct_visible_after(results, truth, "speaker", 35.0, 3.0)
    out["pass"] = (all(c["pass"] for v in cov.values() for c in v) and out["violations"]["pass"]
                   and ("speaker" not in names or (out["speaker_not_visible_after_removal_20s"]["pass"]
                                                   and out["speaker_visible_after_return_35s"]["pass"])))
    return out


def score_controls(results, truth, pause_window=None):
    """Old controls: speaker occlusion [10,18) must never be charged as absence; phone_a removal [18,24) with an
    identical twin present must never show Visible on the twin; pause/offline never become absence."""
    occl = state_distribution(results, "speaker", 10.0, 18.0)
    twin = state_distribution(results, "phone_a", 18.0, 24.0)
    out = {"speaker_occlusion_[10,18)": occl, "speaker_occlusion_no_fabricated_absence": occl.get("not_seen", 0) == 0,
           "phone_a_removed_twin_present_[18,24)": twin, "twin_never_visible": twin.get("visible", 0) == 0,
           "phone_a_[24,30)_after_return": state_distribution(results, "phone_a", 24.0, 30.0),
           "violations": violations(results, truth, ["speaker", "phone_a"])}
    out["pass"] = out["speaker_occlusion_no_fabricated_absence"] and out["twin_never_visible"] and out["violations"]["pass"]
    return out
