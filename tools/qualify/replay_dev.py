# SPDX-License-Identifier: AGPL-3.0-only
"""OFFLINE development replay (CPU, no model): the real ItemEvaluator over the real development detections that a
collect_dev.py session saved, for DEVELOPMENT clips only (D, old main scene, old controls). It checks policy logic
before the source freeze; it is NOT the qualification (full-pipeline real-time sessions are). The collected detections
were produced with four prompts (phone_a, speaker, watch_box, tablet); the replay members are the items a run of that
scenario remembers, and every detection in view is a candidate for them (a stricter setting than one prompt).

  env\\ui\\python.exe -I tools\\qualify\\replay_dev.py --session <dir> --scenario D|scene|controls [--items one|three]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import scoring  # noqa: E402
from common import ROOT, SCENARIOS  # noqa: E402

from echo_watch.engine.contracts import ItemRef  # noqa: E402
from echo_watch.engine.evaluator import ItemEvaluator, prepare_member  # noqa: E402
from echo_watch.frames import Epoch, Frame  # noqa: E402
from echo_watch.matching import default_gate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--scenario", required=True, choices=("D", "scene", "controls"))
    ap.add_argument("--items", default="three")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sdir = ROOT / a.session
    res = json.loads((sdir / "result.json").read_text("utf-8"))
    r = res["scenarios"][a.scenario]
    sc = SCENARIOS[a.scenario]
    names = sc[a.items]
    truth = common.load_truth(sc)
    refs = {x["name"]: x for x in r["refs"]}
    keys = [x["ref_key"] for x in r["refs"]]
    members = {}
    for n in names:
        x = refs[n]
        ref = ItemRef(n, n, "Other", 1, x["ref_key"], tuple(x["box"]), (1280, 720), x["crop_path"], x["context_path"],
                      x["crop_sha256"])
        members[n] = prepare_member(ref)
    class_map = {n: (keys.index(refs[n]["ref_key"]),) for n in names}
    ep = Epoch("replay", "replay", 1, 0.0, 0.0)
    ev = ItemEvaluator(members, (1280, 720), class_map, "replay", {n: True for n in names})
    dets = {}
    with open(sdir / f"detections_{a.scenario}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            dets[d["i"]] = [{"cls": x["cls"], "confidence": x["conf"], "box": x["box"], "ref": x["ref"]} for x in d["dets"]]
    results = []
    pause = sc.get("pause")
    resumed_pending, carried = False, {}
    for i, img in common.decode_frames(sc["video"], sc["fps"], r["every"]):
        t = i / sc["fps"]
        if i not in dets:
            continue
        if pause and pause[0] <= t < pause[0] + pause[1]:
            if not resumed_pending:                      # PAUSE: the session drops its evaluator, carrying ambiguity
                carried = {k: dict(it.ambiguity) for k, it in ev.policy.items.items() if it.ambiguity}
                resumed_pending = True
            continue                                     # the live run is paused here: no evaluation
        if resumed_pending:                              # RESUME: a fresh evaluator (no seeds), carried ambiguity only
            import dataclasses
            ev = ItemEvaluator({k: (dataclasses.replace(m, ambiguity=carried[k]) if k in carried else m)
                                for k, m in members.items()}, (1280, 720), class_map, "replay", {n: False for n in names})
            resumed_pending = False
        f = Frame(ep, i + 1, t, t, np.ascontiguousarray(img))
        fork = ev.fork()
        e = fork.evaluate(f, dets[i])
        ev.adopt(fork)
        results.append({"t": t, "items": {n: {"state": d.status, "box": d.box, "reason": d.reason, "app": d.app,
                                                "admission": d.admission} for n, d in e.decisions.items()}})
    if a.scenario == "D":
        score = scoring.score_phone_variant(results, truth)
    elif a.scenario == "scene":
        score = scoring.score_main_scene(results, truth, names)
    else:
        score = scoring.score_controls(results, truth)
    out = {"scenario": a.scenario, "items": names, "gate": default_gate().to_dict(), "results": len(results),
           "src_tree_sha256": common.src_digest(), "score": score, "kind": "OFFLINE DEVELOPMENT REPLAY (not qualification; Pause/Resume mirrored with a fresh evaluator)"}
    if a.out:
        p = ROOT / a.out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1, default=str), "utf-8")
        (p.with_suffix(".states.jsonl")).write_text("\n".join(json.dumps(x, default=str) for x in results), "utf-8")
    print(json.dumps(out["score"], indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
