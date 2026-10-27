# SPDX-License-Identifier: AGPL-3.0-only
"""Re-score saved runs offline with the CURRENT evaluator (no model, no device). A previous evaluation.json is never
overwritten: it is kept as evaluation.v<N>.json before the new one is written.

  env\\cpu\\python.exe -I tools\\qualify\\reevaluate.py LABEL [LABEL ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "qualify"))
import evaluate  # noqa: E402
from run_session import SCENARIOS  # noqa: E402

CLOSEOUT = ROOT / "closeouts" / "patsight-parity-foundation-20261025-193700"


def main() -> int:
    for lab in sys.argv[1:]:
        d = CLOSEOUT / "sessions" / lab
        res = json.loads((d / "result.json").read_text("utf-8"))
        oc = res.get("outcome")
        if not oc:
            print(lab, "no outcome; nothing to re-score")
            continue
        sc = SCENARIOS[res["scenario"]]
        names = sc[res["items_case"]]
        selected = {n: res["items"][n] for n in names} if sc["truth"] is not None else None
        ev = evaluate.evaluate(Path(oc["run_dir"]), sc["truth"], sc["video"], selected,
                               kind="controls" if res["scenario"] == "controls" else "scene")
        old = d / "evaluation.json"
        if old.is_file():
            n = 1
            while (d / f"evaluation.v{n}.json").exists():
                n += 1
            old.replace(d / f"evaluation.v{n}.json")
        (d / "evaluation.json").write_text(json.dumps(ev, indent=1, default=str), "utf-8")
        print(lab, "re-scored;", "previous kept")
    return 0


if __name__ == "__main__":
    sys.exit(main())
