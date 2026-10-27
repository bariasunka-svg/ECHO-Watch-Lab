# SPDX-License-Identifier: AGPL-3.0-only
"""Print each item's committed state transitions (media time, frame, state, reason, box) from a run's decision trace."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1])
    last: dict[str, tuple] = {}
    n = 0
    for line in open(run / "trace.jsonl", "r", encoding="utf-8"):
        r = json.loads(line)
        if r["k"] != "result":
            continue
        n += 1
        t = (r.get("media") or {}).get("media_pts")
        for o in r["items"]:
            cur = (o["state"], o["reason"] if o["state"] not in ("checking",) else None)
            if last.get(o["item_id"]) != cur:
                last[o["item_id"]] = cur
                box = None if o["box"] is None else [round(v) for v in o["box"]]
                print(f"{o['item_id'][-6:]} t={t if t is None else round(t, 2):>6} seq={r['seq']:>4} {o['state']:<13} {o['reason'] or ''} box={box} "
                      f"dets={len(r['detections'])}")
    print("results", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
