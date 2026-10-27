# SPDX-License-Identifier: AGPL-3.0-only
"""Development-detection collection for the descriptor calibration (EW-LAB-01 section 5). ONE owned GPU session.

For each DEVELOPMENT clip (D, the old main scene, the old controls; never H1/H2): retain its first frame, run the real
prompt-free Scan, select phone_a / speaker / watch_box / tablet among the REAL proposals (evaluator hint = frame-0 truth
region, IoU >= 0.3), Remember them in a qualification-only store, have the worker acknowledge those visual-prompt
references exactly as a run would, then run the real visual-prompt verify on every ``--every``-th decoded frame
(not real time) and save every detection. Truth is NOT used here beyond the selection hint; scoring happens offline in
calibrate.py. Owned, windowless, admitted, deadline 240 s including model startup.

  env\\ui\\python.exe -I tools\\qualify\\collect_dev.py --label C1 --closeout closeouts\\<id>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from common import ROOT, SCENARIOS  # noqa: E402

from echo_watch.config import Settings  # noqa: E402
from echo_watch.engine.api import ItemEngine  # noqa: E402
from echo_watch.ownedjob import OwnedJob  # noqa: E402

DEV = [("D", 1), ("controls", 1), ("scene", 2)]
NAMES = ["phone_a", "speaker", "watch_box", "tablet"]
HARD_DEADLINE_S = 240.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--closeout", required=True)
    a = ap.parse_args()
    t0 = time.monotonic()
    co = ROOT / a.closeout
    out = co / "sessions" / a.label
    out.mkdir(parents=True, exist_ok=False)
    res = {"label": a.label, "kind": "development detection collection", "started": common.now(), "pid": os.getpid(),
           "src_tree_sha256": common.src_digest(), "scenarios": {}}

    def save(status):
        res.update(status=status, total_wall_s=round(time.monotonic() - t0, 2), ended=common.now())
        (out / "result.json").write_text(json.dumps(res, indent=1, default=str), "utf-8")

    job = OwnedJob()
    job.adopt_current_process()
    adm = common.admission(out)
    res["admission"] = {k: adm.get(k) for k in ("admitted", "refusal_reasons", "machine_cpu_mean", "machine_cpu_p95",
                                                "nvml_free_mib_min")}
    if not adm.get("admitted"):
        save("REFUSED_BY_ADMISSION")
        print(json.dumps(res["admission"], indent=1))
        return 10
    data = ROOT / "data" / "qualification" / a.label
    s = Settings(data_dir=data, device="cuda:0", threads=4)
    eng = ItemEngine(s)
    wt0, ww0 = time.monotonic(), time.time()
    done = {"v": False}

    def watchdog():
        while not done["v"]:
            if time.monotonic() - t0 > HARD_DEADLINE_S:
                res["error"] = "hard deadline reached; owned job terminated"
                save("ABORTED_HARD_DEADLINE")
                job.terminate()
                os._exit(99)
            time.sleep(0.5)
    threading.Thread(target=watchdog, daemon=True).start()
    status = "EXCEPTION"
    try:
        res["pinned"] = eng.verify_pinned_artifacts()
        if not all(v["ok"] for v in res["pinned"].values()):
            save("REFUSED_PINNED_HASH")
            return 13
        rcpt = eng.start_worker()
        res["worker_receipt"] = rcpt
        if rcpt.get("event") != "ready":
            save("WORKER_UNAVAILABLE")
            return 14
        for scn, every in DEV:
            sc = SCENARIOS[scn]
            truth = common.load_truth(sc)
            r: dict = {"video": str(sc["video"]), "video_sha256": common.sha256_file(sc["video"]), "every": every}
            frame = eng.open_setup_source(str(sc["video"]), sc["fps"], source_id=f"dev-{scn}")
            scan = eng.scan(frame, f"dev-{scn}")
            r["scan"] = {"scan_id": scan.scan_id, "status": scan.status, "proposals": len(scan.proposals),
                         "infer_s": scan.infer_s, "device": scan.device, "frame_sha256": scan.frame_sha256,
                         "boxes": [[round(v, 1) for v in p.box] + [p.label, p.confidence, p.duplicate_of] for p in scan.proposals]}
            picks, report = common.pick(scan, NAMES, truth)
            r["selection"] = report
            if len(picks) != len(NAMES):
                r["error"] = "discovery did not propose every requested item"
                res["scenarios"][scn] = r
                continue
            refs = eng.remember(f"dev-{a.label}-{scn}", scan.scan_id, picks)
            digest = hashlib.sha256(json.dumps(sorted((x.ref_key, x.crop_sha256) for x in refs)).encode()).hexdigest()
            payload = [{"key": x.ref_key, "image_path": x.context_path,
                        "boxes": [[x.box[0], x.box[1], x.box[0] + x.box[2], x.box[1] + x.box[3]]]} for x in refs]
            ack = eng.worker.set_refs(payload, digest)
            r["refs"] = [{"name": x.name, "item_id": x.item_id, "ref_key": x.ref_key, "box": list(x.box),
                          "crop_path": x.crop_path, "crop_sha256": x.crop_sha256, "context_path": x.context_path} for x in refs]
            r["refs_ack"] = {"classes": ack.get("classes"), "seconds": ack.get("seconds")}
            n = 0
            tv = time.monotonic()
            with open(out / f"detections_{scn}.jsonl", "w", encoding="utf-8") as fh:
                for i, img in common.decode_frames(sc["video"], sc["fps"], every):
                    if time.monotonic() - t0 > HARD_DEADLINE_S - 15:
                        r["truncated_at_frame"] = i
                        break
                    rep = eng.worker.verify(img, f"{scn}:{i}", s.limits.request_timeout_s, s.verify_imgsz, s.verify_conf,
                                            s.verify_iou, s.verify_max_det)
                    fh.write(json.dumps({"i": i, "infer_s": rep.get("infer_s"), "device": rep.get("device"),
                                         "dets": [{"cls": d["cls"], "ref": d["ref"][0], "conf": d["confidence"],
                                                   "box": [round(v, 2) for v in d["box"]]} for d in rep["detections"]]}) + "\n")
                    n += 1
            r["verified_frames"] = n
            r["verify_wall_s"] = round(time.monotonic() - tv, 2)
            res["scenarios"][scn] = r
        status = "COMPLETED"
    except Exception:  # noqa: BLE001 - preserved, never hidden
        import traceback
        res["exception"] = traceback.format_exc()
    finally:
        done["v"] = True
        eng.shutdown()
        secs = time.monotonic() - wt0
        left = job.leftovers(exclude=(os.getpid(),))
        res["worker_process_wall_s"] = round(secs, 2)
        res["leftover_owned_processes_after_shutdown"] = left
        if left:
            job.terminate()
        common.ledger(co, {"label": a.label, "kind": "dev-collection", "wall_start": ww0, "seconds": round(secs, 2)})
    save(status)
    print(json.dumps({k: res.get(k) for k in ("status", "worker_process_wall_s", "total_wall_s")}, indent=1))
    for k, v in res["scenarios"].items():
        print(k, v.get("scan", {}).get("proposals"), v.get("verified_frames"), v.get("error"), v.get("verify_wall_s"))
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
