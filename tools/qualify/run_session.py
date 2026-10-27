# SPDX-License-Identifier: AGPL-3.0-only
"""One owned, bounded qualification session through the REAL product engine path (real Scan, Remember, managed Start,
visual-prompt monitoring, recording, trace, Stop/EOF finalization), scored afterwards by evaluate.py.

Adapted from the Native foundation archive tools/qualify/run_session.py (tree 3d14d650). Lab changes: Lab scenarios
(D/H1/H2 added), the 10 s admission in a windowless env/gpu child, a resource guard fed by worker-reported VRAM, the
corrected scorers (evaluate.py) and scoring.py recognition criteria. The coordinator is the product UI runtime (env/ui:
no Torch); inference is the env/gpu worker only.

Owned and bounded: the process adopts itself into a kill-on-close Job Object, every child is windowless, the whole
session has a hard wall-clock deadline (<= 240 s including model startup and cleanup), 1 Hz resource guard (machine CPU
> 85% for 10 samples or free VRAM < 20% for 3), and an admission refusal is a receipt (no retry, no waiter). The
operator-like selection takes the best-overlapping REAL Scan proposal for a hint region; nothing is injected.

  env\\ui\\python.exe -I tools\\qualify\\run_session.py --label Q1 --scenario scene --items three --closeout closeouts\\<id>
"""
from __future__ import annotations

import argparse
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
from echo_watch.engine.contracts import CaptureOptions  # noqa: E402
from echo_watch.engine.session import SourceSpec  # noqa: E402
from echo_watch.matching import default_gate  # noqa: E402
from echo_watch.metrics import ResourceSampler  # noqa: E402
from echo_watch.ownedjob import OwnedJob  # noqa: E402

HARD_DEADLINE_S = 240.0
SOFT_DEADLINE_S = 232.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--items", required=True, choices=("one", "three"))
    ap.add_argument("--closeout", required=True)
    a = ap.parse_args()
    t_start = time.monotonic()
    sc = SCENARIOS[a.scenario]
    co = ROOT / a.closeout
    out = co / "sessions" / a.label
    out.mkdir(parents=True, exist_ok=False)
    res: dict = {"label": a.label, "scenario": a.scenario, "role": sc["role"], "items_case": a.items, "device": "cuda:0",
                 "started": common.now(), "pid": os.getpid(), "src_tree_sha256": common.src_digest(),
                 "gate": default_gate().to_dict(), "coordinator": sys.executable}

    def save(status: str) -> None:
        res.update(status=status, total_wall_s=round(time.monotonic() - t_start, 2), ended=common.now())
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
    s = Settings(data_dir=data, device="cuda:0", threads=4, record_fps=sc["fps"])
    eng = ItemEngine(s)
    worker_t0, worker_wall0 = time.monotonic(), time.time()
    stop_flag = {"v": False}

    def watchdog() -> None:
        while not stop_flag["v"]:
            el = time.monotonic() - t_start
            if el > SOFT_DEADLINE_S and eng.session.busy:
                eng.session.stop(error="session deadline reached", ended_by="abort")
            if el > HARD_DEADLINE_S:
                res["error"] = "hard deadline reached; owned job terminated"
                save("ABORTED_HARD_DEADLINE")
                job.terminate()
                os._exit(99)
            time.sleep(0.5)
    threading.Thread(target=watchdog, name="deadline", daemon=True).start()
    sampler = None
    status = "EXCEPTION"
    try:
        res["pinned"] = eng.verify_pinned_artifacts()
        if not all(v["ok"] for v in res["pinned"].values()):
            save("REFUSED_PINNED_HASH")
            return 13
        t = time.monotonic()
        rcpt = eng.start_worker()
        res["worker_start_s"] = round(time.monotonic() - t, 2)
        res["worker_receipt"] = rcpt
        if rcpt.get("event") != "ready":
            save("WORKER_UNAVAILABLE")
            return 14
        frame = eng.open_setup_source(str(sc["video"]), sc["fps"])
        from echo_watch.sources.fingerprint import fingerprint
        scope = fingerprint(sc["video"])["scope"]          # EW-LAB-05: Start requires the profile's content scope
        scan = eng.scan(frame, scope)
        res["scan"] = {"status": scan.status, "proposals": len(scan.proposals), "duration_s": round(scan.duration_s, 3),
                       "infer_s": scan.infer_s, "device": scan.device, "frame_sha256": scan.frame_sha256,
                       "proposal_boxes": [[round(v, 1) for v in p.box] + [p.label, p.confidence, p.duplicate_of]
                                          for p in scan.proposals]}
        names = sc[a.items]
        truth = common.load_truth(sc)
        picks, report = common.pick(scan, names, truth, sc.get("hints"))
        res["selection"] = report
        if len(picks) != len(names):
            res["error"] = "discovery did not propose every requested item (IoU >= 0.3)"
            save("DISCOVERY_INCOMPLETE")
            return 15
        refs = eng.remember(f"qual-{a.label}", scan.scan_id, picks)
        item_ids = {r.name: r.item_id for r in refs}
        res["items"] = item_ids
        sampler = ResourceSampler(lambda: job.pids(), on_sample=lambda x: eng.session._t("resource", **x),
                                  on_abort=lambda why: eng.session.stop(error=f"aborted by the resource guard: {why}",
                                                                        ended_by="abort"),
                                  gpu_fn=lambda: eng.worker.last_vram)
        eng.session.sampler = sampler
        sampler.start()
        t = time.monotonic()
        eng.start(CaptureOptions(), [r.item_id for r in refs], SourceSpec(scope, str(sc["video"]), sc["fps"]))
        eng.session.start_driver()
        if "pause" in sc:
            pa, pd = sc["pause"]
            end = time.monotonic() + 120
            while time.monotonic() < end:
                lp = eng.latest_result()
                if lp is not None and (lp.media or {}).get("media_pts", 0) >= pa and eng.session.phase == "active":
                    eng.pause()
                    time.sleep(pd)
                    eng.resume()
                    res["pause_resume"] = {"paused_at_media_s": pa, "paused_for_s": pd}
                    break
                time.sleep(0.02)
        outcome = eng.session.wait_outcome(max(5.0, SOFT_DEADLINE_S + 3 - (time.monotonic() - t_start)))
        res["start_to_outcome_s"] = round(time.monotonic() - t, 2)
        sampler.stop()
        res["outcome"] = None if outcome is None else {
            "kind": outcome.kind, "ended_by": outcome.ended_by, "complete": outcome.complete, "error": outcome.error,
            "run_id": outcome.run_id, "run_dir": outcome.run_dir, "committed": outcome.results_committed,
            "discards": outcome.discarded, "clean": outcome.clean.status, "annotated": outcome.annotated.status,
            "trace": outcome.trace.status}
        if outcome is None:
            save("NO_OUTCOME")
            return 16
        import evaluate
        name_of = {v: k for k, v in item_ids.items()}
        evl = evaluate.evaluate(Path(outcome.run_dir), a.scenario, sc.get("truth"), sc["video"], name_of if truth else None,
                                names)
        (out / "evaluation.json").write_text(json.dumps(evl, indent=1, default=str), "utf-8")
        res["evaluation_file"] = str(out / "evaluation.json")
        res["throughput"] = {k: evl["throughput"].get(k) for k in (
            "status", "mean_results_per_s", "fraction_bins_ge5", "capture_to_commit_ms_p95", "max_unexplained_gap_s",
            "results_committed", "requested_to_active_s", "trailing_gap_s")}
        res["pairing"] = {k: evl["pairing"].get(k) for k in ("status", "results", "fully_evidenced", "problems")}
        res["recognition_pass"] = (evl.get("recognition") or {}).get("pass")
        res["device_counts"] = evl.get("device_counts")
        status = "COMPLETED" if outcome.complete else "COMPLETED_WITH_OUTPUT_PROBLEMS"
    except Exception:  # noqa: BLE001 - preserved, never hidden
        import traceback
        res["exception"] = traceback.format_exc()
    finally:
        stop_flag["v"] = True
        if sampler is not None:
            sampler.stop()
        eng.shutdown()
        worker_secs = time.monotonic() - worker_t0
        left = job.leftovers(exclude=(os.getpid(),))
        res["worker_process_wall_s"] = round(worker_secs, 2)
        res["leftover_owned_processes_after_shutdown"] = left
        if left:
            job.terminate()
        common.ledger(co, {"label": a.label, "kind": "qualification", "scenario": a.scenario, "items": a.items,
                           "wall_start": worker_wall0, "seconds": round(worker_secs, 2)})
    save(status)
    print(json.dumps({k: res.get(k) for k in ("label", "status", "worker_start_s", "outcome", "throughput", "pairing",
                                              "recognition_pass", "total_wall_s",
                                              "leftover_owned_processes_after_shutdown")}, indent=1, default=str))
    return 0 if status.startswith("COMPLETED") else 1


if __name__ == "__main__":
    sys.exit(main())
