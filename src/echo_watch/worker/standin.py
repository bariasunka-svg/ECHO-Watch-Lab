# SPDX-License-Identifier: AGPL-3.0-only
"""DETERMINISTIC STAND-IN WORKER for lifecycle tests and the configuration/lifecycle smoke ONLY.

It speaks the real worker protocol (ready / scan / set_refs / verify / ping / quit, shared-memory frame slot) but runs NO
model: it returns detections scripted by a JSON spec. Anything it produces is test material and is never evidence of
recognition, discovery or monitoring quality; qualification sessions refuse to start with a stand-in (the client's spec
must name the real ``echo_watch.worker.main`` command).

Spec (``--spec FILE``): {"scan": [{"box": [x,y,w,h], "label": str, "confidence": f}],
                         "verify": [{"seq_from": a, "seq_to": b, "delay_s": 0.0, "fail": false,
                                     "items": [{"cls": 0, "box": [x,y,w,h], "confidence": 0.9}]}]}
A verify request whose frame sequence matches no rule returns no detections. ``delay_s`` simulates a slow reply (used by
the Resume-boundary regression); ``fail`` makes the worker exit (used by the worker-death regression).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    a = ap.parse_args()
    spec = json.load(open(a.spec, "r", encoding="utf-8"))
    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout = sys.stderr

    def emit(o: dict) -> None:
        proto.write(json.dumps(o) + "\n")
        proto.flush()

    emit({"event": "ready", "device": spec.get("device", "cuda:0"), "requested_device": spec.get("device", "cuda:0"),
          "precision": "fp32", "torch": "stand-in", "interpreter": sys.executable, "worker_pid": os.getpid(),
          "model_sha256": "stand-in", "stand_in": True, "load_s": 0.0})
    classes: list = []
    for line in sys.stdin:
        recv = time.time()
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd, job = req.get("cmd"), req.get("job_id")
        if cmd == "quit":
            emit({"event": "bye", "job_id": job})
            break
        if cmd == "ping":
            emit({"event": "pong", "job_id": job, "recv_wall": recv, "done_wall": time.time()})
        elif cmd == "scan":
            emit({"event": "result", "job_id": job, "candidates": [
                {"box": c["box"], "polygon": None, "label": c.get("label", "object"), "confidence": c.get("confidence", 0.9)}
                for c in spec.get("scan", [])], "infer_s": 0.001, "device": spec.get("device", "cuda:0"),
                "precision": "fp32", "recv_wall": recv, "done_wall": time.time()})
        elif cmd == "set_refs":
            classes = []
            for ref in req["refs"]:
                for k in range(len(ref["boxes"])):
                    classes.append([ref["key"], k])
            emit({"event": "refs", "job_id": job, "classes": classes, "computed": [r["key"] for r in req["refs"]],
                  "seconds": 0.001, "warm_s": 0.0, "recv_wall": recv, "done_wall": time.time()})
        elif cmd == "verify":
            try:
                seq = int(str(req.get("frame_ref", "x:0")).rsplit(":", 1)[1])
            except (ValueError, IndexError):
                seq = 0
            dets, delay, fail = [], 0.0, False
            for rule in spec.get("verify", []):
                if rule["seq_from"] <= seq <= rule["seq_to"]:
                    delay, fail = rule.get("delay_s", 0.0), rule.get("fail", False)
                    for it in rule.get("items", []):
                        cls = it["cls"]
                        ref = classes[cls] if cls < len(classes) else ["?", 0]
                        dets.append({"cls": cls, "ref": ref, "confidence": it.get("confidence", 0.9), "box": it["box"],
                                     "polygon": None})
            if fail:
                return 7
            if delay:
                time.sleep(delay)
            emit({"event": "verified", "job_id": job, "detections": dets, "infer_s": delay or 0.001, "post_s": 0.0,
                  "frame_ref": req.get("frame_ref"), "device": spec.get("device", "cuda:0"), "precision": "fp32",
                  "recv_wall": recv, "done_wall": time.time()})
        else:
            emit({"event": "error", "job_id": job, "error": "unknown command"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
