# SPDX-License-Identifier: AGPL-3.0-only
"""YOLOE workspace inference worker: prompt-free YOLOE-26s detection + explicit BoT-SORT association (EW-LAB-06).

A separate windowless process in the GPU runtime, started and supervised by ``yoloe.client.YoloeWorkerClient`` (the
existing ``WorkerClient`` protocol: one JSON object per line, one request in flight, frames through the client's
shared-memory slot). It loads ONLY the pinned prompt-free checkpoint - never the visual-prompt model, a text encoder or a
ReID model - on the EXPLICIT device/precision (cuda:0 FP32; no CPU fallback), offline, with auto-install disabled and
network/subprocess egress refused by the audit hooks. ``lap`` is checked BEFORE any tracker module is imported: the
installed matching module would otherwise try to install it.

Commands (typed, additive to the Display Assets worker which is unchanged):
  ping / quit
  reset        {epoch}                         new BoT-SORT instance for this tracking epoch (acknowledged cutover)
  detect_track {epoch, frame, accept, shm, shape}
               settles ``accept`` (the previous provisional update) as accepted, then ONE ``model.predict`` on exactly
               this image, every post-model detection kept with its stable index, then ONE explicit tracker update on
               the same image/result. The update stays provisional until settled.
  settle       {epoch, frame, decision}        accept, or reject = restore the exact pre-update tracker state
No callback is registered on the model (``model.track`` is never used), so no predictor state accumulates.

EW-LAB-06 R1: startup runs through ``bootstrap.start`` (one failure boundary, structured ``fatal`` with stage, exception,
bounded traceback; ready only after the receipt checks) and the detector uses the pinned API's canonical FP32 setting
``quantize=None``. ``RuntimeCapabilities`` below is the ONLY place that touches Torch, Ultralytics, CUDA and the model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from . import bootstrap
from .profile import LAP_VERSION, ULTRALYTICS_VERSION, load_tracker_config


def _emit(stream, obj: dict) -> None:
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else x


class ProtocolError(RuntimeError):
    pass


class YoloeCore:
    """Command semantics, independent of Torch (tests drive it with a spy model and a fake tracker)."""

    def __init__(self, model, names, device: str, quantize, tracker_factory, sync=lambda: None):
        if quantize is not None:
            raise ValueError("the YOLOE profile is FP32 (quantize=None)")
        self.model, self.names, self.device = model, names, device
        self.tracker_factory, self.sync = tracker_factory, sync
        self.tracker = None
        self.epoch: str | None = None
        self.pending: str | None = None       # frame key of the unsettled provisional update
        self.predictions = 0

    def predict_kwargs(self) -> dict:
        return bootstrap.predict_kwargs(self.device)

    def detect(self, img) -> tuple[list[dict], dict]:
        t = time.perf_counter()
        results = self.model.predict(img, **self.predict_kwargs())
        self.sync()
        infer_s = time.perf_counter() - t
        self.predictions += 1
        if len(results) != 1:
            raise ProtocolError(f"one image produced {len(results)} results")
        r = results[0]
        dets: list[dict] = []
        b = r.boxes
        if b is not None and len(b):
            xyxy, conf, cls = _np(b.xyxy), _np(b.conf), _np(b.cls)
            for k in range(len(xyxy)):
                c = int(cls[k])
                dets.append({"i": k, "xyxy": [round(float(v), 2) for v in xyxy[k]], "score": round(float(conf[k]), 4),
                             "cls": c, "name": str(self.names[c])})
        speed = getattr(r, "speed", None) or {}
        return dets, {"infer_s": round(infer_s, 5), "speed_ms": {k: round(float(v), 3) for k, v in speed.items()},
                      "max_det_reached": len(dets) >= bootstrap.DETECTOR["max_det"],
                      "masks_computed": getattr(r, "masks", None) is not None}

    def reset(self, epoch: str) -> dict:
        # the previous tracker (settled or not, restorable or not) is discarded whole: nothing of it reaches the epoch
        self.tracker = self.tracker_factory()
        self.epoch, self.pending = epoch, None
        return {"event": "reset", "epoch": epoch, "tracker": self.tracker.identity()}

    def _check_epoch(self, epoch) -> None:
        if self.tracker is None or epoch != self.epoch:
            raise ProtocolError(f"tracking epoch {epoch!r} is not the acknowledged epoch {self.epoch!r}")

    def detect_track(self, req: dict, img) -> dict:
        self._check_epoch(req.get("epoch"))
        if self.pending is not None:
            if req.get("accept") != self.pending:
                raise ProtocolError(f"provisional update {self.pending!r} is not settled")
            self.tracker.settle(True)
            self.pending = None
        elif req.get("accept") is not None:
            raise ProtocolError(f"nothing provisional to accept for {req.get('accept')!r}")
        from .association import detections_array
        dets, meta = self.detect(img)
        tr = self.tracker.update(detections_array(dets), img)
        self.pending = req["frame"]
        return {"event": "tracked", "epoch": self.epoch, "frame": req["frame"], "detections": dets, **meta,
                "tracker": {k: tr[k] for k in ("rows", "tentative", "frame_id", "lost", "removed_this_frame", "warnings",
                                                "tracker_s")},
                "worker_assoc_errors": tr["errors"], "input_shape": list(img.shape[:2])}

    def settle(self, req: dict) -> dict:
        self._check_epoch(req.get("epoch"))
        if self.pending is None or req.get("frame") != self.pending:
            raise ProtocolError(f"frame {req.get('frame')!r} is not the provisional update ({self.pending!r})")
        accept = req.get("decision") == "accept"
        if req.get("decision") not in ("accept", "reject"):
            raise ProtocolError(f"unknown settle decision {req.get('decision')!r}")
        self.tracker.settle(accept)
        self.pending = None
        return {"event": "settled", "epoch": self.epoch, "frame": req["frame"], "decision": req["decision"],
                "restored": not accept, "tracker_updates": self.tracker.updates}


def _preflight() -> str | None:
    import importlib.metadata as md
    import importlib.util
    if importlib.util.find_spec("lap") is None:
        return ("the tracker dependency 'lap' is not installed in the GPU runtime; it is never installed at run time "
                "(see requirements/gpu-tracker.lock)")
    try:
        v = md.version("lap")
    except md.PackageNotFoundError:
        return "the 'lap' distribution metadata is missing"
    if v != LAP_VERSION:
        return f"lap {v} is installed; the pinned version is {LAP_VERSION}"
    u = md.version("ultralytics")
    if u != ULTRALYTICS_VERSION:
        return f"Ultralytics {u} is installed; the pinned version is {ULTRALYTICS_VERSION}"
    return None


class RuntimeCapabilities:
    """The real effects of startup (process guards, files, Torch/Ultralytics, CUDA, the model). Kept thin: the stage
    order, checks and receipt live in ``bootstrap.start``."""

    def install_guards(self, cfg) -> None:
        from echo_watch import guards
        guards.install(block_subprocess=True)
        cfg_dir = Path(cfg.config_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update(YOLO_OFFLINE="1", YOLO_AUTOINSTALL="False", YOLO_CONFIG_DIR=str(cfg_dir),
                          YOLO_VERBOSE="False")

    def preflight(self) -> str | None:
        return _preflight()

    def load_tracker_config(self, path):
        return load_tracker_config(Path(path))

    def file_sha256(self, path) -> str | None:
        p = Path(path)
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

    def import_runtime(self, cfg):
        import numpy as np
        import torch
        import ultralytics
        from ultralytics import YOLOE
        from ultralytics.utils import SETTINGS
        SETTINGS.update(sync=False)
        torch.set_num_threads(max(1, cfg.threads))
        return {"np": np, "torch": torch, "YOLOE": YOLOE, "ultralytics": ultralytics}

    def cuda_problem(self, rt, device: str) -> str | None:
        torch = rt["torch"]
        if not device.startswith("cuda") or not torch.cuda.is_available():
            return f"device {device} requires CUDA in this interpreter (torch {torch.__version__}); no CPU fallback"
        return None

    def load_model(self, rt, path):
        return rt["YOLOE"](str(Path(path)))

    def tracker_factory(self, tcfg):
        from .tracker import BotSortAdapter
        return lambda: BotSortAdapter(tcfg)

    def neutral_image(self):
        import numpy as np
        return np.full((640, 640, 3), 114, np.uint8)

    def synchronize(self, rt) -> None:
        rt["torch"].cuda.synchronize()

    def after_warmup(self, rt, cfg) -> None:
        rt["torch"].set_num_threads(max(1, cfg.threads))   # select_device resets the CPU thread count

    def backend(self, model):
        backend = model.predictor.model                    # AutoBackend; the module it executes is its .model
        return next(backend.model.parameters()), backend

    def tracking_callbacks(self, model) -> bool:
        cb = [getattr(f, "func", f) for f in model.callbacks.get("on_predict_postprocess_end", [])]
        return any("track" in getattr(f, "__module__", "") for f in cb)

    def runtime_facts(self, rt) -> dict:
        torch = rt["torch"]
        i = torch.cuda.current_device()
        free, total = torch.cuda.mem_get_info(i)
        return {"torch": torch.__version__, "ultralytics": rt["ultralytics"].__version__,
                "threads": torch.get_num_threads(), "gpu_name": torch.cuda.get_device_name(i),
                "vram_total_mib": round(total / 2**20), "vram_free_mib_after_load": round(free / 2**20),
                "interpreter": sys.executable}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--tracker-config", required=True)
    ap.add_argument("--tracker-sha256", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", default="fp32", choices=("fp32",))
    a = ap.parse_args()

    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout = sys.stderr                 # library prints never corrupt the protocol stream

    started = bootstrap.start(a, RuntimeCapabilities(), lambda obj: _emit(proto, obj))
    if started is None:
        return 2
    core, _receipt = started
    import numpy as np
    from echo_watch import guards

    shm = {"name": None, "obj": None}

    def frame_from(req):
        from multiprocessing import shared_memory
        if shm["name"] != req["shm"]:
            if shm["obj"] is not None:
                shm["obj"].close()
            shm["obj"], shm["name"] = shared_memory.SharedMemory(name=req["shm"]), req["shm"]
        h, w = req["shape"]
        return np.ndarray((h, w, 3), np.uint8, buffer=shm["obj"].buf[:h * w * 3]).copy()

    for line in sys.stdin:
        recv_wall = time.time()
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd, job = req.get("cmd"), req.get("job_id")
        if cmd == "quit":
            _emit(proto, {"event": "bye", "job_id": job})
            break
        try:
            if cmd == "ping":
                out = {"event": "pong"}
            elif cmd == "reset":
                out = core.reset(req["epoch"])
            elif cmd == "detect_track":
                out = core.detect_track(req, frame_from(req))
            elif cmd == "settle":
                out = core.settle(req)
            else:
                out = {"event": "error", "error": "unknown command"}
            _emit(proto, {**out, "job_id": job, "recv_wall": recv_wall, "done_wall": time.time(),
                          "egress_refusals": len(guards.snapshot()["egress_refusals"])})
        except Exception as e:  # noqa: BLE001 - reported to the controller, never hidden
            _emit(proto, {"event": "error", "job_id": job, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
