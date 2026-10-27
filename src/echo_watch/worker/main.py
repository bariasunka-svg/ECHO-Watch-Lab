# SPDX-License-Identifier: AGPL-3.0-only
"""Inference worker process: prompt-free YOLOE-26s broad scan and YOLOE-26s visual-prompt verification on an EXPLICIT
device (cpu or cuda:0) and precision.

Adapted from the preserved ECHO Watch reference runtime/echowatch/discovery_worker.py (SHA-256
9c6d07541590aefdb6ef757d0c2717dd45e05b2b2179b3d2a080d1f832cf2d0e; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md). Protocol, model
loading, device/precision verification, reference embedding cache and hash checks are unchanged. Adaptations:
* package names (``echo_watch``); the instance-mask transport (``masks``) is not used - the foundation never draws or
  propagates outlines, so ``verify`` never builds or transmits masks;
* ``verify`` returns polygons only when asked (``polygons``: default False): ``r.masks.xy`` is CPU work per detection
  that the paired-result path does not need;
* wall-clock stamps (``recv_wall``/``done_wall``, ``time.time()`` on the same machine) and ``post_s`` are added so the
  controller can attribute round-trip time to IPC, inference and post-processing separately;
* ``set_refs`` warms the visual-prompt predictor once (a neutral image) so the first real frame is steady-state;
* ``worker_pid`` and the pinned python/torch identities are reported in the ready receipt.

Runs as a separate windowless process so model memory, the CUDA context and stalls are contained and work can be
cancelled by terminating it. One worker owns at most the two selected models. Protocol: one JSON object per line on
stdin/stdout; diagnostics go to stderr. Frames arrive through one named shared-memory slot owned by the controller.
Network egress and child processes are refused by audit hooks; Ultralytics is configured offline with no auto-install
and a project-local settings directory. No text encoder is ever loaded: visual prompts use the pinned
YOLOEVPSegPredictor get_vpe/set_classes contract. The ready receipt reports the ACTUAL device of the model the
predictor runs; nothing is hard-coded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _emit(stream, obj: dict) -> None:
    stream.write(json.dumps(obj) + "\n")
    stream.flush()


def _sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--config-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")             # "cpu" or "cuda:0"; never silently changed
    ap.add_argument("--precision", default="fp32", choices=("fp32", "fp16"))
    ap.add_argument("--vp-model", default=None)            # optional visual-prompt checkpoint
    ap.add_argument("--vp-sha256", default=None)
    a = ap.parse_args()

    proto = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout = sys.stderr  # library prints never corrupt the protocol stream

    from echo_watch import guards
    guards.install(block_subprocess=True)

    cfg = Path(a.config_dir)
    cfg.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_OFFLINE="1", YOLO_AUTOINSTALL="False", YOLO_CONFIG_DIR=str(cfg), YOLO_VERBOSE="False")

    t0 = time.perf_counter()
    model_path = Path(a.model)
    digest = _sha(model_path)
    if digest != a.sha256:
        _emit(proto, {"event": "fatal", "error": f"model hash mismatch or missing: {model_path} ({digest})"})
        return 2
    vp_path = Path(a.vp_model) if a.vp_model else None
    if vp_path is not None and _sha(vp_path) != a.vp_sha256:
        _emit(proto, {"event": "fatal", "error": f"visual-prompt model hash mismatch or missing: {vp_path}"})
        return 2

    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLOE
    from ultralytics.utils import SETTINGS

    SETTINGS.update(sync=False)
    torch.set_num_threads(max(1, a.threads))
    cuda = a.device.startswith("cuda")
    if cuda and not torch.cuda.is_available():
        _emit(proto, {"event": "fatal", "error": f"device {a.device} requested but CUDA is unavailable in this "
                      f"interpreter (torch {torch.__version__}); no CPU fallback"})
        return 3
    if not cuda and a.device != "cpu":
        _emit(proto, {"event": "fatal", "error": f"unsupported device {a.device}"})
        return 3
    half = a.precision == "fp16"
    if half and not cuda:
        _emit(proto, {"event": "fatal", "error": "fp16 is only an admitted profile on CUDA"})
        return 3
    model = YOLOE(str(model_path))
    names = model.names
    from echo_watch.geometry import simplify_polygon

    def predict(m, img, **kw):
        return m.predict(img, device=a.device, half=half, verbose=False, retina_masks=False, **kw)[0]

    def pin_threads() -> None:
        # Ultralytics' select_device resets torch's CPU thread count to its own NUM_THREADS whenever a CPU predictor is
        # built; the configured count must hold, so it is re-applied after every predictor construction.
        torch.set_num_threads(max(1, a.threads))

    # warm-up on a neutral image, then read the device the predictor actually runs on
    warm = np.full((640, 640, 3), 114, np.uint8)
    predict(model, warm, imgsz=640, conf=0.25)
    pin_threads()
    if cuda:
        torch.cuda.synchronize()
    # predictor.model is Ultralytics' AutoBackend; the module it executes is its .model
    run_param = next(model.predictor.model.model.parameters())
    run_dev, run_dtype = str(run_param.device), str(run_param.dtype)
    if half and not getattr(model.predictor.model, "fp16", False):
        _emit(proto, {"event": "fatal", "error": "fp16 requested but the backend did not enable half precision"})
        return 3
    if (cuda and not run_dev.startswith("cuda")) or (not cuda and run_dev != "cpu"):
        _emit(proto, {"event": "fatal", "error": f"model runs on {run_dev}, not the requested {a.device}"})
        return 3

    vm = None
    vp_state: dict = {"cache": {}, "active": [], "names": [], "active_keys": []}
    if vp_path is not None:
        vm = YOLOE(str(vp_path))

    receipt = {"torch": torch.__version__, "device": run_dev, "requested_device": a.device, "dtype": run_dtype,
               "precision": a.precision, "threads": torch.get_num_threads(), "threads_requested": a.threads, "vocabulary": len(names),
               "model_path": str(model_path.resolve()), "model_sha256": digest,
               "vp_model_path": str(vp_path.resolve()) if vp_path else None,
               "vp_model_sha256": a.vp_sha256 if vp_path else None,
               "interpreter": sys.executable, "torch_path": str(Path(torch.__file__).resolve().parent),
               "ultralytics_path": str(Path(sys.modules["ultralytics"].__file__).resolve().parent),
               "worker_pid": os.getpid(), "echo_watch_path": str(Path(sys.modules["echo_watch"].__file__).resolve().parent)}
    if cuda:
        i = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(i)
        free, total = torch.cuda.mem_get_info(i)
        receipt.update({"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "gpu_name": torch.cuda.get_device_name(i), "capability": f"{cap[0]}.{cap[1]}",
                        "vram_total_mib": round(total / 2**20), "vram_free_mib_after_load": round(free / 2**20)})
    receipt["load_s"] = round(time.perf_counter() - t0, 3)
    _emit(proto, {"event": "ready", **receipt})

    shm = {"name": None, "obj": None}

    def frame_from(req) -> "np.ndarray":
        if req.get("image_path"):
            img = cv2.imread(req["image_path"], cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(req["image_path"])
            return img
        from multiprocessing import shared_memory
        if shm["name"] != req["shm"]:
            if shm["obj"] is not None:
                shm["obj"].close()
            shm["obj"], shm["name"] = shared_memory.SharedMemory(name=req["shm"]), req["shm"]
        h, w = req["shape"]
        return np.ndarray((h, w, 3), np.uint8, buffer=shm["obj"].buf[:h * w * 3]).copy()

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def vram() -> dict:
        if not cuda:
            return {}
        free, total = torch.cuda.mem_get_info()
        return {"vram_free_mib": round(free / 2**20), "vram_total_mib": round(total / 2**20),
                "vram_allocated_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)}

    def compute_vpe(img, boxes_xyxy):
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        n = len(boxes_xyxy)
        vm.model.model[-1].nc = n
        vm.model.names = [f"object{i}" for i in range(n)]
        pred = YOLOEVPSegPredictor(overrides={"task": vm.model.task, "mode": "predict", "save": False, "verbose": False,
                                              "batch": 1, "device": a.device, "half": half, "imgsz": 640})
        pred.set_prompts({"bboxes": np.asarray(boxes_xyxy, np.float32), "cls": np.arange(n)})
        pred.setup_model(model=vm.model, verbose=False)
        with torch.no_grad():
            vpe = pred.get_vpe(img)
        pin_threads()
        return vpe.detach()

    def activate() -> None:
        import torch as _t
        if not vp_state["active"]:
            return
        emb = _t.cat([e for e in vp_state["active"]], dim=1)
        vm.model.set_classes(vp_state["names"], emb)
        vm.predictor = None

    for line in sys.stdin:
        recv_wall = time.time()
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")
        job = req.get("job_id")
        if cmd == "ping":
            _emit(proto, {"event": "pong", "job_id": job, "recv_wall": recv_wall, "done_wall": time.time(), **vram()})
            continue
        if cmd == "quit":
            _emit(proto, {"event": "bye", "job_id": job})
            break
        try:
            if cmd == "scan":
                img = frame_from(req)
                ox, oy = 0, 0
                area = req.get("area")
                if area:
                    x, y, w, h = (int(round(v)) for v in area)
                    x, y = max(0, x), max(0, y)
                    img = img[y:y + h, x:x + w]
                    ox, oy = x, y
                    if img.size == 0:
                        raise ValueError("scan area is empty")
                t = time.perf_counter()
                r = predict(model, img, imgsz=req["imgsz"], conf=req["conf"], iou=req["iou"], max_det=req["max_det"])
                sync()
                infer_s = time.perf_counter() - t
                cands = []
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
                clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
                polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
                for i, (bx, cf, cl) in enumerate(zip(boxes, confs, clss)):
                    x0, y0, x1, y1 = (float(v) for v in bx)
                    poly = None
                    if polys[i] is not None and len(polys[i]) >= 3:
                        pts = np.asarray(polys[i], dtype=np.float32) + np.array([ox, oy], dtype=np.float32)
                        poly = simplify_polygon(pts)
                    cands.append({"box": [x0 + ox, y0 + oy, x1 - x0, y1 - y0], "polygon": poly,
                                  "label": str(names[int(cl)]), "confidence": round(float(cf), 4)})
                _emit(proto, {"event": "result", "job_id": job, "candidates": cands, "infer_s": round(infer_s, 4),
                              "input_shape": list(img.shape[:2]), "offset": [ox, oy], "device": run_dev,
                              "precision": a.precision, "recv_wall": recv_wall, "done_wall": time.time(), **vram(),
                              "egress_refusals": guards.snapshot()["egress_refusals"]})
            elif cmd == "set_refs":
                if vm is None:
                    raise RuntimeError("no visual-prompt model loaded in this worker")
                t = time.perf_counter()
                computed = []
                for ref in req["refs"]:  # [{key, image_path, boxes:[[x0,y0,x1,y1]...]}]
                    if ref["key"] not in vp_state["cache"]:
                        img = cv2.imread(ref["image_path"], cv2.IMREAD_COLOR)
                        if img is None:
                            raise FileNotFoundError(ref["image_path"])
                        vp_state["cache"][ref["key"]] = compute_vpe(img, ref["boxes"])
                        computed.append(ref["key"])
                # active class order: one class per (reference key, box index)
                vp_state["active"], vp_state["names"], vp_state["active_keys"] = [], [], []
                for ref in req["refs"]:
                    e = vp_state["cache"][ref["key"]]
                    vp_state["active"].append(e)
                    for k in range(e.shape[1]):
                        vp_state["names"].append(f"{ref['key']}#{k}")
                        vp_state["active_keys"].append([ref["key"], k])
                for k in [k for k in vp_state["cache"] if k not in {r["key"] for r in req["refs"]}]:
                    del vp_state["cache"][k]  # an updated/removed reference is never applied again
                activate()
                sync()
                warm_s = 0.0
                if vp_state["names"]:
                    tw = time.perf_counter()
                    predict(vm, warm, imgsz=640, conf=0.25)   # predictor construction / first-use cost, off the stream
                    pin_threads()
                    sync()
                    warm_s = time.perf_counter() - tw
                _emit(proto, {"event": "refs", "job_id": job, "classes": vp_state["active_keys"], "computed": computed,
                              "seconds": round(time.perf_counter() - t, 4), "warm_s": round(warm_s, 4),
                              "recv_wall": recv_wall, "done_wall": time.time(), **vram()})
            elif cmd == "verify":
                if vm is None or not vp_state["names"]:
                    raise RuntimeError("visual references are not set")
                img = frame_from(req)
                ox, oy = 0, 0
                region = req.get("region")
                if region:
                    x, y, w, h = (int(round(v)) for v in region)
                    x, y = max(0, x), max(0, y)
                    img = img[y:y + h, x:x + w]
                    ox, oy = x, y
                t = time.perf_counter()
                r = predict(vm, img, imgsz=req.get("imgsz", 640), conf=req.get("conf", 0.10), iou=req.get("iou", 0.70),
                            max_det=req.get("max_det", 100))
                sync()
                infer_s = time.perf_counter() - t
                t_post = time.perf_counter()
                dets = []
                if r.boxes is not None:
                    want_polys = bool(req.get("polygons", False))
                    polys = r.masks.xy if (want_polys and r.masks is not None) else [None] * len(r.boxes)
                    for i, (bx, cf, cl) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                                         r.boxes.cls.cpu().numpy().astype(int))):
                        x0, y0, x1, y1 = (float(v) for v in bx)
                        poly = None
                        if polys[i] is not None and len(polys[i]) >= 3:
                            poly = simplify_polygon(np.asarray(polys[i], np.float32) + np.array([ox, oy], np.float32))
                        dets.append({"cls": int(cl), "ref": vp_state["active_keys"][int(cl)],
                                     "confidence": round(float(cf), 4),
                                     "box": [x0 + ox, y0 + oy, x1 - x0, y1 - y0], "polygon": poly})
                _emit(proto, {"event": "verified", "job_id": job, "detections": dets, "infer_s": round(infer_s, 4),
                              "post_s": round(time.perf_counter() - t_post, 4),
                              "frame_ref": req.get("frame_ref"), "device": run_dev, "precision": a.precision,
                              "recv_wall": recv_wall, "done_wall": time.time(), **vram()})
            else:
                _emit(proto, {"event": "error", "job_id": job, "error": "unknown command"})
        except Exception as e:  # noqa: BLE001 - reported to the controller, never hidden
            _emit(proto, {"event": "error", "job_id": job, "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
