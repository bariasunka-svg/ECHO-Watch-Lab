# SPDX-License-Identifier: AGPL-3.0-only
"""YOLOE worker startup: staged bootstrap, the ready-receipt contract and structured startup failures (EW-LAB-06 R1).

Defect repaired: the Lab-06 worker passed the deprecated ``half=False`` to ``predict`` and then read ``predictor.args.half``
for its ready receipt. Pinned Ultralytics 8.4.160 consumes ``half`` in ``cfg._handle_deprecation`` and emits the unified
``quantize`` key (False -> None = FP32); ``default.yaml`` has no ``half``; the predictor selects FP16 only when
``args.quantize == 16``. Reading ``args.half`` raised AttributeError after warm-up, so every start died before ``ready``.

Now:
* prediction uses the installed API's canonical FP32 setting ``quantize=None`` (``predict_kwargs``);
* the receipt keeps three things apart - what was REQUESTED (cuda:0, fp32, thresholds), the EFFECTIVE predictor
  configuration actually read from the pinned schema (missing essential keys are a structured failure, never a
  reassuring default), and the OBSERVED backend (device and dtype of the parameters the predictor runs, AutoBackend's
  fp16 flag). A legacy ``half`` value is DERIVED from the observed dtype and labelled as derived;
* ``start`` runs every stage (guards, preflight, tracker config, model hash, runtime import, CUDA, model load, tracker,
  warm-up, backend check, callbacks, receipt) inside ONE failure boundary. Any exception or failed check emits ONE
  ``fatal`` protocol message with category, stage, exception type, bounded message and traceback (with explicit
  truncation flags) - before any ``ready`` exists - and no ready is ever emitted after a failure.
Model, device and process effects live behind a ``caps`` object (``worker.RuntimeCapabilities`` in production), so the
orchestration and receipt code here are the code tests drive.
"""
from __future__ import annotations

import os
import time
import traceback

from .profile import DETECTOR, LAP_VERSION

STAGE_LABEL = {"guards": "process guards", "preflight": "dependency check", "tracker_config": "tracker configuration",
               "model_hash": "model file check", "runtime_import": "loading the GPU runtime libraries",
               "cuda": "CUDA device check", "model_load": "loading the YOLOE model", "tracker_build": "building the tracker",
               "warmup": "model warm-up", "backend_check": "backend device/precision check",
               "callbacks": "model callback check", "receipt": "ready-receipt construction"}
ESSENTIAL = ("imgsz", "conf", "iou", "max_det", "classes", "quantize", "device")
OPTIONAL = ("agnostic_nms", "retina_masks", "nms", "save", "show", "verbose", "batch")
MESSAGE_MAX = 4000
TRACEBACK_MAX = 16 * 1024


class StartupFailure(Exception):
    """A failed startup CHECK (as opposed to an unexpected exception)."""

    def __init__(self, message: str, category: str = "startup_check"):
        super().__init__(message)
        self.category = category


def bounded(text: str, limit: int) -> tuple[str, bool]:
    """``text`` limited to ``limit`` characters keeping its head and tail, and whether anything was cut."""
    text = str(text)
    if len(text) <= limit:
        return text, False
    head = limit // 3
    tail = limit - head - 60
    return text[:head] + f"\n[... {len(text) - head - tail} characters truncated ...]\n" + text[-tail:], True


def first_line(text: str) -> str:
    return next((ln.strip() for ln in str(text).splitlines() if ln.strip()), "")


def failure_record(stage: str, exc: BaseException) -> dict:
    msg, msg_cut = bounded(str(exc).strip(), MESSAGE_MAX)
    tb, tb_cut = bounded("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)), TRACEBACK_MAX)
    where = None
    frames = traceback.extract_tb(exc.__traceback__) if exc.__traceback__ is not None else []
    if frames:
        f = frames[-1]
        where = f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
    category = getattr(exc, "category", None) or "startup_exception"
    return {"category": category, "stage": stage, "stage_label": STAGE_LABEL.get(stage, stage),
            "exc_type": type(exc).__name__, "message": msg, "message_truncated": msg_cut, "traceback": tb,
            "traceback_truncated": tb_cut, "where": where, "worker_pid": os.getpid(), "wall": time.time()}


def fatal_event(rec: dict) -> dict:
    """The protocol message: ``error`` stays one readable line (older clients show it); ``failure`` has the detail."""
    return {"event": "fatal", "error": f"{rec['stage_label']}: {rec['exc_type']}: {first_line(rec['message'])}"[:500],
            "failure": rec}


def predict_kwargs(device: str) -> dict:
    """Detector call of the frozen profile in the installed API's canonical form (``quantize=None`` = FP32)."""
    return {"imgsz": DETECTOR["imgsz"], "conf": DETECTOR["conf"], "iou": DETECTOR["iou"], "max_det": DETECTOR["max_det"],
            "classes": None, "device": device, "quantize": None, "verbose": False, "save": False, "show": False,
            "stream": False, "retina_masks": False}


def requested(device: str, precision: str) -> dict:
    return {"device": device, "precision": precision, "quantize": None, **DETECTOR, "classes": None}


def effective_settings(pa) -> dict:
    """The predictor's EFFECTIVE configuration from the pinned schema (``vars`` of its namespace; no attribute guessing).
    Missing essential keys raise; missing optional keys are listed, never invented."""
    values = dict(vars(pa))
    missing = [k for k in ESSENTIAL if k not in values]
    if missing:
        raise StartupFailure(f"the predictor configuration has no {', '.join(missing)} (pinned schema mismatch)", "schema")
    eff = {k: values[k] for k in ESSENTIAL}
    eff.update({k: values[k] for k in OPTIONAL if k in values})
    eff["absent_optional"] = [k for k in OPTIONAL if k not in values]
    return eff


def check_effective(eff: dict, req: dict) -> None:
    wrong = {k: (eff.get(k), req[k]) for k in ("imgsz", "conf", "iou", "max_det", "classes", "quantize", "device")
             if eff.get(k) != req[k]}
    if wrong:
        raise StartupFailure(f"effective predictor settings differ from the requested profile: {wrong}", "configuration")


def observed_backend(param, backend) -> dict:
    return {"device": str(param.device), "dtype": str(param.dtype), "autobackend_fp16": getattr(backend, "fp16", None)}


def check_backend(obs: dict, device: str) -> None:
    if obs["device"] != device:
        raise StartupFailure(f"the model runs on {obs['device']}, not the requested {device}; no substitute is accepted",
                             "backend")
    if obs["dtype"] != "torch.float32" or obs["autobackend_fp16"] is not False:
        raise StartupFailure(f"the model runs {obs['dtype']} (fp16 flag {obs['autobackend_fp16']!r}), not FP32", "backend")


def precision_record(req: dict, eff: dict, obs: dict) -> dict:
    return {"requested": req["precision"], "effective_quantize": eff["quantize"], "observed_dtype": obs["dtype"],
            "observed_autobackend_fp16": obs["autobackend_fp16"], "half_derived": obs["dtype"] == "torch.float16",
            "half_derived_from": "observed backend parameter dtype (the pinned configuration has no 'half' key)"}


def start(cfg, caps, emit):
    """Run every startup stage inside one failure boundary. Returns ``(core, receipt)`` after emitting ``ready``, or None
    after emitting exactly one ``fatal``. ``cfg`` has model/sha256/config_dir/tracker_config/tracker_sha256/threads/
    device/precision; ``caps`` provides the effects (see worker.RuntimeCapabilities)."""
    stage = "guards"
    t0 = time.perf_counter()
    try:
        caps.install_guards(cfg)
        stage = "preflight"
        why = caps.preflight()
        if why:
            raise StartupFailure(why, "dependency")
        stage = "tracker_config"
        tcfg, tsha = caps.load_tracker_config(cfg.tracker_config)
        if tsha != cfg.tracker_sha256:
            raise StartupFailure("tracker config hash differs from the one the application pinned", "configuration")
        stage = "model_hash"
        digest = caps.file_sha256(cfg.model)
        if digest != cfg.sha256:
            raise StartupFailure(f"model hash mismatch or missing: {cfg.model} ({digest})", "model_file")
        if cfg.precision != "fp32":
            raise StartupFailure(f"precision {cfg.precision!r} is not the admitted fp32 profile", "configuration")
        stage = "runtime_import"
        rt = caps.import_runtime(cfg)
        stage = "cuda"
        why = caps.cuda_problem(rt, cfg.device)
        if why:
            raise StartupFailure(why, "device")
        stage = "model_load"
        model = caps.load_model(rt, cfg.model)
        names = model.names
        head = model.model.model[-1]
        end2end = bool(getattr(head, "end2end", False))
        stage = "tracker_build"
        tracker_factory = caps.tracker_factory(tcfg)
        tracker_identity = tracker_factory().identity()
        stage = "warmup"
        from .worker import YoloeCore
        core = YoloeCore(model, names, cfg.device, None, tracker_factory, lambda: caps.synchronize(rt))
        core.detect(caps.neutral_image())
        core.predictions = 0
        caps.after_warmup(rt, cfg)
        stage = "backend_check"
        param, backend = caps.backend(model)
        obs = observed_backend(param, backend)
        check_backend(obs, cfg.device)
        stage = "callbacks"
        if caps.tracking_callbacks(model):
            raise StartupFailure("a tracking callback is registered on the model (model.track must not be used)",
                                 "configuration")
        stage = "receipt"
        req = requested(cfg.device, cfg.precision)
        eff = effective_settings(model.predictor.args)
        check_effective(eff, req)
        from .profile import vocabulary_digest
        n_vocab, vocab_sha = vocabulary_digest(names)
        receipt = {"device": obs["device"], "dtype": obs["dtype"], "requested_device": cfg.device,
                   "precision": cfg.precision, "requested": req, "effective": eff, "observed": obs,
                   "precision_detail": precision_record(req, eff, obs), "model_path": str(cfg.model),
                   "model_sha256": digest, "task": getattr(model, "task", None), "vocabulary": n_vocab,
                   "vocabulary_sha256": vocab_sha, "head": type(head).__name__, "end2end": end2end,
                   "iou_applies": not end2end,
                   "nms": ("none: end-to-end one-to-one head; iou is not applied, conf and max_det select rows"
                           if end2end else "torchvision NMS with iou"),
                   "segmentation": "the checkpoint's mask head runs; masks are neither transmitted nor drawn",
                   "tracker_config_sha256": tsha, "tracker": tracker_identity, "lap": LAP_VERSION,
                   **caps.runtime_facts(rt), "worker_pid": os.getpid(), "load_s": round(time.perf_counter() - t0, 3)}
        emit({"event": "ready", **receipt})
        return core, receipt
    except Exception as e:  # noqa: BLE001 - every startup failure is reported, structured, before any ready
        emit(fatal_event(failure_record(stage, e)))
        return None
