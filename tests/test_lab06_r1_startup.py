# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 R1 startup contract against the PINNED Ultralytics 8.4.160 configuration schema (installed code executed via
tests/pinned_ultralytics.py; no Ultralytics/Torch import, no model, device or process)."""
from __future__ import annotations

import ast

import pytest

import pinned_ultralytics as PU
from echo_watch.config import ROOT

WORKER = ROOT / "src" / "echo_watch" / "yoloe" / "worker.py"


@pytest.fixture(scope="module")
def P():
    return PU.load()


def production_predict_kwargs():
    from echo_watch.yoloe.worker import YoloeCore
    return YoloeCore(None, {}, "cuda:0", None, lambda: None).predict_kwargs()


def production_effective(pa):
    """The production effective-settings receipt for these predictor args: the bootstrap helper when it exists, else
    the expression the worker's ready receipt evaluates (baseline Lab-06 had it inline in main())."""
    try:
        from echo_watch.yoloe import bootstrap
    except ImportError:
        tree = ast.parse(WORKER.read_text("utf-8"))
        receipt = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == "receipt" for t in n.targets))
        eff = next(v for k, v in zip(receipt.keys, receipt.values) if isinstance(k, ast.Constant) and k.value == "effective")
        return eval(compile(ast.Expression(eff), "<worker ready receipt: effective>", "eval"), {"pa": pa})
    return bootstrap.effective_settings(pa)


def test_installed_sources_are_the_pinned_ones():
    assert all(PU.pin_report().values()), PU.pin_report()


def test_the_pinned_schema_has_quantize_and_no_half(P):
    assert "quantize" in P.default and "half" not in P.default
    pa = PU.predictor_args(P, {"half": False, "conf": 0.1})
    assert vars(pa)["quantize"] is None and "half" not in vars(pa)
    with pytest.raises(AttributeError, match="has no attribute 'half'"):
        pa.half                                              # the installed namespace's own error (with update advice)


def test_production_predictor_args_build_the_ready_receipt(P):
    """BEFORE R1 this fails exactly like the owner's three starts (AttributeError ... 'half')."""
    pa = PU.predictor_args(P, production_predict_kwargs())
    eff = production_effective(pa)
    assert eff["conf"] == 0.1 and eff["iou"] == 0.7 and eff["imgsz"] == 640 and eff["max_det"] == 300
    assert eff["classes"] is None and eff["quantize"] is None


# ---- production bootstrap orchestration with model/device/process effects intercepted ----------------------------------
import json
from types import SimpleNamespace

import numpy as np

from echo_watch.config import MODEL_PF, MODEL_PF_SHA256
from echo_watch.yoloe import bootstrap as B
from echo_watch.yoloe.profile import TRACKER_YAML, load_tracker_config
from echo_watch.yoloe.worker import RuntimeCapabilities
from lab06_support import ScriptTracker


class PinnedModel:
    """Stands in for ultralytics.YOLOE at the model boundary: ``predict`` builds ``self.predictor.args`` with the
    INSTALLED pinned config code (pinned_ultralytics.predictor_args) and an AutoBackend-like object whose fp16 flag
    follows the pinned predictor rule (``args.quantize == 16``); the parameters report the given device/dtype."""

    def __init__(self, P, dtype="torch.float32", device="cuda:0", predict_error=None, drop_key=None):
        self.P, self.dtype, self.device, self.predict_error, self.drop_key = P, dtype, device, predict_error, drop_key
        self.names = {0: "person", 1: "cup", 2: "bottle"}
        self.task = "segment"
        self.model = SimpleNamespace(model=[SimpleNamespace(end2end=True)])
        self.callbacks = {"on_predict_postprocess_end": []}
        self.predictor = None
        self.calls = []

    def predict(self, img, **kw):
        self.calls.append(kw)
        if self.predict_error is not None:
            raise self.predict_error
        args = PU.predictor_args(self.P, kw)
        param = SimpleNamespace(device=self.device, dtype=self.dtype)
        backend = SimpleNamespace(fp16=args.quantize == 16, model=SimpleNamespace(parameters=lambda: iter([param])))
        if self.drop_key:                                  # a schema without this key reaches the receipt stage
            delattr(args, self.drop_key)
        self.predictor = SimpleNamespace(args=args, model=backend)
        return [SimpleNamespace(boxes=None, speed={"inference": 1.0}, masks=None)]


class Caps(RuntimeCapabilities):
    """PRODUCTION capabilities except the effects: guards (process-wide audit hooks), the GPU-runtime dependency
    check, Torch/Ultralytics import, CUDA, model loading, the real tracker, thread pinning and CUDA facts."""

    def __init__(self, P, **faults):
        self.P, self.f, self.log = P, faults, []

    def install_guards(self, cfg):
        self.log.append("guards")

    def preflight(self):
        return self.f.get("preflight")

    def file_sha256(self, path):
        return self.f.get("sha", MODEL_PF_SHA256)            # the 100 MB checkpoint is not read in this test

    def import_runtime(self, cfg):
        if "import_error" in self.f:
            raise self.f["import_error"]
        return {"stub": True}

    def cuda_problem(self, rt, device):
        return self.f.get("cuda")

    def load_model(self, rt, path):
        self.model = PinnedModel(self.P, **self.f.get("model", {}))
        if self.f.get("callback"):
            self.model.callbacks["on_predict_postprocess_end"].append(type("F", (), {"__module__": "x.track"})())
        return self.model

    def tracker_factory(self, tcfg):
        return ScriptTracker

    def synchronize(self, rt):
        pass

    def after_warmup(self, rt, cfg):
        pass

    def runtime_facts(self, rt):
        return {"torch": "intercepted", "ultralytics": "8.4.160", "gpu_name": "intercepted"}


def cfg(**over):
    c = SimpleNamespace(model=str(MODEL_PF), sha256=MODEL_PF_SHA256, config_dir="unused", tracker_config=str(TRACKER_YAML),
                        tracker_sha256=load_tracker_config()[1], threads=4, device="cuda:0", precision="fp32")
    c.__dict__.update(over)
    return c


def run(P, c=None, **faults):
    events = []
    out = B.start(c or cfg(), Caps(P, **faults), lambda e: events.append(json.loads(json.dumps(e, default=str))))
    return out, events


def test_bootstrap_reaches_ready_with_canonical_fp32_and_observed_backend(P):
    out, ev = run(P)
    assert out is not None and [e["event"] for e in ev] == ["ready"]
    r = ev[0]
    assert r["requested"] == {"device": "cuda:0", "precision": "fp32", "quantize": None, "imgsz": 640, "conf": 0.1,
                              "iou": 0.7, "max_det": 300, "classes": None}
    eff = r["effective"]
    assert eff["quantize"] is None and "half" not in eff and eff["conf"] == 0.1 and eff["device"] == "cuda:0"
    assert eff["absent_optional"] == []                                   # all read from the pinned schema
    assert r["observed"] == {"device": "cuda:0", "dtype": "torch.float32", "autobackend_fp16": False}
    pd = r["precision_detail"]
    assert pd["half_derived"] is False and "observed backend parameter dtype" in pd["half_derived_from"]
    assert r["device"] == "cuda:0" and r["dtype"] == "torch.float32" and r["vocabulary"] == 3 and r["end2end"] is True
    core, receipt = out
    assert core.predictions == 0 and receipt["tracker"]["class"] == "ScriptTracker"
    # the core that reached ready serves the first frame: acknowledged epoch -> one predict -> provisional update
    core.reset("yrun_x/e1")
    rep = core.detect_track({"epoch": "yrun_x/e1", "frame": "1", "accept": None}, np.zeros((48, 64, 3), np.uint8))
    assert rep["event"] == "tracked" and rep["epoch"] == "yrun_x/e1" and rep["detections"] == []


@pytest.mark.parametrize("faults,stage,category,exc_type", [
    ({"preflight": "the tracker dependency lap is not installed"}, "preflight", "dependency", "StartupFailure"),
    ({"sha": "0" * 64}, "model_hash", "model_file", "StartupFailure"),
    ({"import_error": ImportError("DLL load failed while importing _C")}, "runtime_import", "startup_exception",
     "ImportError"),
    ({"cuda": "device cuda:0 requires CUDA in this interpreter"}, "cuda", "device", "StartupFailure"),
    ({"model": {"predict_error": RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be reported")}},
     "warmup", "startup_exception", "RuntimeError"),
    ({"model": {"dtype": "torch.float16"}}, "backend_check", "backend", "StartupFailure"),
    ({"model": {"device": "cpu"}}, "backend_check", "backend", "StartupFailure"),
    ({"callback": True}, "callbacks", "configuration", "StartupFailure"),
    ({"model": {"drop_key": "quantize"}}, "receipt", "schema", "StartupFailure"),
])
def test_every_startup_stage_fails_with_one_structured_fatal_and_never_ready(P, faults, stage, category, exc_type):
    out, ev = run(P, **faults)
    assert out is None and [e["event"] for e in ev] == ["fatal"]
    f = ev[0]["failure"]
    assert (f["stage"], f["category"], f["exc_type"]) == (stage, category, exc_type)
    assert f["stage_label"] == B.STAGE_LABEL[stage] and f["traceback"].startswith("Traceback")
    assert f["message"] and f["where"] and ev[0]["error"].startswith(B.STAGE_LABEL[stage])
    assert len(ev[0]["error"]) <= 500 and "\n" not in ev[0]["error"]


def test_tracker_config_and_precision_mismatches_are_configuration_failures(P):
    _, ev = run(P, cfg(tracker_sha256="f" * 64))
    assert ev[0]["failure"]["stage"] == "tracker_config"
    _, ev = run(P, cfg(precision="fp16"))
    assert ev[0]["failure"]["stage"] == "model_hash" and "fp32" in ev[0]["failure"]["message"]


def test_effective_settings_never_invent_missing_or_mismatched_values(P):
    pa = PU.predictor_args(P, B.predict_kwargs("cuda:0"))
    eff = B.effective_settings(pa)
    B.check_effective(eff, B.requested("cuda:0", "fp32"))
    with pytest.raises(B.StartupFailure, match="differ from the requested"):
        B.check_effective({**eff, "conf": 0.25}, B.requested("cuda:0", "fp32"))
    fp16 = PU.predictor_args(P, {**B.predict_kwargs("cuda:0"), "quantize": 16})
    with pytest.raises(B.StartupFailure, match="quantize"):
        B.check_effective(B.effective_settings(fp16), B.requested("cuda:0", "fp32"))


def test_huge_messages_and_tracebacks_are_bounded_with_explicit_truncation():
    try:
        raise ValueError("x" * 100_000)
    except ValueError as e:
        f = B.failure_record("warmup", e)
    assert f["message_truncated"] is True and len(f["message"]) <= B.MESSAGE_MAX + 80
    assert "characters truncated" in f["message"] and len(f["traceback"]) <= B.TRACEBACK_MAX + 80


def test_no_yoloe_module_reads_a_half_attribute_and_prediction_uses_quantize():
    for py in (ROOT / "src" / "echo_watch" / "yoloe").glob("*.py"):
        attrs = {n.attr for n in ast.walk(ast.parse(py.read_text("utf-8"))) if isinstance(n, ast.Attribute)}
        assert "half" not in attrs, py.name
    kw = B.predict_kwargs("cuda:0")
    assert kw["quantize"] is None and "half" not in kw
    src = WORKER.read_text("utf-8")
    assert "bootstrap.start(" in src and "class RuntimeCapabilities" in src
