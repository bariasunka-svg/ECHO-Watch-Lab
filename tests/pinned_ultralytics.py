# SPDX-License-Identifier: AGPL-3.0-only
"""The INSTALLED, pinned Ultralytics 8.4.160 configuration code, executed in the UI runtime without importing Ultralytics.

The functions/classes below are AST-extracted VERBATIM from the GPU runtime's installed files (hash-checked) and run with
small explicit globals: ``IterableSimpleNamespace`` (utils), ``QUANTIZE_ALIASES`` and the ``CFG_*`` key sets,
``cfg2dict``, ``_handle_deprecation``, ``check_dict_alignment``, ``check_cfg`` and ``get_cfg`` (cfg), plus the literal
``custom`` argument dict of ``Model.predict`` (engine/model.py). ``predictor_args`` mirrors how the pinned
``Model.predict`` merges keyword arguments and how ``BasePredictor`` builds ``self.args`` with ``get_cfg``. Nothing is
imported from Ultralytics/Torch; no model, device, process or file write. A changed installed file fails the pin check.
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

from echo_watch.config import ROOT

LIB = ROOT / "env" / "gpu" / "Lib" / "site-packages" / "ultralytics"
PINS = {"cfg/default.yaml": "0694fa6be3656888b893a67aadc4b17240a0d190cf799b10c347e96150b729b7",
        "cfg/__init__.py": "5f208876c16cb55741f366a721cd73c87ff8b539f30f3d0f06cb6dcce484a015",
        "utils/__init__.py": "6a5af8a2462521a67e43a07f3eec53f2168676ae0ed79ed853b150e5be592b75",
        "engine/model.py": "6e5c0001d5891e42002bd8b1d7f55085c69a1e8dd7ffdcc5dd58b63c2f863d88"}
CFG_NAMES = ("QUANTIZE_ALIASES", "QUANTIZE_DOCS_URL", "QUANTIZE_VALID_VALUES", "CFG_FLOAT_KEYS", "CFG_FRACTION_KEYS",
             "CFG_INT_KEYS", "CFG_INT_MIN", "CFG_BOOL_KEYS", "CFG_STR_KEYS")
CFG_FUNCS = ("cfg2dict", "_handle_deprecation", "check_dict_alignment", "check_cfg", "get_cfg")


def pin_report() -> dict:
    return {k: hashlib.sha256((LIB / k).read_bytes()).hexdigest() == v for k, v in PINS.items()}


def _scalar(v: str):
    v = v.strip()
    if v in ("", "null", "None", "~"):
        return None
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v.strip("'\"")


def default_cfg() -> dict:
    """Top-level ``key: value`` entries of the installed default.yaml (comments stripped)."""
    out = {}
    for line in (LIB / "cfg" / "default.yaml").read_text("utf-8-sig").splitlines():
        m = re.match(r"^([A-Za-z_]\w*):(.*)$", line)
        if m:
            out[m.group(1)] = _scalar(m.group(2).split(" #", 1)[0])
    return out


class _Log:
    def __init__(self):
        self.records = []

    def info(self, *a, **k):
        self.records.append(("info", a))

    def warning(self, *a, **k):
        self.records.append(("warning", a))


def load() -> SimpleNamespace:
    bad = [k for k, ok in pin_report().items() if not ok]
    assert not bad, f"installed Ultralytics files differ from the pinned 8.4.160 sources: {bad}"
    warnings, log = [], _Log()
    dflt = default_cfg()
    ns = {"SimpleNamespace": SimpleNamespace, "Path": Path, "DEFAULT_CFG_PATH": LIB / "cfg" / "default.yaml",
          "DEFAULT_CFG_DICT": dflt, "FLOAT_OR_INT": (float, int), "STR_OR_PATH": (str, Path), "LOGGER": log,
          "colorstr": lambda *a: a[-1], "CLI_HELP_MSG": "", "deprecation_warn": lambda *a: warnings.append(a)}
    utils = ast.parse((LIB / "utils" / "__init__.py").read_text("utf-8-sig"))
    cfg = ast.parse((LIB / "cfg" / "__init__.py").read_text("utf-8-sig"))

    def run(node):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-ultralytics>", "exec"), ns)
    run(next(n for n in utils.body if isinstance(n, ast.ClassDef) and n.name == "IterableSimpleNamespace"))
    for n in cfg.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in CFG_NAMES for t in n.targets):
            run(n)
    for name in CFG_FUNCS:
        run(next(n for n in cfg.body if isinstance(n, ast.FunctionDef) and n.name == name))
    model = ast.parse((LIB / "engine" / "model.py").read_text("utf-8-sig"))
    predict = next(n for n in ast.walk(model) if isinstance(n, ast.FunctionDef) and n.name == "predict")
    custom = next(n.value for n in ast.walk(predict) if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "custom" for t in n.targets))
    ns["PREDICT_CUSTOM"] = eval(compile(ast.Expression(custom), "<Model.predict custom>", "eval"), {"is_cli": False})
    return SimpleNamespace(ns=ns, warnings=warnings, log=log, default=dflt,
                           IterableSimpleNamespace=ns["IterableSimpleNamespace"], get_cfg=ns["get_cfg"],
                           handle_deprecation=ns["_handle_deprecation"], custom=ns["PREDICT_CUSTOM"])


def predictor_args(P: SimpleNamespace, predict_kwargs: dict, overrides: dict | None = None):
    """``model.predictor.args`` after ``Model.predict(img, **predict_kwargs)`` builds a new predictor (pinned flow:
    ``stream`` binds to the named parameter; kwargs pass ``_handle_deprecation``; ``{**overrides, **custom, **kwargs}``
    become the predictor overrides; ``BasePredictor`` calls ``get_cfg(DEFAULT_CFG, overrides)``)."""
    kw = dict(predict_kwargs)
    kw.pop("stream", None)
    kw = P.handle_deprecation(kw)
    args = {**(overrides or {}), **P.custom, **kw}
    return P.get_cfg(dict(P.default), args)
