# SPDX-License-Identifier: AGPL-3.0-only
"""The ONE frozen YOLOE detection + BoT-SORT tracking profile of EW-LAB-06 (Tk-free, Torch-free, importable anywhere).

Detector: the pinned prompt-free checkpoint (``yoloe-26s-seg-pf.pt``) on cuda:0 FP32, every built-in vocabulary class
eligible, ``imgsz=640 conf=0.10 iou=0.70 max_det=300``, no class filter, no saving/showing, no streaming source handed to
Ultralytics. The worker records which of these options actually apply to the checkpoint (an end-to-end head ignores
``iou``); nothing here claims NMS behaviour from a knob.

Tracker: the project-owned ``botsort.yaml`` next to this file (ReID OFF, ``gmc_method=sparseOptFlow``). It is parsed
with a small flat-YAML reader (the UI runtime has no YAML package) and VALIDATED against the package values below;
a different file refuses to run instead of silently tracking with other settings.

Presentation: ``Shown confidence`` (default 0.25, 0.10-0.95) filters what is DRAWN only. It never changes the detector
result, the tracker input or the saved record.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ULTRALYTICS_VERSION = "8.4.160"
LAP_VERSION = "0.5.13"
LAP_WHEEL_SHA256 = "1170bd45958733e3ce00ba116fe5e5f1b49b744310d32eca8bf84f71121b7811"
INSTALLED_BOTSORT_SHA256 = "8cd88930d3be5557be0777f384eecff0a36a7011850fe613fc0a68178bfd5b01"
TRACKER_YAML = Path(__file__).resolve().parent / "botsort.yaml"

DETECTOR = {"imgsz": 640, "conf": 0.10, "iou": 0.70, "max_det": 300}
DEVICE, PRECISION = "cuda:0", "fp32"
EXPECTED_TRACKER = {"tracker_type": "botsort", "track_high_thresh": 0.25, "track_low_thresh": 0.1,
                    "new_track_thresh": 0.25, "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True,
                    "gmc_method": "sparseOptFlow", "proximity_thresh": 0.5, "appearance_thresh": 0.8,
                    "with_reid": False, "model": "auto"}
SHOWN_CONF_DEFAULT, SHOWN_CONF_MIN, SHOWN_CONF_MAX = 0.25, 0.10, 0.95
VIEW_TRACKS, VIEW_DETECTIONS = "tracks", "detections"   # "Detections + tracks" / "Detections only" (presentation only)


class ProfileError(RuntimeError):
    pass


def _scalar(v: str):
    v = v.strip()
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_flat_yaml(text: str) -> dict:
    """``key: value  # comment`` lines only (the shape of the Ultralytics tracker files); anything else is an error."""
    out: dict = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if ":" not in line or line.startswith((" ", "\t", "-")):
            raise ProfileError(f"tracker config line {n} is not a flat 'key: value' entry")
        k, v = line.split(":", 1)
        out[k.strip()] = _scalar(v)
    return out


def load_tracker_config(path: Path = TRACKER_YAML) -> tuple[dict, str]:
    """(validated config, SHA-256 of the file). Raises ``ProfileError`` when it differs from the package values."""
    data = Path(path).read_bytes()
    cfg = parse_flat_yaml(data.decode("utf-8"))
    if cfg != EXPECTED_TRACKER:
        diff = {k: (cfg.get(k), v) for k, v in EXPECTED_TRACKER.items() if cfg.get(k) != v}
        extra = sorted(set(cfg) - set(EXPECTED_TRACKER))
        raise ProfileError(f"tracker config differs from the pinned profile: {diff} extra={extra}")
    return cfg, hashlib.sha256(data).hexdigest()


def validate_shown_conf(value) -> float:
    v = round(float(value), 2)
    if not SHOWN_CONF_MIN <= v <= SHOWN_CONF_MAX:
        raise ValueError(f"shown confidence must be between {SHOWN_CONF_MIN:.2f} and {SHOWN_CONF_MAX:.2f}")
    return v


def vocabulary_digest(names) -> tuple[int, str]:
    """(count, SHA-256) of the model's ACTUAL index -> name mapping in index order (never a hard-coded list)."""
    if isinstance(names, dict):
        items = [names[k] for k in sorted(names)]
    else:
        items = list(names)
    return len(items), hashlib.sha256(json.dumps([str(x) for x in items], ensure_ascii=False).encode("utf-8")).hexdigest()


def profile_identity() -> dict:
    cfg, sha = load_tracker_config()
    doc = {"detector": dict(DETECTOR), "device": DEVICE, "precision": PRECISION, "classes": None,
           "tracker": cfg, "tracker_config_sha256": sha, "tracker_config_source_sha256": INSTALLED_BOTSORT_SHA256,
           "ultralytics": ULTRALYTICS_VERSION, "lap": LAP_VERSION}
    doc["profile_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc
