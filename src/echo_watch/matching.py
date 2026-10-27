# SPDX-License-Identifier: AGPL-3.0-only
"""Reference appearance compatibility (pure, deterministic) and the frozen absolute admission gate.

Adapted (SUBSET) from the Native foundation archive (tree 3d14d650, src/echo_watch/matching.py, SHA-256
09cfe6476c55306f3cac58031f617420dc7652aa49aae5a9485a91a2e4124e9a), itself from the preserved ECHO Watch reference
runtime/echowatch/association.py (SHA-256 a3fd2e7113f8dab849122e24338795755956c454be1cfbb4d4d1d3a77a5bb917;
AGPL-3.0-only). Adopted symbols: QUALIFIED_ANGLES, CORE, _core, reference_appearance, appearance.

EW-LAB-01 additions:
* ``DESCRIPTOR_VERSION`` names the one versioned descriptor the runtime uses; a calibration is valid only for it.
* ``AdmissionGate``: one GLOBAL calibration (no per-object exceptions): P = min over references of the 5th percentile of
  genuine scores, N = max distinct-object distractor score; qualified only if P - N >= 0.05; M = (P + N) / 2. A score
  below M - 0.025 is REJECT, above M + 0.025 is SUPPORT, otherwise (including equalities) UNRESOLVED.
* the gate is loaded from the frozen calibration file ``calibration/admission.json`` (its SHA-256 is recorded in the
  closeout). Without a qualified calibration the gate is still applied but ``qualified`` is False, and the workbench
  must not advertise monitoring as qualified.

``appearance`` is tolerant of rotations within QUALIFIED_ANGLES and of scale: de-rotate, estimate the item's own extent
inside the axis-aligned box from the reference's proportions with ONE fitted scale, and compare. Max over the angles.
This is appearance evidence for remembered items, never physical serial identity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import crop
from .vision import Features, features, similarity

QUALIFIED_ANGLES = (-15.0, 0.0, 15.0)  # the bounded rotation envelope the checkpoint qualifies
CORE = 0.90                             # the de-rotated item extent (90%: excludes rotation corners/edge blur)
DESCRIPTOR_VERSION = "crop-appearance-v1"

SUPPORT = "support"
UNRESOLVED = "unresolved"
REJECT = "reject"
BAND = 0.025
MIN_SEPARATION = 0.05
CALIBRATION_FILE = Path(__file__).resolve().parent / "calibration" / "admission.json"


def _core(img: np.ndarray, w_item: float, h_item: float) -> np.ndarray | None:
    hh, ww = img.shape[:2]
    cw, ch = int(round(w_item * CORE)), int(round(h_item * CORE))
    if cw < 8 or ch < 8:
        return None
    x0, y0 = (ww - cw) // 2, (hh - ch) // 2
    return img[max(0, y0):y0 + ch, max(0, x0):x0 + cw]


def reference_appearance(ref_crop: np.ndarray) -> Features:
    """Appearance key of an approved reference crop: its scale-normalised interior."""
    h, w = ref_crop.shape[:2]
    return features(_core(ref_crop, w, h))


def appearance(frame: np.ndarray, box, ref: Features, ref_size: tuple[float, float] | None = None) -> float:
    c = crop(frame, box)
    H, W = c.shape[:2]
    best = -1.0
    for ang in QUALIFIED_ANGLES:
        a = np.radians(abs(ang))
        cs, sn = np.cos(a), np.sin(a)
        if ref_size is not None:
            rw, rh = ref_size
            bw, bh = rw * cs + rh * sn, rw * sn + rh * cs     # axis-aligned extent of the reference at this angle
            sc = (W * bw + H * bh) / (bw * bw + bh * bh)
            w_item, h_item = rw * sc, rh * sc
        else:
            den = cs * cs - sn * sn
            w_item = (W * cs - H * sn) / den if ang else W
            h_item = (H * cs - W * sn) / den if ang else H
        if w_item < 8 or h_item < 8:
            continue
        img = c if not ang else cv2.warpAffine(
            c, cv2.getRotationMatrix2D((W / 2, H / 2), -ang, 1.0), (W, H), borderMode=cv2.BORDER_REPLICATE)
        core = _core(img, w_item, h_item)
        if core is None:
            continue
        best = max(best, similarity(features(core), ref))
    return best


@dataclass(frozen=True)
class AdmissionGate:
    """Absolute reference-compatibility admission: SUPPORT / UNRESOLVED / REJECT from one global calibration."""

    descriptor: str
    P: float
    N: float
    calibration_id: str
    qualified: bool
    note: str = ""

    @property
    def M(self) -> float:
        return (self.P + self.N) / 2.0

    @property
    def reject_below(self) -> float:
        return self.M - BAND

    @property
    def support_above(self) -> float:
        return self.M + BAND

    def classify(self, score: float | None) -> str:
        if score is None:
            return REJECT
        if score < self.reject_below:
            return REJECT
        if score > self.support_above:
            return SUPPORT
        return UNRESOLVED                     # the middle band and the exact equalities are unresolved

    def to_dict(self) -> dict:
        return {"descriptor": self.descriptor, "P": self.P, "N": self.N, "M": round(self.M, 6),
                "reject_below": round(self.reject_below, 6), "support_above": round(self.support_above, 6),
                "calibration_id": self.calibration_id, "qualified": self.qualified, "note": self.note}


def gate_from_calibration(doc: dict, sha256: str) -> AdmissionGate:
    if doc.get("descriptor") != DESCRIPTOR_VERSION:
        return AdmissionGate(DESCRIPTOR_VERSION, 1.0, 1.0, f"mismatch:{sha256[:12]}", False,
                             f"calibration was for {doc.get('descriptor')!r}; nothing can be admitted")
    P, N = float(doc["P"]), float(doc["N"])
    return AdmissionGate(DESCRIPTOR_VERSION, P, N, sha256[:16], bool(doc.get("qualified")) and P - N >= MIN_SEPARATION,
                         str(doc.get("note", "")))


_GATE: AdmissionGate | None = None


def default_gate() -> AdmissionGate:
    """The frozen calibration, read once. A missing file yields a gate that admits nothing (never a permissive one)."""
    global _GATE
    if _GATE is None:
        try:
            raw = CALIBRATION_FILE.read_bytes()
            _GATE = gate_from_calibration(json.loads(raw), hashlib.sha256(raw).hexdigest())
        except (OSError, ValueError, KeyError):
            _GATE = AdmissionGate(DESCRIPTOR_VERSION, 1.0, 1.0, "missing", False,
                                  "no frozen calibration; nothing can be admitted")
    return _GATE
