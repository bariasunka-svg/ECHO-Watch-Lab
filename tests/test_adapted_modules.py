# SPDX-License-Identifier: AGPL-3.0-only
"""Narrow regressions for the hand-adapted ECHO modules and the standalone/provenance guarantees."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import echo_watch
from echo_watch import config, geometry, guards, matching, vision
from echo_watch.config import ROOT, Settings, inside_root, OutsideRoot
from echo_watch.frames import Epoch, Frame
from echo_watch.fixtures.synthetic_scene import SyntheticScene

LEGACY = Path(r"E:\ECHO Watch")


# ---- geometry (unchanged bodies) ---------------------------------------------------------------------------------
def test_geometry_clamp_iou_expand_crop():
    assert geometry.clamp_box((-5, -5, 20, 20), 100, 100) == (0.0, 0.0, 15.0, 15.0)
    assert geometry.iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert geometry.iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert geometry.iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)
    assert geometry.valid_box((0, 0, 7, 20), 100, 100) is False and geometry.valid_box((0, 0, 8, 8), 100, 100)
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert geometry.crop(img, (10, 20, 30, 40)).shape == (40, 30, 3)
    e = geometry.expand((40, 40, 20, 20), 0.5, 100, 100)
    assert e == (30.0, 30.0, 40.0, 40.0)


def test_letterbox_roundtrip():
    lb = geometry.Letterbox(1280, 720, 640, 640)
    x, y = lb.to_dst(100.0, 200.0)
    sx, sy = lb.to_src(x, y)
    assert (round(sx, 6), round(sy, 6)) == (100.0, 200.0)
    assert lb.box_to_src(lb.box_to_src((0, 0, 640, 640))) is not None


# ---- frames ------------------------------------------------------------------------------------------------------
def test_frame_image_is_read_only_and_identity_is_epoch_and_seq():
    ep = Epoch.begin("s", 1)
    f = Frame(ep, 3, 1.0, 2.0, np.zeros((4, 4, 3), np.uint8))
    assert f.ident == (ep.epoch_id, 3) and f.ref == f"{ep.epoch_id}:3"
    with pytest.raises(ValueError):
        f.image[0, 0, 0] = 1                       # a published frame can never be mutated by a consumer
    assert f.age(now_mono=3.5) == pytest.approx(2.5)


# ---- vision / matching ---------------------------------------------------------------------------------------------
def test_features_similarity_self_is_high_and_different_objects_are_lower():
    sc = SyntheticScene(size=(1280, 720))
    img = sc.render(0.0)
    b_sp, b_tab = sc.item_box_px("speaker"), sc.item_box_px("tablet")
    f_sp = matching.reference_appearance(geometry.crop(img, b_sp))
    f_tab = matching.reference_appearance(geometry.crop(img, b_tab))
    assert vision.similarity(f_sp, f_sp) == pytest.approx(1.0, abs=1e-6)
    assert matching.appearance(img, b_sp, f_sp, (b_sp[2], b_sp[3])) > 0.9
    assert matching.appearance(img, b_tab, f_sp, (b_sp[2], b_sp[3])) < 0.6      # a different item is not compatible


def test_background_geometry_stable_then_moved_for_a_shifted_view():
    sc = SyntheticScene(size=(1280, 720))
    a = vision.anchor_image(sc.render(0.0))
    assert vision.background_geometry(a, sc.render(0.0), [])["state"] == "stable"
    sc.shift = (48, 0)
    g = vision.background_geometry(a, sc.render(0.0), [])
    assert g["state"] == "moved", g          # a whole-view camera shift is recognised from the background consensus


# ---- guards --------------------------------------------------------------------------------------------------------
def test_popen_windowless_sets_no_window_flag(monkeypatch):
    seen = {}

    class P:
        def __init__(self, args, **kw):
            seen.update(kw)
    monkeypatch.setattr(guards.subprocess, "Popen", P)
    guards.popen_windowless(["x"])
    assert seen["creationflags"] & config.CREATE_NO_WINDOW and seen["stdin"] == subprocess.DEVNULL


# ---- standalone / isolation ---------------------------------------------------------------------------------------------
def test_application_modules_resolve_inside_the_new_tree_only():
    root = str(ROOT).lower()
    assert str(Path(echo_watch.__file__).resolve()).lower().startswith(str(ROOT / "src").lower())
    for name, mod in list(sys.modules.items()):
        if name.startswith("echo_watch"):
            assert str(Path(mod.__file__).resolve()).lower().startswith(str(ROOT / "src").lower()), name
        assert not name.startswith("echowatch"), "the legacy package must never be imported"
    for p in sys.path:
        assert not p.lower().startswith(r"e:\echo watch\runtime"), p
        assert "echo watch\\vendor" not in p.lower()


def test_config_ignores_inherited_legacy_overrides(monkeypatch):
    monkeypatch.setenv("ECHOWATCH_DATA", r"E:\ECHO Watch\data")
    monkeypatch.setenv("ECHOWATCH_PORT", "8765")
    s = Settings(data_dir=ROOT / "data" / "tests" / "cfg")
    assert str(s.db_path).lower().startswith(str(ROOT).lower())
    with pytest.raises(OutsideRoot):
        Settings(data_dir=Path(r"E:\ECHO Watch\data"))                # a mutable output outside the root is refused
    with pytest.raises(OutsideRoot):
        inside_root(r"C:\Windows\Temp\x")


def test_pinned_artifacts_hashes_match():
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for c in iter(lambda: fh.read(1 << 22), b""):
                h.update(c)
        return h.hexdigest()
    assert sha(config.MODEL_PF) == config.MODEL_PF_SHA256
    assert sha(config.MODEL_VP) == config.MODEL_VP_SHA256
    assert sha(config.FFMPEG) == config.FFMPEG_SHA256
    assert sha(config.GEMINI_CLIP) == config.GEMINI_CLIP_SHA256


# ---- provenance ----------------------------------------------------------------------------------------------------------
HEADER_RE = re.compile(r"runtime/echowatch/([a-z_/]+\.py)[^\n]*?\(SHA-256\s*\n?\s*([0-9a-f]{64})", re.S)


@pytest.mark.skipif(not LEGACY.exists(), reason="legacy reference not present")
def test_adapted_headers_name_the_exact_original_hash():
    """Every adapted module's header carries the SHA-256 of the preserved original it came from; the preserved original
    still has that hash (nothing in the legacy tree was changed by this work)."""
    checked = 0
    for py in sorted((ROOT / "src" / "echo_watch").rglob("*.py")):
        text = py.read_text("utf-8")
        for m in re.finditer(r"runtime/echowatch/((?:sources/)?[a-z_]+\.py)(?:(?!runtime/echowatch/).){0,80}?SHA-256\s+([0-9a-f]{64})",
                             text[:3000], re.S):
            orig = LEGACY / "runtime" / "echowatch" / m.group(1)
            assert hashlib.sha256(orig.read_bytes()).hexdigest() == m.group(2), (py.name, m.group(1))
            checked += 1
    assert checked >= 8


def test_intake_manifest_lists_adopted_files_with_matching_destination_hashes():
    p = ROOT / "docs" / "INTAKE_ECHO.json"
    if not p.is_file():
        pytest.skip("intake manifest not generated")
    doc = json.loads(p.read_text("utf-8"))
    for e in doc["adopted"]:
        dest = ROOT / e["dest"]
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == e["dest_sha256"], e["dest"]
