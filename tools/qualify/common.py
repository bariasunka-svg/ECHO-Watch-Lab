# SPDX-License-Identifier: AGPL-3.0-only
"""Shared, bounded helpers for EW-LAB-01 qualification harnesses (never imported by the product runtime).

* ``admission`` runs the ONE 10 s GPU admission in a windowless env/gpu child (pynvml lives only there); a refusal is
  a receipt, never a retry or a waiter.
* ``decode_frames`` streams a local video through the pinned FFmpeg (windowless) as raw BGR frames - not real time.
* ``pick`` is the evaluator's operator stand-in: among the REAL proposals of a scan it takes the best-overlapping
  proposal for a hint region (IoU >= 0.3). It never injects a box, profile or state into the runtime.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from echo_watch.config import FFMPEG, GEMINI_CLIP, GPU_PYTHON  # noqa: E402
from echo_watch.geometry import iou  # noqa: E402
from echo_watch.guards import popen_windowless  # noqa: E402

GEN = ROOT / "inputs" / "generated"
SCENARIOS = {
    "scene": {"video": GEN / "ew01_scene_seed7_90s_15fps.mp4", "truth": GEN / "ew01_scene_seed7_90s_15fps.evaluator-truth.json",
              "fps": 15.0, "three": ["speaker", "watch_box", "tablet"], "one": ["speaker"], "role": "development"},
    "controls": {"video": GEN / "ew01_controls_seed7_30s_15fps.mp4", "truth": GEN / "ew01_controls_seed7_30s_15fps.evaluator-truth.json",
                 "fps": 15.0, "three": ["speaker", "phone_a"], "one": ["speaker", "phone_a"], "pause": (5.0, 2.0),
                 "role": "development"},
    "D": {"video": GEN / "lab_d_seed17_45s_15fps.mp4", "truth": GEN / "lab_d_seed17_45s_15fps.evaluator-truth.json",
          "fps": 15.0, "one": ["phone_a"], "three": ["phone_a"], "role": "development"},
    "H1": {"video": GEN / "lab_h1_seed31_45s_15fps.mp4", "truth": GEN / "lab_h1_seed31_45s_15fps.evaluator-truth.json",
           "fps": 15.0, "one": ["phone_a"], "three": ["phone_a"], "role": "held-out"},
    "H2": {"video": GEN / "lab_h2_seed47_45s_15fps.mp4", "truth": GEN / "lab_h2_seed47_45s_15fps.evaluator-truth.json",
           "fps": 15.0, "one": ["phone_a"], "three": ["phone_a"], "role": "held-out"},
    "gemini": {"video": GEMINI_CLIP, "truth": None, "fps": 24.0,
               "hints": {"phone": [298, 238, 130, 258], "mug": [568, 322, 190, 180], "bottle": [858, 120, 125, 375]},
               "three": ["phone", "mug", "bottle"], "one": ["mug"], "role": "all-present check"},
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def src_digest() -> str:
    return hashlib.sha256("".join(f"{p.relative_to(ROOT).as_posix()}\0{sha256_file(p)}\n"
                                  for p in sorted((ROOT / "src").rglob("*.py"))).encode()).hexdigest()


def admission(out_dir: Path) -> dict:
    p = popen_windowless([str(GPU_PYTHON), "-I", str(ROOT / "tools" / "qualify" / "admission.py")],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(ROOT))
    try:
        out, err = p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
        p.communicate(timeout=5)
        rep = {"admitted": False, "refusal_reasons": ["admission child timed out"]}
    else:
        try:
            rep = json.loads(out.decode("utf-8", "replace"))
        except ValueError:
            rep = {"admitted": False, "refusal_reasons": [f"admission child failed: {err.decode('utf-8', 'replace')[-300:]}"]}
    (out_dir / "admission.json").write_text(json.dumps(rep, indent=1), "utf-8")
    return rep


def decode_frames(video: Path, fps: float, every: int = 1):
    """Yield (index, BGR frame) for every ``every``-th frame of the file at its declared rate (not real time)."""
    args = [str(FFMPEG), "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "1", "-i", str(video),
            "-vf", f"fps={fps:g}", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    probe = popen_windowless([str(FFMPEG), "-hide_banner", "-nostdin", "-i", str(video)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
    _, perr = probe.communicate(timeout=30)
    import re
    m = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", perr.decode("utf-8", "replace"))
    w, h = int(m.group(1)), int(m.group(2))
    p = popen_windowless(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = w * h * 3
    i = 0
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            if i % every == 0:
                yield i, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            i += 1
    finally:
        p.kill()
        p.wait(5)


def load_truth(sc: dict) -> dict | None:
    return json.loads(Path(sc["truth"]).read_text("utf-8")) if sc.get("truth") else None


def pick(scan, names: list[str], truth: dict | None, hints: dict | None = None) -> tuple[list[dict], list[dict]]:
    picks, report = [], []
    for name in names:
        hint = truth["items"][name]["box"][0] if truth is not None else hints[name]
        cands = [p for p in scan.proposals if not p.duplicate_of]
        best = max(cands, key=lambda p: iou(p.box, hint), default=None)
        ov = iou(best.box, hint) if best else 0.0
        report.append({"item": name, "hint": hint, "chosen_candidate": best.candidate_id if best else None,
                       "chosen_box": list(best.box) if best else None, "iou_with_hint": round(ov, 4),
                       "label": best.label if best else None, "confidence": best.confidence if best else None})
        if best is not None and ov >= 0.3:
            picks.append({"candidate_id": best.candidate_id, "name": name})
    return picks, report


def ledger(closeout: Path, rec: dict) -> None:
    with open(closeout / "model_process_ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")
