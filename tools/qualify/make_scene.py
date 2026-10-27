# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze the deterministic generated scenarios (no model, no device): encoded by the pinned FFmpeg, hashed, with the
EVALUATOR-ONLY truth written beside them. The runtime never reads the truth.

  env\\cpu\\python.exe -I tools\\qualify\\make_scene.py scene     90 s, 15 fps, seed 7, 1280x720 (the package scenario)
  env\\cpu\\python.exe -I tools\\qualify\\make_scene.py controls  30 s occlusion / look-alike / pause controls (same renderer)

The package scenario (all times in media seconds): three remembered items speaker, watch_box and tablet are present
0-20 s; the speaker is absent 20-35 s; it returns to its home 35-50 s; it moves within view 50-65 s by a LINEAR translation
in design units from (dx, dy) = (0, 0) at 50 s to (120, -140) at 65 s and keeps that offset to 90 s; stationary 65-90 s.
Scale, angle and appearance are never changed; every other scene item (phone_a, phone_b, headphones, watch_box, tablet)
stays as it is. The exact supported SyntheticScene keys and geometry are recorded BEFORE generation.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from echo_watch.config import FFMPEG  # noqa: E402
from echo_watch.fixtures.synthetic_scene import SyntheticScene, default_items  # noqa: E402
from echo_watch.guards import popen_windowless  # noqa: E402

FPS = 15
W, H = 1280, 720
OUT = ROOT / "inputs" / "generated"
TARGET = (120.0, -140.0)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scene_state(sc: SyntheticScene, i: int) -> None:
    t = i / FPS
    sp = sc.item("speaker")
    sp.present = not (20.0 <= t < 35.0)
    if t < 50.0:
        sp.dx, sp.dy = 0.0, 0.0
    elif t <= 65.0:
        f = (t - 50.0) / 15.0
        sp.dx, sp.dy = TARGET[0] * f, TARGET[1] * f
    else:
        sp.dx, sp.dy = TARGET


def controls_state(sc: SyntheticScene, i: int) -> None:
    t = i / FPS
    sc.item("phone_b").hue_x = sc.item("phone_a").box[0]            # identical look-alike phones
    sc.item("speaker").obscured = 10.0 <= t < 18.0                  # a foreground blob covers the speaker and its surroundings
    sc.item("phone_a").present = not (18.0 <= t < 24.0)             # phone_a removed while its identical twin stays


SCENARIOS = {
    "scene": {"file": "ew01_scene_seed7_90s_15fps.mp4", "frames": 90 * FPS, "state": scene_state, "items": ("speaker", "watch_box", "tablet")},
    "controls": {"file": "ew01_controls_seed7_30s_15fps.mp4", "frames": 30 * FPS, "state": controls_state, "items": ("speaker", "phone_a")},
}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "scene"
    sc_def = SCENARIOS[which]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / sc_def["file"]
    if out.exists():
        print(f"{out} exists; preserved, not regenerated")
        return 1
    scene = SyntheticScene(seed=7, size=(W, H), layout="default")
    geometry = {"seed": 7, "size": [W, H], "layout": "default", "fps": FPS,
                "supported_item_keys": [{"name": it.name, "kind": it.kind, "box_design_units": list(it.box), "color_bgr": list(it.color)}
                                        for it in default_items()],
                "selected_for_monitoring": list(sc_def["items"]),
                "mapping": {k: k for k in sc_def["items"]},
                "generated_before_encoding": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / (out.stem + ".geometry.json")).write_text(json.dumps(geometry, indent=1), "utf-8")
    names = [it.name for it in scene.items]
    truth = {"fps": FPS, "frames": sc_def["frames"], "size": [W, H], "items": {n: {"present": [], "box": []} for n in names},
             "note": "EVALUATOR-ONLY truth. Boxes are [x, y, w, h] pixels of the rendered item (including scripted movement). "
                     "Never given to the engine, the policy or the worker."}
    args = [str(FFMPEG), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
            f"{W}x{H}", "-framerate", str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-g", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = popen_windowless(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.monotonic()
    for i in range(sc_def["frames"]):
        sc_def["state"](scene, i)
        for it in scene.items:
            truth["items"][it.name]["present"].append(bool(it.present))
            truth["items"][it.name]["box"].append([round(v, 2) for v in scene.item_box_px(it.name)] if it.present else None)
        p.stdin.write(scene.render(i / FPS).tobytes())
    p.stdin.close()
    rc = p.wait(300)
    err = p.stderr.read().decode("utf-8", "replace")
    if rc != 0:
        print("encode failed", rc, err[-400:])
        return 2
    (OUT / (out.stem + ".evaluator-truth.json")).write_text(json.dumps(truth), "utf-8")
    meta = {"scenario": which, "file": out.name, "sha256": sha(out), "bytes": out.stat().st_size, "frames": sc_def["frames"],
            "fps": FPS, "seconds": sc_def["frames"] / FPS, "size": [W, H], "generator": str(Path(__file__).name),
            "generator_sha256": sha(Path(__file__)), "scene_module_sha256": sha(ROOT / "src" / "echo_watch" / "fixtures" / "synthetic_scene.py"),
            "truth_sha256": sha(OUT / (out.stem + ".evaluator-truth.json")), "encode": "libx264 veryfast crf 18 g30 yuv420p",
            "generation_seconds": round(time.monotonic() - t0, 1), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / (out.stem + ".json")).write_text(json.dumps(meta, indent=1), "utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
