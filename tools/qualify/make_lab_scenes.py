# SPDX-License-Identifier: AGPL-3.0-only
"""Generate the three fixed EW-LAB-01 SyntheticScene clips (no model, no device): D seed17 (development), H1 seed31 and
H2 seed47 (prospective held-out). Each 45 s, 15 fps, 1280x720, default item inventory, encoded by the pinned FFmpeg.

Script (media seconds, all three): phone_a present [0,8), absent [8,16), returned [16,24) and present to 32;
identical twin phone_b (same hue phase as phone_a) appears at 24 and stays through [32,38) while phone_a is removed;
both removed [38,45). phone_b is absent before 24. Every other object stays present. Remember ONLY phone_a.
D keeps the default geometry. H1: phone_a offset (+40,-60), tablet (-80,+170). H2: phone_a (-40,-80), tablet
(-100,-170); offsets in design units applied to the item's box (same scale and angle; ordinary rendering path).
In-frame and non-overlapping geometry is verified BEFORE encoding; a conflict is reported and nothing is encoded.
Existing outputs are preserved, never regenerated over. Truth is EVALUATOR-ONLY.

  env\\ui\\python.exe -I tools\\qualify\\make_lab_scenes.py D|H1|H2
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
from echo_watch.fixtures.synthetic_scene import SyntheticScene  # noqa: E402
from echo_watch.guards import popen_windowless  # noqa: E402

FPS, SECONDS, W, H = 15, 45, 1280, 720
OUT = ROOT / "inputs" / "generated"
VARIANTS = {
    "D": {"seed": 17, "offsets": {}, "role": "development"},
    "H1": {"seed": 31, "offsets": {"phone_a": (40, -60), "tablet": (-80, 170)}, "role": "held-out"},
    "H2": {"seed": 47, "offsets": {"phone_a": (-40, -80), "tablet": (-100, -170)}, "role": "held-out"},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def presence(t: float) -> dict[str, bool]:
    return {"phone_a": (t < 8.0) or (16.0 <= t < 32.0), "phone_b": 24.0 <= t < 38.0}


def check_geometry(scene: SyntheticScene) -> list[str]:
    problems = []
    boxes = {it.name: it.box for it in scene.items}
    for n, (x, y, w, h) in boxes.items():
        if x < 0 or y < 0 or x + w > W or y + h > H:
            problems.append(f"{n} is outside the frame: {boxes[n]}")
    names = list(boxes)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = boxes[names[a]], boxes[names[b]]
            ix = min(A[0] + A[2], B[0] + B[2]) - max(A[0], B[0])
            iy = min(A[1] + A[3], B[1] + B[3]) - max(A[1], B[1])
            if ix > 0 and iy > 0:
                problems.append(f"{names[a]} overlaps {names[b]}")
    return problems


def main() -> int:
    which = sys.argv[1]
    v = VARIANTS[which]
    stem = f"lab_{which.lower()}_seed{v['seed']}_45s_15fps"
    out = OUT / f"{stem}.mp4"
    if out.exists():
        print(f"{out} exists; preserved, not regenerated")
        return 1
    scene = SyntheticScene(seed=v["seed"], size=(W, H), layout="default")
    for name, (dx, dy) in v["offsets"].items():
        it = scene.item(name)
        x, y, w, h = it.box
        it.box = (x + dx, y + dy, w, h)
    scene.item("phone_b").hue_x = scene.item("phone_a").box[0]      # identical twin: phone_a's base hue phase
    problems = check_geometry(scene)
    geometry = {"variant": which, "role": v["role"], "seed": v["seed"], "size": [W, H], "fps": FPS, "seconds": SECONDS,
                "offsets_design_units": {k: list(o) for k, o in v["offsets"].items()},
                "items": [{"name": it.name, "kind": it.kind, "box_design_units": list(it.box), "hue_x": it.hue_x}
                          for it in scene.items],
                "script": "phone_a present [0,8) and [16,32); phone_b (identical twin) present [24,38); all others present",
                "remember": ["phone_a"], "geometry_problems": problems,
                "verified_before_encoding": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / f"{stem}.geometry.json").write_text(json.dumps(geometry, indent=1), "utf-8")
    if problems:
        print("GEOMETRY CONFLICT, nothing encoded:", problems)
        return 3
    frames = FPS * SECONDS
    truth = {"fps": FPS, "frames": frames, "size": [W, H], "items": {it.name: {"present": [], "box": []} for it in scene.items},
             "note": "EVALUATOR-ONLY truth. Boxes are [x, y, w, h] pixels of the rendered item. Never given to the runtime."}
    args = [str(FFMPEG), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
            f"{W}x{H}", "-framerate", str(FPS), "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-g", "30", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    p = popen_windowless(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    t0 = time.monotonic()
    for i in range(frames):
        t = i / FPS
        for n, present in presence(t).items():
            scene.item(n).present = present
        for it in scene.items:
            truth["items"][it.name]["present"].append(bool(it.present))
            truth["items"][it.name]["box"].append([round(x, 2) for x in scene.item_box_px(it.name)] if it.present else None)
        p.stdin.write(scene.render(t).tobytes())
    p.stdin.close()
    rc = p.wait(300)
    err = p.stderr.read().decode("utf-8", "replace")
    if rc != 0:
        print("encode failed", rc, err[-400:])
        return 2
    tp = OUT / f"{stem}.evaluator-truth.json"
    tp.write_text(json.dumps(truth), "utf-8")
    meta = {"variant": which, "role": v["role"], "file": out.name, "sha256": sha(out), "bytes": out.stat().st_size,
            "frames": frames, "fps": FPS, "seconds": SECONDS, "size": [W, H], "generator": Path(__file__).name,
            "generator_sha256": sha(Path(__file__)), "scene_module_sha256": sha(ROOT / "src/echo_watch/fixtures/synthetic_scene.py"),
            "geometry_sha256": sha(OUT / f"{stem}.geometry.json"), "truth_sha256": sha(tp),
            "encode": "libx264 veryfast crf 18 g30 yuv420p", "generation_seconds": round(time.monotonic() - t0, 1),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (OUT / f"{stem}.json").write_text(json.dumps(meta, indent=1), "utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
