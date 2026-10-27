# SPDX-License-Identifier: AGPL-3.0-only
"""Shared test helpers. The stand-in worker (echo_watch.worker.standin) is TEST MATERIAL: no model runs."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from echo_watch.config import CPU_PYTHON, FFMPEG, ROOT, Settings
from echo_watch.engine.api import ItemEngine
from echo_watch.engine.contracts import CaptureOptions
from echo_watch.engine.session import SourceSpec
from echo_watch.fixtures.synthetic_scene import SyntheticScene
from echo_watch.guards import popen_windowless

TEST_DATA = ROOT / "data" / "tests"


def work_dir(name: str) -> Path:
    d = TEST_DATA / time.strftime("%Y%m%d-%H%M%S") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


BARCODE_BITS, CELL, PITCH = 12, 22, 26


def draw_barcode(img, n: int) -> None:
    for k in range(BARCODE_BITS):
        x = 4 + k * PITCH
        img[4:4 + CELL, x:x + CELL] = 255 if (n >> k) & 1 else 0


def read_barcode(img) -> int:
    v = 0
    for k in range(BARCODE_BITS):
        x = 4 + k * PITCH
        if float(img[4 + 4:4 + CELL - 4, x + 4:x + CELL - 4].mean()) >= 127.5:
            v |= 1 << k
    return v


def make_clip(path: Path, frames: int, fps: float = 15.0, mutate=None, size=(1280, 720), seed: int = 7,
              barcode: bool = False) -> SyntheticScene:
    """Encode ``frames`` frames of the generated scene (mutate(scene, i) edits scripted state per frame). With
    ``barcode`` the frame ordinal is written into the top-left corner so a packet's pixels can be tied to its frame."""
    scene = SyntheticScene(seed=seed, size=size)
    p = popen_windowless([str(FFMPEG), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt",
                          "bgr24", "-s", f"{size[0]}x{size[1]}", "-framerate", f"{fps:g}", "-i", "pipe:0", "-c:v", "libx264",
                          "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for i in range(frames):
        if mutate:
            mutate(scene, i)
        img = scene.render(i / fps)
        if barcode:
            draw_barcode(img, i)
        p.stdin.write(img.tobytes())
    p.stdin.close()
    rc = p.wait(120)
    assert rc == 0, p.stderr.read().decode()
    return scene


def standin_cmd(spec_path: Path) -> list[str]:
    return [str(CPU_PYTHON), "-I", "-m", "echo_watch.worker.standin", "--spec", str(spec_path)]


def standin_engine(d: Path, spec: dict, **settings_kw) -> ItemEngine:
    sp = d / "spec.json"
    sp.write_text(json.dumps(spec), "utf-8")
    s = Settings(data_dir=d / "data", record_fps=15.0, **settings_kw)
    return ItemEngine(s, worker_cmd=standin_cmd(sp))


def clip_scope(clip: Path) -> str:
    """The content scope of a replay clip (EW-LAB-05: Start requires the profile's scope to equal the source's)."""
    from echo_watch.sources.fingerprint import fingerprint
    return fingerprint(clip)["scope"]


def setup_one(eng: ItemEngine, clip: Path, fps: float, name: str = "speaker"):
    """Worker -> setup frame -> scan -> remember the first proposal. Returns (item_refs, scan)."""
    r = eng.start_worker()
    assert r.get("event") == "ready", r
    frame = eng.open_setup_source(str(clip), fps)
    scan = eng.scan(frame, clip_scope(clip))
    assert scan.status == "succeeded" and scan.proposals
    refs = eng.remember(f"op-{name}-{time.time_ns()}", scan.scan_id, [{"candidate_id": scan.proposals[0].candidate_id, "name": name}])
    return refs, scan


def run_to_outcome(eng: ItemEngine, refs, clip: Path, fps: float, timeout: float = 60.0, options: CaptureOptions | None = None):
    eng.start(options or CaptureOptions(), [r.item_id for r in refs], SourceSpec(clip_scope(clip), str(clip), fps))
    eng.session.start_driver()
    out = eng.session.wait_outcome(timeout)
    eng.session.stop_driver()
    return out
