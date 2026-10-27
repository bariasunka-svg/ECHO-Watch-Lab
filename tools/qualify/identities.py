# SPDX-License-Identifier: AGPL-3.0-only
"""Write closeouts/<stamp>/identities.json: the exact environment, interpreter, executable, model, input and governing-document
identities this evidence was produced with (all hashes recomputed now)."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLOSEOUT = ROOT / "closeouts" / "patsight-parity-foundation-20261025-193700"
sys.path.insert(0, str(ROOT / "src"))
from echo_watch import config  # noqa: E402


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ident: dict = {"written": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "root": str(ROOT)}
    ident["interpreters"] = {w: {"python_exe": str(ROOT / "env" / w / "python.exe"), "sha256": sha(ROOT / "env" / w / "python.exe"),
                                 "pth": (ROOT / "env" / w / "python313._pth").read_text("ascii"),
                                 "lock": f"requirements/{'gpu-worker' if w == 'gpu' else 'verify'}.lock",
                                 "lock_sha256": sha(ROOT / "requirements" / ("gpu-worker.lock" if w == "gpu" else "verify.lock"))} for w in ("cpu", "gpu")}
    ident["bootstrap_artifacts"] = {n: sha(ROOT / "vendor" / "downloads" / n) for n in ("python-3.13.16-embed-amd64.zip", "pip-26.2.1-py3-none-any.whl")}
    ident["ffmpeg"] = {"path": str(config.FFMPEG), "sha256": sha(config.FFMPEG), "pinned": config.FFMPEG_SHA256}
    ident["models"] = {"prompt_free": {"path": str(config.MODEL_PF), "sha256": sha(config.MODEL_PF), "pinned": config.MODEL_PF_SHA256},
                       "visual_prompt": {"path": str(config.MODEL_VP), "sha256": sha(config.MODEL_VP), "pinned": config.MODEL_VP_SHA256}}
    gen = ROOT / "inputs" / "generated"
    ident["inputs"] = {"gemini_clip": {"path": str(config.GEMINI_CLIP), "sha256": sha(config.GEMINI_CLIP), "pinned": config.GEMINI_CLIP_SHA256}}
    for p in sorted(gen.glob("*")):
        ident["inputs"][p.name] = {"bytes": p.stat().st_size, "sha256": sha(p)}
    ident["governing"] = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "AGENTS.md", ROOT / "docs" / "CURRENT_STATE.md",
                                                                      ROOT / "docs" / "PATRONSIGHT_PARITY_RESET.md",
                                                                      ROOT / "docs" / "PATRONSIGHT_WORKSPACE_ASSESSMENT.md",
                                                                      ROOT / "docs" / "PATRONSIGHT_FROZEN_UX_TECHNICAL_REVIEW_20261023.md",
                                                                      ROOT / "work-packages" / "EW-01_STANDALONE_PARITY_FOUNDATION.md"]}
    ident["package_sha256_dispatched"] = "34e65041f2c0f03a9be7be3f99a264ff4a327964ea5bdc5ba96125973d9a5a82"
    ident["package_sha256_in_new_root"] = ident["governing"]["work-packages\\EW-01_STANDALONE_PARITY_FOUNDATION.md"]
    ident["package_matches"] = ident["package_sha256_dispatched"] == ident["package_sha256_in_new_root"]
    (CLOSEOUT / "identities.json").write_text(json.dumps(ident, indent=1), "utf-8")
    print(json.dumps({"package_matches": ident["package_matches"], "keys": list(ident)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
