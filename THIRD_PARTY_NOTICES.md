# Third-party notices — EW-LAB-01 (working inventory, not a release collection)

| Component | Where | Notice / evidence |
|---|---|---|
| CPython 3.13.16 **full** amd64 ZIP (PSF licence) incl. Tcl/Tk 8.6 (`DLLs/_tkinter.pyd`, `tcl86t.dll`, `tk86t.dll`, `tcl/`) | `env/ui` (from `vendor/downloads/python-3.13.16-amd64.zip`, SHA-256 bbf675bb…, matched the official SPDX file) | `env/ui/LICENSE.txt` (covers bundled components incl. Tcl/Tk) |
| CPython 3.13.16 embeddable (PSF) | `env/gpu` (from the cached embed ZIP, SHA-256 97dae527…) | `docs/donor-notices/python-embed-LICENSE.txt` |
| UI wheels: numpy 2.5.4, opencv-python 4.10.0.84, pillow 13.0.0, psutil 7.2.2, imageio-ffmpeg 0.6.0 | `requirements/ui.lock` (hash-locked, cached, no network) | each wheel's own `*.dist-info/` licence files in `env/ui/Lib/site-packages` |
| Verification only: pytest 9.1.1 + closure | `requirements/dev.lock` → `env/dev-site` (not on the product path) | dist-info licence files |
| GPU worker wheels (torch 2.11.0+cu126, torchvision 0.26.0+cu126, ultralytics 8.4.160, …) | `requirements/gpu-worker.lock` (exact Native lock, SHA-256 c670e4f1…) | dist-info licence files; CUDA/cuDNN runtime inside the torch wheel: development qualification only |
| FFmpeg 7.1 (gyan.dev build inside imageio-ffmpeg, GPL, libx264) | `env/ui/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe` (SHA-256 2ce797a0…) | windowless child only; corresponding-source / native-build obligations remain unresolved (hold) |
| lap 0.5.13 (BSD-2-Clause), EW-LAB-06 tracker assignment solver | `requirements/gpu-tracker.lock`; exact wheel `vendor/downloads/lap-0.5.13/lap-0.5.13-cp313-cp313-win_amd64.whl` (SHA-256 1170bd45…, official PyPI file); GPU runtime only | `env/gpu/Lib/site-packages/lap-0.5.13.dist-info/licenses/LICENSE`; receipts in `closeouts/lab-06-20261026-1936/receipts/` |
| BoT-SORT configuration values (Ultralytics 8.4.160 `cfg/trackers/botsort.yaml`, AGPL-3.0) | `src/echo_watch/yoloe/botsort.yaml` (project copy with provenance header) | Ultralytics AGPL-3.0 route, as for the package itself |
| YOLOE-26s prompt-free / visual-prompt checkpoints (Ultralytics) | `models/*.pt` (SHA-256 in `src/echo_watch/config.py`) | AGPL/Ultralytics route; commercial weight terms unresolved |
| Gemini-generated clip (owner-authorised) | `inputs/gemini_generated_video_448bc16c.mp4` (SHA-256 3a9f0898…) | short all-present check only |

The GPU lock contains libraries that the worker does not import (matplotlib, polars, …); they are inherited from the
exact Native lock and are not on the UI path. Minimising that closure is later supply work.
