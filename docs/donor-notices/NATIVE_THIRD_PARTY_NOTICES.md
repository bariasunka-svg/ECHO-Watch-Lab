# Third-party notices - foundation stage (inventory, not a release collection)

This lists what the stage-1 build actually uses. It is a working inventory for review; it is **not** a claim that any file may be shipped.

| Component | Where | Notice / evidence |
|---|---|---|
| CPython 3.13.16 embeddable (PSF licence) | `env/cpu`, `env/gpu`, archive `vendor/downloads/python-3.13.16-embed-amd64.zip` (SHA-256 `97dae527...`, sigstore bundle kept) | `vendor/notices/python-embed/LICENSE.txt` |
| Python packages (numpy, OpenCV (opencv-python 4.10.0.84), Pillow, psutil, nvidia-ml-py, PyTorch 2.13.0 CPU / 2.11.0+cu126, torchvision, ultralytics 8.4.160, imageio-ffmpeg 0.6.0, pytest, ...) | `requirements/runtime.lock`, `verify.lock`, `gpu-worker.lock` (every wheel SHA-256 locked and verified) | `vendor/notices/python/<package-version>/` (copied for the locked set) |
| Dormant libraries in the locks (fastapi, uvicorn, starlette, websockets, pydantic, matplotlib, polars, ...) | installed because the reference locks list them | **not imported by any application module** (the isolation test imports none); later dependency minimisation needs an actual closure |
| FFmpeg 7.1 (gyan.dev essentials build, GPL, configuration includes libx264) | `env/cpu/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe`, SHA-256 `2ce797a0...` | executed only as a windowless child; corresponding-source/native obligations remain **unresolved** (AGENTS hold) |
| NVIDIA CUDA 12.6 / cuDNN runtime inside the GPU torch wheel | `env/gpu` | development qualification only; excluded from any commercial bundle until supply duties are resolved (AGENTS hold) |
| YOLOE-26s (prompt-free) and YOLOE-26s (visual prompt) checkpoints (Ultralytics) | `models/*.pt`, SHA-256 in `src/echo_watch/config.py`; provenance `vendor/evidence/ultralytics-assets-v8.4.0-release.json` | Ultralytics/AGPL route retained; weights supplier terms unresolved for commercial use |
| Authorised Gemini-generated clip | `inputs/gemini_generated_video_448bc16c.mp4` (SHA-256 `3a9f0898...`) | owner-authorised distinct performance check only; not removal/return evidence |

No Tcl/Tk is present (see `docs/NATIVE_INTAKE_MAP.md` section 4); it needs its own approved supply route before the native stage.
