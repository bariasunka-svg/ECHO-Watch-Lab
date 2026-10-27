# SPDX-License-Identifier: AGPL-3.0-only
"""Owner-initiated launch of the ECHO Watch Lab workbench, plus a GUI-FREE startup path for automated checks.

``startup_config()`` resolves every path, the runtime identity and the pinned artifacts WITHOUT importing a Tk widget
module, constructing Tcl/Tk or touching a default root; automated checks call only this. ``main()`` is the single place
that creates the Tk root and the window; it is reached only by the owner's deliberate manual launch
(``ECHO Watch Lab.lnk`` -> env/ui/pythonw.exe -I -m echo_watch). It never downloads anything, never starts a model or a
device at launch (the GPU worker starts on the first Scan/Start), logs failures to logs/ and shows no crash dialog.

Lifecycle pattern adapted from the frozen PatronSight reference app/application.py (one application lifecycle owner;
title-bar close and exit share one safe close; nothing else is offered if the window cannot be built). Difference: the
Lab window is the Tk root itself (no hidden root), and the process adopts itself into a kill-on-close Job Object so
every child it starts (decoder, encoders, GPU worker) ends with it.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from ..config import FFMPEG, FFMPEG_SHA256, LOG_DIR, MODEL_PF, MODEL_PF_SHA256, MODEL_VP, MODEL_VP_SHA256, OPERATOR_DATA, ROOT

TITLE = "ECHO Watch Lab - Display Assets"


def startup_config() -> dict:
    """Everything the window needs, resolved without Tk. Safe for tests (no GUI, no model, no device, no hashing)."""
    return {"root": str(ROOT), "operator_data": str(OPERATOR_DATA), "log_dir": str(LOG_DIR),
            "interpreter": sys.executable, "ffmpeg": str(FFMPEG), "ffmpeg_present": FFMPEG.is_file(),
            "models_present": MODEL_PF.is_file() and MODEL_VP.is_file(),
            "pins": {"ffmpeg": FFMPEG_SHA256, "model_pf": MODEL_PF_SHA256, "model_vp": MODEL_VP_SHA256},
            "title": TITLE, "tk_loaded": "tkinter" in sys.modules and getattr(sys.modules["tkinter"], "_default_root", None) is not None}


def _process_local_env() -> None:
    tmp = ROOT / "cache" / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ["TEMP"] = os.environ["TMP"] = str(tmp)


def _logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / f"workbench-{time.strftime('%Y%m%d-%H%M%S')}.log"
    logging.basicConfig(filename=str(p), level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return p


def _dpi_aware() -> None:
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)      # crisp text at 125% / 150%; Tk then scales fonts itself
    except Exception:  # noqa: BLE001 - older Windows: default scaling
        pass


def _yoloe(settings, engine, ctl, logger):
    """EW-LAB-06: one application owner of GPU analysis (Display Assets or YOLOE) and the YOLOE workspace controller.
    Building them loads no model and starts no process. A failure leaves Display Assets working and is logged."""
    from .resources import AnalysisOwner, release_display_worker
    owner = AnalysisOwner()
    ctl.resources = owner

    def release_da():
        release_display_worker(engine.worker)
        ctl.worker_status = "not started"
    owner.register("display_assets", "Display Assets", ctl.analysis_busy, engine.worker.alive, release_da)
    try:
        from .yoloe_controller import YoloeController
        return YoloeController(settings, owner)
    except Exception:  # noqa: BLE001 - the YOLOE page then says it is unavailable
        logger.exception("the YOLOE workspace could not be prepared")
        return None


def main() -> int:
    _process_local_env()
    log = _logging()
    logger = logging.getLogger("echo_watch.launch")
    logger.info("manual launch: %s", startup_config())
    job = None
    try:
        from ..ownedjob import OwnedJob
        job = OwnedJob()
        job.adopt_current_process()
    except Exception:  # noqa: BLE001 - the app still runs; children are stopped by the normal lifecycle
        logger.exception("could not adopt the process into an owned job")
    try:
        from ..config import Settings
        from ..engine.api import ItemEngine
        from .controller import WorkbenchController
        settings = Settings(data_dir=OPERATOR_DATA)
        engine = ItemEngine(settings)
        ctl = WorkbenchController(engine)
        yc = _yoloe(settings, engine, ctl, logger)
        _dpi_aware()
        import tkinter as tk
        from .tk_app import WorkbenchWindow
        root = tk.Tk()
        try:
            root.state("zoomed")
        except tk.TclError:
            root.geometry("1440x860")
        WorkbenchWindow(root, ctl, OPERATOR_DATA / "ui-metrics", TITLE, yoloe=yc)
        root.mainloop()
        logger.info("window closed normally")
        return 0
    except Exception:  # noqa: BLE001 - logged locally; no crash dialog
        logger.exception("the workbench failed; see %s", log)
        return 1
    finally:
        if job is not None:
            try:
                left = job.leftovers(exclude=(os.getpid(),))
                if left:
                    logger.warning("owned children still alive at exit: %s", left)
            except Exception:  # noqa: BLE001
                pass
