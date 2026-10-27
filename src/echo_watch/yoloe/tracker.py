# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit BoT-SORT adapter over the INSTALLED Ultralytics 8.4.160 ``BOTSORT`` (GPU-runtime process only).

Imported lazily by the YOLOE worker and by the narrow synthetic CPU tracker check; never by the Tk process. Building it
loads no model: ReID is OFF (``build_encoder`` returns None, asserted) and motion compensation is sparse optical flow
(OpenCV on the CPU). That CPU association is the tracker itself, not a CPU inference fallback.

What this adapter adds to ``BOTSORT.update``:
* the input is the frame's FULL post-model detection array (``association.DetArray``); BoT-SORT's own high/low split
  decides what it uses, and its returned ``idx`` column is the row in that full array (8.4.160 sets ``track.idx`` in
  full detection-set space);
* tracks created this frame but not yet confirmed (``is_activated`` False) are reported as TENTATIVE with their index;
* every update is PROVISIONAL until settled: ``settle(accept=False)`` restores the exact pre-update tracker (deep copy,
  including GMC's previous frame/keypoints) and the class-level track-id counter, so discarded work cannot seed later
  tracking. An update cannot start while another is unsettled;
* ``reset`` builds a NEW tracker instance (ids restart at 1; the caller's tracking epoch keeps histories apart);
* library warnings during an update (e.g. GMC failures, too few matching points) are captured and returned.
"""
from __future__ import annotations

import copy
import logging
import time

import numpy as np

from .association import DetArray, join


class TrackerStateError(RuntimeError):
    pass


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage()[:200])


class BotSortAdapter:
    def __init__(self, cfg: dict):
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import IterableSimpleNamespace
        if cfg.get("tracker_type") != "botsort" or cfg.get("with_reid") is not False:
            raise TrackerStateError("only BoT-SORT with ReID OFF is permitted")
        self.cfg = dict(cfg)
        self._ns = IterableSimpleNamespace(**cfg)
        self._cls, self._base = BOTSORT, BaseTrack
        self.tracker = None
        self._pending = None
        self.updates = 0
        self.reset()

    def reset(self) -> None:
        self.tracker = self._cls(args=self._ns)
        self._pending = None
        self.updates = 0
        t = self.tracker
        if t.encoder is not None:
            raise TrackerStateError("a ReID encoder was built although with_reid is False")
        if t.gmc.method != "sparseOptFlow":
            raise TrackerStateError(f"motion compensation is {t.gmc.method!r}, not sparseOptFlow")

    def identity(self) -> dict:
        t = self.tracker
        return {"class": f"{type(t).__module__}.{type(t).__name__}", "encoder": None if t.encoder is None else
                type(t.encoder).__name__, "gmc_method": t.gmc.method, "gmc_downscale": t.gmc.downscale,
                "max_frames_lost": t.max_frames_lost, "with_reid": bool(t.args.with_reid)}

    @property
    def unsettled(self) -> bool:
        return self._pending is not None

    def update(self, dets: np.ndarray, image: np.ndarray) -> dict:
        if self._pending is not None:
            raise TrackerStateError("the previous tracker update is not settled")
        snap = (copy.deepcopy(self.tracker), self._base._count)
        t = self.tracker
        log = logging.getLogger("ultralytics")
        cap, old_level = _Capture(), log.level
        log.addHandler(cap)
        log.setLevel(min(old_level or logging.WARNING, logging.WARNING))
        t0 = time.perf_counter()
        try:
            arr = DetArray(dets)
            out = t.update(arr, image)
        except Exception:
            self.tracker, self._base._count = snap
            raise
        finally:
            log.removeHandler(cap)
            log.setLevel(old_level)
        dt = time.perf_counter() - t0
        rows = np.asarray(out, np.float64).reshape(-1, 8) if len(out) else np.zeros((0, 8))
        tentative = [(int(s.track_id), float(s.idx), [float(v) for v in s.xyxy], float(s.score), float(s.cls))
                     for s in t.tracked_stracks if not s.is_activated and s.frame_id == t.frame_id]
        assoc, errors = join(len(arr), rows, tentative)
        self._pending = snap
        self.updates += 1
        return {"rows": [[round(float(v), 4) for v in r] for r in rows], "tentative": tentative,
                "assoc": assoc, "errors": errors, "frame_id": int(t.frame_id), "lost": len(t.lost_stracks),
                "removed_this_frame": len(getattr(t, "removed_stracks_frame", []) or []),
                "warnings": cap.messages, "tracker_s": round(dt, 5)}

    def settle(self, accept: bool) -> bool:
        """Accept (drop the snapshot) or reject (restore the exact pre-update state). True when something was settled."""
        if self._pending is None:
            return False
        if not accept:
            self.tracker, self._base._count = self._pending
            self.updates -= 1
        self._pending = None
        return True
