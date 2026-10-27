# SPDX-License-Identifier: AGPL-3.0-only
"""ONE application-level owner of GPU analysis (EW-LAB-06): Display Assets or YOLOE, never both.

Each mode registers how to tell whether it is busy (preparing / running / paused / finalizing, a discovery scan or a
pending start), whether its own model worker is alive, and how to release that worker while idle. Rules:
* a mode may start analysis only while the OTHER mode is not busy (``blocked_by`` gives the operator's explanation);
* before a mode starts its worker, ``handover`` releases the other mode's IDLE worker (bounded, its own child only -
  no process-name sweeps) and confirms it is gone, so two model workers never compete for GPU memory;
* navigation never touches either mode. ``holder`` is the mode whose worker may currently be alive.
``handover`` blocks (worker quit is bounded to a few seconds) and is called from background tasks, never the Tk thread.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Participant:
    mode: str
    label: str
    busy: Callable[[], bool]
    worker_alive: Callable[[], bool]
    release_worker: Callable[[], None]


class ResourceBusy(RuntimeError):
    pass


class AnalysisOwner:
    def __init__(self):
        self._parts: dict[str, Participant] = {}
        self._lock = threading.Lock()
        self.holder: str | None = None
        self.log: list[dict] = []

    def register(self, mode: str, label: str, busy, worker_alive, release_worker) -> None:
        self._parts[mode] = Participant(mode, label, busy, worker_alive, release_worker)

    def label(self, mode: str) -> str:
        return self._parts[mode].label

    def active_mode(self) -> str | None:
        return next((m for m, p in self._parts.items() if p.busy()), None)

    def blocked_by(self, mode: str) -> str | None:
        """The operator-facing reason ``mode`` may not start analysis now, or None."""
        for m, p in self._parts.items():
            if m != mode and p.busy():
                return (f"{p.label} is still active. Stop it there before starting {self._parts[mode].label} "
                        "(one analysis at a time).")
        return None

    def handover(self, mode: str, wait_s: float = 10.0) -> dict:
        """Release every other mode's idle worker, then record ``mode`` as the holder. Raises ``ResourceBusy``."""
        with self._lock:
            rec = {"mode": mode, "wall": time.time(), "released": []}
            for m, p in self._parts.items():
                if m == mode:
                    continue
                if p.busy():
                    raise ResourceBusy(self.blocked_by(mode) or f"{p.label} is busy")
                if p.worker_alive():
                    p.release_worker()
                    end = time.monotonic() + wait_s
                    while p.worker_alive() and time.monotonic() < end:
                        time.sleep(0.05)
                    if p.worker_alive():
                        raise ResourceBusy(f"the {p.label} GPU worker did not stop in time; try again")
                    rec["released"].append(m)
            self.holder = mode
            self.log.append(rec)
            del self.log[:-50]
            return rec


def release_display_worker(worker) -> None:
    """Bounded release of the Display Assets worker while it is IDLE (its own child only). The client stays
    restartable: the next Scan/Start starts a new worker process (a new worker session) and the session sets its
    references again at every Start. ``WorkerClient`` itself is unchanged; this uses its own quit/kill path."""
    import subprocess
    p = worker.proc
    if p is not None and p.poll() is None:
        try:
            worker._send({"cmd": "quit", "job_id": "quit"})
        except Exception:  # noqa: BLE001
            pass
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            pass
    worker._kill()
    worker.refs_classes = worker.refs_digest = None
    if worker.state != "stopped":
        worker.state, worker.reason = "not started", None
