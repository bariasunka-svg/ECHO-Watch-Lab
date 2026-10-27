# SPDX-License-Identifier: AGPL-3.0-only
"""Supervised client for the YOLOE workspace worker (``echo_watch.yoloe.worker``).

A thin subclass of the existing ``worker.client.WorkerClient`` (unchanged): the same windowless owned child, minimal
project-local environment, one request in flight with an absolute deadline, bounded replies, per-start ``session``
counter and shared-memory frame slot. Only the command line (PF model + pinned tracker config; no visual-prompt
model) and the typed YOLOE commands are added here.

``release()`` ends THIS client's own child when the application's resource owner hands the GPU to the other mode; unlike
``stop()`` the client stays restartable for the next explicit Start.
"""
from __future__ import annotations

import subprocess

import numpy as np

from ..config import GPU_PYTHON, MODEL_PF, MODEL_PF_SHA256, Limits, Settings
from ..worker.client import WorkerClient, WorkerSpec
from .profile import DEVICE, PRECISION, TRACKER_YAML, load_tracker_config


def yoloe_spec(settings: Settings, cmd_override: list[str] | None = None) -> WorkerSpec:
    return WorkerSpec(GPU_PYTHON, MODEL_PF, MODEL_PF_SHA256, None, None, settings.dirs()["ultralytics"], DEVICE,
                      PRECISION, settings.threads, settings.data_dir / "runtime" / "home", cmd_override,
                      settings.limits.worker_ready_timeout_s)


class YoloeWorkerClient(WorkerClient):
    def __init__(self, spec: WorkerSpec, limits: Limits | None = None):
        super().__init__(spec, limits)
        _cfg, self.tracker_sha256 = load_tracker_config()

    def _cmd(self) -> list[str]:
        s = self.spec
        if s.cmd_override:
            return list(s.cmd_override)
        return [str(s.interpreter), "-I", "-m", "echo_watch.yoloe.worker", "--model", str(s.model),
                "--sha256", s.model_sha256, "--config-dir", str(s.config_dir), "--threads", str(s.threads),
                "--device", s.device, "--precision", s.precision, "--tracker-config", str(TRACKER_YAML),
                "--tracker-sha256", self.tracker_sha256]

    def reset_epoch(self, epoch: str, timeout_s: float = 15.0) -> dict:
        reply = self.request({"cmd": "reset", "epoch": epoch}, timeout_s)
        if reply.get("event") != "reset" or reply.get("epoch") != epoch:
            raise RuntimeError(f"the worker did not acknowledge tracking epoch {epoch}: {reply.get('event')}")
        return reply

    def detect_track(self, image: np.ndarray, epoch: str, frame: str, accept: str | None, timeout_s: float) -> dict:
        return self.request({"cmd": "detect_track", "epoch": epoch, "frame": frame, "accept": accept}, timeout_s,
                            prepare=lambda: self.put_frame(image))

    def settle(self, epoch: str, frame: str, decision: str, timeout_s: float = 5.0) -> dict:
        return self.request({"cmd": "settle", "epoch": epoch, "frame": frame, "decision": decision}, timeout_s)

    def release(self, wait_s: float = 5.0) -> None:
        """Bounded end of this client's own worker process; the client can be started again later."""
        p = self.proc
        if p is not None and p.poll() is None:
            try:
                self._send({"cmd": "quit", "job_id": "quit"})
            except Exception:  # noqa: BLE001
                pass
            try:
                p.wait(wait_s)
            except subprocess.TimeoutExpired:
                pass
        self._kill()
        if self.state != "stopped":
            self.state, self.reason = "not started", None
