# SPDX-License-Identifier: AGPL-3.0-only
"""Rates, percentiles and 1 Hz resource sampling.

``RollingRate`` is adopted from the frozen PatronSight reference (src/patronsight/live_metrics.py, SHA-256
8955cca3839cb5dc80f8656ae1f22866bdb03305d352013d6a95a94f2c5d830a; LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only). Its body is unchanged:
one events-per-second definition shared by capture, analysis, display and encode so the four rates are never
conflated. ``percentile``, ``Distribution`` and ``ResourceSampler`` are new.

Actual capture, analysed, displayed and encoded rates are DISTINCT measurements; this module only supplies the
arithmetic. Deterministic and test-friendly: every method accepts an explicit timestamp and the clock is injectable.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable

DEFAULT_WINDOW_SECONDS = 2.0


class RollingRate:
    """Events-per-second over a trailing time window.

    Only the timestamps inside the window are kept (a bounded deque, not a running total), so a rate that has
    genuinely dropped to zero is reported as zero rather than smoothed by ancient history.
    """

    def __init__(self, *, window_seconds: float = DEFAULT_WINDOW_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = float(window_seconds)
        self._clock = clock
        self._times: deque[float] = deque()

    def reset(self) -> None:
        self._times.clear()

    def record(self, when: float | None = None) -> None:
        now = self._clock() if when is None else when
        self._times.append(now)
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self._window
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    def rate(self, *, now: float | None = None) -> float:
        """Events per second across the trailing window. ``0.0`` when fewer than two samples remain in the
        window - never a fabricated instantaneous value from a single sample."""
        now = self._clock() if now is None else now
        self._trim(now)
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span

    def sample_count(self, *, now: float | None = None) -> int:
        now = self._clock() if now is None else now
        self._trim(now)
        return len(self._times)


def percentile(values, q: float) -> float | None:
    """Linear-interpolated percentile (q in 0..100) of a sequence; None when empty."""
    xs = sorted(float(v) for v in values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(values) -> dict:
    xs = [float(v) for v in values]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": sum(xs) / len(xs), "min": min(xs), "p50": percentile(xs, 50),
            "p95": percentile(xs, 95), "p99": percentile(xs, 99), "max": max(xs)}


class ResourceSampler:
    """1 Hz machine / owned-process-family / GPU sampling on one daemon thread.

    ``family`` returns the pids of the processes this run owns (controller + worker + decoder + encoders); their CPU
    and resident set are summed. Machine CPU is the whole-machine figure. NVML (if available) supplies the GPU memory
    and utilisation of device 0. Samples are kept (bounded) and may also be forwarded to ``on_sample``.
    Abort policy (the package's): machine CPU > cpu_abort for ``cpu_run`` consecutive samples or NVML free memory
    below ``free_abort_frac`` of total for ``free_run`` consecutive samples calls ``on_abort(reason)`` once.
    """

    def __init__(self, family: Callable[[], list[int]], on_sample: Callable[[dict], None] | None = None,
                 on_abort: Callable[[str], None] | None = None, cpu_abort: float = 85.0, cpu_run: int = 10,
                 free_abort_frac: float = 0.20, free_run: int = 3, period_s: float = 1.0, maxlen: int = 3600,
                 gpu_fn: Callable[[], tuple | None] | None = None) -> None:
        import psutil
        self._ps = psutil
        self.family, self.on_sample, self.on_abort = family, on_sample, on_abort
        self.cpu_abort, self.cpu_run, self.free_abort_frac, self.free_run = cpu_abort, cpu_run, free_abort_frac, free_run
        self.period_s = period_s
        self.samples: deque[dict] = deque(maxlen=maxlen)
        self.aborted: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._procs: dict[int, "psutil.Process"] = {}
        self._nvml = None
        self._handle = None
        # EW-LAB-01: the UI process has no NVML binding; device-wide free/total MiB then come from the GPU worker's
        # own replies (torch.cuda.mem_get_info), labelled as such
        self.gpu_fn = gpu_fn

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # noqa: BLE001 - GPU figures are then simply absent (reported as such)
            self._nvml = self._handle = None

    def start(self) -> None:
        self._init_nvml()
        self._ps.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(3)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass

    def one(self) -> dict:
        ps = self._ps
        s: dict = {"t_mono": time.monotonic(), "t_wall": time.time(), "machine_cpu": ps.cpu_percent(None)}
        fam_cpu = fam_rss = 0.0
        pids = []
        try:
            pids = list(self.family())
        except Exception:  # noqa: BLE001
            pass
        alive = {}
        for pid in pids:
            p = self._procs.get(pid)
            try:
                if p is None:
                    p = ps.Process(pid)
                    p.cpu_percent(None)
                    self._procs[pid] = p
                    continue
                fam_cpu += p.cpu_percent(None)
                fam_rss += p.memory_info().rss
                alive[pid] = p
            except (ps.Error, OSError):
                continue
        for pid in [k for k in self._procs if k not in pids]:
            self._procs.pop(pid, None)
        s.update(family_pids=len(pids), family_cpu=round(fam_cpu, 1), family_rss_mib=round(fam_rss / 2 ** 20, 1))
        if self._nvml is not None and self._handle is not None:
            try:
                mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                s.update(gpu_free_mib=round(mem.free / 2 ** 20), gpu_total_mib=round(mem.total / 2 ** 20),
                         gpu_used_mib=round(mem.used / 2 ** 20), gpu_util=util.gpu)
            except Exception:  # noqa: BLE001
                pass
        elif self.gpu_fn is not None:
            try:
                g = self.gpu_fn()
            except Exception:  # noqa: BLE001
                g = None
            if g and g[0] is not None and g[1]:
                s.update(gpu_free_mib=int(g[0]), gpu_total_mib=int(g[1]), gpu_source="worker reply (cuda mem_get_info)")
        return s

    def _run(self) -> None:
        hot = low = 0
        nxt = time.monotonic() + self.period_s
        while not self._stop.wait(max(0.0, nxt - time.monotonic())):
            nxt += self.period_s
            s = self.one()
            self.samples.append(s)
            if self.on_sample is not None:
                try:
                    self.on_sample(s)
                except Exception:  # noqa: BLE001 - instrumentation never breaks a run
                    pass
            hot = hot + 1 if s["machine_cpu"] > self.cpu_abort else 0
            tot = s.get("gpu_total_mib")
            low = low + 1 if tot and s["gpu_free_mib"] < self.free_abort_frac * tot else 0
            reason = None
            if hot >= self.cpu_run:
                reason = f"machine CPU above {self.cpu_abort:g}% for {hot} consecutive 1 Hz samples"
            elif low >= self.free_run:
                reason = f"NVML free memory below {self.free_abort_frac * 100:g}% for {low} consecutive samples"
            if reason and self.aborted is None:
                self.aborted = reason
                if self.on_abort is not None:
                    try:
                        self.on_abort(reason)
                    except Exception:  # noqa: BLE001
                        pass

    def summary(self) -> dict:
        xs = list(self.samples)
        out = {"samples": len(xs), "aborted": self.aborted}
        for k in ("machine_cpu", "family_cpu", "family_rss_mib", "gpu_free_mib", "gpu_used_mib", "gpu_util"):
            vals = [x[k] for x in xs if k in x]
            if vals:
                out[k] = summarize(vals)
        return out
