# SPDX-License-Identifier: AGPL-3.0-only
"""One 10 s admission immediately before each GPU session (package: 'Execution discipline and stop decisions').

Admit only if over a 10 s window of 1 Hz samples: mean machine CPU <= 25%, p95 machine CPU <= 50%, and NVML free memory
>= 2200 MiB (the MINIMUM over the window is used, so a momentary dip refuses). A refusal ends GPU work for the dispatch:
there is no waiter, no retry loop and no owner process is ever stopped to make room.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from echo_watch.metrics import percentile  # noqa: E402

MEAN_MAX, P95_MAX, FREE_MIN_MIB, SECONDS = 25.0, 50.0, 2200, 10


def admit(seconds: int = SECONDS) -> dict:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    psutil.cpu_percent(None)
    cpu, free, used = [], [], []
    t0 = time.monotonic()
    for _ in range(seconds):
        time.sleep(1.0)
        cpu.append(psutil.cpu_percent(None))
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        free.append(round(m.free / 2 ** 20)), used.append(round(m.used / 2 ** 20))
    total = round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2 ** 20)
    name = pynvml.nvmlDeviceGetName(h)
    pynvml.nvmlShutdown()
    rep = {"wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seconds": round(time.monotonic() - t0, 2), "machine_cpu_samples": cpu,
           "machine_cpu_mean": round(sum(cpu) / len(cpu), 2), "machine_cpu_p95": round(percentile(cpu, 95), 2),
           "nvml_free_mib_samples": free, "nvml_free_mib_min": min(free), "nvml_used_mib_samples": used,
           "nvml_total_mib": total, "gpu": name if isinstance(name, str) else name.decode(),
           "thresholds": {"cpu_mean_max": MEAN_MAX, "cpu_p95_max": P95_MAX, "nvml_free_min_mib": FREE_MIN_MIB}}
    why = []
    if rep["machine_cpu_mean"] > MEAN_MAX:
        why.append(f"mean machine CPU {rep['machine_cpu_mean']}% > {MEAN_MAX}%")
    if rep["machine_cpu_p95"] > P95_MAX:
        why.append(f"p95 machine CPU {rep['machine_cpu_p95']}% > {P95_MAX}%")
    if rep["nvml_free_mib_min"] < FREE_MIN_MIB:
        why.append(f"NVML free {rep['nvml_free_mib_min']} MiB < {FREE_MIN_MIB} MiB")
    rep["admitted"] = not why
    rep["refusal_reasons"] = why
    return rep


if __name__ == "__main__":
    r = admit()
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["admitted"] else 10)
