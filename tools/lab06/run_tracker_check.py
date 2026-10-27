# SPDX-License-Identifier: AGPL-3.0-only
"""Launch the EW-LAB-06 synthetic tracker check as ONE bounded, windowless, owned child of the GPU runtime.

usage: env/ui/python.exe -I tools/lab06/run_tracker_check.py <out dir>
The child gets a minimal project-local environment (worker.client.child_env), CREATE_NO_WINDOW, a 120 s bound, and is
the only process this launcher ever terminates.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from echo_watch.config import GPU_PYTHON  # noqa: E402
from echo_watch.guards import popen_windowless  # noqa: E402
from echo_watch.worker.client import child_env  # noqa: E402

out_dir = Path(sys.argv[1]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
scratch = ROOT / "data" / "tests" / f"{time.strftime('%Y%m%d-%H%M%S')}-lab06-tracker"
cmd = [str(GPU_PYTHON), "-I", str(ROOT / "tools" / "lab06" / "tracker_synthetic_check.py"),
       str(out_dir / "tracker_synthetic_results.json"), str(scratch)]
env = child_env(scratch / "home")
t0 = time.monotonic()
p = popen_windowless(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=str(ROOT))
try:
    out, err = p.communicate(timeout=120)
    rc = p.returncode
except subprocess.TimeoutExpired:
    p.kill()
    out, err = p.communicate(timeout=10)
    rc = "timeout"
rec = {"command": cmd, "pid": p.pid, "exit": rc, "seconds": round(time.monotonic() - t0, 2), "scratch": str(scratch),
       "env_keys": sorted(env), "stdout": out.decode("utf-8", "replace"), "stderr_tail": err.decode("utf-8", "replace")[-4000:],
       "child_alive_after": p.poll() is None}
(out_dir / "tracker_synthetic_launch.json").write_text(json.dumps(rec, indent=1), "utf-8")
print(rec["stdout"])
print("exit:", rc, "seconds:", rec["seconds"], "alive after:", rec["child_alive_after"])
if rec["stderr_tail"].strip():
    print("stderr tail:\n" + rec["stderr_tail"][-1500:])
sys.exit(0 if rc == 0 else 1)
