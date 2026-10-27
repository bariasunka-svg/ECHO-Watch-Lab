# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 controlled supply of the ONE approved tracker dependency: lap 0.5.13 (CPython 3.13, Windows x64).

usage (UI interpreter, isolated):
  env/ui/python.exe -I tools/provision/lap_wheel.py fetch   <receipt.json>
  env/ui/python.exe -I tools/provision/lap_wheel.py inspect <receipt.json>

``fetch`` prefers an exact local copy in the destination; otherwise it retrieves official PyPI metadata (<=1 MiB) and the
exact wheel URL (<=2 MiB per attempt, <=60 s, at most one transport retry, total budget <=5 MiB) over HTTPS into
``vendor/downloads/lap-0.5.13/`` and verifies the pinned SHA-256 AND the metadata digest before keeping it. Nothing is
executed or installed here. ``inspect`` reads METADATA / RECORD / license text from the wheel as a ZIP (no import, no
execution) and checks every RECORD hash.
"""
from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NAME = "lap-0.5.13-cp313-cp313-win_amd64.whl"
SHA256 = "1170bd45958733e3ce00ba116fe5e5f1b49b744310d32eca8bf84f71121b7811"
URL = ("https://files.pythonhosted.org/packages/cc/d7/613db6729e31c945f31d3875d9b00e4b43aff7aa4c53557054bdec29bc95/"
       + NAME)
META_URL = "https://pypi.org/pypi/lap/0.5.13/json"
DEST = ROOT / "vendor" / "downloads" / "lap-0.5.13"
WHEEL_MAX, META_MAX, TOTAL_MAX, TIMEOUT_S = 2 * 2 ** 20, 2 ** 20, 5 * 2 ** 20, 60.0


def _get(url: str, cap: int, budget: dict) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "echo-watch-lab-provision/1 (exact pinned artifact)"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        if r.geturl().split("://", 1)[0] != "https":
            raise RuntimeError(f"redirected away from HTTPS: {r.geturl()}")
        out = bytearray()
        while True:
            if time.monotonic() - t0 > TIMEOUT_S:
                raise TimeoutError(f"{url} exceeded {TIMEOUT_S:g} s")
            chunk = r.read(65536)
            if not chunk:
                break
            out += chunk
            budget["bytes"] += len(chunk)
            if len(out) > cap or budget["bytes"] > TOTAL_MAX:
                raise RuntimeError(f"{url} exceeded its size bound ({cap} B / total {TOTAL_MAX} B)")
        return bytes(out)


def _attempt(url: str, cap: int, budget: dict, log: list) -> bytes:
    for n in (1, 2):                              # at most one transport retry
        try:
            data = _get(url, cap, budget)
            log.append({"url": url, "attempt": n, "ok": True, "bytes": len(data)})
            return data
        except (OSError, TimeoutError) as e:      # transport errors only; size/redirect violations are final
            log.append({"url": url, "attempt": n, "ok": False, "error": f"{type(e).__name__}: {e}"})
            if n == 2:
                raise
    raise AssertionError


def fetch(receipt: Path) -> int:
    rec: dict = {"tool": "tools/provision/lap_wheel.py fetch", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "pinned": {"name": NAME, "sha256": SHA256, "url": URL, "metadata_url": META_URL},
                 "dest": str(DEST), "attempts": []}
    target = DEST / NAME
    budget = {"bytes": 0}
    try:
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
            rec.update(route="existing exact local copy", ok=True)
            return 0
        meta = json.loads(_attempt(META_URL, META_MAX, budget, rec["attempts"]))
        files = [u for u in meta.get("urls", []) if u.get("filename") == NAME]
        rec["metadata"] = {"version": meta.get("info", {}).get("version"), "license": meta.get("info", {}).get("license"),
                           "license_expression": meta.get("info", {}).get("license_expression"),
                           "requires_dist": meta.get("info", {}).get("requires_dist"),
                           "requires_python": meta.get("info", {}).get("requires_python"),
                           "file": files[0] if files else None}
        if not files or files[0].get("digests", {}).get("sha256") != SHA256 or files[0].get("url") != URL:
            rec.update(ok=False, error="official metadata does not list the pinned file/hash/URL; nothing downloaded")
            return 2
        data = _attempt(URL, WHEEL_MAX, budget, rec["attempts"])
        got = hashlib.sha256(data).hexdigest()
        rec["downloaded_sha256"] = got
        if got != SHA256:
            rec.update(ok=False, error="downloaded bytes do not match the pinned SHA-256; not kept")
            return 3
        DEST.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(target)
        rec.update(route="official HTTPS download", ok=True, bytes=len(data), path=str(target))
        return 0
    except Exception as e:  # noqa: BLE001 - recorded, never hidden
        rec.update(ok=False, error=f"{type(e).__name__}: {e}")
        return 4
    finally:
        rec["total_bytes_transferred"] = budget["bytes"]
        rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        receipt.write_text(json.dumps(rec, indent=1), "utf-8")
        print(json.dumps({k: rec.get(k) for k in ("ok", "route", "error", "total_bytes_transferred")}))


def inspect(receipt: Path) -> int:
    target = DEST / NAME
    rec: dict = {"tool": "tools/provision/lap_wheel.py inspect", "path": str(target)}
    data = target.read_bytes()
    rec["sha256"] = hashlib.sha256(data).hexdigest()
    rec["sha256_matches_pin"] = rec["sha256"] == SHA256
    with zipfile.ZipFile(target) as z:
        names = z.namelist()
        rec["files"] = names
        dist = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
        rec["metadata_text"] = z.read(f"{dist}/METADATA").decode("utf-8", "replace")
        rec["wheel_text"] = z.read(f"{dist}/WHEEL").decode("utf-8", "replace")
        lic = [n for n in names if "licen" in n.lower() or "copying" in n.lower()]
        rec["license_files"] = {n: z.read(n).decode("utf-8", "replace") for n in lic}
        bad, checked = [], 0
        for line in z.read(f"{dist}/RECORD").decode("utf-8").splitlines():
            path, h, _size = line.rsplit(",", 2)
            if not h:
                continue
            algo, want = h.split("=", 1)
            got = base64.urlsafe_b64encode(hashlib.new(algo, z.read(path)).digest()).rstrip(b"=").decode()
            checked += 1
            if got != want:
                bad.append(path)
        rec["record_checked"], rec["record_mismatches"] = checked, bad
        rec["executables_or_scripts"] = [n for n in names if n.lower().endswith((".exe", ".bat", ".cmd", ".ps1"))
                                         or "/scripts/" in n.lower()]
    meta = rec["metadata_text"].splitlines()
    rec["requires_dist"] = [ln.split(":", 1)[1].strip() for ln in meta if ln.startswith("Requires-Dist:")]
    rec["requires_python"] = [ln.split(":", 1)[1].strip() for ln in meta if ln.startswith("Requires-Python:")]
    rec["license_fields"] = [ln for ln in meta if ln.startswith(("License", "Classifier: License"))]
    receipt.write_text(json.dumps(rec, indent=1), "utf-8")
    print(json.dumps({k: rec[k] for k in ("sha256_matches_pin", "requires_dist", "requires_python", "license_fields",
                                           "record_checked", "record_mismatches", "executables_or_scripts")}, indent=1))
    print("files:", names)
    return 0 if rec["sha256_matches_pin"] and not bad else 5


if __name__ == "__main__":
    sys.exit({"fetch": fetch, "inspect": inspect}[sys.argv[1]](Path(sys.argv[2])))
