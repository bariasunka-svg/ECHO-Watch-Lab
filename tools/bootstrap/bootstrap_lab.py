"""Standard-library-only bootstrap for ECHO Watch Lab (EW-LAB-01).

Runs once under a reference interpreter with -I.  It never imports product code.
Stages (each writes a JSON receipt under the closeout receipts directory):

  donors     verify + copy the exact Native source ZIP and frozen PatronSight ZIP
  caches     verify + copy embed ZIP, pip wheel, models and inputs
  official   acquire the exact official full Python ZIP + SPDX JSON (bounded)
  extract    extract env/ui (full ZIP) and env/gpu (embed ZIP) with traversal checks
  wheels     copy exact locked wheels from the Native caches (hash-verified)

Nothing here executes downloaded code.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import pathlib
import shutil
import sys
import time
import urllib.request
import zipfile

LAB = pathlib.Path(r"E:\ECHO Watch Lab")
NATIVE = pathlib.Path(r"E:\ECHO Watch Native")
PS_ZIP = pathlib.Path(r"C:\Users\User\Documents\ChatGPT\Restaurant Vision\patsight-freeze-20261020\Patsight_PS27D_C3_reference_source.zip")
NATIVE_ZIP = NATIVE / "dist/source/echo-watch-native-0.2.0-foundation-3d14d6502c9e-source.zip"

PINS = {
    "native_zip": "29332384727d47cb09386778f5e0e3e24c78a780390ac8f657b116daec759f97",
    "ps_zip": "72efc34a1852bf9beced72edd26119d6f5f8bc2295efa5e35740c66f371703ea",
    "embed_zip": "97dae5274cc54867065e8d5a3226e48c35017ed332a0fdb0e27d5b5821961297",
    "pip_whl": "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e",
    "full_zip": "bbf675bb5e763c1efbb09a3a461b259d81598a63c30c4b0d7ea11b9f063df159",
    "model_pf": "01605da3f2e0e5325df45d5f5eef8e40b5b5c7d085399be3c33ac638258b150a",
    "model_vp": "48f24206bc8680d60cbbfa296b0140da849669b9515058b72f5a945142df0654",
    "gemini": "3a9f089891ae8c08175bad5dec1730df506659777d93777d895ddeb820ba3d07",
}
FULL_URL = "https://www.python.org/ftp/python/3.13.16/python-3.13.16-amd64.zip"
SPDX_URL = FULL_URL + ".spdx.json"
FULL_BYTES = 31153407
AGG_LIMIT = 40 * 1024 * 1024

GENERATED = {
    "ew01_controls_seed7_30s_15fps.evaluator-truth.json": "60cada43df4633f9668af8894a365c02025e68e5d4002e5ab3431a80d439fce5",
    "ew01_controls_seed7_30s_15fps.geometry.json": "950dacc7ef3d92893e3a9b3473c9e27609c61c41551b5669eee3ebbc5ae4f0cf",
    "ew01_controls_seed7_30s_15fps.json": "8b76e7da213d4c65329d10cc4c9d041e10e67c27da232c2eaf7145d47b1e3065",
    "ew01_controls_seed7_30s_15fps.mp4": "9cd3abcfed24f7a3ce721480f8391d2636bc697968c8e1dae9ad76d6f08f9396",
    "ew01_scene_seed7_90s_15fps.evaluator-truth.json": "2adb2d4cbf9459741c5a82405ae159346f56ea6d8e1bb2e13aa67853e2b7deb9",
    "ew01_scene_seed7_90s_15fps.geometry.json": "f1a7d4e2f5cc80544aa7d130422c6c5d1f64cc65df066be30e66637d79e25679",
    "ew01_scene_seed7_90s_15fps.json": "d93a5a2831a434764d4349018ec74dfbc13274df9128f843590f58d78c1984b8",
    "ew01_scene_seed7_90s_15fps.mp4": "14ba31460b7fa99c8ce3619fed5187021f2cd27095e2173634c7fc4b48d79a7b",
}


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return _dt.datetime.now().astimezone().isoformat()


class Receipt:
    def __init__(self, closeout: pathlib.Path, stage: str):
        self.path = closeout / "receipts" / f"bootstrap_{stage}.json"
        if self.path.exists():
            # never overwrite an earlier attempt's receipt
            n = 2
            while (closeout / "receipts" / f"bootstrap_{stage}.attempt{n}.json").exists():
                n += 1
            self.path = closeout / "receipts" / f"bootstrap_{stage}.attempt{n}.json"
        self.data = {"stage": stage, "started": now(), "interpreter": sys.executable,
                     "interpreter_sha256": sha256_file(pathlib.Path(sys.executable)),
                     "isolated_flag": sys.flags.isolated, "items": [], "status": "RUNNING"}

    def add(self, **kw):
        self.data["items"].append(kw)
        self.flush()

    def finish(self, status: str, **kw):
        self.data.update(kw)
        self.data["status"] = status
        self.data["finished"] = now()
        self.flush()

    def flush(self):
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")


def verified_copy(src: pathlib.Path, dst: pathlib.Path, pin: str | None, rec: Receipt, kind: str):
    src_hash = sha256_file(src)
    if pin is not None and src_hash != pin:
        rec.add(kind=kind, source=str(src), sha256=src_hash, pinned=pin, result="PIN_MISMATCH")
        raise SystemExit(f"pin mismatch for {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        existing = sha256_file(dst)
        if existing != src_hash:
            rec.add(kind=kind, source=str(src), destination=str(dst), result="DESTINATION_DIFFERS", existing=existing)
            raise SystemExit(f"refusing to overwrite differing {dst}")
        rec.add(kind=kind, source=str(src), destination=str(dst), sha256=src_hash, bytes=dst.stat().st_size,
                result="ALREADY_PRESENT_IDENTICAL", method="immutable artifact read; no environment reuse")
        return
    shutil.copyfile(src, dst)
    dst_hash = sha256_file(dst)
    if dst_hash != src_hash:
        raise SystemExit(f"copy hash mismatch {dst}")
    rec.add(kind=kind, source=str(src), destination=str(dst), sha256=dst_hash, bytes=dst.stat().st_size,
            pinned=pin, result="COPIED", method="immutable artifact read; no environment reuse")


def stage_donors(co):
    rec = Receipt(co, "donors")
    verified_copy(NATIVE_ZIP, LAB / "vendor/donors" / NATIVE_ZIP.name, PINS["native_zip"], rec, "engine_donor_zip")
    verified_copy(PS_ZIP, LAB / "vendor/donors" / PS_ZIP.name, PINS["ps_zip"], rec, "ui_donor_zip")
    rec.finish("PASS")


def stage_caches(co):
    rec = Receipt(co, "caches")
    dl = NATIVE / "vendor/downloads"
    verified_copy(dl / "python-3.13.16-embed-amd64.zip", LAB / "vendor/downloads/python-3.13.16-embed-amd64.zip", PINS["embed_zip"], rec, "gpu_runtime_zip")
    verified_copy(dl / "pip-26.2.1-py3-none-any.whl", LAB / "vendor/downloads/pip-26.2.1-py3-none-any.whl", PINS["pip_whl"], rec, "pip_wheel")
    verified_copy(NATIVE / "models/yoloe-26s-seg-pf.pt", LAB / "models/yoloe-26s-seg-pf.pt", PINS["model_pf"], rec, "model")
    verified_copy(NATIVE / "models/yoloe-26s-seg.pt", LAB / "models/yoloe-26s-seg.pt", PINS["model_vp"], rec, "model")
    verified_copy(NATIVE / "inputs/gemini_generated_video_448bc16c.mp4", LAB / "inputs/gemini_generated_video_448bc16c.mp4", PINS["gemini"], rec, "input")
    for name, pin in GENERATED.items():
        verified_copy(NATIVE / "inputs/generated" / name, LAB / "inputs/generated" / name, pin, rec, "input")
    rec.finish("PASS")


def _fetch(url: str, dest: pathlib.Path, rec: Receipt, budget: list[int]):
    attempts = 0
    while True:
        attempts += 1
        t0 = time.monotonic()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ECHO-Watch-Lab-bootstrap/1"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = resp.status
                final_url = resp.geturl()
                data = bytearray()
                deadline = t0 + 60
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    data.extend(chunk)
                    if budget[0] + len(data) > AGG_LIMIT:
                        raise SystemExit("aggregate download limit exceeded")
                    if time.monotonic() > deadline:
                        raise TimeoutError("request exceeded 60 s")
            budget[0] += len(data)
            dest.write_bytes(bytes(data))
            rec.add(kind="download", url=url, final_url=final_url, http_status=status, bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(), seconds=round(time.monotonic() - t0, 3),
                    attempt=attempts, destination=str(dest))
            return
        except (OSError, TimeoutError) as exc:  # transport failure: at most one retry
            rec.add(kind="download_failure", url=url, attempt=attempts, error=repr(exc),
                    seconds=round(time.monotonic() - t0, 3))
            if attempts >= 2:
                raise SystemExit(f"download failed twice: {url}")


def stage_official(co):
    rec = Receipt(co, "official")
    dl = LAB / "vendor/downloads"
    dl.mkdir(parents=True, exist_ok=True)
    zdest = dl / "python-3.13.16-amd64.zip"
    sdest = dl / "python-3.13.16-amd64.zip.spdx.json"
    if zdest.exists():
        raise SystemExit("official ZIP already staged; refusing a second acquisition")
    budget = [0]
    _fetch(SPDX_URL, sdest, rec, budget)
    _fetch(FULL_URL, zdest, rec, budget)
    actual = sha256_file(zdest)
    size = zdest.stat().st_size
    spdx = json.loads(sdest.read_text(encoding="utf-8"))
    spdx_hashes = set()

    def walk(o):
        if isinstance(o, dict):
            if str(o.get("algorithm", "")).upper() == "SHA256" and "checksumValue" in o:
                spdx_hashes.add(o["checksumValue"].lower())
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(spdx)
    ok = actual == PINS["full_zip"] and size == FULL_BYTES
    rec.finish("PASS" if ok else "FAIL", zip_sha256=actual, zip_bytes=size, pinned=PINS["full_zip"],
               pinned_bytes=FULL_BYTES, spdx_contains_zip_hash=actual in spdx_hashes,
               spdx_sha256=sha256_file(sdest), aggregate_bytes=budget[0])
    if not ok:
        raise SystemExit("official ZIP hash/size mismatch; route stopped, evidence preserved")


def safe_extract(zpath: pathlib.Path, target: pathlib.Path, rec: Receipt, pin: str):
    if sha256_file(zpath) != pin:
        raise SystemExit(f"hash mismatch before extraction: {zpath}")
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"target not empty, refusing to extract over it: {target}")
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    n = 0
    total = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename
            if name.startswith(("/", "\\")) or ":" in name or ".." in pathlib.PurePosixPath(name.replace("\\", "/")).parts:
                raise SystemExit(f"unsafe member {name!r}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise SystemExit(f"symlink member {name!r}")
            dest = (target / name).resolve()
            if root not in dest.parents and dest != root:
                raise SystemExit(f"escape member {name!r}")
            if name.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
            n += 1
            total += info.file_size
    rec.add(kind="extract", archive=str(zpath), sha256=pin, target=str(target), files=n, bytes=total)


def stage_extract(co):
    rec = Receipt(co, "extract")
    safe_extract(LAB / "vendor/downloads/python-3.13.16-amd64.zip", LAB / "env/ui", rec, PINS["full_zip"])
    safe_extract(LAB / "vendor/downloads/python-3.13.16-embed-amd64.zip", LAB / "env/gpu", rec, PINS["embed_zip"])
    for p in ("DLLs/_tkinter.pyd", "DLLs/tcl86t.dll", "DLLs/tk86t.dll", "Lib/tkinter/__init__.py",
              "tcl/tcl8.6/init.tcl", "tcl/tk8.6/tk.tcl", "python.exe", "pythonw.exe", "python313.dll"):
        f = LAB / "env/ui" / p
        rec.add(kind="ui_member", path=p, present=f.exists(), sha256=sha256_file(f) if f.exists() else None)
    rec.finish("PASS")


def stage_donorsrc(co):
    """Unpack donor source archives into a read-only reference area for inspection."""
    rec = Receipt(co, "donorsrc")
    d = LAB / "vendor/donors"
    safe_extract(d / NATIVE_ZIP.name, d / "native-src", rec, PINS["native_zip"])
    safe_extract(d / PS_ZIP.name, d / "patronsight-src", rec, PINS["ps_zip"])
    rec.finish("PASS")


def parse_lock(path: pathlib.Path):
    """Return [(name, version, [sha256...])] from a pip --require-hashes lock."""
    out = []
    cur = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        body = line.rstrip("\\").strip()
        if "==" in body and not body.startswith("--hash"):
            spec = body.split()[0]
            name, ver = spec.split("==", 1)
            name = name.split(";")[0].strip()
            ver = ver.split(";")[0].strip()
            cur = [name, ver, []]
            out.append(cur)
            rest = body.split()[1:]
        else:
            rest = body.split()
        for tok in rest:
            if tok.startswith("--hash=sha256:") and cur is not None:
                cur[2].append(tok.split(":", 1)[1])
    return out


def stage_wheels(co, locks: list[str]):
    rec = Receipt(co, "wheels")
    caches = [NATIVE / "vendor/wheels", NATIVE / "vendor/wheels-gpu"]
    index = {}
    for c in caches:
        for w in c.glob("*.whl"):
            index.setdefault(sha256_file(w), w)
    for lock in locks:
        lp = LAB / lock
        dest_dir = LAB / ("vendor/wheels-gpu" if "gpu" in lp.name else "vendor/wheels")
        for name, ver, hashes in parse_lock(lp):
            hit = next((index[h] for h in hashes if h in index), None)
            if hit is None:
                rec.add(kind="wheel", lock=lock, name=name, version=ver, result="NOT_IN_CACHE", hashes=hashes)
                continue
            verified_copy(hit, dest_dir / hit.name, sha256_file(hit), rec, f"wheel:{lock}")
    missing = [i for i in rec.data["items"] if i.get("result") == "NOT_IN_CACHE"]
    rec.finish("PASS" if not missing else "BLOCKED", missing=len(missing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["donors", "caches", "official", "extract", "donorsrc", "wheels"])
    ap.add_argument("--closeout", required=True)
    ap.add_argument("--lock", action="append", default=[])
    a = ap.parse_args()
    co = LAB / a.closeout
    if not co.is_dir():
        raise SystemExit("closeout dir missing")
    if a.stage == "wheels":
        stage_wheels(co, a.lock)
    else:
        globals()["stage_" + a.stage](co)
    print(a.stage, "done")


if __name__ == "__main__":
    main()
