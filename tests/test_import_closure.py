# SPDX-License-Identifier: AGPL-3.0-only
"""The application's actual third-party import closure: dormant libraries in the locks are not application dependencies,
no UI toolkit is imported, no network client is imported, and no legacy package name appears."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

from echo_watch.config import ROOT

ALLOWED_THIRD_PARTY = {"numpy", "cv2", "psutil", "pynvml", "torch", "ultralytics", "PIL"}
FORBIDDEN = {"fastapi", "uvicorn", "starlette", "websockets", "pydantic", "matplotlib", "polars", "requests", "httpx", "tkinter", "_tkinter",
             "flask", "aiohttp", "echowatch", "patronsight", "imageio_ffmpeg"}


def third_party_imports(py: Path) -> set[str]:
    tree = ast.parse(py.read_text("utf-8"))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names.add(n.module.split(".")[0])
    return {n for n in names if n not in sys.stdlib_module_names and n != "echo_watch" and n != "__future__"}


def test_source_imports_only_the_allowed_third_party_set_and_nothing_forbidden():
    seen: dict[str, set[str]] = {}
    for py in sorted((ROOT / "src" / "echo_watch").rglob("*.py")):
        for n in third_party_imports(py):
            seen.setdefault(n, set()).add(py.name if py.parent.name != "yoloe" else "yoloe/" + py.name)
    assert not (set(seen) & FORBIDDEN), {k: v for k, v in seen.items() if k in FORBIDDEN}
    assert set(seen) <= ALLOWED_THIRD_PARTY, {k: v for k, v in seen.items() if k not in ALLOWED_THIRD_PARTY}
    # the heavy model libraries are confined to the worker process modules (EW-LAB-06 adds the YOLOE worker and its
    # BoT-SORT adapter, which run only in that GPU worker process)
    worker_modules = {"main.py", "yoloe/worker.py", "yoloe/tracker.py"}
    assert seen.get("torch", set()) <= worker_modules and seen.get("ultralytics", set()) <= worker_modules, seen


def test_no_module_of_the_application_path_names_a_legacy_location():
    import re
    pat = re.compile(r"E:\\\\?ECHO Watch(?! Native)|PatronSight Lab|runtime\\\\echowatch", re.I)
    offenders = []
    for py in sorted((ROOT / "src" / "echo_watch").rglob("*.py")):
        if pat.search(py.read_text("utf-8")):
            offenders.append(py.name)
    assert offenders == []                       # the legacy locations appear only in tools/tests that READ them for evidence
