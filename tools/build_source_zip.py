# SPDX-License-Identifier: AGPL-3.0-only
"""Build the Lab source ZIP + manifest (source, tests, tools, requirements, docs, notices, launch shortcut). Excludes
environments, vendor caches, models, inputs, data, logs and closeout evidence. Never overwrites an existing archive."""
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "qualify"))
import common  # noqa: E402

INCLUDE = ["src", "tests", "tools", "requirements", "docs", "Work Packages"]
FILES = ["AGENTS.md", "README.md", "LICENSE", "NOTICE.md", "THIRD_PARTY_NOTICES.md", "pytest.ini", "ECHO Watch Lab.lnk"]
tree = common.src_digest()
out = ROOT / "dist" / "source" / f"echo-watch-lab-0.3.0-{tree[:12]}-source.zip"
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    print("exists; preserved", out)
    sys.exit(1)
entries = []
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    paths = [ROOT / f for f in FILES] + [p for d in INCLUDE for p in sorted((ROOT / d).rglob("*"))]
    for p in paths:
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        rel = p.relative_to(ROOT).as_posix()
        data = p.read_bytes()
        z.writestr(f"echo-watch-lab/{rel}", data)
        entries.append({"path": rel, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
man = {"archive": out.name, "src_tree_sha256": tree, "files": len(entries), "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
       "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "entries": entries}
out.with_suffix(".manifest.json").write_text(json.dumps(man, indent=1), "utf-8")
print(out, man["sha256"], man["files"])
