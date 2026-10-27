# SPDX-License-Identifier: AGPL-3.0-only
"""Saved display setups: versioned, validated JSON artifacts (EW-LAB-05).

A setup version captures exactly what one Start will use: display/setup name, the recording's content identity and
metadata, the exact item ids and reference versions (with their crop SHA-256), output options, the supported-Not-visible
review allowance and the model/calibration identity. Files live under ``<operator data>/setups/<setup_id>/v<N>.json``,
are written atomically (temp file + replace) and only by an explicit Save configuration or a Start that first saves
this configuration. A version file is never overwritten or deleted: a changed configuration becomes version N+1 and
old versions keep the reference versions they actually saved. Saving an unchanged configuration returns the existing
version. No database schema change or migration is involved.

Loading validates the schema name/version, rejects unknown fields, checks the content digest and the file's own
id/version, and checks value types and ranges. Link validation against the Library (reference versions present, crop
hashes unchanged, recording compatible) is done by the caller with engine.compat; a failing link is a visible,
recoverable state and never alters the saved file. Selecting a setup never opens a device or starts a run.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..frames import new_id
from .files import write_atomic

SCHEMA = "echo-watch.display-setup"
SCHEMA_VERSION = 1
ALLOWANCE_MIN, ALLOWANCE_MAX, ALLOWANCE_DEFAULT = 1, 3600, 60
OUTPUT_KEYS = ("clean_source", "record_annotated", "decision_trace", "repeat")
FIELDS = ("schema", "schema_version", "setup_id", "version", "name", "created_wall", "previous_version", "source",
          "items", "outputs", "review_allowance_s", "model", "calibration", "policy_version", "digest")
SOURCE_FIELDS = ("scope", "name", "path", "bytes", "width", "height", "fps")
ITEM_FIELDS = ("item_id", "name", "ref_version", "crop_sha256")
_ID_RE = re.compile(r"^setup_[0-9a-f]{16}$")
_FILE_RE = re.compile(r"^v([1-9][0-9]*)\.json$")


class SetupError(ValueError):
    """A setup could not be saved or a saved setup is invalid (the message is operator-facing)."""


def validate_allowance(value) -> int:
    """The review allowance: an integer number of seconds, 1-3600."""
    if isinstance(value, bool):
        raise SetupError("The review allowance must be a whole number of seconds")
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise SetupError("The review allowance must be a whole number of seconds") from None
    if f != int(f):
        raise SetupError("The review allowance must be a whole number of seconds")
    n = int(f)
    if not ALLOWANCE_MIN <= n <= ALLOWANCE_MAX:
        raise SetupError(f"The review allowance must be between {ALLOWANCE_MIN} and {ALLOWANCE_MAX} seconds")
    return n


def content_digest(doc: dict) -> str:
    body = {k: v for k, v in doc.items() if k not in ("digest", "setup_id", "version", "previous_version", "created_wall")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SavedSetup:
    setup_id: str
    version: int
    doc: dict
    path: str

    @property
    def name(self) -> str:
        return self.doc["name"]

    @property
    def scope(self) -> str:
        return self.doc["source"]["scope"]

    @property
    def items(self) -> tuple[tuple[str, int], ...]:
        return tuple((i["item_id"], int(i["ref_version"])) for i in self.doc["items"])

    @property
    def allowance_s(self) -> int:
        return int(self.doc["review_allowance_s"])

    @property
    def outputs(self) -> dict:
        return dict(self.doc["outputs"])

    @property
    def digest(self) -> str:
        return self.doc["digest"]

    def identity(self) -> dict:
        """What a run records about the configuration it used (and what the session binds against)."""
        return {"setup_id": self.setup_id, "version": self.version, "digest": self.digest, "name": self.name,
                "scope": self.scope, "items": [list(x) for x in self.items], "review_allowance_s": self.allowance_s}


def build(name: str, source: dict, items: list[dict], outputs: dict, allowance_s, model: dict, calibration: dict,
          policy_version: str) -> dict:
    """An unsaved setup document (validated). ``items``: [{item_id, name, ref_version, crop_sha256}]."""
    name = (name or "").strip()
    if not name or len(name) > 80:
        raise SetupError("The display name needs 1-80 characters")
    if not items:
        raise SetupError("A setup needs at least one remembered item")
    doc = {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "name": name,
           "source": {k: source.get(k) for k in SOURCE_FIELDS},
           "items": [{k: (int(i[k]) if k == "ref_version" else i[k]) for k in ITEM_FIELDS} for i in items],
           "outputs": {k: bool(outputs.get(k, k != "repeat")) for k in OUTPUT_KEYS},
           "review_allowance_s": validate_allowance(allowance_s),
           "model": dict(model or {}), "calibration": dict(calibration or {}), "policy_version": policy_version}
    _check_values(doc)
    return doc


def _check_values(doc: dict) -> None:
    src = doc.get("source")
    if not isinstance(src, dict) or not isinstance(src.get("scope"), str) or not src.get("scope"):
        raise SetupError("The saved setup has no recording identity")
    if set(src) - set(SOURCE_FIELDS):
        raise SetupError(f"The saved setup's recording has unknown fields: {sorted(set(src) - set(SOURCE_FIELDS))}")
    items = doc.get("items")
    if not isinstance(items, list) or not items:
        raise SetupError("The saved setup lists no items")
    seen = set()
    for i in items:
        if not isinstance(i, dict) or set(i) != set(ITEM_FIELDS):
            raise SetupError("A saved item entry is malformed")
        if not isinstance(i["item_id"], str) or not isinstance(i["ref_version"], int) or isinstance(i["ref_version"], bool) \
                or i["ref_version"] < 1 or not isinstance(i["crop_sha256"], str):
            raise SetupError("A saved item entry has an invalid id, reference version or hash")
        if i["item_id"] in seen:
            raise SetupError("The saved setup lists an item twice")
        seen.add(i["item_id"])
    outs = doc.get("outputs")
    if not isinstance(outs, dict) or set(outs) != set(OUTPUT_KEYS) or not all(isinstance(v, bool) for v in outs.values()):
        raise SetupError("The saved setup's output options are malformed")
    validate_allowance(doc.get("review_allowance_s"))
    if isinstance(doc.get("review_allowance_s"), bool) or not isinstance(doc.get("review_allowance_s"), int):
        raise SetupError("The saved review allowance is not a whole number of seconds")
    for k in ("model", "calibration"):
        if not isinstance(doc.get(k), dict):
            raise SetupError(f"The saved setup's {k} identity is malformed")


class SetupStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _dir(self, setup_id: str) -> Path:
        if not _ID_RE.match(setup_id or ""):
            raise SetupError("Unknown setup id")
        return self.root / setup_id

    def versions(self, setup_id: str) -> list[int]:
        d = self._dir(setup_id)
        if not d.is_dir():
            return []
        return sorted(int(m.group(1)) for p in d.iterdir() if (m := _FILE_RE.match(p.name)))

    def save(self, doc: dict, setup_id: str | None = None) -> SavedSetup:
        """Atomically save ``doc`` as the next version of ``setup_id`` (a new setup when None). An unchanged
        configuration returns the existing latest version; an existing version file is never overwritten."""
        _check_values(doc)
        digest = content_digest(doc)
        with self._lock:
            sid = setup_id or new_id("setup")
            d = self._dir(sid)
            vers = self.versions(sid)
            if vers:
                latest = self.load(sid, vers[-1])
                if latest.digest == digest:
                    return latest
            version = (vers[-1] if vers else 0) + 1
            path = d / f"v{version}.json"
            if path.exists():
                raise SetupError("A setup version with this number already exists; nothing was overwritten")
            out = {**doc, "setup_id": sid, "version": version, "previous_version": vers[-1] if vers else None,
                   "created_wall": time.time(), "digest": digest}
            write_atomic(path, json.dumps(out, indent=1, sort_keys=True).encode("utf-8"))
        return self.load(sid, version)

    def load(self, setup_id: str, version: int | None = None) -> SavedSetup:
        vers = self.versions(setup_id)
        if not vers:
            raise SetupError("This saved setup no longer exists")
        v = vers[-1] if version is None else int(version)
        path = self._dir(setup_id) / f"v{v}.json"
        try:
            doc = json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            raise SetupError(f"Version {v} of this setup does not exist") from None
        except (OSError, ValueError) as e:
            raise SetupError(f"The saved setup file could not be read ({type(e).__name__})") from None
        if not isinstance(doc, dict):
            raise SetupError("The saved setup file is malformed")
        if doc.get("schema") != SCHEMA:
            raise SetupError("This file is not an ECHO Watch display setup")
        if doc.get("schema_version") != SCHEMA_VERSION:
            raise SetupError(f"This setup was saved by an unsupported version (schema {doc.get('schema_version')})")
        unknown = sorted(set(doc) - set(FIELDS))
        if unknown:
            raise SetupError(f"The saved setup has unknown fields {unknown}; it is kept unchanged but cannot be used")
        missing = sorted(set(FIELDS) - set(doc))
        if missing:
            raise SetupError(f"The saved setup is missing {missing}")
        if doc["setup_id"] != setup_id or doc["version"] != v:
            raise SetupError("The saved setup file does not match its own id/version")
        _check_values(doc)
        if content_digest(doc) != doc["digest"]:
            raise SetupError("The saved setup's content does not match its digest (it was edited outside ECHO Watch)")
        return SavedSetup(setup_id, v, doc, str(path))

    def list_latest(self) -> list[dict]:
        """Every setup's latest version (or its load problem). Never raises for one bad setup."""
        out = []
        if not self.root.is_dir():
            return out
        for d in sorted(self.root.iterdir()):
            if not d.is_dir() or not _ID_RE.match(d.name):
                continue
            try:
                s = self.load(d.name)
                out.append({"setup_id": s.setup_id, "version": s.version, "name": s.name, "scope": s.scope,
                            "source_name": s.doc["source"].get("name"), "items": len(s.items),
                            "created_wall": s.doc["created_wall"], "versions": len(self.versions(d.name)), "error": None})
            except SetupError as e:
                out.append({"setup_id": d.name, "version": None, "name": d.name, "scope": None, "source_name": None,
                            "items": 0, "created_wall": None, "versions": len(self.versions(d.name)), "error": str(e)})
        out.sort(key=lambda r: -(r["created_wall"] or 0))
        return out
