# SPDX-License-Identifier: AGPL-3.0-only
"""Saved YOLOE test evidence: ``data/operator/yoloe-runs/<run_id>/manifest.json`` + ``trace.jsonl`` (EW-LAB-06).

Separate from the Library, saved setups and Display Assets runs (no database, no schema, no migration). The manifest is
replaced atomically (temp file + rename) at start, at each lifecycle change and at the end; a partial run keeps its
files and says so. The trace is append-only JSONL written by ONE background thread from a BOUNDED queue:
* ``put`` blocks up to a bound (back-pressure on the analysis loop: playback slows, nothing is buffered without limit);
  a put that still cannot be queued, a write error, the free-space floor (default 2 GiB) or the per-run cap (default
  10 GiB) stops accepting records and is reported as an evidence failure - never as silent loss;
* ``note`` (paint notes from the UI thread) never blocks; a full queue counts the note as dropped and discloses it;
* the writer keeps the running SHA-256, line and byte counts of what it actually wrote.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import shutil
import threading
import time
from pathlib import Path

SCHEMA = "echo-watch-lab/yoloe-run/1"
_STOP = object()


class EvidenceError(RuntimeError):
    pass


def atomic_json(path: Path, doc: dict) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


class TraceWriter:
    def __init__(self, path: Path, *, queue_max: int = 256, min_free_bytes: int = 2 * 2 ** 30,
                 cap_bytes: int = 10 * 2 ** 30, check_every: int = 64, disk_usage=shutil.disk_usage):
        self.path = Path(path)
        self.min_free, self.cap, self.check_every = min_free_bytes, cap_bytes, check_every
        self._disk_usage = disk_usage
        self._q: "queue.Queue" = queue.Queue(maxsize=queue_max)
        self._sha = hashlib.sha256()
        self.lines = 0
        self.bytes = 0
        self.notes_dropped = 0
        self.blocked_s = 0.0
        self.error: str | None = None
        self._closed = False
        self._fh = open(self.path, "ab")
        self._thread = threading.Thread(target=self._run, name="yoloe-trace", daemon=True)
        self._thread.start()

    @property
    def ok(self) -> bool:
        return self.error is None

    def put(self, rec: dict, block_s: float = 5.0) -> bool:
        """Queue one REQUIRED record (blocks up to ``block_s``). False = evidence failure (see ``error``)."""
        if self.error is not None or self._closed:
            return False
        t0 = time.monotonic()
        try:
            self._q.put(rec, timeout=block_s)
        except queue.Full:
            self.error = f"the trace writer could not keep up for {block_s:g} s (bounded queue full)"
            return False
        self.blocked_s += time.monotonic() - t0
        return True

    def note(self, rec: dict) -> bool:
        """Queue an OPTIONAL record without blocking (paint notes); a full queue counts it as dropped."""
        if self.error is not None or self._closed:
            return False
        try:
            self._q.put_nowait(rec)
            return True
        except queue.Full:
            self.notes_dropped += 1
            return False

    def _check_space(self) -> None:
        try:
            free = self._disk_usage(self.path.parent).free
        except OSError as e:
            self.error = f"free space could not be read: {e}"
            return
        if free < self.min_free:
            self.error = f"less than {self.min_free / 2 ** 30:.1f} GiB free disk space; evidence stopped"
        run_bytes = sum(p.stat().st_size for p in self.path.parent.iterdir() if p.is_file())
        if run_bytes > self.cap:
            self.error = f"this test's evidence reached its {self.cap / 2 ** 30:.1f} GiB cap; evidence stopped"

    def _run(self) -> None:
        while True:
            rec = self._q.get()
            if rec is _STOP:
                break
            if self.error is not None:
                continue
            try:
                line = (json.dumps(rec, separators=(",", ":"), default=str) + "\n").encode("utf-8")
                self._fh.write(line)
                self._fh.flush()
                self._sha.update(line)
                self.lines += 1
                self.bytes += len(line)
                if self.lines % self.check_every == 0:
                    self._check_space()
            except Exception as e:  # noqa: BLE001 - an evidence failure is reported, never hidden
                self.error = f"trace write failed: {type(e).__name__}: {e}"

    def close(self, timeout: float = 10.0) -> dict:
        """Flush what was queued (bounded) and report exactly what was written."""
        if not self._closed:
            self._closed = True
            try:
                self._q.put(_STOP, timeout=timeout)
            except queue.Full:
                self.error = self.error or "the trace writer did not drain before close"
            self._thread.join(timeout)
            if self._thread.is_alive():
                self.error = self.error or "the trace writer did not finish in time"
            try:
                self._fh.close()
            except OSError as e:
                self.error = self.error or f"trace close failed: {e}"
        return {"path": self.path.name, "lines": self.lines, "bytes": self.bytes, "sha256": self._sha.hexdigest(),
                "notes_dropped": self.notes_dropped, "blocked_s": round(self.blocked_s, 3), "error": self.error,
                "complete": self.error is None}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
