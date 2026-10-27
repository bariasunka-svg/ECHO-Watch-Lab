# SPDX-License-Identifier: AGPL-3.0-only
"""Decision trace: a bounded, asynchronous, append-only JSONL record of every capture, attempt, decision and output.

Modelled on the lifecycle the frozen PatronSight reference gives its identity trace (src/patronsight/live_session.py:
required output; terminal states finalized | partially_finalized | failed; dropped-record counts reported, never hidden)
and on the preserved ECHO reference's run trace (flush interval <= 250 ms, sticky overflow counters). One writer thread
owns the file: ``emit`` never blocks the capture or analysis threads - a full queue drops the record and counts it, and
the final state is then ``partial`` (never silently complete). ``finish`` drains, flushes, fsyncs and returns the file's
SHA-256 so an output's existence is never mistaken for its completeness.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from pathlib import Path

STATE_IDLE = "idle"
STATE_RECORDING = "recording"
STATE_FINALIZING = "finalizing"
STATE_FINALIZED = "finalized"
STATE_PARTIAL = "partial"
STATE_FAILED = "failed"
TERMINAL = (STATE_FINALIZED, STATE_PARTIAL, STATE_FAILED)
FLUSH_S = 0.25


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001
        pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return repr(o)


class TraceWriter:
    def __init__(self, path: Path, maxq: int = 50_000):
        self.path = Path(path)
        self._q: "queue.Queue[tuple | None]" = queue.Queue(maxsize=maxq)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.state = STATE_IDLE
        self.error: str | None = None
        self.records = 0          # written
        self.offered = 0
        self.dropped = 0
        self.bytes = 0
        self.sha256: str | None = None
        self._n = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8", newline="\n")
        self.state = STATE_RECORDING
        self._thread = threading.Thread(target=self._run, name="decision-trace", daemon=True)
        self._thread.start()

    def emit(self, kind: str, **fields) -> bool:
        """Non-blocking. Returns False (and counts a sticky drop) when the bounded queue is full or not recording."""
        if self.state != STATE_RECORDING:
            return False
        with self._lock:
            self._n += 1
            n = self._n
        rec = {"n": n, "k": kind, "t": time.monotonic(), "w": time.time(), **fields}
        self.offered += 1
        try:
            self._q.put_nowait((rec,))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _run(self) -> None:
        last_flush = time.monotonic()
        try:
            while True:
                try:
                    item = self._q.get(timeout=FLUSH_S)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    self._fh.flush()
                    continue
                if item is None:
                    break
                line = json.dumps(item[0], separators=(",", ":"), default=_default) + "\n"
                self._fh.write(line)
                self.records += 1
                self.bytes += len(line.encode("utf-8"))
                if time.monotonic() - last_flush >= FLUSH_S:
                    self._fh.flush()
                    last_flush = time.monotonic()
        except Exception as e:  # noqa: BLE001 - a trace failure is recorded; it never propagates into the run
            self.error = f"{type(e).__name__}: {e}"
        finally:
            try:
                self._fh.flush()
                import os
                os.fsync(self._fh.fileno())
                self._fh.close()
            except Exception as e:  # noqa: BLE001
                self.error = self.error or f"{type(e).__name__}: {e}"

    def finish(self, timeout: float = 30.0) -> dict:
        """Stop accepting records, drain the queue, close the file; terminal state is set only by the writer's exit."""
        if self.state == STATE_RECORDING:
            self.state = STATE_FINALIZING
        self._stop.set()
        try:
            self._q.put(None, timeout=1.0)
        except queue.Full:
            pass
        t = self._thread
        if t is not None:
            t.join(timeout)
            if t.is_alive():
                self.error = self.error or f"trace writer did not finish within {timeout:g}s"
                self.state = STATE_FAILED
                return self.status()
        if self.error:
            self.state = STATE_FAILED
        else:
            self.state = STATE_PARTIAL if self.dropped else STATE_FINALIZED
        try:
            h = hashlib.sha256()
            with open(self.path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            self.sha256 = h.hexdigest()
        except OSError as e:
            self.error = self.error or str(e)
            self.state = STATE_FAILED
        return self.status()

    def status(self) -> dict:
        return {"state": self.state, "path": str(self.path), "records": self.records, "offered": self.offered,
                "dropped": self.dropped, "bytes": self.bytes, "sha256": self.sha256, "error": self.error}
