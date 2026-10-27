# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded background tasks for the workbench (Tk-free).

Every slow operation (source preparation, Scan, worker start, Remember, Start, history/Library/run queries, review
decoding, export, finalization waits) runs on a small fixed pool of daemon threads, never on the Tk event thread.
Each submission carries a ``token`` (the controller's generation for that concern); completions are drained by the UI
thread and a completion whose token is no longer current is DISCARDED, so a late Scan or history load can never
repopulate a changed view. Shutdown is bounded. Pattern adapted from the frozen PatronSight reference
app/review_player.py (LatestWorker: outbox drained by ``after`` polling, the worker never touches Tk).
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Completion:
    kind: str
    token: Any
    ok: bool
    value: Any
    error: str | None
    seconds: float
    detail: str | None = None


class TaskRunner:
    def __init__(self, workers: int = 3, max_queued: int = 64):
        self._q: "queue.Queue[tuple | None]" = queue.Queue(maxsize=max_queued)
        self._out: "queue.Queue[Completion]" = queue.Queue()
        self._stop = threading.Event()
        self._threads = [threading.Thread(target=self._loop, name=f"workbench-task-{i}", daemon=True) for i in range(workers)]
        for t in self._threads:
            t.start()
        self.running: dict[str, int] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, token: Any, fn: Callable[[], Any]) -> bool:
        if self._stop.is_set():
            return False
        try:
            self._q.put_nowait((kind, token, fn))
        except queue.Full:
            self._out.put(Completion(kind, token, False, None, "too many pending operations; try again", 0.0))
            return False
        with self._lock:
            self.running[kind] = self.running.get(kind, 0) + 1
        return True

    def busy(self, kind: str) -> bool:
        with self._lock:
            return self.running.get(kind, 0) > 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            kind, token, fn = item
            t0 = time.monotonic()
            try:
                v = fn()
                c = Completion(kind, token, True, v, None, time.monotonic() - t0)
            except Exception as e:  # noqa: BLE001 - reported to the UI, never raised into Tk
                c = Completion(kind, token, False, None, f"{type(e).__name__}: {e}", time.monotonic() - t0,
                               traceback.format_exc(limit=6))
            with self._lock:
                self.running[kind] = max(0, self.running.get(kind, 1) - 1)
            self._out.put(c)

    def drain(self, limit: int = 50) -> list[Completion]:
        out = []
        while len(out) < limit:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                break
        return out

    def shutdown(self, timeout: float = 2.0) -> bool:
        self._stop.set()
        for _ in self._threads:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
        end = time.monotonic() + timeout
        for t in self._threads:
            t.join(max(0.0, end - time.monotonic()))
        return not any(t.is_alive() for t in self._threads)
