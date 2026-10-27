# SPDX-License-Identifier: AGPL-3.0-only
"""Single-item mailboxes: the latest-only handoff between producer and consumer threads.

Adopted from the frozen PatronSight reference (src/patronsight/live_inference.py ``_LatestItemSlot``, SHA-256
7638cabf4e98261c62dfee631f27b60b1b2d818e032ea03503ee0a1132a3dd1e, and the identical contract in
src/patronsight/live_source.py ``_LatestFrameSlot``, SHA-256 cdfed33cc46d32e6e0c7ab51dd9aa8fb5c47246d75b932a7f27cd181d5d13268;
LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only). publish() always keeps only the newest item; replacing an item nobody has taken yet counts as
dropped; take() returns the newest only once; peek() never consumes. Capacity is exactly one: there is never a backlog.

Addition: ``wait_take`` blocks on a condition instead of polling, so the consumer wakes the instant a newer item arrives
(the reference polled every 20 ms). Behaviour of publish/take/peek/clear/dropped is unchanged.
"""
from __future__ import annotations

import threading
from typing import Generic, TypeVar

T = TypeVar("T")


class LatestSlot(Generic[T]):
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._item: T | None = None
        self._consumed = True
        self._closed = False
        self.dropped = 0
        self.published = 0
        self.taken = 0

    def publish(self, item: T) -> None:
        with self._cond:
            if self._item is not None and not self._consumed:
                self.dropped += 1
            self._item = item
            self._consumed = False
            self.published += 1
            self._cond.notify_all()

    def take(self) -> T | None:
        """The newest item, only if it has not already been taken."""
        with self._cond:
            if self._item is None or self._consumed:
                return None
            self._consumed = True
            self.taken += 1
            return self._item

    def wait_take(self, timeout: float) -> T | None:
        """Block until a not-yet-taken item exists (or ``timeout``/close), then take it."""
        with self._cond:
            if (self._item is None or self._consumed) and not self._closed:
                self._cond.wait(timeout)
            if self._item is None or self._consumed:
                return None
            self._consumed = True
            self.taken += 1
            return self._item

    def peek(self) -> T | None:
        """The newest item regardless of whether it was already taken."""
        with self._cond:
            return self._item

    def clear(self) -> None:
        with self._cond:
            self._item = None
            self._consumed = True

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def reopen(self) -> None:
        with self._cond:
            self._closed = False
