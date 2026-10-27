# SPDX-License-Identifier: AGPL-3.0-only
"""Replay-source content scope (EW-LAB-02 repair B).

For this file-replay product the correction scope of a profile is the exact content of the video file it was scanned
from: ``file-sha256:<64 hex>``. Same bytes at another path are the same replay scope; different bytes at the same
resolution are different. This is a narrow replay-scope rule, not a claim about physical stores.

Hashing is bounded chunked I/O, cancellable between chunks, and runs only in background tasks (never on the Tk
thread). The file's size and modification time are captured before and after hashing; a change means the selection
is no longer trustworthy and the caller must invalidate the setup and ask for the file again.
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Callable

SCOPE_PREFIX = "file-sha256:"
CHUNK = 1 << 20
_SCOPE_RE = re.compile(r"^file-sha256:[0-9a-f]{64}$")


class SourceChanged(RuntimeError):
    """The selected file changed while (or after) it was prepared."""


class Cancelled(RuntimeError):
    """A newer selection superseded this preparation."""


def is_known_scope(scope) -> bool:
    return isinstance(scope, str) and bool(_SCOPE_RE.match(scope))


def file_state(path) -> tuple[int, int]:
    st = os.stat(path)
    return (st.st_size, st.st_mtime_ns)


def fingerprint(path, cancelled: Callable[[], bool] = lambda: False, chunk: int = CHUNK) -> dict:
    p = Path(path)
    before = file_state(p)
    h = hashlib.sha256()
    n = 0
    with open(p, "rb") as fh:
        while True:
            if cancelled():
                raise Cancelled("a newer video selection superseded this one")
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
            n += len(b)
    after = file_state(p)
    if after != before or n != before[0]:
        raise SourceChanged("the video file changed while it was being read; choose it again")
    d = h.hexdigest()
    return {"scope": SCOPE_PREFIX + d, "sha256": d, "bytes": n, "state": list(after)}


def verify_unchanged(path, state) -> None:
    try:
        now = list(file_state(path))
    except OSError as e:
        raise SourceChanged(f"the video file is no longer available ({e}); choose it again") from e
    if now != list(state):
        raise SourceChanged("the video file changed after it was chosen; choose it again and Scan")
