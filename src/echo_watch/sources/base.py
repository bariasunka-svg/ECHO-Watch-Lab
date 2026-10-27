# SPDX-License-Identifier: AGPL-3.0-only
"""Frame source contract.

Adapted from the preserved ECHO Watch reference runtime/echowatch/sources/base.py (SHA-256
09e59e682d4fd63008f751db4bcc3422b5a782f6debfe89a37e74e9b4bc2b6f4; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md).
Adaptation: imports only, plus ``SourceEnded`` lives here so every source can signal a normal end.
"""
from __future__ import annotations

import abc
import re

from ..frames import Epoch, Frame

_CRED_RE = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<user>[^:@/\s]+)(?::(?P<pw>[^@/\s]*))?@")


def redact(text: str) -> str:
    """Remove credentials from any URL-like text before it is logged or returned to a caller."""
    return _CRED_RE.sub(lambda m: f"{m.group('scheme')}{m.group('user')}:***@", text)


class SourceError(RuntimeError):
    pass


class SourceEnded(SourceError):
    """A non-repeating video file reached its end: the source is Ended, not failed and not reconnecting."""


class FrameSource(abc.ABC):
    synthetic = False
    kind = "base"

    def __init__(self, source_id: str):
        self.source_id = source_id
        self.epoch: Epoch | None = None

    @abc.abstractmethod
    def start(self, epoch: Epoch) -> None: ...

    @abc.abstractmethod
    def read(self, timeout: float) -> Frame | None:
        """Return the next decoded frame, or None if none arrived within ``timeout``."""

    @abc.abstractmethod
    def stop(self) -> None: ...

    def alive(self) -> bool:
        return True

    def diagnostics(self) -> dict:
        return {}
