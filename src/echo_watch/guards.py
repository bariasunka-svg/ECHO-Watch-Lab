# SPDX-License-Identifier: AGPL-3.0-only
"""Process-level guards: loopback-only networking and windowless child processes.

Adapted from the preserved ECHO Watch reference runtime/echowatch/guards.py (SHA-256
81ff9c807b4229e78931e8a6d07cece79258a93d066d8f1121e03ae60f272fc5; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md).
Adaptation: package/import names and messages only. Behaviour is identical (see tests/test_adapted_modules.py).

The audit hook refuses outbound connections and name lookups to anything other than loopback and records each
refusal, so diagnostics can show that no package attempted a silent download or telemetry call.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from collections import deque

from .config import CREATE_NO_WINDOW

_LOOPBACK = {"127.0.0.1", "::1", "localhost", None, ""}
_lock = threading.Lock()
egress_refusals: deque[tuple[str, str]] = deque(maxlen=200)
subprocess_log: deque[str] = deque(maxlen=200)
_installed = False
_block_subprocess = False


def _host_of(addr) -> str | None:
    if isinstance(addr, tuple) and addr:
        return str(addr[0])
    if isinstance(addr, (str, bytes)):
        return None  # AF_UNIX / pipe style addresses are local
    return None


def _hook(event: str, args) -> None:
    if event == "socket.connect":
        host = _host_of(args[1])
        if host not in _LOOPBACK:
            with _lock:
                egress_refusals.append(("connect", str(host)))
            raise PermissionError(f"ECHO Watch Lab egress guard: outbound connection to {host} refused")
    elif event == "socket.getaddrinfo":
        host = args[0]
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if host not in _LOOPBACK:
            with _lock:
                egress_refusals.append(("getaddrinfo", str(host)))
            raise PermissionError(f"ECHO Watch Lab egress guard: name lookup for {host} refused")
    elif event == "subprocess.Popen":
        exe = str(args[0])
        with _lock:
            subprocess_log.append(exe)
        if _block_subprocess:
            raise PermissionError(f"ECHO Watch Lab guard: subprocess {exe} refused in this process")


def install(block_subprocess: bool = False) -> None:
    """Install once per process. Audit hooks cannot be removed afterwards."""
    global _installed, _block_subprocess
    _block_subprocess = block_subprocess
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


def popen_windowless(args: list[str], **kw) -> subprocess.Popen:
    """Every child process the engine starts goes through here: no console window, no stdin."""
    kw.setdefault("stdin", subprocess.DEVNULL)
    flags = kw.pop("creationflags", 0) | CREATE_NO_WINDOW
    return subprocess.Popen(args, creationflags=flags, close_fds=True, **kw)


def snapshot() -> dict:
    with _lock:
        return {"egress_refusals": list(egress_refusals), "subprocesses": list(subprocess_log)}
