# SPDX-License-Identifier: AGPL-3.0-only
"""Owned process trees through a Windows Job Object (qualification and application shutdown ownership).

Adapted from the preserved ECHO Watch reference tools/owned_jobs.py (AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md): ownership is exactly
"the job's live members" - never a command-line or start-time scan - so an unrelated process of the owner (even one from
the same folder) is never decided to be ours and never touched. Differences: the CURRENT process can be adopted into a
fresh kill-on-close job (every later child it spawns, windowless, is then a member and dies with the job), and the member
list is exposed to the resource sampler so the 'owned family' CPU / RSS is measured, not guessed.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes

import psutil

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateJobObjectW.restype = wintypes.HANDLE
_k32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
_k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
_k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
_k32.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD,
                                           ctypes.POINTER(wintypes.DWORD))
_k32.CloseHandle.argtypes = (wintypes.HANDLE,)


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rb", "wb", "ob")]


class _BASIC(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _EXT(ctypes.Structure):
    _fields_ = [("Basic", _BASIC), ("Io", _IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


class OwnedJob:
    def __init__(self, kill_on_close: bool = True) -> None:
        self.job = _k32.CreateJobObjectW(None, None)
        if not self.job:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        if kill_on_close:
            info = _EXT()
            info.Basic.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _k32.SetInformationJobObject(self.job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")

    def adopt_current_process(self) -> None:
        if not _k32.AssignProcessToJobObject(self.job, _k32.GetCurrentProcess()):
            raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject(current process) failed")

    def pids(self) -> list[int]:
        n = 4096

        class L(ctypes.Structure):
            _fields_ = [("assigned", wintypes.DWORD), ("in_list", wintypes.DWORD), ("ids", ctypes.c_size_t * n)]
        buf = L()
        if not _k32.QueryInformationJobObject(self.job, 3, ctypes.byref(buf), ctypes.sizeof(buf), None):
            raise OSError(ctypes.get_last_error(), "QueryInformationJobObject failed")
        return [int(buf.ids[i]) for i in range(buf.in_list)]

    def leftovers(self, exclude: tuple[int, ...] = ()) -> list[dict]:
        out = []
        for pid in self.pids():
            if pid in exclude:
                continue
            try:
                q = psutil.Process(pid)
                if q.is_running() and q.status() != psutil.STATUS_ZOMBIE:
                    out.append({"pid": pid, "name": q.name(), "cmd": " ".join(q.cmdline())[:200]})
            except psutil.Error:
                pass
        return out

    def terminate(self) -> None:
        _k32.TerminateJobObject(self.job, 1)

    def close(self) -> None:
        if self.job:
            _k32.CloseHandle(self.job)
            self.job = None
