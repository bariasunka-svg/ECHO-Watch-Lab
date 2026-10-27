# SPDX-License-Identifier: AGPL-3.0-only
"""The ONE enrollment compatibility predicate (EW-LAB-05).

A saved reference may take part in a setup for the current recording only when all of these hold:
* its profile's content scope is a known replay scope (``file-sha256:<hex>``) EQUAL to the recording's scope - same
  bytes at another path are the same scope; a different file is a different scope even at identical resolution;
* its reference frame size equals the recording's frame size;
* its reference was taken with the same discovery model identity (when the recording's model identity is known);
* the profile is active.

Unknown legacy scopes (for example the old universal "setup") stay readable in Library and history but are never
compatible with anything - not even with an equal unknown string. Same resolution alone is never enough. This is a
replay-scope rule; it does not claim or reject physical identity across videos.

The predicate is applied to Scan overlap hints, saved-item reuse in Select, reference replacement, saved-setup
loading and the engine-facing Start boundary. Compatibility never means Visible and is never a recognition result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..sources.fingerprint import is_known_scope


@dataclass(frozen=True)
class Scope:
    """The current recording's enrollment scope."""

    content: str | None                       # file-sha256:<hex>
    size: tuple[int, int] | None = None       # (width, height) of the recording's frames
    model_sha256: str | None = None           # discovery model identity of the current scan / pinned model


@dataclass(frozen=True)
class Verdict:
    ok: bool
    code: str          # ok | inactive | unscoped_source | legacy_scope | other_source | size | model
    reason: str


OK = Verdict(True, "ok", "saved from this recording")


def check(item_scope, ref_size, ref_model, scope: Scope, status: str | None = "active") -> Verdict:
    if status not in (None, "active"):
        return Verdict(False, "inactive", "this profile is no longer active")
    if not is_known_scope(scope.content):
        return Verdict(False, "unscoped_source", "the current recording has no content identity; prepare the video again")
    if not is_known_scope(item_scope):
        return Verdict(False, "legacy_scope", "saved before recordings were identified by their content; its history "
                                              "stays readable, but it cannot be reused - remember the item again for "
                                              "this recording")
    if item_scope != scope.content:
        return Verdict(False, "other_source", "saved from a different recording (different file content); remember the "
                                              "item freshly for this recording - identities are not carried across videos")
    if scope.size is not None and ref_size is not None and tuple(int(v) for v in ref_size) != tuple(int(v) for v in scope.size):
        return Verdict(False, "size", f"its reference is {ref_size[0]}x{ref_size[1]} but this recording is "
                                      f"{scope.size[0]}x{scope.size[1]}")
    if scope.model_sha256 is not None and ref_model != scope.model_sha256:
        return Verdict(False, "model", "its reference was taken with a different discovery model" if ref_model else
                       "its reference does not record which discovery model produced it")
    return OK


def check_row(row: Mapping, scope: Scope) -> Verdict:
    """``row``: a profile/reference row with source_id, frame_width, frame_height, model_sha256 and status."""
    w, h = row.get("frame_width"), row.get("frame_height")
    return check(row.get("source_id"), (w, h) if w is not None and h is not None else None, row.get("model_sha256"),
                 scope, row.get("status", "active"))
