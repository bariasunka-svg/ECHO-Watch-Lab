# SPDX-License-Identifier: AGPL-3.0-only
"""Same item / Undo / Reconcile interaction state for the workbench (Tk-free; EW-LAB-03 R1/R2).

The Tk page only renders what this returns. Rules:
* A save CAPTURES its immutable request (operation id, both profiles, primary, acknowledgement, the comparison and its
  binding). A retry repeats exactly that request with the same operation id; to change anything the operator compares
  again (a new comparison gets a new operation id).
* Background results carry (what, operation id, comparison generation, Outcome-or-exception). Known validation refusals
  (``CorrectionError`` family) say the request was refused and nothing was written by it. ANY other exception keeps
  the operation id and says the outcome is unknown - never "nothing was saved".
* A completion that belongs to an older comparison may report its operation's outcome but never clears or overwrites
  the newer comparison or its retry identity.
* Durable-but-undone, pending and uncertain outcomes are worded distinctly; the operator reaches pending/uncertain
  operations through the existing history list and "Reconcile selected".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..frames import new_id
from ..review.corrections import CorrectionError, Corrections, OperationConflict, Outcome, Preview, StalePreview

TONE = {"confirmed": "success", "legacy": "success", "confirmed_undone": "warning", "legacy_undone": "warning",
        "pending": "warning", "uncertain": "warning", "not_saved": "warning"}


@dataclass(frozen=True)
class Request:
    op_id: str
    gen: int
    preview: Preview
    primary: str
    acknowledged: bool


def _run(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - classified by CorrectionFlow.on_result, never raised into Tk
        return e


class CorrectionFlow:
    def __init__(self, corrections: Corrections, submit: Callable[[str, object, Callable[[], object]], object]):
        self.cor, self.submit = corrections, submit
        self.gen = 0
        self.preview: Preview | None = None
        self.request: Request | None = None
        self.undo_ops: dict[str, str] = {}          # target correction -> its Undo operation id (retries reuse it)

    # ---- comparison ----------------------------------------------------------------------------------------------------
    def compare(self, a: str, b: str) -> None:
        self.gen += 1
        self.preview, self.request = None, None      # a new comparison gets a new operation id when saved
        cor, g = self.cor, self.gen
        self.submit("corr_preview", ("corr", g), lambda: cor.preview(a, b))

    def on_preview(self, token, ok: bool, value, error: str | None) -> str | None:
        """Text for the comparison panel, or None when this completion belongs to an older comparison."""
        if token != ("corr", self.gen):
            return None
        if not ok:
            return f"Comparison failed: {error}"
        self.preview = value
        return describe(value, names=None)

    # ---- save / retry ----------------------------------------------------------------------------------------------------
    def save(self, primary_choice: str, acknowledged: bool) -> str | None:
        """Submit (or retry) the captured request; returns a notice for the operator, or None."""
        p = self.preview
        if p is None:
            return "Compare the two profiles first."
        primary = p.a if primary_choice == "a" else p.b
        notice = None
        r = self.request
        if r is None or r.gen != self.gen:
            r = self.request = Request(new_id("same"), self.gen, p, primary, bool(acknowledged))
        elif (r.primary, r.acknowledged) != (primary, bool(acknowledged)):
            notice = (f"Retrying the original request (operation {r.op_id}, primary {r.primary}). To change the primary "
                      "or the acknowledgement, Compare again.")
        cor = self.cor
        self.submit("corr_apply", ("corr", r.gen),
                    lambda: ("same", r.op_id, r.gen, _run(lambda: cor.apply_same_item(r.op_id, r.preview, r.primary,
                                                                                      r.acknowledged))))
        return notice

    def undo(self, target_correction: str) -> None:
        op = self.undo_ops.setdefault(target_correction, new_id("undo"))
        cor, g = self.cor, self.gen
        self.submit("corr_apply", ("corr", g), lambda: ("undo", op, g, _run(lambda: cor.undo(op, target_correction))))

    def reconcile(self, op_id: str) -> None:
        cor, g = self.cor, self.gen
        self.submit("corr_apply", ("corr", g), lambda: ("reconcile", op_id, g, _run(lambda: cor.reconcile(op_id))))

    def retry_op(self) -> str | None:
        return self.request.op_id if self.request is not None else None

    # ---- completion ------------------------------------------------------------------------------------------------------
    def on_result(self, value) -> tuple[str, str]:
        """(message, tone) for one finished save/undo/reconcile; updates retry identity only for its own operation."""
        what, op, gen, res = value
        mine = self.request is not None and self.request.op_id == op and gen == self.gen
        if isinstance(res, Exception):
            if isinstance(res, StalePreview):
                if mine:
                    self.preview, self.request = None, None
                return (f"Refused - nothing was written by this request: {res}. Compare the profiles again.", "danger")
            if isinstance(res, OperationConflict):
                return (f"Refused: {res}.", "danger")
            if isinstance(res, CorrectionError):
                return (f"Refused - nothing was written by this request: {res}.", "danger")
            return (f"Outcome unknown for operation {op} ({type(res).__name__}: {res}). It may already be saved or in "
                    "effect; select it in the history and use Reconcile, or press Save again to retry the same "
                    "operation.", "warning")
        out: Outcome = res
        msg = out.message
        if out.status == "not_saved" and what != "reconcile":
            msg += " Press Save again to retry the same operation." if what == "same" else ""
        if out.durable:
            if what in ("same", "reconcile") and mine:
                self.preview, self.request = None, None          # done: the next Save needs a fresh comparison
            if what == "undo":
                self.undo_ops = {k: v for k, v in self.undo_ops.items() if v != op}
        if out.effective and out.kind != "undo" and what != "undo":
            msg += " Activity shows the effective grouping; original records are unchanged."
        return msg, TONE.get(out.status, "warning")


def describe(p: Preview, names: dict | None = None) -> str:
    """The comparison panel text: both EFFECTIVE groups with every affected member, primaries, problems and the
    actual conflict evidence (display rounding only)."""
    by_id = {m["item_id"]: m for m in p.members}

    def label(i):
        m = by_id.get(i)
        return f"{m['name']} ({i})" if m else i

    def group(ids, prim):
        return ", ".join(label(i) + (" [primary]" if i == prim else "") for i in ids)
    lines = [f"First group ({len(p.group_a)}): {group(p.group_a, p.primary_a)}",
             f"Second group ({len(p.group_b)}): {group(p.group_b, p.primary_b)}"]
    for m in p.members:
        scope = (m.get("scope") or "unknown")[-12:]
        lines.append(f"  {m['name']} ({m['item_id']}): source ...{scope}, reference v{m['ref_version']}, "
                     f"{len(m['runs'])} run(s)")
    if p.problems:
        lines.append("Cannot combine: " + "; ".join(p.problems))
    for c in p.conflicts:
        a, b = c["items"]
        lines.append(f"CONFLICT: run {c['run_id']}: {label(a)} and {label(b)} were Visible at the same time "
                     f"({c['seen_together_s']} s over {len(c['evidence'])} interval pair(s)).")
    if p.compatible:
        total = len(set(p.group_a) | set(p.group_b))
        lines.append(f"Saving makes all {total} profiles one effective item, with the primary you choose below. "
                     "Original records stay unchanged." + (" Tick the conflict acknowledgement first." if p.conflicts else ""))
    return "\n".join(lines)
