# SPDX-License-Identifier: AGPL-3.0-only
"""Append-only operator corrections for remembered items: "Same item" grouping and Undo (correction format 2).

Patterns adapted (no code imported) from the frozen PatronSight reference src/patronsight/visitor_corrections.py and
docs/VISITOR_CORRECTIONS.md (LICENSE: 'Original PatSight contributions: Copyright (c) 2026 Bari Asunka', AGPL-3.0-only):
an effective grouping above preserved original observations; compare before saving; stale previews refused; save and
read back before apply; idempotent retries; Undo appends a reversal and stays reachable.

EW-LAB-02:
* SCOPE is the replay source's exact content (``file-sha256:<digest>``, bound when the video was chosen). Equal frame
  size is geometry, not scope. A profile whose scope is unknown (legacy shared "setup" id) or whose current reference
  came from a different scope is never grouped; nothing is inferred from names, dimensions or re-hashing old paths.
* DURABLE OPERATIONS: (1) the operation is written as PENDING with its stable id and immutable request digest;
  (2) it is read back and verified; (3) an append-only CONFIRMATION is written. Only confirmed operations (and legacy
  rows) affect grouping - for Undo too. Retries and reconciliation use the SAME operation id.
* LEGACY RULE: format-1 rows (``format`` NULL) were effective immediately; they stay effective as saved ("earlier format,
  not readback-confirmed"). They are not silently unapplied or relabelled.

EW-LAB-04 (Lab-03 review C1): a known rollback proves only what THAT transaction wrote. The operation id's durable
state is then read afresh: absent -> nothing saved by anyone at that check; present -> its reconciled status;
unreadable -> uncertain. A missing/mismatched/unreadable readback never demotes a known confirmation (uncertain instead).

EW-LAB-03 (independent review findings R1 and R2):
* TRUTHFUL OUTCOMES (R1). Every persistence read (initial lookup, readback, confirmation lookup, reconciliation) and
  every write acknowledgement is guarded: an exception never becomes "not saved". Only a successful authoritative read
  that finds no record, or a KNOWN rollback before commit, may say "not saved"; anything else is ``uncertain`` with the
  stable operation id and Reconcile as the route. Inside the confirmation transaction the exact operation and its own
  confirmation are re-read FIRST: if another caller already confirmed it, its durable status is returned before any
  freshness check, so a confirmed operation is never reported pending/not applied.
* CONFIRMED vs IN EFFECT (R1). A confirmed Same item later undone reports ``confirmed_undone`` ("saved and later undone;
  not in effect"); reconciliation never redoes it. History, Activity, reopen and messages read the same fold.
* GROUP BINDING (R2). A comparison is between the two EFFECTIVE GROUPS containing the chosen profiles. The binding
  digests every affected member (status, reference version + crop hash, scope sources), both groups' memberships and
  effective primaries, every correction/confirmation/Undo revision that involves an affected member (unrelated ones
  are excluded), and the exact conflict evidence (interval ids and unrounded times) across the groups' member pairs.
  It is recomputed in the pending-save transaction and again in the confirmation transaction (excluding the operation's
  own row). Membership growth with an unchanged primary, merge-then-Undo, reference/scope changes and new conflicts all
  require a fresh comparison; an identical, already confirmed same-id retry stays idempotent.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from ..engine.store import Store, dumps, loads
from ..frames import new_id
from ..sources.fingerprint import is_known_scope

KIND_SAME = "same_item"
KIND_UNDO = "undo"
FORMAT = 2
EFFECTIVE = ("confirmed", "legacy")


class CorrectionError(ValueError):
    """A known validation refusal: nothing was written by this call."""


class StalePreview(CorrectionError):
    pass


class OperationConflict(CorrectionError):
    """The operation id was already used for a DIFFERENT request."""


class SaveNotConfirmed(CorrectionError):
    """Kept for callers of format 1; format 2 returns an ``Outcome`` with status ``uncertain`` instead."""


@dataclass(frozen=True)
class Outcome:
    # confirmed | confirmed_undone | legacy | legacy_undone | pending | uncertain | not_saved
    status: str
    op_id: str
    correction_id: str | None
    message: str
    kind: str | None = None

    @property
    def effective(self) -> bool:
        """The operation's effect is part of the current grouping."""
        return self.status in EFFECTIVE

    @property
    def durable(self) -> bool:
        """The operation is durably saved and confirmed (or legacy), whether or not it is still in effect."""
        return self.status in ("confirmed", "confirmed_undone", "legacy", "legacy_undone")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


@dataclass
class Grouping:
    parent: dict[str, str] = field(default_factory=dict)
    primary: dict[str, str] = field(default_factory=dict)     # root -> primary item id
    active: list[dict] = field(default_factory=list)
    reversed_ids: set[str] = field(default_factory=set)

    def _find(self, x: str) -> str:
        while self.parent.get(x, x) != x:
            x = self.parent[x]
        return x

    def effective(self, item_id: str) -> str:
        r = self._find(item_id)
        return self.primary.get(r, item_id if r == item_id else r)

    def members(self, item_id: str) -> list[str]:
        r = self._find(item_id)
        out = {item_id}
        for x in list(self.parent):
            if self._find(x) == r:
                out.add(x)
        return sorted(out)


@dataclass(frozen=True)
class Preview:
    a: str
    b: str
    binding: str                    # digest of every fact the decision depends on (see module docstring)
    compatible: bool
    problems: tuple[str, ...]
    conflicts: tuple[dict, ...]
    profiles: tuple[dict, ...]      # the two chosen profiles
    group_a: tuple[str, ...] = ()   # every member of the first effective group (incl. a)
    group_b: tuple[str, ...] = ()
    primary_a: str | None = None    # current effective primaries
    primary_b: str | None = None
    members: tuple[dict, ...] = ()  # profiles of every affected member

    @property
    def token(self) -> str:         # format-1 name
        return self.binding


class _BeforeCommit(Exception):
    """Rolled back before commit (known: nothing written by this transaction)."""


class _CommitUncertain(Exception):
    """COMMIT (or the rollback after a failure) could not be acknowledged: the outcome is unknown."""


def ensure_schema(store: Store) -> None:
    """Additive, transactional upgrade to correction format 2 (idempotent)."""
    with store.tx("corrections format 2") as c:
        cols = [r["name"] for r in c.execute("PRAGMA table_info(corrections)").fetchall()]
        if "format" not in cols:
            c.execute("ALTER TABLE corrections ADD COLUMN format INTEGER")
        if "request_sha256" not in cols:
            c.execute("ALTER TABLE corrections ADD COLUMN request_sha256 TEXT")
        c.execute("""CREATE TABLE IF NOT EXISTS correction_confirmations(
                     correction_id TEXT PRIMARY KEY, op_id TEXT UNIQUE, wall REAL, request_sha256 TEXT, note TEXT)""")
        c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('corrections_format', ?)", (str(FORMAT),))


def _what(kind: str | None) -> str:
    return "The Undo" if kind == KIND_UNDO else "The Same item correction"


class Corrections:
    def __init__(self, store: Store):
        self.store = store
        ensure_schema(store)

    # ---- durable state reads -----------------------------------------------------------------------------------------
    def rows(self) -> list[dict]:
        return [dict(r) for r in self.store.query("SELECT * FROM corrections ORDER BY wall, rowid")]

    def _confirmed(self) -> set[str]:
        return {r["correction_id"] for r in self.store.query("SELECT correction_id FROM correction_confirmations")}

    @staticmethod
    def _status(row: dict, confirmed: set[str]) -> str:
        if row.get("format") is None:
            return "legacy"
        return "confirmed" if row["correction_id"] in confirmed else "pending"

    def _undone(self, rows: list[dict], conf: set[str]) -> set[str]:
        return {r["reverses"] for r in rows if r["kind"] == KIND_UNDO and r["reverses"]
                and self._status(r, conf) in EFFECTIVE}

    def fold(self) -> Grouping:
        rows, conf = self.rows(), self._confirmed()
        undone = self._undone(rows, conf)
        g = Grouping(reversed_ids=undone)
        for r in rows:
            if self._status(r, conf) not in EFFECTIVE or r["kind"] != KIND_SAME or r["correction_id"] in undone:
                continue
            p = loads(r["payload"])
            a, b, prim = p["a"], p["b"], p["primary"]
            g.parent.setdefault(a, a)
            g.parent.setdefault(b, b)
            ra, rb = g._find(a), g._find(b)
            if ra != rb:
                g.parent[rb] = ra
            g.primary[g._find(a)] = prim
            g.active.append(r)
        return g

    def history(self, item_id: str) -> list[dict]:
        """Every operation that ever involved this item's correction history (Same item rows naming it, and the
        Undo rows reversing them), with durable status, whether it is currently in effect, and operator text."""
        rows, conf = self.rows(), self._confirmed()
        undone = self._undone(rows, conf)
        out, ids = [], set()
        for r in rows:
            if r["kind"] == KIND_SAME:
                p = loads(r["payload"])
                if item_id in (p["a"], p["b"]):
                    ids.add(r["correction_id"])
                    out.append(r)
        for r in rows:
            if r["kind"] == KIND_UNDO and r["reverses"] in ids:
                out.append(r)
        for r in out:
            r["payload_obj"] = loads(r["payload"])
            st = self._status(r, conf)
            r["status"] = st
            r["undone"] = r["kind"] == KIND_SAME and r["correction_id"] in undone
            r["in_effect"] = st in EFFECTIVE and not r["undone"]
            if r["kind"] == KIND_UNDO:
                r["status_text"] = {"legacy": "Undo (earlier format) - in effect",
                                    "pending": "Undo saved, NOT confirmed - no effect until reconciled",
                                    "confirmed": "Undo confirmed - in effect"}[st]
            else:
                base = {"legacy": "saved (earlier format, not readback-confirmed)",
                        "pending": "saved, NOT confirmed - no effect until reconciled",
                        "confirmed": "confirmed"}[st]
                r["status_text"] = base + (" - later undone, NOT in effect" if r["undone"] else
                                           (" - in effect" if st in EFFECTIVE else ""))
        return sorted(out, key=lambda r: r["wall"])

    # ---- preview: the two effective groups -----------------------------------------------------------------------------
    def _profile(self, item_id: str) -> dict | None:
        it = self.store.one("SELECT * FROM items WHERE item_id=?", (item_id,))
        if it is None:
            return None
        ref = self.store.one("SELECT * FROM item_references WHERE item_id=? AND version=?", (item_id, it["current_ref_version"]))
        ref_scope = None
        if ref is not None and ref["scan_id"]:
            sc = self.store.one("SELECT source_id FROM scans WHERE scan_id=?", (ref["scan_id"],))
            ref_scope = sc["source_id"] if sc else None
        scope = it["source_id"] if (is_known_scope(it["source_id"]) and ref_scope == it["source_id"]) else None
        iv = self.store.one("SELECT MIN(start_wall) AS first, MAX(end_wall) AS last, COUNT(*) AS n FROM run_intervals "
                            "WHERE item_id=? AND state='visible'", (item_id,))
        runs = [r["run_id"] for r in self.store.query("SELECT run_id FROM run_items WHERE item_id=? ORDER BY run_id", (item_id,))]
        return {"item_id": item_id, "name": it["name"], "status": it["status"], "ref_version": it["current_ref_version"],
                "crop_path": ref["crop_path"] if ref else None, "crop_sha256": ref["crop_sha256"] if ref else None,
                "frame_size": [ref["frame_width"], ref["frame_height"]] if ref else None,
                "recorded_source": it["source_id"], "reference_source": ref_scope, "scope": scope,
                "first_visible_wall": iv["first"], "last_visible_wall": iv["last"], "visible_intervals": iv["n"],
                "runs": runs}

    def _revisions(self, members: set[str], exclude: str | None) -> list[dict]:
        """Ordered correction/confirmation/Undo revisions that involve an affected member (unrelated ones excluded)."""
        rows, conf = self.rows(), self._confirmed()
        rel, out = set(), []
        for r in rows:
            if r["correction_id"] == exclude or r["kind"] != KIND_SAME:
                continue
            p = loads(r["payload"])
            if p["a"] in members or p["b"] in members:
                rel.add(r["correction_id"])
        for r in rows:
            if r["correction_id"] == exclude:
                continue
            if r["correction_id"] in rel or (r["kind"] == KIND_UNDO and r["reverses"] in rel):
                out.append({"id": r["correction_id"], "kind": r["kind"], "status": self._status(r, conf),
                            "request": r.get("request_sha256") or _sha(r["payload"]), "reverses": r["reverses"]})
        return out

    def _conflicts(self, group_a: list[str], group_b: list[str]) -> list[dict]:
        """Exact co-visibility evidence across the two groups' member pairs (interval ids and unrounded times)."""
        out = []
        for x in group_a:
            for y in group_b:
                runs = [r["run_id"] for r in self.store.query(
                    "SELECT run_id FROM run_items WHERE item_id=? INTERSECT SELECT run_id FROM run_items WHERE item_id=? "
                    "ORDER BY run_id", (x, y))]
                for run_id in runs:
                    ix = self.store.query("SELECT interval_id,start_mono,end_mono FROM run_intervals WHERE run_id=? AND "
                                          "item_id=? AND state='visible' ORDER BY start_mono, interval_id", (run_id, x))
                    iy = self.store.query("SELECT interval_id,start_mono,end_mono FROM run_intervals WHERE run_id=? AND "
                                          "item_id=? AND state='visible' ORDER BY start_mono, interval_id", (run_id, y))
                    ev = []
                    for u in ix:
                        for v in iy:
                            lo, hi = max(u["start_mono"], v["start_mono"]), min(u["end_mono"], v["end_mono"])
                            if hi > lo:
                                ev.append({"intervals": [u["interval_id"], v["interval_id"]], "from": lo, "to": hi})
                    if ev:
                        total = sum(e["to"] - e["from"] for e in ev)
                        out.append({"run_id": run_id, "items": [x, y], "evidence": ev, "seen_together_s": round(total, 2),
                                    "why": "both were Visible at the same time in this run"})
        return out

    def _state(self, a: str, b: str, exclude: str | None = None) -> dict:
        """Every fact a Same item decision depends on; recomputed at preview, pending save and confirmation."""
        problems = []
        g = self.fold()
        pa, pb = self._profile(a), self._profile(b)
        if pa is None or pb is None:
            problems.append("one of the profiles does not exist")
        if a == b:
            problems.append("choose two different profiles")
        ga, gb = g.members(a), g.members(b)
        if pa and pb and a != b and g.effective(a) == g.effective(b):
            problems.append("these profiles are already one effective item")
        member_ids = sorted(set(ga) | set(gb))
        profs = {m: (pa if m == a else pb if m == b else self._profile(m)) for m in member_ids}
        for m in member_ids:
            p = profs[m]
            if p is None:
                problems.append(f"group member {m} no longer exists")
            elif p["scope"] is None:
                problems.append(f"{p['name']} has no known source scope (it was created before source fingerprints, or its "
                                "reference comes from another recording); it cannot be grouped. Choose the video, Scan "
                                "and Remember it again to create a scoped profile")
        scopes = {p["scope"] for p in profs.values() if p and p["scope"]}
        if len(scopes) > 1:
            problems.append("the affected profiles come from different recordings (different source content); they "
                            "cannot be combined, even at the same resolution")
        conflicts = self._conflicts(ga, gb) if (pa and pb and a != b and not set(ga) & set(gb)) else []
        bind = {"groups": [ga, gb], "primaries": [g.effective(a), g.effective(b)],
                "members": [None if profs[m] is None else {k: profs[m][k] for k in (
                    "item_id", "status", "ref_version", "crop_sha256", "scope", "recorded_source", "reference_source")}
                    for m in member_ids],
                "revisions": self._revisions(set(member_ids), exclude),
                "conflicts": [{k: c[k] for k in ("run_id", "items", "evidence")} for c in conflicts],
                "problems": problems}
        return {"pa": pa, "pb": pb, "problems": problems, "conflicts": conflicts, "binding": _sha(bind),
                "group_a": ga, "group_b": gb, "primary_a": g.effective(a), "primary_b": g.effective(b),
                "members": [profs[m] for m in member_ids if profs[m] is not None]}

    def preview(self, a: str, b: str) -> Preview:
        st = self._state(a, b)
        return Preview(a, b, st["binding"], not st["problems"], tuple(st["problems"]), tuple(st["conflicts"]),
                       tuple(p for p in (st["pa"], st["pb"]) if p is not None), tuple(st["group_a"]), tuple(st["group_b"]),
                       st["primary_a"], st["primary_b"], tuple(st["members"]))

    # ---- durable write pipeline --------------------------------------------------------------------------------------
    def _write(self, fn) -> None:
        """BEGIN IMMEDIATE / fn(conn) / COMMIT. A failure before COMMIT whose ROLLBACK succeeded raises _BeforeCommit
        (known: nothing written); a CorrectionError passes through after rollback; a failed COMMIT or a failed
        ROLLBACK raises _CommitUncertain (outcome unknown)."""
        st = self.store
        with st._lock:
            conn = st._conn
            try:
                conn.execute("BEGIN IMMEDIATE")
            except Exception as e:  # noqa: BLE001
                raise _BeforeCommit(e) from e
            try:
                fn(conn)
            except Exception as e:  # noqa: BLE001
                try:
                    conn.execute("ROLLBACK")
                except Exception as e2:  # noqa: BLE001
                    raise _CommitUncertain(e2) from e
                if isinstance(e, CorrectionError):
                    raise
                raise _BeforeCommit(e) from e
            try:
                conn.execute("COMMIT")
            except Exception as e:  # noqa: BLE001
                try:
                    conn.execute("ROLLBACK")
                except Exception:  # noqa: BLE001
                    pass
                raise _CommitUncertain(e) from e

    def _readback(self, conn, op_id: str) -> dict | None:
        r = self.store.one("SELECT * FROM corrections WHERE op_id=?", (op_id,))
        return dict(r) if r else None

    def _is_confirmed(self, conn, correction_id: str) -> bool:
        return self.store.one("SELECT 1 FROM correction_confirmations WHERE correction_id=?", (correction_id,)) is not None

    def _insert_pending(self, conn, cid: str, op_id: str, kind: str, actor: str, payload: dict, reverses: str | None,
                        req_sha: str) -> None:
        conn.execute("INSERT INTO corrections(correction_id,op_id,wall,kind,actor,payload,reverses,format,request_sha256) "
                     "VALUES(?,?,?,?,?,?,?,?,?)", (cid, op_id, time.time(), kind, actor, dumps(payload), reverses, FORMAT, req_sha))

    def _confirm_row(self, conn, row: dict) -> None:
        conn.execute("INSERT OR IGNORE INTO correction_confirmations(correction_id,op_id,wall,request_sha256,note) "
                     "VALUES(?,?,?,?,?)", (row["correction_id"], row["op_id"], time.time(), row["request_sha256"],
                                         "read back and verified"))

    @staticmethod
    def _request(row_or_kind, payload: dict | None = None) -> dict:
        if isinstance(row_or_kind, dict):
            kind, payload = row_or_kind["kind"], loads(row_or_kind["payload"])
        else:
            kind = row_or_kind
        if kind == KIND_SAME:
            return {"kind": kind, "a": payload["a"], "b": payload["b"], "primary": payload["primary"],
                    "acknowledged": bool(payload["acknowledged"]), "binding": payload["binding"]}
        return {"kind": kind, "reverses": payload["reverses"]}

    def _verify(self, row: dict | None, req_sha: str) -> str | None:
        """None when the saved record is exactly the requested operation, else the problem."""
        if row is None:
            return "the saved record could not be found"
        if row.get("request_sha256") != req_sha:
            return "the saved record does not match the requested operation"
        try:
            if _sha(self._request(row)) != req_sha:
                return "the saved record's content does not match its request digest"
        except (KeyError, TypeError, ValueError):
            return "the saved record is unreadable"
        return None

    @staticmethod
    def _uncertain(op_id: str, cid: str | None, why: str, kind: str | None) -> Outcome:
        return Outcome("uncertain", op_id, cid, f"Save outcome unknown for operation {op_id}: {why}. Nothing is assumed: "
                       "it may already be saved or in effect. Use Reconcile (same operation) to read the stored result; "
                       "do not repeat it as a new operation.", kind)

    def _durable(self, row: dict) -> Outcome:
        """The durable status of a saved, confirmed (or legacy) operation, including whether it is still in effect."""
        rows, conf = self.rows(), self._confirmed()
        st = self._status(row, conf)
        if st == "pending":
            return Outcome("pending", row["op_id"], row["correction_id"], f"{_what(row['kind'])} is saved but NOT "
                           "confirmed; it has no effect.", row["kind"])
        if row["kind"] == KIND_UNDO:
            return Outcome(st, row["op_id"], row["correction_id"], "The Undo is saved and confirmed; the grouping it "
                           "reverses is no longer in effect (no second operation was created)." if st == "confirmed" else
                           "The Undo (earlier format) is saved and in effect.", row["kind"])
        if row["correction_id"] in self._undone(rows, conf):
            return Outcome(f"{st}_undone", row["op_id"], row["correction_id"], "This Same item correction was saved "
                           "and confirmed earlier and later UNDONE; it is NOT in effect now. Reconcile does not redo "
                           "it - compare the profiles again to group them anew.", row["kind"])
        return Outcome(st, row["op_id"], row["correction_id"], "The Same item correction is saved, confirmed and in "
                       "effect (no second operation was created)." if st == "confirmed" else
                       "Saved in the earlier format; it is in effect as saved.", row["kind"])

    def _check_confirmable(self, row: dict) -> None:
        """Pre-conditions re-checked under the confirmation transaction (the operation's own row excluded)."""
        if row["kind"] == KIND_SAME:
            p = loads(row["payload"])
            st = self._state(p["a"], p["b"], exclude=row["correction_id"])
            if st["problems"] or st["binding"] != p["binding"]:
                raise StalePreview("the affected profiles, groups, references or conflicts changed since this comparison; "
                                   "it was not applied - compare the profiles again")
        else:
            conf, rows = self._confirmed(), self.rows()
            target = next((r for r in rows if r["correction_id"] == row["reverses"]), None)
            if target is None or self._status(target, conf) not in EFFECTIVE:
                raise CorrectionError("the correction to undo is not in effect")
            for r in rows:
                if r["kind"] == KIND_UNDO and r["reverses"] == row["reverses"] and r["correction_id"] != row["correction_id"] \
                        and self._status(r, conf) != "pending":
                    raise CorrectionError("this correction was already undone")

    def _finish(self, op_id: str, req_sha: str, kind: str) -> Outcome:
        """Read back, verify and confirm an operation that was committed as pending (also the reconciliation path).

        EW-LAB-04 C1: CONFIRMATION IS ESTABLISHED FIRST. A missing, mismatched or unreadable readback is not evidence
        of no confirmation or no effect: a known confirmation (or legacy row) is never demoted by a bad read - such a
        read anomaly is ``uncertain`` with the operation id. Only an operation proven unconfirmed by a successful read
        of the confirmation table can be reported pending/no effect."""
        what = _what(kind)
        try:
            row = self._readback(None, op_id)
            already = row is not None and (row.get("format") is None or self._is_confirmed(None, row["correction_id"]))
            problem = self._verify(row, req_sha)
        except Exception as e:  # noqa: BLE001 - a read failure never establishes that nothing was saved
            return self._uncertain(op_id, None, f"the saved record could not be read back ({type(e).__name__})", kind)
        if row is None:
            return self._uncertain(op_id, None, "the record was committed but could not be found on read-back", kind)
        if already and problem is not None:
            return self._uncertain(op_id, row["correction_id"], f"it is recorded as confirmed, but its saved record read "
                                   f"back inconsistently ({problem}); a read anomaly does not revoke a confirmation", kind)
        if problem is not None:
            return Outcome("pending", op_id, row["correction_id"], f"{what} is saved but NOT confirmed ({problem}); no "
                           "confirmation exists for it at this check, so it has no effect. It stays in the history for "
                           "review.", kind)
        if not already:
            def confirm(conn):
                fresh = self._readback(conn, op_id)
                if fresh is not None and self._is_confirmed(conn, fresh["correction_id"]):
                    return                       # another caller confirmed THIS operation: its durable status wins
                if self._verify(fresh, req_sha) is not None:
                    # not a refusal: an inconsistent read inside the transaction leaves the outcome unknown
                    raise RuntimeError("the saved record read back inconsistently inside the confirmation transaction")
                self._check_confirmable(fresh)
                self._confirm_row(conn, fresh)
            try:
                self._write(confirm)
            except CorrectionError as e:
                return Outcome("pending", op_id, row["correction_id"], f"{what} is saved but NOT applied: {e}.", kind)
            except _BeforeCommit as e:
                return self._uncertain(op_id, row["correction_id"], f"the confirmation could not be completed "
                                       f"({type(e.__cause__).__name__}); another session may have completed it", kind)
            except _CommitUncertain as e:
                return self._uncertain(op_id, row["correction_id"], f"the confirmation could not be acknowledged "
                                       f"({type(e.__cause__).__name__})", kind)
        try:
            return self._durable(row)            # status/effect from the confirmation and correction tables
        except Exception as e:  # noqa: BLE001
            return self._uncertain(op_id, row["correction_id"], f"it was confirmed but its current effect could not be "
                                   f"re-read ({type(e).__name__})", kind)

    def _after_rollback(self, op_id: str, req_sha: str, kind: str, err: Exception) -> Outcome:
        """EW-LAB-04 C1: a known rollback proves only that THIS attempt wrote nothing. The operation id's durable state
        comes from a successful read afterwards; without one the outcome is uncertain."""
        cause = type(err.__cause__).__name__ if err.__cause__ is not None else type(err).__name__
        try:
            row = self._readback(None, op_id)
        except Exception as e2:  # noqa: BLE001
            return self._uncertain(op_id, None, f"this attempt was rolled back ({cause}) and the operation's stored state "
                                   f"could not be read afterwards ({type(e2).__name__})", kind)
        if row is None:
            return Outcome("not_saved", op_id, None, f"This attempt was rolled back ({cause}) and wrote nothing; no saved "
                           f"operation {op_id} was found when checked afterwards. You may retry the same operation.", kind)
        if row["kind"] != kind or row.get("request_sha256") not in (req_sha, None):
            raise OperationConflict(f"operation {op_id} was already used for a different request; it was not changed")
        out = self.reconcile(op_id)
        return Outcome(out.status, out.op_id, out.correction_id, f"This attempt was rolled back ({cause}), but operation "
                       f"{op_id} was already saved by another session. {out.message}", out.kind)

    def _submit(self, op_id: str, kind: str, payload: dict, reverses: str | None, actor: str, revalidate) -> Outcome:
        if not op_id:
            raise CorrectionError("missing operation id")
        req_sha = _sha(self._request(kind, payload))
        try:
            prior = self._readback(None, op_id)
        except Exception as e:  # noqa: BLE001
            return self._uncertain(op_id, None, f"the store could not be read to check this operation "
                                   f"({type(e).__name__})", kind)
        if prior is not None:
            if prior["kind"] != kind or prior.get("request_sha256") not in (req_sha, None):
                raise OperationConflict(f"operation {op_id} was already used for a different request; it was not changed")
            return self.reconcile(op_id)
        cid = new_id("cor")
        existed = {}

        def write(conn):
            again = self._readback(conn, op_id)          # concurrent duplicate retry: the first writer wins
            if again is not None:
                existed["row"] = again
                return
            revalidate()
            self._insert_pending(conn, cid, op_id, kind, actor, payload, reverses, req_sha)
        try:
            self._write(write)
        except _BeforeCommit as e:
            return self._after_rollback(op_id, req_sha, kind, e)
        except _CommitUncertain as e:
            return self._uncertain(op_id, cid, f"the save could not be acknowledged ({type(e.__cause__).__name__})", kind)
        if "row" in existed:
            if existed["row"]["kind"] != kind or existed["row"].get("request_sha256") != req_sha:
                raise OperationConflict(f"operation {op_id} was already used for a different request; it was not changed")
            return self.reconcile(op_id)
        return self._finish(op_id, req_sha, kind)

    # ---- public operations ------------------------------------------------------------------------------------------------
    def apply_same_item(self, op_id: str, preview: Preview, primary: str, acknowledge_conflicts: bool,
                        actor: str = "operator") -> Outcome:
        if primary not in (preview.a, preview.b):
            raise CorrectionError("the primary profile must be one of the two compared profiles")
        payload = {"a": preview.a, "b": preview.b, "primary": primary, "acknowledged": bool(acknowledge_conflicts),
                   "binding": preview.binding, "conflicts": list(preview.conflicts),
                   "groups": [list(preview.group_a), list(preview.group_b)],
                   "effect": "historical grouping only; references and recognition templates are unchanged"}
        try:
            prior = self._readback(None, op_id) if op_id else None
        except Exception as e:  # noqa: BLE001
            return self._uncertain(op_id, None, f"the store could not be read to check this operation "
                                   f"({type(e).__name__})", KIND_SAME)
        if prior is None:
            if not preview.compatible:
                raise CorrectionError("; ".join(preview.problems))
            if preview.conflicts and not acknowledge_conflicts:
                raise CorrectionError("review and acknowledge the conflicts before saving")

        def revalidate():
            st = self._state(preview.a, preview.b)
            if st["problems"]:
                raise CorrectionError("; ".join(st["problems"]))
            if st["binding"] != preview.binding:
                raise StalePreview("the affected profiles, groups, references or conflicts changed since this comparison; "
                                   "compare them again")
        return self._submit(op_id, KIND_SAME, payload, None, actor, revalidate)

    def undo(self, op_id: str, correction_id: str, actor: str = "operator") -> Outcome:
        payload = {"reverses": correction_id}

        def revalidate():
            conf = self._confirmed()
            target = self.store.one("SELECT * FROM corrections WHERE correction_id=?", (correction_id,))
            if target is None or target["kind"] != KIND_SAME:
                raise CorrectionError("only a Same item correction can be undone")
            if self._status(dict(target), conf) not in EFFECTIVE:
                raise CorrectionError("that correction is not in effect (it was never confirmed); reconcile it first")
            for r in self.rows():
                if r["kind"] == KIND_UNDO and r["reverses"] == correction_id:
                    raise CorrectionError("an Undo of this correction is already saved"
                                          + ("; reconcile it" if self._status(r, conf) == "pending" else ""))
        return self._submit(op_id, KIND_UNDO, payload, correction_id, actor, revalidate)

    def reconcile(self, op_id: str) -> Outcome:
        """Read the durable state of ONE operation and finish it if it is saved but not yet confirmed. Never creates a
        second operation and never redoes an undone one."""
        try:
            row = self._readback(None, op_id)
        except Exception as e:  # noqa: BLE001
            return self._uncertain(op_id, None, f"the store could not be read ({type(e).__name__})", None)
        if row is None:
            return Outcome("not_saved", op_id, None, f"No saved record exists for operation {op_id}; nothing changed. "
                           "You may retry it.")
        if row.get("format") is None:
            try:
                return self._durable(row)
            except Exception as e:  # noqa: BLE001
                return self._uncertain(op_id, row["correction_id"], f"its current effect could not be read "
                                       f"({type(e).__name__})", row["kind"])
        return self._finish(op_id, row["request_sha256"], row["kind"])
