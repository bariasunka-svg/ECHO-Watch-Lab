# SPDX-License-Identifier: AGPL-3.0-only
"""Real broad discovery (Scan), operator selection and enrollment -> versioned item references.

Adapted from the preserved ECHO Watch reference runtime/echowatch/discovery.py (SHA-256
a666d32ea1b1836f076d064cdcf2b636c0609baff9bba2b196d8c421869340ac) and runtime/echowatch/registry.py (SHA-256
630bcf37c67a41f8abd050dc2be4f0638dc84469a68862125ef5815b4b1e4fa0); AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md. Behaviour kept:
* each scan retains the EXACT frame it analysed (lossless PNG + SHA-256) and its candidates are in that frame's source
  coordinates; candidates are suggestions - nothing becomes an item until the operator selects it and Remembers;
* duplicate proposals (IoU > 0.8) are marked; proposals overlapping a COMPATIBLE remembered item (IoU > 0.5) carry a
  suggestion hint;
* enrollment is transactional and idempotent (operation id), refuses outlines outside the frame / smaller than 8 px and
  two new selections of the same region;
* a reference is retained appearance memory of a selected region, never proof of serial identity, and no model is
  trained; Update reference creates a NEW version.
EW-LAB-05 changes (repeat-run enrollment):
* the geometry-only veto ("overlaps remembered item ... Use Update reference") is REMOVED. Spatial overlap is only a
  suggestion: it neither merges two selections nor forbids a distinct item at an old location;
* ``enroll`` is ONE immutable command per operation id, bound to the digest of its request: NEW profiles from scan
  candidates plus explicit REUSE of compatible saved items at their exact current reference version. It is
  all-or-nothing (one transaction, one decision record) and its outcome is a value: confirmed / rejected (validation,
  nothing written) / not_saved (this attempt wrote nothing) / uncertain (outcome unknown; reconcile with the SAME id) /
  conflict (the id is bound to a different request). Reuse is an operator decision recorded as such, never a
  recognition result and never Visible;
* every compatibility decision (hints, reuse, replacement, Start) uses engine.compat.check - content scope, reference
  size and model identity; resolution alone is never enough and unknown legacy scopes are never compatible.
Adaptation: the legacy SQLite schema, event bus, compute manager and commissioning/support tables are not used; rows go
to the fresh engine store (no schema change, no migration). A scan runs through the supervised worker client, never a
hidden CPU substitute.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings
from ..frames import Frame, new_id
from ..geometry import clamp_box, crop, iou, valid_box
from ..worker.client import WorkerClient
from . import compat
from .contracts import ItemRef
from .files import decode_image, encode_png, sha256_bytes, write_atomic
from .store import Store, dumps, loads


class LibraryError(ValueError):
    pass


@dataclass(frozen=True)
class Proposal:
    candidate_id: str
    idx: int
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    duplicate_of: str | None
    overlaps_item: str | None


@dataclass(frozen=True)
class ScanResult:
    scan_id: str
    status: str
    frame_path: str
    frame_sha256: str
    size: tuple[int, int]
    proposals: tuple[Proposal, ...]
    duration_s: float
    infer_s: float | None
    device: str | None
    worker_session: int | None
    error: str | None = None
    model_sha256: str | None = None


@dataclass(frozen=True)
class EnrolledItem:
    item_id: str
    name: str
    ref_version: int
    disposition: str                # new | reused
    candidate_id: str | None = None


@dataclass(frozen=True)
class EnrollOutcome:
    """The durable outcome of one enrollment command (never an exception for operational failures)."""

    status: str                     # confirmed | rejected | not_saved | uncertain | conflict
    op_id: str
    digest: str
    items: tuple[EnrolledItem, ...] = ()
    message: str = ""
    refs: tuple[ItemRef, ...] = ()

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"

    @property
    def new_count(self) -> int:
        return sum(1 for i in self.items if i.disposition == "new")

    @property
    def reused_count(self) -> int:
        return sum(1 for i in self.items if i.disposition == "reused")


def request_digest(scan_id: str, new: list[dict], reuse: list[dict]) -> str:
    """Canonical digest of an enrollment request (binds an operation id to exactly this request)."""
    n = [{"candidate_id": s.get("candidate_id"), "name": (s.get("name") or "").strip(),
          "category": (s.get("category") or "Other").strip()[:40] or "Other",
          "box": None if s.get("box") is None else [round(float(v), 1) for v in s["box"]]} for s in new]
    r = [{"item_id": s.get("item_id"), "ref_version": int(s.get("ref_version") or 0), "candidate_id": s.get("candidate_id")}
         for s in reuse]
    return hashlib.sha256(json.dumps({"scan_id": scan_id, "new": n, "reuse": r}, sort_keys=True).encode()).hexdigest()


class Library:
    def __init__(self, settings: Settings, store: Store):
        self.s, self.store = settings, store

    # ---- Scan ---------------------------------------------------------------------------------------------
    def scan(self, frame: Frame, source_id: str, worker: WorkerClient, actor: str = "operator") -> ScanResult:
        """Retain ``frame`` losslessly, run the real prompt-free discovery on exactly that image, store proposals."""
        data = encode_png(frame.image)
        scan_id = new_id("scan")
        d = self.s.dirs()["scans"] / scan_id
        fpath = d / "frame.png"
        write_atomic(fpath, data)
        frame_sha = sha256_bytes(data)
        now = time.time()
        model_sha = worker.receipt.get("model_sha256") if worker.receipt else None
        settings = {"imgsz": self.s.scan_imgsz, "conf": self.s.scan_conf, "iou": self.s.scan_iou,
                    "max_det": self.s.scan_max_det, "mode": "prompt-free", "device": self.s.device,
                    "precision": self.s.precision}
        with self.store.tx("create scan") as c:
            c.execute("""INSERT INTO scans(scan_id,source_id,epoch_id,frame_seq,frame_wall,frame_t_mono,media,frame_path,
                         frame_sha256,width,height,status,model_sha256,settings,created_wall)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,'running',?,?,?)""",
                      (scan_id, source_id, frame.epoch.epoch_id, frame.seq, frame.t_wall, frame.t_mono,
                       dumps(frame.media), str(fpath), frame_sha, frame.width, frame.height, model_sha, dumps(settings), now))
        t0 = time.monotonic()
        try:
            msg = worker.scan(fpath, self.s.scan_imgsz, self.s.scan_conf, self.s.scan_iou, self.s.scan_max_det,
                              self.s.limits.scan_timeout_s)
        except Exception as e:  # noqa: BLE001 - a visible failure, never a substitute result
            with self.store.tx("scan failed") as c:
                c.execute("UPDATE scans SET status='failed', error=?, finished_wall=?, duration_s=? WHERE scan_id=?",
                          (f"{type(e).__name__}: {e}"[:500], time.time(), round(time.monotonic() - t0, 3), scan_id))
            return ScanResult(scan_id, "failed", str(fpath), frame_sha, (frame.width, frame.height), (),
                              time.monotonic() - t0, None, None, worker.session, f"{type(e).__name__}: {e}", model_sha)
        duration = time.monotonic() - t0
        w, h = frame.width, frame.height
        cleaned = []
        for cnd in msg["candidates"]:
            box = clamp_box(cnd["box"], w, h)
            if box[2] < 2 or box[3] < 2:
                continue
            cleaned.append({**cnd, "box": [round(v, 1) for v in box]})
        cleaned.sort(key=lambda c: -(c.get("confidence") or 0))
        # overlap hints only against COMPATIBLE saved references (same content scope, size and model) - a suggestion
        scope = compat.Scope(source_id, (w, h), model_sha)
        existing = [e for e in self._active_refs() if compat.check_row(e, scope).ok]
        out, dups = [], 0
        for i, c in enumerate(cleaned):
            cid = new_id("cand")
            dup = next((p["candidate_id"] for p in out if iou(p["box"], c["box"]) > 0.8), None)
            dups += 1 if dup else 0
            overlap = next((e["item_id"] for e in existing if iou(json.loads(e["box"]), c["box"]) > 0.5), None)
            out.append({**c, "candidate_id": cid, "idx": i, "duplicate_of": dup, "overlaps_item": overlap})
        with self.store.tx("store candidates") as c:
            for p in out:
                c.execute("""INSERT INTO candidates(candidate_id,scan_id,idx,label,confidence,box,polygon,duplicate_of,
                             overlaps_item) VALUES(?,?,?,?,?,?,?,?,?)""",
                          (p["candidate_id"], scan_id, p["idx"], p["label"], p["confidence"], dumps(p["box"]),
                           dumps(p["polygon"]) if p.get("polygon") else None, p["duplicate_of"], p["overlaps_item"]))
            c.execute("""UPDATE scans SET status='succeeded', finished_wall=?, duration_s=?, candidate_count=?,
                         duplicate_count=?, infer_s=?, device=?, worker_session=? WHERE scan_id=?""",
                      (time.time(), round(duration, 3), len(out), dups, msg.get("infer_s"), msg.get("device"),
                       msg.get("_session"), scan_id))
            Store.record_decision(c, new_id("dec"), "scan", actor, None,
                                  {"scan_id": scan_id, "candidates": len(out), "frame": [frame.epoch.epoch_id, frame.seq]})
        props = tuple(Proposal(p["candidate_id"], p["idx"], p["label"], p["confidence"], tuple(p["box"]),
                               p["duplicate_of"], p["overlaps_item"]) for p in out)
        return ScanResult(scan_id, "succeeded", str(fpath), frame_sha, (w, h), props, duration, msg.get("infer_s"),
                          msg.get("device"), msg.get("_session"), None, model_sha)

    def _active_refs(self) -> list[dict]:
        rows = self.store.query(
            """SELECT i.item_id, i.name, i.status, i.source_id, i.current_ref_version, r.box, r.frame_width,
                      r.frame_height, r.model_sha256, r.crop_path, r.crop_sha256, r.scan_id, r.candidate_id
               FROM items i JOIN item_references r ON r.item_id=i.item_id AND r.version=i.current_ref_version
               WHERE i.status='active'""")
        return [dict(r) for r in rows]

    def scan_scope(self, scan_id: str) -> compat.Scope:
        scan = self.store.one("SELECT * FROM scans WHERE scan_id=?", (scan_id,))
        if scan is None:
            raise LibraryError("Unknown scan")
        return compat.Scope(scan["source_id"], (scan["width"], scan["height"]), scan["model_sha256"])

    # ---- saved items for the current recording ----------------------------------------------------------------
    def saved_for_scope(self, scope: compat.Scope) -> tuple[list[dict], list[dict]]:
        """(compatible, incompatible) saved profiles for ``scope``. Each row carries ``compat_code``/``compat_reason``.
        Incompatible rows are listed so the view can explain WHY an old profile cannot be reused here."""
        rows = self.store.query(
            """SELECT i.item_id, i.name, i.category, i.status, i.source_id, i.current_ref_version, i.created_wall,
                      r.box, r.frame_width, r.frame_height, r.model_sha256, r.crop_path, r.crop_sha256, r.scan_id,
                      r.candidate_id, r.context_path,
                      (SELECT COUNT(*) FROM run_items x WHERE x.item_id=i.item_id) AS runs
               FROM items i JOIN item_references r ON r.item_id=i.item_id AND r.version=i.current_ref_version
               WHERE i.status='active' ORDER BY i.created_wall, i.item_id""")
        ok, bad = [], []
        for r in rows:
            p = dict(r)
            v = compat.check_row(p, scope)
            p["box"] = json.loads(p["box"]) if isinstance(p["box"], str) else p["box"]
            p.update(compat_code=v.code, compat_reason=v.reason)
            (ok if v.ok else bad).append(p)
        return ok, bad

    def enrolled_from_scan(self, scan_id: str) -> dict[str, dict]:
        """candidate_id -> {item_id, ref_version, name} for references created from THIS scan's candidates (exact
        same-scan/candidate reuse can then be proposed instead of a duplicate profile)."""
        rows = self.store.query("""SELECT r.candidate_id, r.item_id, r.version, i.name, i.current_ref_version
                                   FROM item_references r JOIN items i ON i.item_id=r.item_id
                                   WHERE r.scan_id=? AND i.status='active'""", (scan_id,))
        return {r["candidate_id"]: {"item_id": r["item_id"], "ref_version": r["version"], "name": r["name"],
                                    "current_ref_version": r["current_ref_version"]} for r in rows}

    # ---- Select / enroll --------------------------------------------------------------------------------------
    def plan_enrollment(self, scan_id: str, new: list[dict], reuse: list[dict]) -> dict:
        """Validate a request WITHOUT writing anything. Raises ``LibraryError`` with an operator-facing reason."""
        new, reuse = list(new or []), list(reuse or [])
        if not new and not reuse:
            raise LibraryError("Select at least one item to remember")
        scan = self.store.one("SELECT * FROM scans WHERE scan_id=?", (scan_id,))
        if scan is None or scan["status"] != "succeeded":
            raise LibraryError("Unknown or unfinished scan")
        w, h = scan["width"], scan["height"]
        scope = compat.Scope(scan["source_id"], (w, h), scan["model_sha256"])
        cands = {r["candidate_id"]: r for r in self.store.query("SELECT * FROM candidates WHERE scan_id=?", (scan_id,))}
        seen_c: set[str] = set()
        prepared = []
        for sel in new:
            name = (sel.get("name") or "").strip()
            if not name or len(name) > 80:
                raise LibraryError("Each item needs a name of 1-80 characters")
            c = cands.get(sel.get("candidate_id"))
            if c is None:
                raise LibraryError("A selected proposal does not belong to this scan")
            if c["candidate_id"] in seen_c:
                raise LibraryError(f"{name}: the same suggestion is selected twice")
            seen_c.add(c["candidate_id"])
            box = sel.get("box") or json.loads(c["box"])
            if not valid_box(box, w, h):
                raise LibraryError(f"{name}: outline is outside the frame or smaller than 8 px")
            box = [round(v, 1) for v in clamp_box(box, w, h)]
            for p in prepared:
                if iou(p["box"], box) > 0.8:
                    raise LibraryError(f"{name} and {p['name']} outline the same region; keep one")
            prepared.append({"name": name, "category": (sel.get("category") or "Other").strip()[:40] or "Other",
                             "candidate_id": c["candidate_id"], "box": box,
                             "polygon": loads(c["polygon"]), "label": c["label"], "confidence": c["confidence"]})
        reused, seen_i = [], set()
        for sel in reuse:
            item_id = sel.get("item_id")
            if not item_id or item_id in seen_i:
                raise LibraryError("A saved item is selected twice; keep one")
            seen_i.add(item_id)
            cid = sel.get("candidate_id")
            if cid is not None:
                if cid not in cands:
                    raise LibraryError("A compared suggestion does not belong to this scan")
                if cid in seen_c:
                    raise LibraryError("One suggestion cannot be both a new item and a saved item; choose one")
                seen_c.add(cid)
            row = self.store.one("""SELECT i.item_id, i.name, i.status, i.source_id, i.current_ref_version, r.version,
                                           r.frame_width, r.frame_height, r.model_sha256, r.crop_sha256
                                    FROM items i LEFT JOIN item_references r ON r.item_id=i.item_id AND r.version=?
                                    WHERE i.item_id=?""", (int(sel.get("ref_version") or 0), item_id))
            if row is None:
                raise LibraryError("A chosen saved item no longer exists")
            row = dict(row)
            v = compat.check_row(row, scope)
            if not v.ok:
                raise LibraryError(f"{row['name']} cannot be reused for this recording: {v.reason}")
            want = int(sel.get("ref_version") or 0)
            if row["version"] is None:
                raise LibraryError(f"{row['name']}: reference version v{want} does not exist")
            if want != row["current_ref_version"]:
                raise LibraryError(f"{row['name']}: its reference changed to v{row['current_ref_version']} after you chose "
                                   f"v{want}; review the selection")
            reused.append({"item_id": item_id, "name": row["name"], "ref_version": want, "candidate_id": cid})
        return {"scan": dict(scan), "new": prepared, "reuse": reused, "scope": scope}

    def enroll(self, op_id: str, scan_id: str, new: list[dict], reuse: list[dict], actor: str = "operator") -> EnrollOutcome:
        """One captured, idempotent, all-or-nothing enrollment command (see module docstring)."""
        new, reuse = list(new or []), list(reuse or [])
        if not op_id:
            return EnrollOutcome("rejected", op_id or "", "", (), "Missing operation id")
        digest = request_digest(scan_id, new, reuse)
        try:
            prior = self._read_enrollment(op_id)
        except Exception as e:  # noqa: BLE001
            return EnrollOutcome("uncertain", op_id, digest, (), f"Outcome unknown: whether operation {op_id} was already "
                                 f"saved could not be checked ({type(e).__name__}). Check the saved outcome; nothing is "
                                 "repeated under a new id.")
        if prior is not None:
            return self._from_prior(op_id, digest, prior)
        try:
            plan = self.plan_enrollment(scan_id, new, reuse)
        except LibraryError as e:
            return EnrollOutcome("rejected", op_id, digest, (), f"Not saved: {e}")
        except Exception as e:  # noqa: BLE001
            return self._after_failure(op_id, digest, e, "validation read failed")
        written: list[Path] = []
        committed = False
        try:
            scan = plan["scan"]
            frame = None
            if plan["new"]:
                frame_data = Path(scan["frame_path"]).read_bytes()
                if hashlib.sha256(frame_data).hexdigest() != scan["frame_sha256"]:
                    return EnrollOutcome("rejected", op_id, digest, (), "Not saved: the retained scan frame failed its "
                                         "integrity check; discover items again before saving")
                frame = decode_image(frame_data)
            now = time.time()
            for p in plan["new"]:
                item_id = new_id("item")
                data = encode_png(crop(frame, p["box"]))
                path = self.s.dirs()["assets"] / item_id / "ref_v1.png"
                write_atomic(path, data)
                written.append(path)
                p.update(item_id=item_id, crop_path=str(path), crop_sha=sha256_bytes(data))
            items = ([{"item_id": p["item_id"], "name": p["name"], "ref_version": 1, "disposition": "new",
                       "candidate_id": p["candidate_id"]} for p in plan["new"]]
                     + [{"item_id": r["item_id"], "name": r["name"], "ref_version": r["ref_version"], "disposition": "reused",
                         "candidate_id": r["candidate_id"]} for r in plan["reuse"]])
            with self.store.tx("enroll items") as c:
                self._write_enrollment(c, op_id, digest, scan, plan, items, now, actor)
            committed = True
        except Exception as e:  # noqa: BLE001
            if not committed:
                for pth in written:
                    pth.unlink(missing_ok=True)
                return self._after_failure(op_id, digest, e, "the save was rolled back")
        return self._readback_outcome(op_id, digest, fresh=True)

    def enroll_outcome(self, op_id: str, digest: str) -> EnrollOutcome:
        """Reconcile: read the durable outcome of ``op_id`` (the SAME operation; never a new one)."""
        try:
            prior = self._read_enrollment(op_id)
        except Exception as e:  # noqa: BLE001
            return EnrollOutcome("uncertain", op_id, digest, (), f"Outcome still unknown: operation {op_id} could not be "
                                 f"read ({type(e).__name__}). Try Check saved outcome again.")
        if prior is None:
            return EnrollOutcome("not_saved", op_id, digest, (), f"Operation {op_id} is not saved: nothing was added. You "
                                 "can save the selection again.")
        return self._from_prior(op_id, digest, prior)

    # test seams (monkeypatched by tests to force failures; never switched in production)
    def _read_enrollment(self, op_id: str) -> dict | None:
        return self.store.op_result(op_id)

    def _write_enrollment(self, c, op_id, digest, scan, plan, items, now, actor) -> None:
        for p in plan["new"]:
            c.execute("""INSERT INTO items(item_id,name,category,status,current_ref_version,source_id,created_wall,
                         updated_wall,created_op) VALUES(?,?,?,'active',1,?,?,?,?)""",
                      (p["item_id"], p["name"], p["category"], scan["source_id"], now, now, op_id))
            c.execute("""INSERT INTO item_references(item_id,version,scan_id,candidate_id,origin,context_path,
                         crop_path,crop_sha256,box,polygon,label,confidence,frame_width,frame_height,model_sha256,
                         reason,created_wall) VALUES(?,1,?,?,'model',?,?,?,?,?,?,?,?,?,?,'initial selection',?)""",
                      (p["item_id"], scan["scan_id"], p["candidate_id"], scan["frame_path"], p["crop_path"], p["crop_sha"],
                       dumps(p["box"]), dumps(p["polygon"]) if p["polygon"] else None, p["label"],
                       p["confidence"], scan["width"], scan["height"], scan["model_sha256"], now))
        Store.record_decision(c, new_id("dec"), "remember", actor, op_id,
                              {"scan_id": scan["scan_id"], "digest": digest, "items": items,
                               "selected_candidates": [p["candidate_id"] for p in plan["new"]],
                               "reused": [[r["item_id"], r["ref_version"]] for r in plan["reuse"]]})

    def _from_prior(self, op_id: str, digest: str, prior: dict) -> EnrollOutcome:
        bound = prior.get("digest")
        if bound is not None and bound != digest:
            return EnrollOutcome("conflict", op_id, digest, (), f"Operation {op_id} is already bound to a different "
                                 "selection; it was not changed. Review the selection and save it as a new operation.")
        return self._readback_outcome(op_id, digest, fresh=False, prior=prior)

    def _readback_outcome(self, op_id: str, digest: str, fresh: bool, prior: dict | None = None) -> EnrollOutcome:
        saved_note = "saved" if fresh else "already saved (same operation)"
        try:
            rec = prior if prior is not None else self._read_enrollment(op_id)
            if rec is None or (rec.get("digest") is not None and rec.get("digest") != digest):
                raise LibraryError("the saved record read back inconsistently")
            items = tuple(EnrolledItem(a["item_id"], a.get("name", ""), int(a.get("ref_version") or 1),
                                       a.get("disposition", "new"), a.get("candidate_id")) for a in rec["items"])
            refs = tuple(self.item_ref(a.item_id, a.ref_version) for a in items)
        except Exception as e:  # noqa: BLE001
            head = "The save committed, but" if fresh else f"Operation {op_id} exists, but"
            return EnrollOutcome("uncertain", op_id, digest, (), f"{head} its saved record could not be read back "
                                 f"({type(e).__name__}: {e}). Outcome uncertain: Check saved outcome reads the SAME "
                                 "operation; nothing is created again.")
        n_new = sum(1 for i in items if i.disposition == "new")
        n_re = len(items) - n_new
        return EnrollOutcome("confirmed", op_id, digest, items, f"{len(items)} item(s) {saved_note}: {n_new} new, "
                             f"{n_re} reused. Saved, not monitoring.", refs)

    def _after_failure(self, op_id: str, digest: str, err: Exception, what: str) -> EnrollOutcome:
        """A failed attempt proves only that THIS attempt wrote nothing; the operation id is read afresh."""
        why = f"{type(err).__name__}: {err}"[:300]
        try:
            prior = self._read_enrollment(op_id)
        except Exception as e2:  # noqa: BLE001
            return EnrollOutcome("uncertain", op_id, digest, (), f"Outcome unknown: {what} ({why}) and the operation "
                                 f"could not be checked afterwards ({type(e2).__name__}). Check saved outcome for "
                                 f"operation {op_id}.")
        if prior is None:
            return EnrollOutcome("not_saved", op_id, digest, (), f"Not saved: {what} ({why}). This attempt wrote nothing "
                                 "and no saved operation was found afterwards; your selection is kept - try again.")
        out = self._from_prior(op_id, digest, prior)
        if out.confirmed:
            return EnrollOutcome("confirmed", op_id, digest, out.items, f"This attempt failed ({why}), but the same "
                                 f"operation was already saved. {out.message}", out.refs)
        return out

    def remember(self, op_id: str, scan_id: str, selections: list[dict], actor: str = "operator") -> list[ItemRef]:
        """Remember NEW profiles from selected proposals (compatibility wrapper over ``enroll``; idempotent per op id).
        ``selections``: [{candidate_id, name, category?, box?}]. Raises ``LibraryError`` unless confirmed."""
        prior = None
        try:
            prior = self.store.op_result(op_id) if op_id else None
        except Exception:  # noqa: BLE001
            prior = None
        if prior is not None and prior.get("digest") is None and prior.get("items"):
            return [self.item_ref(a["item_id"]) for a in prior["items"]]     # a legacy (pre-Lab-05) record
        out = self.enroll(op_id, scan_id, selections, [], actor)
        if not out.confirmed:
            raise LibraryError(out.message.removeprefix("Not saved: "))
        return list(out.refs)

    # ---- Start boundary ------------------------------------------------------------------------------------------
    def resolve_refs(self, items, scope: compat.Scope) -> list[ItemRef]:
        """Exact references for a run. ``items``: [(item_id, ref_version)] (or item ids = their CURRENT version, for
        engine harnesses). Every reference must be compatible with ``scope``, exist at that version and its saved crop
        must still match its recorded SHA-256. Raises ``LibraryError`` naming the first problem."""
        refs, seen = [], set()
        for it in items:
            item_id, ver = (it, None) if isinstance(it, str) else (it[0], int(it[1]))
            if item_id in seen:
                raise LibraryError("an item is listed twice for one run")
            seen.add(item_id)
            row = self.store.one("""SELECT i.item_id, i.name, i.status, i.source_id, i.current_ref_version,
                                           r.version, r.frame_width, r.frame_height, r.model_sha256, r.crop_path,
                                           r.crop_sha256
                                    FROM items i LEFT JOIN item_references r ON r.item_id=i.item_id
                                         AND r.version=COALESCE(?, i.current_ref_version)
                                    WHERE i.item_id=?""", (ver, item_id))
            if row is None or row["version"] is None:
                raise LibraryError(f"{item_id}: the saved reference version is missing")
            row = dict(row)
            v = compat.check_row(row, scope)
            if not v.ok:
                raise LibraryError(f"{row['name']} cannot be monitored on this recording: {v.reason}")
            try:
                data = Path(row["crop_path"]).read_bytes()
            except OSError as e:
                raise LibraryError(f"{row['name']}: its saved reference image is missing ({e})") from e
            if sha256_bytes(data) != row["crop_sha256"]:
                raise LibraryError(f"{row['name']}: its saved reference image changed on disk; replace the reference")
            refs.append(self.item_ref(item_id, row["version"]))
        return refs

    # ---- Reads --------------------------------------------------------------------------------------------
    def item_ref(self, item_id: str, version: int | None = None) -> ItemRef:
        it = self.store.one("SELECT * FROM items WHERE item_id=?", (item_id,))
        if it is None:
            raise LibraryError("Unknown item")
        ver = version or it["current_ref_version"]
        r = self.store.one("SELECT * FROM item_references WHERE item_id=? AND version=?", (item_id, ver))
        if r is None:
            raise LibraryError("Unknown reference version")
        return ItemRef(item_id, it["name"], it["category"], ver, f"{item_id}@v{ver}", tuple(json.loads(r["box"])),
                       (r["frame_width"], r["frame_height"]), r["crop_path"], r["context_path"], r["crop_sha256"])

    def list_items(self) -> list[dict]:
        rows = self.store.query("""SELECT i.item_id, i.name, i.category, i.status, i.current_ref_version, r.box,
                                   r.crop_sha256, r.created_wall FROM items i JOIN item_references r
                                   ON r.item_id=i.item_id AND r.version=i.current_ref_version
                                   ORDER BY i.created_wall, i.item_id""")
        return [dict(r) for r in rows]

    def update_reference(self, op_id: str, item_id: str, scan_id: str, candidate_id: str, actor: str = "operator",
                         box=None) -> ItemRef:
        """Explicit, versioned reference change: a NEW version; incompatible cached representations are invalidated
        (the worker keys embeddings by '<item>@v<version>', so an old key is never applied again). Only from a scan of
        the SAME recording (engine.compat), so a replacement never carries an identity across videos."""
        prior = self.store.op_result(op_id)
        if prior is not None:
            return self.item_ref(item_id)
        it = self.store.one("SELECT * FROM items WHERE item_id=? AND status='active'", (item_id,))
        scan = self.store.one("SELECT * FROM scans WHERE scan_id=? AND status='succeeded'", (scan_id,))
        c = self.store.one("SELECT * FROM candidates WHERE candidate_id=? AND scan_id=?", (candidate_id, scan_id))
        if it is None or scan is None or c is None:
            raise LibraryError("Unknown item, scan or proposal")
        cur = self.store.one("SELECT frame_width, frame_height, model_sha256 FROM item_references WHERE item_id=? AND "
                             "version=?", (item_id, it["current_ref_version"]))
        v = compat.check(it["source_id"], (cur["frame_width"], cur["frame_height"]) if cur else None,
                         cur["model_sha256"] if cur else None,
                         compat.Scope(scan["source_id"], (scan["width"], scan["height"]), scan["model_sha256"]))
        if not v.ok:
            raise LibraryError(f"The reference of {it['name']} cannot be replaced from this recording: {v.reason}")
        w, h = scan["width"], scan["height"]
        box = [round(v, 1) for v in clamp_box(box if box is not None else json.loads(c["box"]), w, h)]
        if not valid_box(box, w, h):
            raise LibraryError("Outline is outside the frame or too small")
        frame = decode_image(Path(scan["frame_path"]).read_bytes())
        version = it["current_ref_version"] + 1
        data = encode_png(crop(frame, box))
        path = self.s.dirs()["assets"] / item_id / f"ref_v{version}.png"
        write_atomic(path, data)
        now = time.time()
        with self.store.tx("update reference") as conn:
            conn.execute("""INSERT INTO item_references(item_id,version,scan_id,candidate_id,origin,context_path,crop_path,
                            crop_sha256,box,polygon,label,confidence,frame_width,frame_height,model_sha256,reason,
                            created_wall) VALUES(?,?,?,?,'model',?,?,?,?,?,?,?,?,?,?,'reference updated by operator',?)""",
                         (item_id, version, scan_id, candidate_id, scan["frame_path"], str(path), sha256_bytes(data),
                          dumps(box), c["polygon"], c["label"], c["confidence"], w, h, scan["model_sha256"], now))
            conn.execute("UPDATE items SET current_ref_version=?, updated_wall=? WHERE item_id=?", (version, now, item_id))
            Store.record_decision(conn, new_id("dec"), "update_reference", actor, op_id,
                                  {"item_id": item_id, "version": version, "items": [{"item_id": item_id}]})
        return self.item_ref(item_id)

    # ---- EW-LAB-01 additions ----------------------------------------------------------------------------------
    def add_manual_candidate(self, scan_id: str, box, name_hint: str = "manual region") -> str:
        """A manual region RECOVERS a miss of a completed automatic Scan (it never replaces Scan): it becomes one more
        candidate of that scan, marked manual, in the scan frame's coordinates."""
        scan = self.store.one("SELECT * FROM scans WHERE scan_id=?", (scan_id,))
        if scan is None or scan["status"] != "succeeded":
            raise LibraryError("Run an automatic Scan first; a manual region only recovers a missed item")
        w, h = scan["width"], scan["height"]
        if not valid_box(box, w, h):
            raise LibraryError("The region is outside the frame or smaller than 8 px")
        b = [round(v, 1) for v in clamp_box(box, w, h)]
        n = self.store.one("SELECT COALESCE(MAX(idx), -1) + 1 AS n FROM candidates WHERE scan_id=?", (scan_id,))["n"]
        cid = new_id("cand")
        with self.store.tx("manual region") as c:
            c.execute("""INSERT INTO candidates(candidate_id,scan_id,idx,label,confidence,box,polygon,duplicate_of,
                         overlaps_item) VALUES(?,?,?,?,?,?,NULL,NULL,NULL)""", (cid, scan_id, n, name_hint, None, dumps(b)))
            Store.record_decision(c, new_id("dec"), "manual_region", "operator", None, {"scan_id": scan_id, "candidate_id": cid,
                                                                                         "box": b})
        return cid

    def rename(self, op_id: str, item_id: str, name: str, actor: str = "operator") -> None:
        """Friendly-name change: the item's current name is updated and the change is APPENDED (item_names, decisions);
        references, versions and history are untouched."""
        name = (name or "").strip()
        if not name or len(name) > 80:
            raise LibraryError("A name needs 1-80 characters")
        if self.store.one("SELECT 1 FROM item_names WHERE op_id=?", (op_id,)):
            return
        now = time.time()
        with self.store.tx("rename") as c:
            c.execute("INSERT INTO item_names(item_id,wall,name,op_id) VALUES(?,?,?,?)", (item_id, now, name, op_id))
            c.execute("UPDATE items SET name=?, updated_wall=? WHERE item_id=?", (name, now, item_id))
            Store.record_decision(c, new_id("dec"), "rename", actor, op_id, {"item_id": item_id, "name": name})

    def profiles(self) -> list[dict]:
        """Saved Library profiles (never the current run's membership)."""
        rows = self.store.query("""SELECT i.item_id, i.name, i.status, i.current_ref_version, i.created_wall, r.crop_path, r.box,
                                   r.frame_width, r.frame_height, r.crop_sha256, r.scan_id, i.source_id, r.model_sha256,
                                   (SELECT COUNT(*) FROM run_items x WHERE x.item_id=i.item_id) AS runs,
                                   (SELECT COUNT(*) FROM item_references v WHERE v.item_id=i.item_id) AS versions
                                   FROM items i JOIN item_references r ON r.item_id=i.item_id AND r.version=i.current_ref_version
                                   WHERE i.status='active' ORDER BY i.created_wall DESC, i.item_id""")
        return [dict(r) for r in rows]
