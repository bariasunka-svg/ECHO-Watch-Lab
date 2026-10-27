# SPDX-License-Identifier: AGPL-3.0-only
"""The fresh engine store (EW-LAB-01: plus append-only corrections and indexes for paged activity) (SQLite, standard library): scans, candidates, remembered items and versioned references,
append-only operator decisions, runs, per-run item intervals and activity events.

A NEW database in a project-local, attempt-local data directory. It is never the legacy ECHO database and nothing is
migrated: the legacy schema is not imported. Writes are short serialised transactions; reads use the same connection
under the same lock (the volume here is small: operator commands, run summaries, state intervals).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS scans(
  scan_id TEXT PRIMARY KEY, source_id TEXT, epoch_id TEXT, frame_seq INTEGER, frame_wall REAL, frame_t_mono REAL,
  media TEXT, frame_path TEXT, frame_sha256 TEXT, width INTEGER, height INTEGER, status TEXT, error TEXT,
  model_sha256 TEXT, settings TEXT, candidate_count INTEGER, duplicate_count INTEGER, duration_s REAL, infer_s REAL,
  device TEXT, worker_session INTEGER, created_wall REAL, finished_wall REAL);
CREATE TABLE IF NOT EXISTS candidates(
  candidate_id TEXT PRIMARY KEY, scan_id TEXT, idx INTEGER, label TEXT, confidence REAL, box TEXT, polygon TEXT,
  duplicate_of TEXT, overlaps_item TEXT);
CREATE TABLE IF NOT EXISTS items(
  item_id TEXT PRIMARY KEY, name TEXT, category TEXT, status TEXT, current_ref_version INTEGER, source_id TEXT,
  created_wall REAL, updated_wall REAL, created_op TEXT);
CREATE TABLE IF NOT EXISTS item_references(
  item_id TEXT, version INTEGER, scan_id TEXT, candidate_id TEXT, origin TEXT, context_path TEXT, crop_path TEXT,
  crop_sha256 TEXT, box TEXT, polygon TEXT, label TEXT, confidence REAL, frame_width INTEGER, frame_height INTEGER,
  model_sha256 TEXT, reason TEXT, created_wall REAL, PRIMARY KEY(item_id, version));
CREATE TABLE IF NOT EXISTS decisions(
  decision_id TEXT PRIMARY KEY, wall REAL, kind TEXT, actor TEXT, op_id TEXT UNIQUE, payload TEXT);
CREATE TABLE IF NOT EXISTS item_ambiguity(
  item_id TEXT, ref_version INTEGER, created_wall REAL, detail TEXT, resolved_wall REAL);
CREATE TABLE IF NOT EXISTS runs(
  run_id TEXT PRIMARY KEY, created_wall REAL, started_wall REAL, ended_wall REAL, phase TEXT, options TEXT,
  source TEXT, policy_version TEXT, refs_digest TEXT, outcome TEXT, dir TEXT);
CREATE TABLE IF NOT EXISTS run_items(run_id TEXT, item_id TEXT, ref_version INTEGER, PRIMARY KEY(run_id, item_id));
CREATE TABLE IF NOT EXISTS run_intervals(
  interval_id TEXT PRIMARY KEY, run_id TEXT, item_id TEXT, state TEXT, start_mono REAL, end_mono REAL, start_wall REAL,
  end_wall REAL, start_ident TEXT, end_ident TEXT, samples INTEGER, box TEXT, track_id TEXT, reason TEXT, closed INTEGER);
CREATE TABLE IF NOT EXISTS run_events(
  event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, item_id TEXT, wall REAL, kind TEXT, payload TEXT);
CREATE INDEX IF NOT EXISTS run_events_item_wall ON run_events(item_id, wall);
CREATE INDEX IF NOT EXISTS run_events_run ON run_events(run_id, wall);
CREATE INDEX IF NOT EXISTS run_intervals_run_item ON run_intervals(run_id, item_id, start_mono);
-- EW-LAB-01: append-only operator corrections (Same item / Undo). Original observations are never rewritten.
CREATE TABLE IF NOT EXISTS corrections(
  correction_id TEXT PRIMARY KEY, op_id TEXT UNIQUE, wall REAL, kind TEXT, actor TEXT, payload TEXT, reverses TEXT);
-- EW-LAB-01: append-only profile edits (friendly-name changes) beside the decisions log
CREATE TABLE IF NOT EXISTS item_names(item_id TEXT, wall REAL, name TEXT, op_id TEXT UNIQUE);
"""


class StoreError(RuntimeError):
    pass


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), default=str)


def loads(s):
    return None if s is None else json.loads(s)


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('schema_version',?)", (str(SCHEMA_VERSION),))

    @contextmanager
    def tx(self, why: str = ""):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def one(self, sql: str, args: tuple = ()):
        rows = self.query(sql, args)
        return rows[0] if rows else None

    def op_result(self, op_id: str) -> dict | None:
        row = self.one("SELECT payload FROM decisions WHERE op_id=?", (op_id,))
        return loads(row["payload"]) if row else None

    @staticmethod
    def record_decision(conn, decision_id: str, kind: str, actor: str, op_id: str | None, payload: dict) -> None:
        conn.execute("INSERT INTO decisions(decision_id,wall,kind,actor,op_id,payload) VALUES(?,?,?,?,?,?)",
                     (decision_id, time.time(), kind, actor, op_id, dumps(payload)))

    def schema_version(self) -> int:
        return int(self.one("SELECT value FROM meta WHERE key='schema_version'")["value"])

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
