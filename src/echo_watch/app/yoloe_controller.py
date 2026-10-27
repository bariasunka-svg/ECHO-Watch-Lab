# SPDX-License-Identifier: AGPL-3.0-only
"""GUI-free state and commands of the YOLOE workspace page (EW-LAB-06). Tk never touches the session directly.

Commands: choose_source (content hash in the background; never starts inference) -> start (resource-owner check,
idle-worker handover, then the session prepares: worker, tracking epoch, ordered source) -> pause / resume -> stop
(bounded) -> finished panel: View test results / Run again (new run, new tracking epoch) / Choose another video.
Presentation: ``view`` (Detections + tracks | Detections only) and ``shown_conf`` change only what is drawn; both are
recorded in the trace as view records. Selection is (packet, detection) and follows a TRACK to newer packets only
when that track is associated there; otherwise it becomes historical text.
Results: bounded list, paged frame records and exact original frames through ``yoloe.review`` (no inference, no
Library profile, no item run). Every slow step runs as a bounded background task with a generation token.

EW-LAB-06 R1: a refused/failed Start is kept as ``start_error``; a source change bumps ``source_gen`` (the page drops
attempt-specific presentation); ``details_text`` / saved-result ``details`` expose the bounded structured failure; a
zero-frame saved test is reopened from its manifest alone (no source hashing, no worker, no decoding).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from ..config import Settings
from ..review.frames import LatestWorker
from ..yoloe import review as R
from ..yoloe.client import YoloeWorkerClient, yoloe_spec
from ..yoloe.profile import SHOWN_CONF_DEFAULT, VIEW_DETECTIONS, VIEW_TRACKS, validate_shown_conf
from ..yoloe.session import IDLE, PAUSED, RUNNING, SourceInfo, YoloeError, YoloeSession
from . import yoloe_views as YV
from .tasks import Completion, TaskRunner

MODE = "yoloe"
LABEL = "YOLOE test"
PAGE = 50


class YoloeController:
    def __init__(self, settings: Settings, owner=None, *, tasks: TaskRunner | None = None, client=None, session=None,
                 review_decode=None):
        self.s = settings
        self.tasks = tasks or TaskRunner(workers=2)
        self.client = client if client is not None else YoloeWorkerClient(yoloe_spec(settings), settings.limits)
        self.runs_dir = Path(settings.data_dir) / "yoloe-runs"
        self.session = session if session is not None else YoloeSession(runs_dir=self.runs_dir, client=self.client,
                                                                         limits=settings.limits)
        self.owner = owner
        if owner is not None:
            owner.register(MODE, LABEL, self.busy, self.client.alive, self.release_worker)
        self.gen = {"source": 0, "run": 0, "results": 0, "review": 0, "record": 0}
        self.source_path: str | None = None
        self.source_name: str | None = None
        self.source_info: SourceInfo | None = None
        self.source_status = "none"            # none | hashing | ready | failed
        self.source_error: str | None = None
        self._hash_cancel = threading.Event()
        self.start_pending = False
        self.start_error: str | None = None    # a Start refused or failed before the session began
        self.source_gen = 0
        self.finished = None                   # the Outcome shown in the finish panel
        self._seen_outcome = None
        self.view = VIEW_TRACKS
        self.shown_conf = SHOWN_CONF_DEFAULT
        self.selection: dict | None = None
        self.selection_note: str | None = None
        self.messages: list[tuple[str, str, float]] = []
        self.panel = "live"                    # live | results
        self.results: list[dict] = []
        self.results_status = "idle"
        self.review: dict | None = None        # {run_id, manifest, pager, source, page_start, rows, record, frame}
        self.review_worker = LatestWorker("yoloe-review-frame", review_decode or R.decode_original)
        self.closed = False
        self._closing = False

    # ---- helpers ------------------------------------------------------------------------------------------------
    def say(self, text: str, tone: str = "neutral") -> None:
        self.messages.append((text, tone, time.time()))
        del self.messages[:-40]

    def _submit(self, kind: str, concern: str, fn) -> int:
        self.gen[concern] += 1
        self.tasks.submit(kind, (concern, self.gen[concern]), fn)
        return self.gen[concern]

    def _current(self, c: Completion) -> bool:
        concern, g = c.token
        return self.gen.get(concern) == g

    def busy(self) -> bool:
        return self.start_pending or self.session.busy

    @property
    def phase(self) -> str:
        return self.session.phase

    def release_worker(self) -> None:
        self.client.release()

    def can(self) -> dict[str, bool]:
        ph = self.session.phase
        idle = ph == IDLE and not self.start_pending
        return {"choose": idle, "start": idle and self.source_status == "ready",
                "pause": ph == RUNNING, "resume": ph == PAUSED,
                "stop": ph in ("preparing", RUNNING, PAUSED) and not self.session.snapshot()["stop_requested"],
                "results": True, "run_again": idle and self.finished is not None and self.source_status == "ready",
                "choose_another": idle}

    # ---- source --------------------------------------------------------------------------------------------------------
    def choose_source(self, path: str) -> bool:
        if self.busy():
            self.say("Stop the current test before choosing another video.", "warning")
            return False
        self._hash_cancel.set()
        self._hash_cancel = cancel = threading.Event()
        self.source_path, self.source_name = path, Path(path).name
        self.source_info, self.source_status, self.source_error = None, "hashing", None
        self.source_gen += 1
        self.start_error = None
        self.finished = None
        self.selection, self.selection_note = None, None

        def work():
            from ..sources.fingerprint import fingerprint
            from ..sources.ffmpeg_source import validate_file
            validate_file(path)
            fp = fingerprint(path, cancel.is_set)
            return SourceInfo(path, Path(path).name, fp["bytes"], fp["sha256"], tuple(fp["state"]))
        self._submit("y_source", "source", work)
        return True

    # ---- lifecycle -------------------------------------------------------------------------------------------------------
    def start(self) -> bool:
        if not self.can()["start"]:
            if self.source_status != "ready":
                self.say("Choose a video first (its content is checked before a test can start).", "warning")
            return False
        why = self.owner.blocked_by(MODE) if self.owner is not None else None
        if why:
            self.start_error = why
            self.say(why, "warning")
            return False
        self.start_error = None
        self.start_pending = True
        self.finished = None
        self.selection, self.selection_note = None, None
        info, owner, sess = self.source_info, self.owner, self.session

        def work():
            if self.closed or self._closing:
                raise RuntimeError("the application is closing")
            if owner is not None:
                owner.handover(MODE)
            return sess.start(info)
        self._submit("y_start", "run", work)
        return True

    def pause(self) -> bool:
        return self.session.pause()

    def resume(self) -> bool:
        return self.session.resume()

    def stop(self) -> bool:
        ok = self.session.stop("operator")
        if ok:
            self.say("Stopping: the rest of the video is not analysed; evidence is being finalized.", "neutral")
        return ok

    def run_again(self) -> bool:
        return self.start()

    # ---- presentation (never inference) --------------------------------------------------------------------------------
    def set_view(self, view: str) -> None:
        if view not in (VIEW_TRACKS, VIEW_DETECTIONS) or view == self.view:
            return
        self.view = view
        self.session.note_view(view, self.shown_conf)

    def set_shown_conf(self, value) -> bool:
        try:
            v = validate_shown_conf(value)
        except (TypeError, ValueError) as e:
            self.say(str(e), "warning")
            return False
        if v != self.shown_conf:
            self.shown_conf = v
            self.session.note_view(self.view, v)
        return True

    def select(self, pk, i: int | None) -> None:
        self.selection = YV.select(pk, i)
        self.selection_note = None

    def follow(self, pk) -> None:
        sel, note = YV.follow_selection(self.selection, pk)
        if sel is not self.selection or note:
            self.selection = sel
            if note:
                self.selection_note = note

    # ---- results -----------------------------------------------------------------------------------------------------------
    def show_results(self) -> None:
        self.panel = "results"
        self.refresh_results()

    def show_live(self) -> None:
        self.panel = "live"

    def refresh_results(self) -> None:
        self.results_status = "loading"
        root = self.runs_dir
        self._submit("y_results", "results", lambda: R.list_runs(root))

    def open_result(self, run_id: str) -> bool:
        row = next((r for r in self.results if r["run_id"] == run_id), None)
        if row is None or row.get("status") == "unreadable":
            self.say("This saved test cannot be opened (its manifest is unreadable).", "warning")
            return False
        if self.session.busy and run_id == self.session.run_id:
            self.say("This test is still running; open it after it ends.", "warning")
            return False
        man = row["manifest"]
        zero = row.get("processed") == 0
        self.review = {"run_id": run_id, "manifest": man, "pager": None, "source": {"ok": None, "why": "Checking the "
                       "original video..."}, "page_start": 0, "rows": [], "record": None, "frame": None,
                       "status": "loading", "details": YV.result_details(man), "zero_frames": zero}
        getattr(self, "_review_cancel", threading.Event()).set()     # a newer open supersedes the previous one
        cancel = threading.Event()
        self._review_cancel = cancel

        def work():
            pager = R.TracePager(Path(row["dir"]))
            n = pager.index(cancel)
            if n == 0:      # nothing was analysed: there is no frame to locate, so the video is neither hashed nor read
                src = {"ok": None, "why": "No video frame was analysed in this test, so there is no frame to show."}
            else:
                src = R.verify_source(man, cancel.is_set)
            return {"pager": pager, "n": n, "source": src, "rows": pager.page(0, PAGE)}
        self._submit("y_review", "review", work)
        return True

    def page(self, delta: int) -> None:
        rv = self.review
        if not rv or rv.get("pager") is None:
            return
        n = len(rv["pager"])
        start = min(max(0, rv["page_start"] + delta * PAGE), max(0, (n - 1) // PAGE * PAGE))
        if start == rv["page_start"] and rv["rows"]:
            return
        rv["page_start"] = start
        pager = rv["pager"]
        self._submit("y_page", "record", lambda: ("page", start, pager.page(start, PAGE)))

    def select_record(self, n: int, max_w: int = 1280, max_h: int = 720) -> None:
        rv = self.review
        if not rv or rv.get("pager") is None:
            return
        pager, src = rv["pager"], rv["source"]
        rv["record"], rv["frame"] = None, None

        def work():
            rec = pager.record(n)
            return ("record", n, rec)
        self._submit("y_record", "record", work)
        rv["want_frame"] = (n, max_w, max_h) if src.get("ok") else None

    # ---- read model --------------------------------------------------------------------------------------------------------
    def details_text(self) -> str:
        return YV.details_text(self.session.snapshot(), self)

    def worker_text(self) -> str:
        """The YOLOE worker's own status line (the header shows it on this page)."""
        st = self.client.state
        lf = getattr(self.client, "last_failure", None) or {}
        if st == "ready":
            return f"YOLOE GPU worker: ready ({(self.client.receipt or {}).get('device', '-')})"
        if st == "unavailable":
            where = lf.get("stage_label") or lf.get("category") or "start"
            return f"YOLOE GPU worker: failed ({where})"
        return f"YOLOE GPU worker: {st}"

    # ---- pump (UI thread; cheap) -------------------------------------------------------------------------------------------
    def pump(self) -> None:
        for c in self.tasks.drain():
            self._apply(c)
        for _tok, res in self.review_worker.drain():
            rv = self.review
            if rv is not None and isinstance(res, R.ReviewFrame) and res.request.token == (rv["run_id"], rv.get("record_n")):
                rv["frame"] = res
            elif isinstance(res, Exception) and rv is not None:
                rv["frame_error"] = f"{type(res).__name__}: {res}"
        out = self.session.last_outcome
        if out is not None and out is not self._seen_outcome and self.session.phase == IDLE:
            self._seen_outcome = out
            self.finished = out
            self.say(out.summary_text(), "success" if out.status == "completed" and out.complete else
                     "danger" if out.status == "failed" or not out.complete else "neutral")

    def _apply(self, c: Completion) -> None:
        if not self._current(c):
            return
        k = c.kind
        if k == "y_source":
            if c.ok:
                self.source_info, self.source_status = c.value, "ready"
                self.say(f"{self.source_name} is ready. Start analyses it frame by frame; nothing runs until then.",
                         "neutral")
            else:
                self.source_status, self.source_error = "failed", c.error
                self.say(f"This video cannot be used: {c.error}", "danger")
        elif k == "y_start":
            self.start_pending = False
            if not c.ok:
                self.start_error = c.error
                self.say(f"The test did not start: {c.error}", "danger")
        elif k == "y_results":
            self.results_status = "done" if c.ok else "failed"
            self.results = c.value if c.ok else []
            if not c.ok:
                self.say(f"Saved tests could not be listed: {c.error}", "danger")
        elif k == "y_review":
            rv = self.review
            if rv is None:
                return
            if not c.ok:
                rv.update(status="failed", error=c.error)
                return
            rv.update(pager=c.value["pager"], source=c.value["source"], rows=c.value["rows"], status="done",
                      count=c.value["n"])
        elif k in ("y_page", "y_record"):
            rv = self.review
            if rv is None or not c.ok:
                if rv is not None:
                    rv["error"] = c.error
                return
            what, n, val = c.value
            if what == "page" and n == rv["page_start"]:
                rv["rows"] = val
            elif what == "record":
                rv["record"], rv["record_n"] = val, n
                want = rv.get("want_frame")
                md = val.get("media") or {}
                if want and want[0] == n and md.get("pts") is not None and rv["source"].get("ok"):
                    req = R.FrameRequest(rv["source"]["path"], float(md["pts"]), tuple(val["size"]), val.get("sig"),
                                         want[1], want[2], (rv["run_id"], n))
                    self.gen["review"] += 1
                    self.review_worker.submit(self.gen["review"], req)
                elif not rv["source"].get("ok"):
                    rv["frame_gap"] = rv["source"].get("why")
                elif md.get("pts") is None:
                    rv["frame_gap"] = "This record has no verified timestamp; its frame cannot be located."

    # ---- exit -----------------------------------------------------------------------------------------------------------------
    def shutdown(self, wait_s: float = 30.0) -> dict:
        self._closing = True                     # a Start still queued behind the owner handover will not begin
        rec = {"busy": self.session.busy}
        if self.session.busy:
            self.session.stop("application close")
            rec["settled"] = self.session.wait_idle(wait_s)
            oc = self.session.last_outcome
            rec["outcome"] = None if oc is None else oc.summary_text()
        self.client.stop()
        self.review_worker.shutdown(1.0)
        self.tasks.shutdown(2.0)
        self._hash_cancel.set()
        self.closed = True
        rec["worker_state"] = self.client.state
        return rec
