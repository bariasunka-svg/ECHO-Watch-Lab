# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 R1 end-to-end failure readback (GUI-free, no model/decoder/worker process):
real bootstrap fatal -> real YOLOE client (pipe mocked) -> real session -> saved manifest/trace -> controller reopen ->
read model -> production page renderers against inert sinks; plus pre-R1 owner manifests (byte copies from the
review evidence), evidence-write failure, failure after frames, retry and source change."""
from __future__ import annotations

import json
import shutil
import threading
import time
import uuid
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

import echo_watch.app.yoloe_page as page_mod
import echo_watch.worker.client as WC
from echo_watch.app import yoloe_views as YV
from echo_watch.app.display_schedule import DisplayScheduler
from echo_watch.app.tasks import TaskRunner
from echo_watch.app.yoloe_controller import YoloeController
from echo_watch.config import ROOT, Settings
from echo_watch.yoloe import review as R
from echo_watch.yoloe.client import YoloeWorkerClient, yoloe_spec
from echo_watch.yoloe.session import YoloeSession
from lab05_view_harness import NoTk, Sink
from lab06_support import InProcessClient, ScriptSource, media_frames, read_trace, wait_for, work_dir
from test_lab06_presentation import Canvas, Tree
from test_lab06_r1_transport import J, Proc, worker_fatal

OWNER = ROOT / "reviews" / "20261026-lab06-startup-independent" / "owner-evidence"
Usage = namedtuple("Usage", "total used free")


class Rig:
    def __init__(self, client=None, frames=None, procs=None, disk=None):
        self.d = work_dir("lab06r1-readback")
        self.s = Settings(data_dir=self.d / "operator")
        self.client = client or YoloeWorkerClient(yoloe_spec(self.s, cmd_override=["scripted"]))
        self.sources = []

        def factory(path):
            src = ScriptSource(frames or media_frames(5, fps=500))
            self.sources.append(src)
            return src
        kw = {"disk_usage": disk} if disk else {}
        self.session = YoloeSession(runs_dir=self.s.data_dir / "yoloe-runs", client=self.client, source_factory=factory,
                                    **kw)
        self.decoded = []
        self.y = YoloeController(self.s, None, tasks=TaskRunner(2), client=self.client, session=self.session,
                                 review_decode=lambda req, cancel=None: self.decoded.append(req))
        self.video = self.d / "chosen clip.mp4"
        self.video.write_bytes(b"x" * 128)

    def pump_until(self, cond, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.y.pump()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def choose_and_start(self):
        assert self.y.choose_source(str(self.video))
        assert self.pump_until(lambda: self.y.source_status == "ready")
        assert self.y.start()

    def close(self):
        if self.session.busy:
            self.session.stop()
            self.session.wait_idle(5)
        self.y.tasks.shutdown(1.0)
        self.y.review_worker.shutdown(1.0)


@pytest.fixture()
def procs(monkeypatch):
    queue = []
    monkeypatch.setattr(WC, "popen_windowless", lambda args, **kw: queue.pop(0))
    return queue


def failed_start(procs):
    procs.append(Proc(out=[J(worker_fatal())], err=["WARNING 'half' is deprecated. Use 'quantize' instead."], rc=2))
    r = Rig()
    r.choose_and_start()
    assert r.pump_until(lambda: r.y.finished is not None)
    return r


def test_a_worker_startup_failure_is_carried_into_outcome_manifest_trace_and_headline(procs):
    r = failed_start(procs)
    try:
        out = r.y.finished
        f = out.failure
        assert out.status == "failed" and out.frames == 0 and out.complete is True          # evidence complete != success
        assert f["stage"] == "worker_start" and f["stage_label"] == "starting the YOLOE GPU worker"
        w = f["worker"]
        assert (w["stage"], w["exc_type"]) == ("warmup", "AttributeError") and w["traceback"].startswith("Traceback")
        assert "has no attribute 'half'" in w["message"] and "deprecated" in w["stderr"]["text"]
        assert out.headline().startswith("Analysis could not start: starting the YOLOE GPU worker failed - model warm-up: "
                                          "AttributeError:") and "No video frame was analysed." in out.headline()
        assert r.sources == []                                     # the decoder was never constructed or started
        man = json.loads((Path(out.run_dir) / "manifest.json").read_text("utf-8"))
        oc = man["outcome"]
        assert oc["status"] == "failed" and oc["frames_analysed"] == 0 and oc["details_captured"] is True
        assert oc["failure"]["worker"]["exc_type"] == "AttributeError" and man["worker"] is None
        tr = read_trace(out.run_dir)
        kinds = [x["k"] for x in tr]
        assert kinds[0] == "start" and "worker_start" in kinds and kinds[-2:] == ["failure", "end"]
        assert tr[-2]["worker"]["traceback"] == w["traceback"]
        assert any("model warm-up" in m[0] for m in r.y.messages)
    finally:
        r.close()


def test_a_zero_frame_failure_reopens_with_its_details_without_hashing_decoding_or_a_worker(procs, monkeypatch):
    r = failed_start(procs)
    try:
        calls = list(procs)
        monkeypatch.setattr(R, "verify_source", lambda *a, **k: pytest.fail("the source must not be hashed"))
        r.y.show_results()
        assert r.pump_until(lambda: r.y.results_status == "done")
        row = r.y.results[0]
        assert row["status"] == "failed" and row["processed"] == 0 and row["failure"]["stage"] == "worker_start"
        assert YV.results_rows(r.y.results)[0][1][2] == "failed (not started)"
        assert r.y.open_result(row["run_id"])
        assert r.pump_until(lambda: (r.y.review or {}).get("status") == "done")
        rv = r.y.review
        assert rv["count"] == 0 and rv["zero_frames"] and rv["source"]["ok"] is None
        det = rv["details"]
        assert "no video frame was analysed" in det and "Failed stage: starting the YOLOE GPU worker" in det
        assert "Worker stage: model warm-up · AttributeError" in det and "Worker traceback:" in det
        assert "has no attribute 'half'" in det and "Worker stderr" in det
        assert r.decoded == [] and procs == calls and r.client.attempts == 1    # nothing started or decoded
    finally:
        r.close()


def test_pre_r1_owner_failures_stay_readable_and_say_their_details_were_not_captured():
    d = work_dir("lab06r1-owner-copies") / "yoloe-runs"
    before = {p: p.read_bytes() for p in OWNER.rglob("*") if p.is_file()}
    for run in OWNER.iterdir():
        if run.is_dir():
            shutil.copytree(run, d / run.name)
    rows = R.list_runs(d)
    assert len(rows) == 3 and all(x["status"] == "failed" and x["processed"] == 0 for x in rows)
    for x in rows:
        assert x["failure"] is None and x["details_captured"] is False
        det = YV.result_details(x["manifest"])
        assert "Failure details: NOT CAPTURED" in det and "before EW-LAB-06 R1" in det
        assert "Reason: YoloeError: the YOLOE GPU worker could not start" in det
        assert "AttributeError" not in det                           # the lost exception is not reconstructed
        assert YV.results_rows([x])[0][1][2] == "failed (not started)"
    assert {p: p.read_bytes() for p in OWNER.rglob("*") if p.is_file()} == before


def test_low_disk_before_any_evidence_is_a_failed_start_with_explicitly_incomplete_evidence():
    r = Rig(client=InProcessClient(), disk=lambda p: Usage(1, 1, 10))
    try:
        r.choose_and_start()
        assert r.pump_until(lambda: r.y.finished is not None)
        out = r.y.finished
        assert out.status == "failed" and not out.complete and out.failure["stage"] == "disk_space"
        ps = YV.page_state(r.y, r.session.state())
        assert ps["kind"] == "failed_before_image" and "INCOMPLETE" in ps["text"] and "free disk" in ps["text"]
        assert r.client.calls == []
    finally:
        r.close()


def test_failure_after_frames_keeps_the_last_paired_image_explicitly_historical():
    c = InProcessClient()
    c.fail_at = "4"
    r = Rig(client=c, frames=media_frames(10, fps=500))
    try:
        r.choose_and_start()
        assert r.pump_until(lambda: r.y.finished is not None)
        out = r.y.finished
        pk = r.session.latest_packet()
        assert out.status == "failed" and out.frames == 3 and pk.seq == 3     # frame 4 failed in the worker
        ps = YV.page_state(r.y, r.session.state(), True)
        assert ps["kind"] == "failed_after_frames" and "after 3 analysed frames" in ps["text"]
        live, banner = YV.authority(pk, r.session.state(), time.monotonic())
        assert not live and banner.startswith("FAILED while analysing frames") and "frame 3" in banner
    finally:
        r.close()


def test_retry_after_a_failed_start_reaches_a_first_paired_packet_with_fresh_ownership_and_evidence():
    """In-process production worker core (spy detector): attempt 1 fails before ready, Run again reaches ready ->
    acknowledged epoch -> first paired packet -> EOF. (Pipe-level restart is covered in test_lab06_r1_transport.)"""
    c = InProcessClient(ready=False)
    r = Rig(client=c, frames=media_frames(4, fps=500))
    try:
        r.choose_and_start()
        assert r.pump_until(lambda: r.y.finished is not None)
        first = r.y.finished
        assert first.status == "failed" and first.frames == 0 and first.failure["stage"] == "worker_start"
        assert YV.page_state(r.y, r.session.state())["kind"] == "failed_before_image"
        c.ready = True
        assert r.y.can()["run_again"] and r.y.run_again()
        assert r.y.finished is None                                  # the failed attempt's presentation is gone
        assert r.pump_until(lambda: r.y.finished is not None, 15)
        second = r.y.finished
        assert second.status == "completed" and second.frames == 4 and second.failure is None
        assert second.run_id != first.run_id and second.run_dir != first.run_dir
        pk = r.session.latest_packet()
        assert pk.run_id == second.run_id and pk.epoch == f"{second.run_id}/e1" and int(pk.image[0, 0, 0]) == 3
        assert YV.page_state(r.y, r.session.state(), True)["kind"] == "ended"
        assert Path(first.run_dir, "manifest.json").is_file()        # the failed attempt stays in history
        man2 = json.loads(Path(second.run_dir, "manifest.json").read_text("utf-8"))
        assert man2["outcome"]["failure"] is None and man2["worker"]["device"] == "cuda:0"
        gen = r.y.source_gen
        other = r.d / "another.mp4"
        other.write_bytes(b"y" * 64)
        assert r.y.choose_source(str(other))
        assert r.y.source_gen == gen + 1 and r.y.finished is None and r.y.start_error is None
        assert r.pump_until(lambda: r.y.source_status == "ready")
        assert YV.page_state(r.y, r.session.state())["kind"] == "ready"
    finally:
        r.close()


@pytest.mark.parametrize("setup,kind,label", [
    (lambda y, s: None, "no_source", "Idle"),
    (lambda y, s: setattr(y, "source_status", "hashing"), "checking", "Checking"),
    (lambda y, s: setattr(y, "source_status", "ready"), "ready", "Ready"),
    (lambda y, s: setattr(y, "start_error", "Display Assets is still active."), "start_refused", "Not started"),
    (lambda y, s: s.update(phase="preparing", stage_label="starting the YOLOE GPU worker"), "preparing", "Preparing"),
    (lambda y, s: s.update(phase="running"), "running_no_image", "Running"),
])
def test_page_state_covers_the_idle_and_preparing_states(setup, kind, label):
    y = SimpleNamespace(source_name="clip.mp4", source_status="none", source_error=None, finished=None,
                        start_pending=False, start_error=None)
    st = {"phase": "idle"}
    setup(y, st)
    ps = YV.page_state(y, st)
    assert (ps["kind"], ps["label"]) == (kind, label)
    if kind == "preparing":
        assert ps["text"].startswith("Starting the YOLOE GPU worker...")
    if kind == "no_source":
        assert "Choose a video" in ps["text"]


def test_stopped_and_ended_zero_and_nonzero_frame_states_are_distinct():
    from echo_watch.yoloe.session import Outcome
    y = SimpleNamespace(source_name="clip.mp4", source_status="ready", source_error=None, start_pending=False,
                        start_error=None)
    for status, n, kind in (("stopped", 0, "stopped_before_image"), ("stopped", 7, "stopped"), ("completed", 9, "ended")):
        y.finished = Outcome("r", status, "x", True, {"processed": n}, "d")
        assert YV.page_state(y, {"phase": "idle"})["kind"] == kind
    assert "Stopped before any video frame" in YV.page_state(
        SimpleNamespace(**{**y.__dict__, "finished": Outcome("r", "stopped", "x", True, {"processed": 0}, "d")}),
        {"phase": "idle"})["text"]


# ---- production page renderers against inert sinks ----------------------------------------------------------------------
@pytest.fixture()
def failed_page(procs, monkeypatch):
    guard = NoTk()
    for name in ("tk", "ttk", "filedialog"):
        monkeypatch.setattr(page_mod, name, guard)
    r = failed_start(procs)
    p = page_mod.YoloePage.__new__(page_mod.YoloePage)
    sinks = {n: Sink(n) for n in ("status", "file_lbl", "b_choose", "b_start", "b_pause", "b_stop", "v_tracks", "v_dets",
                                  "footer", "fin_title", "fin_lbl", "f_again", "f_other", "detail", "frame_lbl",
                                  "details_txt", "details_view", "list_view", "s_dets", "s_details", "rinfo", "page_lbl",
                                  "rdetails_txt", "rdetails_view", "rcanvas", "r_frame_btn", "r_details_btn")}
    p.__dict__.update(sinks)
    p.__dict__.update(root=None, c=r.y, canvas=Canvas(), tree=Tree(), runs_tree=Tree(), rec_tree=Tree(),
                      prep=SimpleNamespace(drain=lambda: []), rate=page_mod.DisplayedRate(), _photo=None,
                      _img_item=None, _painted=None, _fit=None, _overlay_key=None, _rows_key=None, _placeholder=None,
                      _photo_fn=lambda root, ppm: object(), _lay=YV.layout(1280, 720, 1.0), _side_view="detections",
                      _details_shown=None, _rdetails_shown=None, _rview="details", _source_gen=r.y.source_gen,
                      _counts_blank=False, _panel="live", _review_key=None, _review_item=None, _review_photo=None,
                      tiles={k: (Sink(k + "n"), Sink(k + "d")) for k in ("detections", "associated", "untracked")})
    p.schedule = DisplayScheduler(SimpleNamespace(submit=lambda tok, req: None))
    yield p, r
    r.close()


def inserted(sink):
    return "".join(a[1] for name, a, _k in sink.calls if name == "insert")


def test_the_failed_page_names_the_stage_and_cause_keeps_the_file_and_never_shows_choose_video(failed_page):
    p, r = failed_page
    p._monitor(time.monotonic())
    texts = [kw["text"] for _a, kw in p.canvas.of("text", "yph")]
    assert texts[0] == "Analysis could not start"
    body = texts[1]
    assert body.startswith("Starting the YOLOE GPU worker failed: model warm-up: AttributeError:")
    assert "No video frame was analysed. chosen clip.mp4 is still selected" in body
    assert not any("Choose a video" in t for t in texts)
    assert [p.tiles[k][0].cget("text") for k in ("detections", "associated", "untracked")] == ["—"] * 3
    assert p.tiles["detections"][1].cget("text") == "detections · not analysed"
    p._chrome(time.monotonic(), True)
    assert p.status.cget("text") == "Failed" and p.fin_title.cget("text") == "Analysis could not start"
    assert "model warm-up" in p.fin_lbl.cget("text") and p.file_lbl.cget("text").startswith("chosen clip.mp4")
    assert p.f_again.cget("state") == "normal" and p.f_other.cget("state") == "normal"


def test_details_are_reachable_inline_and_hold_the_full_bounded_failure(failed_page):
    p, r = failed_page
    p._show_side("details")
    text = inserted(p.details_txt)
    assert "Failed stage: starting the YOLOE GPU worker" in text and "Worker traceback:" in text
    assert "has no attribute 'half'" in text and "Saved test:" in text and "evidence complete" in text
    assert ("tkraise", (), {}) in p.details_view.calls and len(text) <= YV.DETAILS_MAX + 200
    n = len(p.details_txt.calls)
    p._refresh_details()                                         # unchanged text: no rewrite (keeps selection/scroll)
    assert len(p.details_txt.calls) == n


def test_reopened_zero_frame_result_shows_its_details_in_the_results_panel(failed_page):
    p, r = failed_page
    r.y.show_results()
    assert r.pump_until(lambda: r.y.results_status == "done")
    assert r.y.open_result(r.y.results[0]["run_id"])
    assert r.pump_until(lambda: (r.y.review or {}).get("status") == "done")
    p._panel = "results"
    p._render_results()
    assert "Worker traceback:" in inserted(p.rdetails_txt) and ("tkraise", (), {}) in p.rdetails_view.calls
    assert "Reason:" in p.rinfo.cget("text") and r.decoded == []


def test_a_source_change_clears_attempt_specific_presentation(failed_page):
    p, r = failed_page
    p._monitor(time.monotonic())
    other = r.d / "another.mp4"
    other.write_bytes(b"z" * 32)
    assert r.y.choose_source(str(other))
    p._monitor(time.monotonic())
    texts = [kw["text"] for _a, kw in p.canvas.of("text", "yph")]
    assert texts[0] == "Checking the video" and not any("could not start" in t for t in texts)


def test_the_header_describes_the_yoloe_attempt_on_its_page_and_routes_controls_unchanged():
    import ast
    src = (ROOT / "src/echo_watch/app/tk_app.py").read_text("utf-8")
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_render_chrome")
    body = ast.unparse(fn)
    owns = next(ast.unparse(n.value) for n in ast.walk(fn) if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "yoloe_page_owns_header")
    assert "self.page == 'yoloe'" in owns and "self.yc is not None" in owns and "not self.c.analysis_busy()" in owns
    assert "self._yoloe_busy() or yoloe_page_owns_header" in body and "ns = self.yc.worker_text()" in body
    hs = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_header_stop")
    assert "if self._yoloe_busy():" in ast.unparse(hs)          # Stop still goes to the mode that holds analysis


def test_the_worker_status_line_names_the_yoloe_worker_and_its_failed_stage(procs):
    r = failed_start(procs)
    try:
        assert r.y.worker_text() == "YOLOE GPU worker: failed (model warm-up)"
    finally:
        r.close()
