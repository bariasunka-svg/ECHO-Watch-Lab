# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 matrix 4 and 6 (controller part): ONE application owner of GPU analysis between the REAL Display Assets
controller (over the existing FakeEngine) and the REAL YOLOE controller/session (in-process worker core, scripted
source); handover of idle workers; navigation-independent sessions; presentation changes without inference; finish
actions; saved results reopened and paged with no inference, no Library profile and no item run."""
from __future__ import annotations

import time
import uuid

import pytest

from echo_watch.app.controller import WorkbenchController
from echo_watch.app.resources import AnalysisOwner, ResourceBusy, release_display_worker
from echo_watch.app.tasks import TaskRunner
from echo_watch.app.yoloe_controller import YoloeController
from echo_watch.config import Settings
from echo_watch.yoloe import review as R
from echo_watch.yoloe.session import YoloeSession
from lab06_support import InProcessClient, ScriptSource, media_frames, wait_for, work_dir
from test_workbench_controller import FakeEngine, settle


class Rig:
    def __init__(self, frames=40, fps=20.0, decode=None):
        self.d = work_dir(f"lab06-owner-{uuid.uuid4().hex[:6]}")
        self.order: list[str] = []
        self.eng = FakeEngine(self.d / "da")
        video = self.d / "clip.mp4"
        video.write_bytes(b"x" * 64)
        self.video = str(video)
        self.da = WorkbenchController(self.eng, TaskRunner(3),
                                      probe=lambda p: {"effective_fps": 15.0, "width": 1280, "height": 720})
        self.owner = AnalysisOwner()
        self.da.resources = self.owner
        eng, da, order = self.eng, self.da, self.order
        start_worker = eng.start_worker

        def da_start():
            order.append("da_worker_start")
            return start_worker()
        eng.start_worker = da_start

        def release_da():
            order.append("da_worker_released")
            eng.worker.started = False
            da.worker_status = "not started"
        self.owner.register("display_assets", "Display Assets", da.analysis_busy, eng.worker.alive, release_da)
        self.client = InProcessClient()
        orig_start, orig_release = self.client.start, self.client.release

        def y_start():
            order.append("yoloe_worker_start")
            return orig_start()

        def y_release(wait_s=5.0):
            order.append("yoloe_worker_released")
            return orig_release(wait_s)
        self.client.start, self.client.release = y_start, y_release
        s = Settings(data_dir=self.d / "operator")
        self.frames = media_frames(frames, fps=fps)
        self.session = YoloeSession(runs_dir=s.data_dir / "yoloe-runs", client=self.client,
                                    source_factory=lambda p: ScriptSource(self.frames))
        self.y = YoloeController(s, self.owner, tasks=TaskRunner(2), client=self.client, session=self.session,
                                 review_decode=decode)

    def pump_until(self, cond, timeout=10.0):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.y.pump()
            self.da.pump()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def ready_yoloe(self):
        assert self.y.choose_source(self.video)
        assert self.pump_until(lambda: self.y.source_status == "ready")

    def close(self):
        if self.session.busy:
            self.session.stop()
            self.session.wait_idle(5)
        self.y.tasks.shutdown(1.0)
        self.y.review_worker.shutdown(1.0)
        self.da.tasks.shutdown(1.0)
        self.eng.store.close()


@pytest.fixture()
def rig():
    r = Rig()
    yield r
    r.close()


def test_choosing_a_video_hashes_it_and_starts_nothing(rig):
    rig.ready_yoloe()
    assert rig.y.source_info.sha256 and rig.client.calls == [] and not rig.session.busy and rig.y.can()["start"]


def test_a_running_yoloe_test_blocks_display_assets_scan_and_start_with_an_explanation(rig):
    rig.ready_yoloe()
    assert rig.y.start()
    assert rig.pump_until(lambda: rig.session.counts["published"] >= 1)
    assert rig.da.choose_source(rig.video) and settle(rig.da, lambda: rig.da.source_status == "ready")
    assert rig.da.scan() is False
    assert "YOLOE test is still active" in rig.da.messages[-1].text
    assert "da_worker_start" not in rig.order and rig.da.scan_status == "idle"
    rig.y.show_results()                                      # switching panels/pages never touches the session
    rig.y.show_live()
    assert rig.session.phase == "running"


def test_display_assets_activity_blocks_a_yoloe_start(rig):
    rig.ready_yoloe()
    rig.da.start_pending = True
    assert rig.y.start() is False and "Display Assets is still active" in rig.y.messages[-1][0]
    assert rig.client.calls == []
    rig.da.start_pending = False
    rig.da.scan_status = "running"
    assert rig.y.start() is False
    rig.da.scan_status = "idle"


def test_an_idle_display_worker_is_released_before_the_yoloe_worker_starts_and_vice_versa(rig):
    assert rig.da.choose_source(rig.video) and settle(rig.da, lambda: rig.da.source_status == "ready")
    assert rig.da.scan() and settle(rig.da, lambda: rig.da.scan_status == "done")
    assert rig.eng.worker.alive()
    rig.ready_yoloe()
    assert rig.y.start()
    assert rig.pump_until(lambda: rig.session.counts["published"] >= 2)
    assert rig.order.index("da_worker_released") < rig.order.index("yoloe_worker_start")
    assert not rig.eng.worker.alive() and rig.da.worker_status == "not started"
    rig.y.stop()
    assert rig.pump_until(lambda: not rig.y.busy() and rig.y.finished is not None)
    assert rig.client.alive()                                 # idle YOLOE worker stays until another mode needs the GPU
    assert rig.da.scan() and settle(rig.da, lambda: rig.da.scan_status == "done")
    assert rig.order.index("yoloe_worker_released") < len(rig.order) - 1 and rig.order[-1] == "da_worker_start"
    assert not rig.client.alive() and rig.owner.holder == "display_assets"


def test_owner_refuses_handover_while_the_other_mode_is_busy_or_its_worker_does_not_stop():
    o = AnalysisOwner()
    state = {"busy": False, "alive": True}
    o.register("a", "Mode A", lambda: state["busy"], lambda: state["alive"], lambda: None)
    o.register("b", "Mode B", lambda: False, lambda: False, lambda: None)
    state["busy"] = True
    assert o.blocked_by("b").startswith("Mode A is still active") and o.blocked_by("a") is None
    with pytest.raises(ResourceBusy):
        o.handover("b")
    state["busy"] = False
    with pytest.raises(ResourceBusy, match="did not stop"):
        o.handover("b", wait_s=0.2)                           # its own worker would not stop: no second worker
    state["alive"] = False
    assert o.handover("b")["released"] == [] and o.holder == "b"


def test_the_display_worker_release_keeps_the_client_restartable():
    class W:
        proc, state, reason, refs_classes, refs_digest = None, "ready", None, [1], "d"
        killed = 0

        def _kill(self):
            self.killed += 1
    w = W()
    release_display_worker(w)
    assert w.killed == 1 and w.state == "not started" and w.refs_digest is None and w.refs_classes is None


def test_a_failed_yoloe_start_releases_ownership_only_after_the_session_settles(rig):
    rig.client.ready = False
    rig.ready_yoloe()
    assert rig.y.start()
    assert rig.pump_until(lambda: rig.y.finished is not None)
    assert rig.y.finished.status == "failed" and not rig.y.busy() and rig.owner.blocked_by("display_assets") is None
    assert "failed" in rig.y.messages[-1][0].lower()


def test_presentation_switches_never_run_the_model_and_are_recorded_as_view_changes(rig):
    rig.ready_yoloe()
    rig.y.start()
    assert rig.pump_until(lambda: rig.session.counts["published"] >= 2)
    rig.y.pause()
    time.sleep(0.2)
    n = len(rig.client.model.calls)
    rig.y.set_view("detections")
    assert rig.y.set_shown_conf(0.6) and not rig.y.set_shown_conf(1.5)
    rig.y.set_view("tracks")
    time.sleep(0.1)
    assert len(rig.client.model.calls) == n and len(rig.client.trackers) == 1     # no inference, no tracker reset
    rig.y.stop()
    assert rig.pump_until(lambda: rig.y.finished is not None)
    from lab06_support import read_trace
    views = [r for r in read_trace(rig.y.finished.run_dir) if r["k"] == "view"]
    assert [(v["view"], v["shown_conf"]) for v in views] == [("detections", 0.25), ("detections", 0.6), ("tracks", 0.6)]


def test_selection_follows_a_track_and_otherwise_becomes_historical(rig):
    rig.ready_yoloe()
    rig.y.start()
    assert rig.pump_until(lambda: rig.session.counts["published"] >= 2)
    rig.y.pause()
    time.sleep(0.1)
    pk = rig.session.latest_packet()
    rig.y.select(pk, 0)
    tid = rig.y.selection["track_id"]
    assert tid is not None
    rig.y.resume()
    assert rig.pump_until(lambda: rig.session.latest_packet().seq > pk.seq + 1)
    pk2 = rig.session.latest_packet()
    rig.y.follow(pk2)
    assert rig.y.selection["key"] == pk2.key and rig.y.selection["track_id"] == tid
    rig.y.select(pk2, 1)                                       # an untracked detection cannot be followed
    assert rig.pump_until(lambda: rig.session.latest_packet().seq > pk2.seq)
    rig.y.follow(rig.session.latest_packet())
    assert rig.y.selection is None and "not in the current frame" in rig.y.selection_note


def test_eof_offers_finish_actions_and_run_again_is_a_new_run_and_epoch():
    r = Rig(frames=5, fps=500)
    try:
        r.ready_yoloe()
        r.y.start()
        assert r.pump_until(lambda: r.y.finished is not None)
        first = r.y.finished
        assert first.status == "completed" and r.y.can()["run_again"] and r.y.can()["choose_another"]
        e1 = r.session.latest_packet().epoch
        assert r.y.run_again()
        assert r.pump_until(lambda: r.y.finished is not None and r.y.finished.run_id != first.run_id)
        assert r.session.latest_packet().epoch != e1 and len(r.client.trackers) == 2
    finally:
        r.close()


def test_saved_results_reopen_page_and_show_the_original_frame_without_inference():
    decoded = []

    def fake_decode(req, cancel=None):
        decoded.append(req)
        import numpy as np
        return R.ReviewFrame(req, np.zeros((req.size[1], req.size[0], 3), np.uint8), req.pts, 0.0)
    r = Rig(frames=60, fps=5000, decode=fake_decode)
    try:
        r.ready_yoloe()
        r.y.start()
        assert r.pump_until(lambda: r.y.finished is not None)
        n_pred, n_calls = len(r.client.model.calls), len(r.client.calls)
        r.y.show_results()
        assert r.pump_until(lambda: r.y.results_status == "done")
        assert r.y.results[0]["run_id"] == r.y.finished.run_id and r.y.results[0]["processed"] == 60
        assert r.y.open_result(r.y.finished.run_id)
        assert r.pump_until(lambda: (r.y.review or {}).get("status") == "done")
        rv = r.y.review
        assert rv["count"] == 60 and len(rv["rows"]) == 50 and rv["source"]["ok"] is True
        r.y.page(1)
        assert r.pump_until(lambda: rv["rows"] and rv["rows"][0]["n"] == 50)
        r.y.select_record(55)
        assert r.pump_until(lambda: rv.get("frame") is not None)
        assert rv["record"]["seq"] == 56 and decoded[-1].pts == pytest.approx(rv["record"]["media"]["pts"])
        assert decoded[-1].expect_sig == rv["record"]["sig"] and decoded[-1].size == (64, 48)
        assert len(r.client.model.calls) == n_pred and len(r.client.calls) == n_calls   # no inference, no worker use
        assert not (r.d / "da" / "assets").exists() or not any((r.d / "da" / "assets").iterdir())
        # the original video changes: the review reports an evidence gap instead of decoding another file
        open(r.video, "ab").write(b"more")
        assert r.y.open_result(r.y.finished.run_id)
        assert r.pump_until(lambda: (r.y.review or {}).get("status") == "done")
        assert r.y.review["source"]["ok"] is False and "different file" in r.y.review["source"]["why"]
        r.y.select_record(0)
        assert r.pump_until(lambda: r.y.review.get("frame_gap"))
        assert len(decoded) == 1
    finally:
        r.close()


def test_yoloe_shutdown_stops_a_running_test_and_its_worker(rig):
    rig.ready_yoloe()
    rig.y.start()
    assert rig.pump_until(lambda: rig.session.counts["published"] >= 1)
    rec = rig.y.shutdown(wait_s=5)
    assert rec["busy"] and rec["settled"] and rig.client.state == "stopped" and not rig.session.busy
    assert rig.session.last_outcome.status == "stopped" and "application close" in rig.session.last_outcome.reason
