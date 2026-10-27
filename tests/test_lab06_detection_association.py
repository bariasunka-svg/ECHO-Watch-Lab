# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 matrix 1-2 (UI-runtime part): detection/association separation, the production worker core's detector
call configuration with a SPY model (wiring verification, not GPU recognition), protocol settlement rules and the
pinned profile. The real BoT-SORT adapter is checked separately in the GPU runtime (tools/lab06)."""
from __future__ import annotations

import ast

import numpy as np
import pytest

from echo_watch.config import MODEL_PF, MODEL_PF_SHA256, ROOT, Settings
from echo_watch.yoloe import association as A
from echo_watch.yoloe import profile as P
from echo_watch.yoloe.client import YoloeWorkerClient, yoloe_spec
from echo_watch.yoloe.worker import ProtocolError, YoloeCore
from lab06_support import NAMES, ScriptTracker, SpyModel, work_dir

IMG = np.zeros((48, 64, 3), np.uint8)


# ---- join by returned index -------------------------------------------------------------------------------------
def test_join_attaches_only_by_returned_index_and_keeps_every_detection_countable():
    rows = [[10, 10, 30, 40, 7, 0.9, 1, 0], [40, 5, 60, 30, 9, 0.8, 2, 2]]
    assoc, errors = A.join(4, rows, [(11, 3.0, [0, 0, 5, 5], 0.6, 4.0)])
    assert errors == []
    assert assoc[0]["track_id"] == 7 and assoc[2]["track_id"] == 9 and assoc[3]["state"] == A.TENTATIVE
    assert 1 not in assoc                                   # untracked detection: present, attached to nothing
    assert A.summarize(4, assoc) == {"detections": 4, "associated": 2, "tentative": 1, "untracked": 2}


@pytest.mark.parametrize("rows,tent,needle", [
    ([[0, 0, 1, 1, 5, 0.9, 1, 4]], [], "outside"),
    ([[0, 0, 1, 1, 5, 0.9, 1, 0.5]], [], "not an integer"),
    ([[0, 0, 1, 1, 5, 0.9, 1, float("nan")]], [], "not an integer"),
    ([[0, 0, 1, 1, 5, 0.9, 1, 0], [0, 0, 1, 1, 6, 0.9, 1, 0]], [], "claimed by tracks"),
    ([[0, 0, 1, 1, 5, 0.9, 1, 0], [0, 0, 1, 1, 5, 0.9, 1, 1]], [], "more than one detection"),
    ([[0, 0, 1, 1, 0, 0.9, 1, 0]], [], "invalid track id"),
    ([], [(5, 0, [0, 0, 1, 1], 0.5, 1), (6, 0, [0, 0, 1, 1], 0.5, 1)], "claimed by tracks"),
])
def test_invalid_tracker_output_is_reported_and_never_attached(rows, tent, needle):
    assoc, errors = A.join(2, rows, tent)
    assert any(needle in e for e in errors), errors
    claimed = [r[7] for r in rows] + [t[1] for t in tent]
    bad = {int(c) for c in claimed if c == c and float(c).is_integer() and 0 <= c < 2}
    # nothing questionable is attached: a contested detection keeps no association at all
    if needle in ("claimed by tracks", "more than one detection"):
        assert not (set(assoc) & bad)


def test_bad_tracker_array_shape_is_an_error_not_a_guess():
    assoc, errors = A.join(3, np.zeros((2, 5)), [])
    assert assoc == {} and "expected M x 8" in errors[0]


def test_detections_array_requires_detector_order():
    good = [{"i": 0, "xyxy": [0, 0, 1, 1], "score": 0.5, "cls": 1, "name": "cup"}]
    assert A.detections_array(good).shape == (1, 6) and A.detections_array([]).shape == (0, 6)
    with pytest.raises(ValueError):
        A.detections_array([{**good[0], "i": 1}])


def test_detarray_is_the_results_view_botsort_uses():
    d = A.DetArray([[10, 20, 30, 60, 0.9, 3], [0, 0, 4, 2, 0.2, 1]])
    assert len(d) == 2 and not hasattr(d, "xywhr")
    assert d.xywh[0].tolist() == [20, 40, 20, 40] and d[d.conf > 0.5].cls.tolist() == [3]
    assert len(A.DetArray(np.zeros((0, 6)))) == 0


# ---- worker core with a spy model ------------------------------------------------------------------------------------
def core(script=None):
    spy = SpyModel(script)
    c = YoloeCore(spy, NAMES, "cuda:0", None, ScriptTracker)
    c.reset("r/e1")
    return c, spy


def test_one_predict_per_frame_with_the_frozen_profile_and_no_class_filter():
    c, spy = core()
    rep = c.detect_track({"epoch": "r/e1", "frame": "1", "accept": None}, IMG)
    c.detect_track({"epoch": "r/e1", "frame": "2", "accept": "1"}, IMG)
    assert len(spy.calls) == 2                                   # exactly one detector pass per frame
    kw = spy.calls[0]
    assert {k: kw[k] for k in ("imgsz", "conf", "iou", "max_det")} == {"imgsz": 640, "conf": 0.10, "iou": 0.70, "max_det": 300}
    assert kw["classes"] is None and kw["device"] == "cuda:0" and kw["quantize"] is None and "half" not in kw
    assert kw["save"] is False and kw["show"] is False and kw["stream"] is False and kw["verbose"] is False
    assert rep["detections"] == [
        {"i": 0, "xyxy": [10.0, 10.0, 30.0, 40.0], "score": 0.9, "cls": 1, "name": "cup"},
        {"i": 1, "xyxy": [40.0, 5.0, 60.0, 30.0], "score": 0.15, "cls": 2, "name": "bottle"}]
    assert rep["masks_computed"] is True and rep["max_det_reached"] is False


def test_low_score_detections_reach_the_tracker_input_and_survive_the_display_filter():
    c, _spy = core()
    rep = c.detect_track({"epoch": "r/e1", "frame": "1", "accept": None}, IMG)
    assert [d["score"] for d in rep["detections"]] == [0.9, 0.15]       # nothing pre-filtered at 0.25
    assoc, _ = A.join(2, rep["tracker"]["rows"], [tuple(t) for t in rep["tracker"]["tentative"]])
    assert 0 in assoc and 1 not in assoc                                # untracked, still a detection
    from echo_watch.app import yoloe_views as YV
    from echo_watch.yoloe.session import YoloePacket
    pk = YoloePacket("r", "r/e1", 1, 1, {}, IMG, (64, 48), tuple(rep["detections"]), assoc, (),
                     A.summarize(2, assoc))
    shown = [e for e in YV.det_entries(pk, P.VIEW_TRACKS, 0.25) if e["shown"]]
    assert len(shown) == 1 and YV.counts(pk, 0.25) == {"detections": 2, "associated": 1, "tentative": 0, "untracked": 1,
                                                        "shown": 1}


def test_max_det_reached_is_disclosed_without_claiming_omissions():
    c, _ = core(lambda n, img: [((0, 0, 2, 2), 0.5, 1)] * 300)
    rep = c.detect_track({"epoch": "r/e1", "frame": "1", "accept": None}, IMG)
    assert rep["max_det_reached"] is True and len(rep["detections"]) == 300


def test_label_changes_are_kept_as_produced():
    c, _ = core(lambda n, img: [((10, 10, 30, 40), 0.8, 1 if n % 2 == 0 else 4)])
    names, prev = [], None
    for k in range(1, 5):
        rep = c.detect_track({"epoch": "r/e1", "frame": str(k), "accept": prev}, IMG)
        prev = str(k)
        names.append(rep["detections"][0]["name"])
    assert names == ["cup", "mug", "cup", "mug"]


def test_settlement_protocol_refuses_unsettled_work_wrong_epochs_and_restores_rejections():
    c, _ = core()
    c.detect_track({"epoch": "r/e1", "frame": "1", "accept": None}, IMG)
    with pytest.raises(ProtocolError, match="not settled"):
        c.detect_track({"epoch": "r/e1", "frame": "2", "accept": None}, IMG)
    with pytest.raises(ProtocolError, match="not the acknowledged epoch"):
        c.detect_track({"epoch": "r/e9", "frame": "2", "accept": "1"}, IMG)
    before = (dict(c.tracker.ids), c.tracker.frame_id)
    c.detect_track({"epoch": "r/e1", "frame": "2", "accept": "1"}, IMG)
    out = c.settle({"epoch": "r/e1", "frame": "2", "decision": "reject"})
    assert out["restored"] is True and (dict(c.tracker.ids), c.tracker.frame_id) == before
    with pytest.raises(ProtocolError, match="nothing provisional"):
        c.detect_track({"epoch": "r/e1", "frame": "3", "accept": "2"}, IMG)
    with pytest.raises(ProtocolError):
        c.settle({"epoch": "r/e1", "frame": "3", "decision": "maybe"})


def test_reset_is_a_new_tracker_instance_and_discards_unsettled_work():
    c, _ = core()
    t1 = c.tracker
    c.detect_track({"epoch": "r/e1", "frame": "1", "accept": None}, IMG)
    ack = c.reset("r/e2")
    assert ack == {"event": "reset", "epoch": "r/e2", "tracker": c.tracker.identity()}
    assert c.tracker is not t1 and c.pending is None and c.tracker.updates == 0     # the old tracker is discarded whole


# ---- pins: model, profile, worker command ----------------------------------------------------------------------------
def test_tracker_profile_is_the_pinned_project_copy_and_refuses_changes():
    cfg, sha = P.load_tracker_config()
    assert cfg == P.EXPECTED_TRACKER and cfg["with_reid"] is False and cfg["gmc_method"] == "sparseOptFlow"
    bad = work_dir("lab06-profile") / "botsort.yaml"
    bad.write_text(P.TRACKER_YAML.read_text("utf-8").replace("with_reid: False", "with_reid: True"), "utf-8")
    with pytest.raises(P.ProfileError, match="with_reid"):
        P.load_tracker_config(bad)
    assert P.vocabulary_digest({1: "b", 0: "a"}) == P.vocabulary_digest(["a", "b"])
    assert P.validate_shown_conf("0.25") == 0.25
    for v in (0.05, 0.99, "x"):
        with pytest.raises(ValueError):
            P.validate_shown_conf(v)


def test_worker_command_loads_only_the_pf_model_with_the_pinned_tracker():
    s = Settings(data_dir=work_dir("lab06-cmd"))
    c = YoloeWorkerClient(yoloe_spec(s))
    cmd = c._cmd()
    assert cmd[1:4] == ["-I", "-m", "echo_watch.yoloe.worker"]
    assert cmd[cmd.index("--model") + 1] == str(MODEL_PF) and cmd[cmd.index("--sha256") + 1] == MODEL_PF_SHA256
    assert "--vp-model" not in cmd and cmd[cmd.index("--device") + 1] == "cuda:0"
    assert cmd[cmd.index("--precision") + 1] == "fp32"
    assert cmd[cmd.index("--tracker-sha256") + 1] == P.load_tracker_config()[1]
    assert c.state == "not started" and c.proc is None                # constructing the client starts nothing


def test_worker_source_never_uses_model_track_prompts_or_reid():
    src = (ROOT / "src/echo_watch/yoloe/worker.py").read_text("utf-8")
    tree = ast.parse(src)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not attrs & {"track", "set_classes", "get_text_pe", "get_vpe", "add_callback", "register_tracker"}
    assert "YOLOEVPSegPredictor" not in src and "vp_model" not in src
    assert "_preflight()" in src and src.index("_preflight()") < src.index("from ultralytics import YOLOE")
    trk = (ROOT / "src/echo_watch/yoloe/tracker.py").read_text("utf-8")
    assert "BOTSORT" in trk and "build_encoder" not in trk.split('"""', 2)[2]
