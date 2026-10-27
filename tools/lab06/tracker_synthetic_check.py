# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-06 narrow synthetic CPU tracker check, run ONLY in the GPU runtime as a bounded windowless test child:

  env/gpu/python.exe -I tools/lab06/tracker_synthetic_check.py <out.json> <scratch dir>

It imports the INSTALLED BoT-SORT / lap / NumPy / Torch modules and drives the PRODUCTION adapter
(``echo_watch.yoloe.tracker.BotSortAdapter``) and the production worker core (``echo_watch.yoloe.worker.YoloeCore``, with
a scripted stand-in detector) on generated, textured arrays. No YOLO/model instance, no checkpoint, no CUDA (devices are
hidden and every CUDA entry point is trapped and recorded), no source file, no codec, no network (egress and child
processes are refused by the project's audit hooks), auto-install off, every writable path inside <scratch dir>.
This is association/optical-flow verification on synthetic data, not recognition or real-video tracking evidence.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

OUT, SCRATCH = Path(sys.argv[1]), Path(sys.argv[2])
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.update(YOLO_OFFLINE="1", YOLO_AUTOINSTALL="False", YOLO_CONFIG_DIR=str(SCRATCH / "ultralytics"),
                  YOLO_VERBOSE="False", CUDA_VISIBLE_DEVICES="", TEMP=str(SCRATCH), TMP=str(SCRATCH),
                  TORCH_HOME=str(SCRATCH / "torch"), MPLCONFIGDIR=str(SCRATCH / "mpl"))

from echo_watch import guards  # noqa: E402

guards.install(block_subprocess=True)
IMPORTS: list[str] = []


def _audit(event, args):
    if event == "import":
        IMPORTS.append(str(args[0]))


sys.addaudithook(_audit)

import numpy as np  # noqa: E402
import torch  # noqa: E402

CUDA_CALLS: list[str] = []


def _trap(name):
    def f(*a, **k):
        CUDA_CALLS.append(name)
        raise RuntimeError(f"CUDA entry point {name} is forbidden in this synthetic check")
    return f


for _n in ("is_available", "device_count", "init", "_lazy_init", "current_device", "synchronize", "mem_get_info"):
    setattr(torch.cuda, _n, _trap(f"torch.cuda.{_n}"))

from echo_watch.yoloe import association as A  # noqa: E402
from echo_watch.yoloe.profile import load_tracker_config, vocabulary_digest  # noqa: E402
from echo_watch.yoloe.tracker import BotSortAdapter  # noqa: E402
from echo_watch.yoloe.worker import YoloeCore  # noqa: E402

CFG, CFG_SHA = load_tracker_config()
RNG = np.random.default_rng(20261026)
H, W = 360, 640
results: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tracker_config_sha256": CFG_SHA, "cases": {}}


def texture(h=H + 200, w=W + 400) -> np.ndarray:
    import cv2
    base = RNG.integers(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    big = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = RNG.integers(0, 40, (h, w, 3), dtype=np.uint8)
    return cv2.add(big, noise)


BG = texture()


def frame(pan_x=0, pan_y=0, boxes=()):
    """A W x H view of the textured world shifted by the camera pan, with solid objects drawn at ``boxes``."""
    img = BG[100 + pan_y:100 + pan_y + H, 200 + pan_x:200 + pan_x + W].copy()
    for (x1, y1, x2, y2), color in boxes:
        img[int(y1):int(y2), int(x1):int(x2)] = color
    return img


def dets(rows):
    """rows: [(xyxy, score, cls)] -> production detection records (index = row)."""
    return [{"i": k, "xyxy": [float(v) for v in b], "score": float(s), "cls": int(c), "name": f"class{int(c)}"}
            for k, (b, s, c) in enumerate(rows)]


def case(name):
    def deco(fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            results["cases"][name] = {"pass": True, "detail": detail, "s": round(time.perf_counter() - t0, 3)}
        except AssertionError as e:
            results["cases"][name] = {"pass": False, "detail": f"assertion: {e}", "s": round(time.perf_counter() - t0, 3)}
        except Exception as e:  # noqa: BLE001
            results["cases"][name] = {"pass": False, "detail": f"{type(e).__name__}: {e}"}
        return fn
    return deco


def step(ad: BotSortAdapter, records, img, accept=True):
    out = ad.update(A.detections_array(records), img)
    ad.settle(accept)
    return out


@case("identity_reid_off_gmc_sparse_optical_flow")
def _():
    ad = BotSortAdapter(CFG)
    idn = ad.identity()
    assert idn["encoder"] is None and ad.tracker.encoder is None, idn
    assert idn["gmc_method"] == "sparseOptFlow" and idn["with_reid"] is False, idn
    assert idn["max_frames_lost"] == 30, idn
    assert type(ad.tracker).__module__ == "ultralytics.trackers.bot_sort", idn
    return idn


@case("easy_motion_keeps_ids_and_every_detection_index")
def _():
    ad = BotSortAdapter(CFG)
    ids_a, ids_b = set(), set()
    per_frame = []
    for t in range(20):
        a = (100 + 4 * t, 100, 160 + 4 * t, 180)
        b = (400 - 3 * t, 200, 470 - 3 * t, 290)
        recs = dets([(a, 0.9, 41), (b, 0.8, 63)])
        out = step(ad, recs, frame(boxes=[(a, (0, 0, 255)), (b, (0, 255, 0))]))
        assert not out["errors"], out["errors"]
        assert set(out["assoc"]) <= {0, 1}
        if 0 in out["assoc"]:
            ids_a.add(out["assoc"][0]["track_id"])
        if 1 in out["assoc"]:
            ids_b.add(out["assoc"][1]["track_id"])
        per_frame.append(sorted((i, a_["track_id"], a_["state"]) for i, a_ in out["assoc"].items()))
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a != ids_b, (ids_a, ids_b)
    assert per_frame[0] == [(0, 1, "tracked"), (1, 2, "tracked")], per_frame[0]   # frame 1 activates immediately
    return {"ids": [sorted(ids_a), sorted(ids_b)], "first": per_frame[0], "last": per_frame[-1]}


@case("new_object_is_tentative_then_confirmed_and_low_scores_stay_represented")
def _():
    ad = BotSortAdapter(CFG)
    seen = []
    for t in range(8):
        a = (100 + 3 * t, 100, 160 + 3 * t, 180)
        rows = [(a, 0.9, 1)]
        if t >= 3:
            rows.append(((300 + 2 * t, 150, 360 + 2 * t, 230), 0.7, 2))     # appears at frame 4
        rows.append(((520, 40, 560, 80), 0.15, 3))                          # low score, never a track by itself
        rows.append(((20, 300, 50, 330), 0.05, 4))                          # below the low threshold
        recs = dets(rows)
        out = step(ad, recs, frame(boxes=[(r[0], (40 * k, 200, 90)) for k, r in enumerate(rows)]))
        assert not out["errors"], out["errors"]
        states = {i: (x["track_id"], x["state"]) for i, x in out["assoc"].items()}
        summ = A.summarize(len(recs), out["assoc"])
        assert summ["detections"] == len(recs) and summ["untracked"] == len(recs) - summ["associated"]
        seen.append({"frame": t + 1, "assoc": states, "summary": summ})
    f4, f5 = seen[3]["assoc"], seen[4]["assoc"]
    assert f4.get(1, (None, None))[1] == "tentative", f4
    assert f5.get(1, (None, None))[1] == "tracked" and f5[1][0] == f4[1][0], (f4, f5)
    for s in seen:
        assert 2 not in s["assoc"] and 3 not in s["assoc"], s      # low-score detections: represented as untracked
    return seen[2:6]


@case("low_score_detection_keeps_an_existing_track_in_second_association")
def _():
    ad = BotSortAdapter(CFG)
    ids = []
    for t in range(10):
        b = (200 + 3 * t, 120, 270 + 3 * t, 210)
        score = 0.9 if t < 5 else 0.18                     # drops below track_high_thresh but above track_low_thresh
        out = step(ad, dets([(b, score, 5)]), frame(boxes=[(b, (255, 0, 0))]))
        ids.append(out["assoc"].get(0, {}).get("track_id"))
    assert len(set(ids)) == 1 and ids[0] is not None, ids
    return ids


@case("empty_frames_advance_loss_expiry_and_new_id_after_expiry")
def _():
    ad = BotSortAdapter(CFG)
    b = (100, 100, 170, 180)
    for _t in range(3):
        out = step(ad, dets([(b, 0.9, 1)]), frame(boxes=[(b, (0, 0, 255))]))
    first_id = out["assoc"][0]["track_id"]
    lost_series = []
    for _t in range(32):
        out = step(ad, [], frame())
        assert out["rows"] == [] and out["assoc"] == {}
        lost_series.append(out["lost"])
    assert lost_series[0] == 1 and lost_series[-1] == 0, lost_series   # lost, then removed after > track_buffer updates
    out = step(ad, dets([(b, 0.9, 1)]), frame(boxes=[(b, (0, 0, 255))]))
    a = out["assoc"].get(0)
    new_id = a["track_id"] if a else None
    assert new_id is not None and new_id != first_id, (first_id, a)
    return {"first_id": first_id, "lost_after_1": lost_series[0], "lost_after_32": lost_series[-1],
            "frame_id": out["frame_id"], "reappeared_as": (new_id, a["state"])}


@case("lost_track_within_buffer_is_refound_with_same_id")
def _():
    ad = BotSortAdapter(CFG)
    b = (100, 100, 170, 180)
    for _t in range(3):
        out = step(ad, dets([(b, 0.9, 1)]), frame(boxes=[(b, (0, 0, 255))]))
    tid = out["assoc"][0]["track_id"]
    for _t in range(5):
        step(ad, [], frame())
    out = step(ad, dets([(b, 0.9, 1)]), frame(boxes=[(b, (0, 0, 255))]))
    assert out["assoc"][0]["track_id"] == tid and out["assoc"][0]["state"] == "tracked", out["assoc"]
    return {"track_id": tid}


@case("label_flips_keep_the_track_and_are_never_relabelled")
def _():
    core = YoloeCore(None, {7: "cup", 9: "mug"}, "synthetic", None, lambda: BotSortAdapter(CFG))
    core.reset("synthetic/e1")
    seq = []

    class R:
        def __init__(self, rows):
            self.boxes = type("B", (), {"xyxy": np.array([r[0] for r in rows], np.float32),
                                        "conf": np.array([r[1] for r in rows], np.float32),
                                        "cls": np.array([r[2] for r in rows], np.float32),
                                        "__len__": lambda s: len(rows)})()
            self.speed = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
            self.masks = None

    rows_by_frame = []
    for t in range(8):
        b = (150 + 3 * t, 140, 220 + 3 * t, 230)
        rows_by_frame.append([(b, 0.85, 7 if t % 2 == 0 else 9)])

    class Spy:
        calls = []

        def predict(self, img, **kw):
            Spy.calls.append(kw)
            return [R(rows_by_frame[len(Spy.calls) - 1])]

    core.model = Spy()
    prev = None
    for t in range(8):
        b = rows_by_frame[t][0][0]
        rep = core.detect_track({"epoch": "synthetic/e1", "frame": str(t + 1), "accept": prev},
                                frame(boxes=[(b, (0, 200, 200))]))
        prev = str(t + 1)
        assoc, errs = A.join(len(rep["detections"]), rep["tracker"]["rows"], [tuple(x) for x in rep["tracker"]["tentative"]])
        assert not errs, errs
        seq.append((rep["detections"][0]["name"], assoc.get(0, {}).get("track_id")))
    core.settle({"epoch": "synthetic/e1", "frame": prev, "decision": "accept"})
    names = [n for n, _ in seq]
    tids = {t for _, t in seq}
    assert names == ["cup", "mug"] * 4, names
    assert len(tids) == 1 and None not in tids, seq
    assert len(Spy.calls) == 8 and all(c["classes"] is None and c["conf"] == 0.10 for c in Spy.calls)
    return {"per_frame": seq, "predict_calls": len(Spy.calls)}


@case("camera_pan_with_texture_keeps_ids_using_sparse_optical_flow")
def _():
    ad = BotSortAdapter(CFG)
    ids = set()
    warn = []
    for t in range(15):
        pan = 9 * t                                         # the whole view moves; objects are static in the world
        objs = [((300 - pan, 120, 360 - pan, 200), (0, 0, 255)), ((460 - pan, 220, 530 - pan, 300), (255, 255, 0))]
        recs = dets([(o[0], 0.9, 1) for o in objs])
        out = step(ad, recs, frame(pan_x=pan, boxes=objs))
        warn += out["warnings"]
        ids |= {(i, a["track_id"]) for i, a in out["assoc"].items()}
    per_index = {}
    for i, tid in ids:
        per_index.setdefault(i, set()).add(tid)
    assert all(len(v) == 1 for v in per_index.values()) and len(per_index) == 2, per_index
    # the installed sparse optical flow itself measures the pan on this texture (a separate GMC of the same class)
    from ultralytics.trackers.utils.gmc import GMC
    g = GMC(method="sparseOptFlow")
    g.apply(frame(pan_x=0))
    warp = g.apply(frame(pan_x=9))
    assert abs(warp[0, 2] + 9) < 1.5 and abs(warp[1, 2]) < 1.5, warp.tolist()
    return {"ids": {k: sorted(v) for k, v in per_index.items()}, "warnings": warn[:5],
            "gmc_initialized": bool(ad.tracker.gmc.initializedFirstFrame),
            "gmc_translation_for_9px_pan": [round(float(warp[0, 2]), 3), round(float(warp[1, 2]), 3)],
            "note": "IDs would also survive this pan by IoU alone; the warp estimate shows GMC ran and measured it"}



@case("rejected_update_restores_exact_state_and_counter")
def _():
    ad = BotSortAdapter(CFG)
    for t in range(4):
        b = (100 + 4 * t, 100, 160 + 4 * t, 180)
        step(ad, dets([(b, 0.9, 1)]), frame(boxes=[(b, (0, 0, 255))]))
    from ultralytics.trackers.basetrack import BaseTrack

    def state():
        tr = ad.tracker
        return (tr.frame_id, BaseTrack._count, [(s.track_id, s.state, s.mean.round(4).tolist()) for s in tr.tracked_stracks],
                [s.track_id for s in tr.lost_stracks], None if tr.gmc.prevFrame is None else int(tr.gmc.prevFrame.sum()))
    before = state()
    b = (116, 100, 176, 180)
    nb = (400, 250, 460, 320)
    recs = dets([(b, 0.9, 1), (nb, 0.9, 2)])
    img = frame(boxes=[(b, (0, 0, 255)), (nb, (0, 255, 0))])
    out1 = ad.update(A.detections_array(recs), img)
    try:
        ad.update(A.detections_array(recs), img)
        raise AssertionError("an unsettled update was not refused")
    except Exception as e:  # noqa: BLE001
        assert "not settled" in str(e), e
    assert ad.settle(False) is True
    assert state() == before, (state(), before)
    out2 = ad.update(A.detections_array(recs), img)
    ad.settle(True)
    assert out1["rows"] == out2["rows"] and out1["tentative"] == out2["tentative"], (out1, out2)
    return {"restored_frame_id": before[0], "counter": before[1], "replayed_rows_equal": True}


@case("reset_starts_a_new_instance_and_id_namespace")
def _():
    ad = BotSortAdapter(CFG)
    for t in range(3):
        b = (100 + 4 * t, 100, 160 + 4 * t, 180)
        step(ad, dets([(b, 0.9, 1), ((300, 50, 340, 90), 0.9, 2)]), frame(boxes=[(b, (0, 0, 255))]))
    old = ad.tracker
    ad.reset()
    assert ad.tracker is not old and ad.tracker.frame_id == 0 and not ad.tracker.tracked_stracks
    assert ad.tracker.gmc.prevFrame is None
    out = step(ad, dets([((10, 10, 60, 60), 0.9, 1)]), frame())
    assert out["assoc"][0]["track_id"] == 1, out["assoc"]
    return {"first_id_after_reset": 1}


@case("invalid_tracker_output_is_reported_and_attached_to_nothing")
def _():
    ad = BotSortAdapter(CFG)
    recs = dets([((100, 100, 160, 180), 0.9, 1), ((300, 100, 360, 180), 0.9, 2)])
    out = step(ad, recs, frame())
    rows = [list(r) for r in out["rows"]]
    good, _ = A.join(2, rows, [])
    bad_rows = [rows[0][:7] + [7.0], rows[1][:7] + [0.0]]            # out of range + steals detection 0
    dup_rows = [rows[0], rows[0][:4] + [99.0] + rows[0][5:]]          # two tracks claim detection 0
    frac = [rows[0][:7] + [0.5]]
    a1, e1 = A.join(2, bad_rows, [])
    a2, e2 = A.join(2, dup_rows, [])
    a3, e3 = A.join(2, frac, [])
    assert set(good) == {0, 1}
    assert 0 in a1 and 1 not in a1 and any("outside" in e for e in e1), (a1, e1)
    assert a2 == {} and any("claimed" in e for e in e2), (a2, e2)
    assert a3 == {} and any("not an integer" in e for e in e3), (a3, e3)
    return {"errors": e1 + e2 + e3}


@case("no_cuda_no_model_no_network")
def _():
    assert not CUDA_CALLS, CUDA_CALLS
    import torch as _t
    assert not _t.cuda.is_initialized(), "CUDA was initialized"
    snap = guards.snapshot()
    assert not snap["egress_refusals"] and not snap["subprocesses"], snap
    mods = sorted({m for m in sys.modules if m.startswith("ultralytics")})
    assert "ultralytics.models.yolo.yoloe" not in sys.modules, "a YOLOE model module was imported"
    return {"cuda_calls": CUDA_CALLS, "ultralytics_modules_imported": len(mods),
            "autobackend_module_imported_not_instantiated": "ultralytics.nn.autobackend" in sys.modules,
            "lap": __import__("lap").__version__, "numpy": np.__version__, "torch": _t.__version__,
            "ultralytics": __import__("ultralytics").__version__}


results["vocabulary_digest_selftest"] = vocabulary_digest({0: "a", 1: "b"})
results["all_pass"] = all(c["pass"] for c in results["cases"].values())
results["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
results["scratch_files"] = sorted(str(p.relative_to(SCRATCH)) for p in SCRATCH.rglob("*") if p.is_file())
OUT.write_text(json.dumps(results, indent=1, default=str), "utf-8")
for k, v in results["cases"].items():
    print(("PASS " if v["pass"] else "FAIL ") + k + ("" if v["pass"] else f" :: {v['detail']}"))
print("ALL PASS" if results["all_pass"] else "FAILURES")
sys.exit(0 if results["all_pass"] else 1)
