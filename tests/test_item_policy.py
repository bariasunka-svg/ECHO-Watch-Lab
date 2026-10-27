# SPDX-License-Identifier: AGPL-3.0-only
# PORTED REGRESSION. Source: the preserved ECHO Watch reference tests/test_item_memory_policy.py (SHA-256 ab29efd5d7a9c278dc812baafc7bb788ce4cfdad614fcf62753217f1574f64c2;
# Bari Asunka, AGPL-3.0-only). Only the four import lines were renamed to the standalone namespace; every test body is
# unchanged (EW-LAB-01 exception: the fragmented-track move no longer places the speaker under the watch box), so the adopted item_policy / matching / synthetic-scene code is held to the reference's own expectations
# (one-to-one confirmation after three supports over 1 s, look-alike ambiguity, blind-swap uncertainty, continuous movement
# keeps identity, Cannot-assess when unassessable, review-timer charging/freezing/recovery).
# SPDX-License-Identifier: AGPL-3.0-only
"""EW-01 Item Memory checkpoint A: the pure item-memory-v1 policy and the review timer.

SCRIPTED-POLICY evidence: synthetic ECHO pixels with scripted detections (boxes from the synthetic
scene's own geometry, classes as the visual-prompt worker would label them). No model, GPU or device.
Identity is never read from track ids or object names; the scorer compares decisions with truth only
after they are produced.
"""
from __future__ import annotations

import pytest

from echo_watch.geometry import crop, iou
from echo_watch.item_policy import ItemTimeline, MemoryPolicy, MemoryParams, Sample, WatchedItem
from echo_watch.matching import reference_appearance
from echo_watch.fixtures.synthetic_scene import SyntheticScene

REGION = (0.0, 300.0, 1920.0, 600.0)


def scene(identical_phones=True):
    sc = SyntheticScene(size=(1920, 1080))
    if identical_phones:
        sc.item("phone_b").hue_x = sc.item("phone_a").box[0]
    return sc


def watched(sc, names, seeds=()):
    ref = sc.render(0.0)
    out = []
    for i, n in enumerate(names):
        b = sc.item_box_px(n)
        out.append(WatchedItem(n.upper(), (i,), reference_appearance(crop(ref, b)), (b[2], b[3]), REGION,
                               seed_box=tuple(b) if n in seeds else None))
    return out


def dets(sc, visible: dict[str, int], conf=0.9):
    """visible: scene object name -> class index the prompt model reports for it."""
    return [{"box": sc.item_box_px(n), "cls": c, "confidence": conf} for n, c in visible.items()]


def run(pol, sc, t, visible, assess=None):
    return pol.step(t, sc.render(0.0), dets(sc, visible), assess)


def test_distinct_items_confirm_one_to_one_after_three_supports():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["speaker", "tablet", "watch_box"]))
    statuses = []
    for k in range(4):
        r = run(pol, sc, k * 0.5, {"speaker": 0, "tablet": 1, "watch_box": 2})
        statuses.append({i: d.status for i, d in r.items()})
    assert statuses[0] == {"SPEAKER": "checking", "TABLET": "checking", "WATCH_BOX": "checking"}
    assert statuses[2] == {"SPEAKER": "visible", "TABLET": "visible", "WATCH_BOX": "visible"}  # 3 supports, 1 s
    boxes = {i: d.box for i, d in r.items()}
    assert iou(boxes["SPEAKER"], sc.item_box_px("speaker")) > 0.9
    assert len({d.track_id for d in r.values()}) == 3   # one observation never belongs to two items


def test_duplicate_hits_on_one_object_are_combined_not_uncertain():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["speaker", "tablet"]))
    for k in range(4):
        b = sc.item_box_px("speaker")
        d = dets(sc, {"tablet": 1}) + [{"box": b, "cls": 0, "confidence": 0.9},
                                       {"box": [b[0] + 3, b[1] + 2, b[2] - 4, b[3] - 3], "cls": 1, "confidence": 0.4}]
        r = pol.step(k * 0.5, sc.render(0.0), d)
    assert r["SPEAKER"].status == "visible" and r["TABLET"].status == "visible"
    assert r["SPEAKER"].track_id != r["TABLET"].track_id


def _establish_phone_a_with_distractor(pol, sc, t0=0.0, n=6):
    r = None
    for k in range(n):  # A (remembered, operator-selected) and identical unremembered B, co-visible
        r = run(pol, sc, t0 + k * 0.5, {"phone_a": 0, "phone_b": 0})
    return r


def test_identical_unremembered_distractor_is_excluded_while_covisible_and_never_takes_identity():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["phone_a"], seeds=("phone_a",)))
    r = _establish_phone_a_with_distractor(pol, sc)
    a = r["PHONE_A"]
    assert a.status == "visible" and iou(a.box, sc.item_box_px("phone_a")) > 0.9
    assert pol.items["PHONE_A"].ambiguity is not None           # persisted reference-basis ambiguity
    facts = pol.drain_facts()
    assert any(f["kind"] == "covisible_distinct" for f in facts)
    # A removed, B stays: B is excluded (seen together with A) -> Not seen by ECHO, not B's location
    sc.item("phone_a").present = False
    for k in range(6):
        r = run(pol, sc, 3.0 + k * 0.5, {"phone_b": 0})
        assert r["PHONE_A"].status != "visible"
    assert r["PHONE_A"].status == "not_seen", r["PHONE_A"]
    assert any(x["why"].startswith("excluded") for x in r["PHONE_A"].alternatives)
    # A returns while B is still continuously observed (live exclusion = distinguishing evidence)
    sc.item("phone_a").present = True
    seen = []
    for k in range(5):
        r = run(pol, sc, 6.0 + k * 0.5, {"phone_a": 0, "phone_b": 0})
        seen.append(r["PHONE_A"].status)
    assert seen[-1] == "visible" and iou(r["PHONE_A"].box, sc.item_box_px("phone_a")) > 0.9


def test_blind_swap_of_identical_objects_stays_unresolved():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["phone_a"], seeds=("phone_a",)))
    _establish_phone_a_with_distractor(pol, sc)
    # cover both for 3 s (no observations: tracks expire), swap them blindly, reveal
    for k in range(6):
        r = run(pol, sc, 3.0 + k * 0.5, {})
    a, b = sc.item("phone_a"), sc.item("phone_b")
    a.dx, b.dx = b.box[0] - a.box[0], a.box[0] - b.box[0]
    for k in range(20):
        r = run(pol, sc, 6.0 + k * 0.5, {"phone_a": 0, "phone_b": 0})
        assert r["PHONE_A"].status != "visible", f"identity claimed after a blind swap at step {k}"
    assert r["PHONE_A"].status == "uncertain"
    # a restart (new policy instance) with the persisted ambiguity record cannot resolve it either
    items = watched(scene(), ["phone_a"])
    items[0].ambiguity = pol.items["PHONE_A"].ambiguity
    pol2 = MemoryPolicy(items)
    for k in range(10):
        r = run(pol2, sc, 20.0 + k * 0.5, {"phone_a": 0, "phone_b": 0})
        assert r["PHONE_A"].status != "visible"


def test_two_remembered_identical_items_need_attribution_and_blind_swap_is_uncertain():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["phone_a", "phone_b"], seeds=("phone_a", "phone_b")))
    for k in range(6):
        r = run(pol, sc, k * 0.5, {"phone_a": 0, "phone_b": 1})
    assert r["PHONE_A"].status == "visible" and r["PHONE_B"].status == "visible"
    assert iou(r["PHONE_A"].box, sc.item_box_px("phone_a")) > 0.9
    assert iou(r["PHONE_B"].box, sc.item_box_px("phone_b")) > 0.9
    for k in range(6):
        run(pol, sc, 3.0 + k * 0.5, {})
    a, b = sc.item("phone_a"), sc.item("phone_b")
    a.dx, b.dx = b.box[0] - a.box[0], a.box[0] - b.box[0]
    for k in range(20):
        r = run(pol, sc, 6.0 + k * 0.5, {"phone_a": 0, "phone_b": 1})
        assert r["PHONE_A"].status != "visible" and r["PHONE_B"].status != "visible"
    assert {r["PHONE_A"].status, r["PHONE_B"].status} == {"uncertain"}


def test_without_attribution_identical_remembered_items_never_confirm():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["phone_a", "phone_b"]))
    for k in range(10):
        r = run(pol, sc, k * 0.5, {"phone_a": 0, "phone_b": 1})
        assert r["PHONE_A"].status != "visible" and r["PHONE_B"].status != "visible"


def test_fragmented_distinct_track_recovers_only_by_fresh_qualifying_appearance():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["speaker", "tablet"]))
    for k in range(4):
        run(pol, sc, k * 0.5, {"speaker": 0, "tablet": 1})
    sc.item("speaker").present = False
    for k in range(6):
        r = run(pol, sc, 2.0 + k * 0.5, {"tablet": 1})
    assert r["SPEAKER"].status == "not_seen"
    sc.item("speaker").present = True
    # EW-LAB-01: the ported dx=120 moved the speaker underneath the watch box (partly occluded; descriptor v1 score
    # 0.351, admitted only by the former 0.35 floor). dx=-60 keeps the same move-and-rotate check without occlusion.
    sc.item("speaker").dx, sc.item("speaker").angle = -60.0, 15.0     # moved while unseen
    seq = []
    for k in range(5):
        r = run(pol, sc, 5.0 + k * 0.5, {"speaker": 0, "tablet": 1})
        seq.append(r["SPEAKER"].status)
    assert seq[:2] == ["checking", "checking"] and seq[2] == "visible"
    assert iou(r["SPEAKER"].box, sc.item_box_px("speaker")) > 0.9


def test_continuous_movement_keeps_identity():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["speaker", "tablet", "watch_box"]))
    for k in range(4):
        run(pol, sc, k * 0.5, {"speaker": 0, "tablet": 1, "watch_box": 2})
    it = sc.item("watch_box")
    for k in range(8):  # 2 Hz samples, 48 px/s, 192 px total displacement, rotating to +15 degrees
        it.dx, it.dy = 10.0 * (k + 1), 12.5 * (k + 1)
        it.angle = min(15.0, 2.0 * (k + 1))
        r = run(pol, sc, 2.0 + 0.5 * (k + 1), {"speaker": 0, "tablet": 1, "watch_box": 2})
        assert r["WATCH_BOX"].status == "visible", (k, r["WATCH_BOX"])
        assert iou(r["WATCH_BOX"].box, sc.item_box_px("watch_box")) > 0.8


def test_crossing_identical_objects_marks_chain_suspect():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["phone_a"], seeds=("phone_a",)))
    _establish_phone_a_with_distractor(pol, sc)
    a, b = sc.item("phone_a"), sc.item("phone_b")
    gap = (b.box[0] - a.box[0])
    for k in range(1, 9):  # converge at the midpoint, overlapping
        a.dx, b.dx = gap / 2 * k / 8, -gap / 2 * k / 8
        r = run(pol, sc, 3.0 + k * 0.25, {"phone_a": 0, "phone_b": 0})
    for k in range(1, 9):  # separate again (which is which is not knowable)
        a.dx, b.dx = gap / 2 - gap / 2 * k / 8, -gap / 2 + gap / 2 * k / 8
        r = run(pol, sc, 5.0 + k * 0.25, {"phone_a": 0, "phone_b": 0})
    for k in range(8):
        r = run(pol, sc, 7.0 + k * 0.5, {"phone_a": 0, "phone_b": 0})
        assert r["PHONE_A"].status != "visible"


def test_operator_attribution_is_explicit_new_evidence():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["phone_a"], seeds=("phone_a",)))
    _establish_phone_a_with_distractor(pol, sc)
    for k in range(6):
        run(pol, sc, 3.0 + k * 0.5, {})
    r = run(pol, sc, 6.0, {"phone_a": 0, "phone_b": 0})
    assert r["PHONE_A"].status == "uncertain"
    target = next(x for x in r["PHONE_A"].alternatives if iou(x["box"], sc.item_box_px("phone_a")) > 0.9)
    assert pol.attribute(target["track_id"], "PHONE_A")
    r = run(pol, sc, 6.5, {"phone_a": 0, "phone_b": 0})
    assert r["PHONE_A"].status == "visible" and iou(r["PHONE_A"].box, sc.item_box_px("phone_a")) > 0.9


def test_cannot_assess_when_view_unassessable_and_bounds_never_force_matches():
    sc = scene()
    pol = MemoryPolicy(watched(sc, ["speaker"]))
    for k in range(4):
        run(pol, sc, k * 0.5, {"speaker": 0})
    sc.item("speaker").present = False
    r = run(pol, sc, 2.0, {}, {"SPEAKER": (False, "view obstructed")})
    assert r["SPEAKER"].status == "cannot_assess"
    pol2 = MemoryPolicy(watched(sc, ["speaker"]), MemoryParams(max_obs=2))
    many = [{"box": [100 + 60 * k, 500, 40, 40], "cls": 0, "confidence": 0.9} for k in range(3)]
    r = pol2.step(0.0, sc.render(0.0), many)
    assert r["SPEAKER"].status == "cannot_assess" and "bound" in r["SPEAKER"].reason


# ---- review timer --------------------------------------------------------------------------------
def feed(tl, t0, t1, cat, epoch="e1", dt=0.5, wall0=1000.0):
    ev, t = [], t0
    while t < t1 - 1e-9:
        ev += tl.sample(Sample(t, wall0 + t, epoch, cat, int(t * 2)))
        t += dt
    return ev


def confirmed(allowance=60.0):
    tl = ItemTimeline("i", allowance)
    tl.sample(Sample(0.0, 1000.0, "e1", "visible", 0, supports=[0.0, 0.5, 1.0]))
    return tl


def test_gap_with_outage_charges_only_eligible_time_and_requests_one_review():
    tl = confirmed()
    ev = feed(tl, 1.0, 41.0, "not_seen")            # 40 s eligible gap
    assert [e["kind"] for e in ev] == ["gap_started"]
    ev = feed(tl, 41.0, 71.0, "offline")            # 30 s source outage (samples say offline)
    assert not ev and abs(tl.gap["charged_s"] - 39.5) < 1e-6
    ev = feed(tl, 71.0, 92.0, "not_seen")           # 20 s more
    kinds = [e["kind"] for e in ev]
    assert kinds.count("review_requested") == 1
    g = tl.gap
    assert g["charged_s"] >= 60.0 and g["review"]["interrupted"] is True
    assert abs(g["unknown_s"] - 29.5) < 1e-6 and abs(g["boundary_s"] - 1.0) < 1e-6
    assert g["review"]["requested_wall"] - g["onset_wall"] > 60   # wall elapsed separate from charged
    ev = feed(tl, 92.0, 120.0, "not_seen")
    assert not any(e["kind"] == "review_requested" for e in ev)    # one continuing episode, one review


def test_below_allowance_return_does_not_request_review_and_recovers_on_three_supports():
    tl = confirmed(10.0)
    feed(tl, 1.0, 9.0, "not_seen")
    ev = feed(tl, 9.0, 10.0, "visible")
    assert not any(e["kind"] == "recovered" for e in ev)     # only 2 supports in 0.5 s
    ev = feed(tl, 10.0, 10.5, "visible")
    assert [e["kind"] for e in ev] == ["recovered"] and ev[0]["gap"]["review"] is None


def test_uncertainty_pause_and_new_epoch_freeze_but_never_charge_or_recover():
    tl = confirmed(10.0)
    feed(tl, 1.0, 5.0, "not_seen")
    c0 = tl.gap["charged_s"]
    feed(tl, 5.0, 30.0, "uncertain")
    feed(tl, 30.0, 40.0, "paused")
    assert tl.gap["charged_s"] == c0 and tl.gap is not None
    ev = feed(tl, 0.0, 1.0, "not_seen", epoch="e2", wall0=2000.0)   # fresh epoch: prior clock never charged
    assert tl.gap["charged_s"] == c0 + 0.5 and tl.gap["unobserved_wall_s"] > 0
    assert not ev


def test_first_ever_not_seen_is_setup_not_a_disappearance():
    tl = ItemTimeline("i", 10.0)
    ev = feed(tl, 0.0, 30.0, "not_seen")
    assert ev == [] and tl.gap is None and tl.needs_setup


def test_basis_change_segments_episode_and_requires_fresh_setup():
    tl = confirmed(10.0)
    feed(tl, 1.0, 20.0, "not_seen")
    assert tl.gap["review"] is not None
    ev = tl.basis_changed("reference changed to version 2", 2000.0)
    assert ev[0]["kind"] == "episode_segmented" and tl.gap is None and tl.needs_setup
    assert feed(tl, 30.0, 60.0, "not_seen") == []
