# SPDX-License-Identifier: AGPL-3.0-only
"""Throughput must not change the configured attention times: confirmation, co-visible exclusion and recovery need
supports that SPAN real elapsed time, whatever the analysis rate; repeated results inside that span are not extra
confirmations. (Found by the foundation's scripted removal/return run: the reference retained evidence by COUNT.)"""
from __future__ import annotations

import pytest

from echo_watch.fixtures.synthetic_scene import SyntheticScene
from echo_watch.geometry import crop
from echo_watch.item_policy import ItemTimeline, MemoryPolicy, Sample, WatchedItem
from echo_watch.matching import reference_appearance

REGION = (0.0, 0.0, 1280.0, 720.0)


def setup(names=("speaker",), seeds=()):
    sc = SyntheticScene(size=(1280, 720))
    ref = sc.render(0.0)
    items = []
    for i, n in enumerate(names):
        b = sc.item_box_px(n)
        items.append(WatchedItem(n.upper(), (i,), reference_appearance(crop(ref, b)), (b[2], b[3]), REGION,
                                 seed_box=tuple(b) if n in seeds else None))
    return sc, items


@pytest.mark.parametrize("hz", [2, 5, 8, 15, 30, 60])
def test_confirmation_needs_one_second_of_elapsed_support_at_any_rate(hz):
    sc, items = setup()
    pol = MemoryPolicy(items, width=1280)
    img = sc.render(0.0)
    box = sc.item_box_px("speaker")
    first_visible = None
    n_to_visible = 0
    for k in range(int(hz * 4)):
        t = k / hz
        r = pol.step(t, img, [{"box": box, "cls": 0, "confidence": 0.9}])
        if r["SPEAKER"].status == "visible" and first_visible is None:
            first_visible, n_to_visible = t, k + 1
        if first_visible is None:
            assert r["SPEAKER"].status == "checking"
    assert first_visible is not None, f"never confirmed at {hz} Hz"
    assert 1.0 - 1e-9 <= first_visible <= 1.0 + 1.0 / hz + 1e-9         # not earlier than 1 s, and promptly at 1 s
    assert n_to_visible >= 3


@pytest.mark.parametrize("hz", [2, 8, 15, 30])
def test_a_burst_of_results_inside_one_second_never_confirms(hz):
    """Many results in 0.5 s are many looks at the same moment, not 1 s of independent support."""
    sc, items = setup()
    pol = MemoryPolicy(items, width=1280)
    img = sc.render(0.0)
    box = sc.item_box_px("speaker")
    for k in range(int(hz * 0.5)):
        r = pol.step(k / hz, img, [{"box": box, "cls": 0, "confidence": 0.9}])
        assert r["SPEAKER"].status != "visible"


@pytest.mark.parametrize("hz", [2, 8, 15, 30])
def test_recovery_and_covisible_exclusion_complete_at_high_rates(hz):
    tl = ItemTimeline("i", 10.0)
    tl.sample(Sample(0.0, 1000.0, "e", "visible", 0, supports=[0.0, 0.5, 1.0]))
    t = 0.2
    while t < 3.0:
        tl.sample(Sample(t, 1000 + t, "e", "not_seen", int(t * hz)))
        t += 1.0 / hz
    assert tl.gap is not None
    ev = []
    t0 = t
    for k in range(int(hz * 2.5)):
        t = t0 + k / hz
        ev += tl.sample(Sample(t, 1000 + t, "e", "visible", 1000 + k, supports=[t]))
    rec = [e for e in ev if e["kind"] == "recovered"]
    assert len(rec) == 1                                                  # recovers once, >= 1 s after the first return
    # co-visible distinct pair: established after >= 1 s together, at any rate
    sc, items = setup(("phone_a",), seeds=("phone_a",))
    sc.item("phone_b").hue_x = sc.item("phone_a").box[0]
    pol = MemoryPolicy(items, width=1280)
    img = sc.render(0.0)
    dets = [{"box": sc.item_box_px("phone_a"), "cls": 0, "confidence": 0.9}, {"box": sc.item_box_px("phone_b"), "cls": 0, "confidence": 0.9}]
    for k in range(int(hz * 3)):
        pol.step(k / hz, img, dets)
    assert any(f["kind"] == "covisible_distinct" for f in pol.drain_facts())
