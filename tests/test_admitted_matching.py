# SPDX-License-Identifier: AGPL-3.0-only
"""EW-LAB-01 section 5: remembered-item acceptance is admitted evidence, not repeated weak proposals.

Synthetic pixels with scripted detections (no model). The Native S2 failure: the tablet, labelled with phone_a's prompt
class, was confirmed as phone_a at appearance 0.398 because a unique candidate above the 0.35 floor accumulated elapsed
supports. Here the same pattern is replayed under explicit gates.
"""
from __future__ import annotations

from echo_watch import matching
from echo_watch.fixtures.synthetic_scene import SyntheticScene
from echo_watch.geometry import crop
from echo_watch.item_policy import MemoryPolicy, WatchedItem
from echo_watch.matching import AdmissionGate, reference_appearance

REGION = (0.0, 0.0, 1280.0, 720.0)


def phone_policy(gate: AdmissionGate, seed=False):
    sc = SyntheticScene(size=(1280, 720))
    ref = sc.render(0.0)
    b = sc.item_box_px("phone_a")
    it = WatchedItem("PHONE_A", (0,), reference_appearance(crop(ref, b)), (b[2], b[3]), REGION,
                     seed_box=tuple(b) if seed else None)
    return sc, MemoryPolicy([it], gate=gate)


def det(sc, name, cls=0, conf=0.47):
    return {"box": sc.item_box_px(name), "cls": cls, "confidence": conf}


def test_tablet_labelled_as_phone_never_confirms_under_a_separating_gate():
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, 0.80, 0.45, "t", True)
    sc, pol = phone_policy(gate)
    sc.item("phone_a").present = False
    img = sc.render(0.0)
    states = [pol.step(k / 15.0, img, [det(sc, "tablet")])["PHONE_A"].status for k in range(150)]   # 10 s at 15 Hz
    assert "visible" not in states and "checking" not in states
    assert set(states) == {"not_seen"}                  # the tablet is rejected: phone_a's absence is assessable


def test_weak_unresolved_evidence_is_uncertainty_never_identity_or_absence():
    sc0 = SyntheticScene(size=(1280, 720))
    ref = sc0.render(0.0)
    tablet_score = matching.appearance(ref, sc0.item_box_px("tablet"),
                                       reference_appearance(crop(ref, sc0.item_box_px("phone_a"))),
                                       tuple(sc0.item_box_px("phone_a")[2:]))
    # a gate whose middle band contains the tablet's score: its evidence is UNRESOLVED
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, tablet_score + 0.02, tablet_score - 0.02, "t", False)
    assert gate.classify(tablet_score) == matching.UNRESOLVED
    sc, pol = phone_policy(gate)
    sc.item("phone_a").present = False
    img = sc.render(0.0)
    out = [pol.step(k / 15.0, img, [det(sc, "tablet")])["PHONE_A"] for k in range(150)]
    assert {d.status for d in out} == {"uncertain"}
    assert all(d.box is None for d in out)               # an alternative never becomes the item's live box
    assert "weak" in out[-1].reason


def test_operator_selection_does_not_bypass_admission():
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, 0.80, 0.45, "t", True)
    sc, pol = phone_policy(gate, seed=False)
    # seed the operator's selection ON the tablet's position while phone_a is absent: still rejected evidence
    pol.items["PHONE_A"].seed_box = tuple(sc.item_box_px("tablet"))
    sc.item("phone_a").present = False
    img = sc.render(0.0)
    states = [pol.step(k / 15.0, img, [det(sc, "tablet")])["PHONE_A"].status for k in range(60)]
    assert "visible" not in states and "checking" not in states


def test_owned_chain_ends_when_appearance_is_rejected_and_unresolved_hold_is_bounded():
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, 0.80, 0.45, "t", True)
    sc, pol = phone_policy(gate, seed=True)
    img = sc.render(0.0)
    for k in range(20):
        d = pol.step(k / 15.0, img, [det(sc, "phone_a", conf=0.9)])["PHONE_A"]
    assert d.status == "visible" and d.admission == matching.SUPPORT
    # the same track's appearance becomes unresolved: held at most hold_unresolved_s (1 s), then the chain ends
    weak = AdmissionGate(matching.DESCRIPTOR_VERSION, 2.0, 0.0, "t", False)      # everything below 0.975: unresolved
    pol.gate = weak
    seq = [pol.step(20 / 15.0 + k / 15.0, img, [det(sc, "phone_a", conf=0.9)])["PHONE_A"].status for k in range(30)]
    assert seq[0] == "visible" and "uncertain" in seq
    assert seq.index("uncertain") <= 16                  # <= 1 s (15 Hz) after the last strong frame
    assert "visible" not in seq[seq.index("uncertain"):]


def test_policy_clone_isolates_mutable_state():
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, 0.80, 0.45, "t", True)
    sc, pol = phone_policy(gate, seed=True)
    img = sc.render(0.0)
    pol.step(0.0, img, [det(sc, "phone_a", conf=0.9)])
    snap = {k: (t.pending, t.owner) for k, t in pol.tracks.items()}
    c = pol.clone()
    for k in range(1, 20):
        c.step(k / 15.0, img, [det(sc, "phone_a", conf=0.9)])
    assert {k: (t.pending, t.owner) for k, t in pol.tracks.items()} == snap      # the original never saw those steps
    assert c.owned("PHONE_A") is not None and pol.owned("PHONE_A") is None


def test_low_confidence_resembling_proposal_blocks_absence_but_never_creates_identity():
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, 0.80, 0.45, "t", True)
    sc, pol = phone_policy(gate)
    img = sc.render(0.0)
    out = [pol.step(k / 15.0, img, [det(sc, "phone_a", conf=0.15)])["PHONE_A"] for k in range(45)]
    assert {d.status for d in out} == {"uncertain"} and all(d.box is None for d in out)
    sc.item("phone_a").present = False
    img = sc.render(0.0)
    # a low-confidence REJECTED proposal (the tablet) does not block absence
    assert pol.step(3.0, img, [det(sc, "tablet", conf=0.15)])["PHONE_A"].status == "not_seen"


def test_unresolved_identical_twins_never_hand_identity_to_the_remaining_one():
    """Q3 controls failure (preserved): without a seed (e.g. after Resume) two identical phones fit phone_a equally; when
    one disappears the other must stay Match uncertain, never become phone_a."""
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, 0.80, 0.45, "t", True)
    sc, pol = phone_policy(gate, seed=False)
    sc.item("phone_b").hue_x = sc.item("phone_a").box[0]
    img = sc.render(0.0)
    for k in range(30):
        d = pol.step(k / 15.0, img, [det(sc, "phone_a", conf=0.5), det(sc, "phone_b", conf=0.5)])["PHONE_A"]
    assert d.status == "uncertain"
    sc.item("phone_a").present = False
    img = sc.render(0.0)
    states = [pol.step(2.0 + k / 15.0, img, [det(sc, "phone_b", conf=0.5)])["PHONE_A"].status for k in range(60)]
    assert "visible" not in states and "checking" not in states and states[-1] == "uncertain"


def test_track_ids_are_unique_across_policy_incarnations():
    gate = AdmissionGate(matching.DESCRIPTOR_VERSION, 0.80, 0.45, "t", True)
    sc, p1 = phone_policy(gate)
    _, p2 = phone_policy(gate)
    img = sc.render(0.0)
    p1.step(0.0, img, [det(sc, "phone_a")])
    p2.step(0.0, img, [det(sc, "phone_a")])
    assert set(p1.tracks).isdisjoint(p2.tracks)
