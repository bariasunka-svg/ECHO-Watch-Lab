# SPDX-License-Identifier: AGPL-3.0-only
"""Defects B and C (EW-LAB-01 section 4): the qualification scorers. The review's own counterexamples are replayed against
the preserved Native scorer (which wrongly PASSES them) and the Lab scorer (which must not)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "qualify"))
import evaluate  # noqa: E402


def _results(rate_hz: float, t_from: float, t_to: float, active: float):
    out, seq, t = [], 0, t_from
    while t < t_to - 1e-9:
        seq += 1
        out.append({"seq": seq, "frame": ["ep", seq], "frame_t_mono": active + t - 0.05,
                    "timing": {"commit_mono": active + t}, "media": {"media_n": seq - 1}, "flags": {}})
        t += 1.0 / rate_hz
    return out


def _manifest(active, end, kind="finalized", basis=None):
    return {"times": {"active_ack_mono": active, "stop_requested_mono": end, "requested_to_active_s": 3.0},
            "outcome": {"kind": kind, "ended_by": "operator"}, "basis_history": basis or []}


def test_tail_counterexample_fails_with_full_interval_rate():
    active = 100.0
    res = _results(10.0, 0.0, 10.0, active)                 # 10 Hz to 10 s while the session stays active to 20 s
    tp = evaluate.throughput(_manifest(active, active + 20.0), res, [active + x / 15 for x in range(300)])
    assert tp["status"] == "FAIL"
    assert abs(tp["mean_results_per_s"] - 3.4) < 0.11       # 51 results over the 15 s steady window
    assert tp["trailing_gap_s"] > 9.9 and tp["unhealthy_gaps"][-1]["kind"] == "processing_stall"


def test_failed_session_is_never_throughput_pass_despite_fast_prefix():
    active = 100.0
    res = _results(15.0, 0.0, 20.0, active)
    tp = evaluate.throughput(_manifest(active, active + 20.0, kind="stopped_after_failure"), res, [])
    assert tp["criteria"]["mean_ge_8"] and tp["status"] == "FAIL"


def test_deliberate_pause_is_separated_from_stalls():
    active = 100.0
    res = _results(15.0, 0.0, 8.0, active) + [dict(r, seq=r["seq"] + 1000) for r in _results(15.0, 10.0, 20.0, active)]
    basis = [{"reason": "operator pause", "t_mono": active + 8.0}, {"reason": "operator resume", "t_mono": active + 10.0}]
    tp = evaluate.throughput(_manifest(active, active + 20.0, basis=basis), res, [])
    assert tp["paused_s"] == 2.0 and not tp["unhealthy_gaps"] and tp["status"] == "PASS"


def test_missing_pairing_evidence_prevents_pass():
    sigs = [np.full(144, 100, np.uint8), np.full(144, 120, np.uint8)]
    sig0 = sigs[0].tobytes().hex()
    good = {"seq": 1, "frame": ["ep", 1], "media": {"media_n": 0}, "flags": {"image_sig": sig0}}
    missing = {"seq": 2, "frame": ["ep", 2], "media": {}, "flags": {}}
    assert evaluate.pairing([good, missing], sigs, None)["status"] == "FAIL"
    neg = {"seq": 3, "frame": ["ep", 3], "media": {"media_n": -1}, "flags": {"image_sig": sig0}}
    malformed = {"seq": 4, "frame": ["ep", 4], "media": {"media_n": 0}, "flags": {"image_sig": "abc"}}
    rep = evaluate.pairing([good, neg, malformed], sigs, None)
    assert rep["status"] == "FAIL" and rep["problems"]["missing_or_invalid_media_ordinal"] == 1
    assert rep["problems"]["missing_or_malformed_signature"] == 1
    # a clean frame-map without the result's identity is missing evidence
    rep = evaluate.pairing([good], sigs, {"entries": [{"ident": ["ep", 9], "sig": sig0}]})
    assert rep["status"] == "FAIL" and "no_clean_frame_map_entry" in rep["problems"]
    rep = evaluate.pairing([good], sigs, {"entries": [{"ident": ["ep", 1], "sig": sig0}]})
    assert rep["status"] == "PASS"
    assert evaluate.pairing([good], None, None, decode_error="boom")["status"] == "FAIL"


def test_native_v0_scorer_baseline_reproduces_the_reviewed_defects():
    """The preserved Native scorer PASSES both counterexamples (baseline kept, not overwritten)."""
    import evaluate_native_v0 as v0
    active = 100.0
    res = _results(10.0, 0.0, 10.0, active)
    run = {"manifest": {"times": {"active_ack_mono": active, "requested_to_active_s": 3.0}}, "results": res, "frames": []}
    assert v0.throughput(run)["status"] == "PASS"
