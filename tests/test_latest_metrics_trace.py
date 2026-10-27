# SPDX-License-Identifier: AGPL-3.0-only
"""Donor-derived primitives (latest slot, rolling rate) and the decision trace."""
from __future__ import annotations

import json
import threading
import time

import pytest

from echo_watch.engine.latest import LatestSlot
from echo_watch.engine.trace import STATE_FAILED, STATE_FINALIZED, STATE_PARTIAL, TraceWriter
from echo_watch.metrics import RollingRate, percentile, summarize
from support import work_dir


def test_latest_slot_keeps_only_the_newest_and_counts_replaced_unconsumed_items():
    s = LatestSlot()
    assert s.take() is None
    s.publish("a")
    s.publish("b")                      # 'a' was never taken: replaced and counted
    assert s.dropped == 1 and s.peek() == "b"
    assert s.take() == "b" and s.take() is None and s.peek() == "b"      # taken once; peek never consumes
    s.publish("c")
    assert s.dropped == 1               # 'b' had been taken, so it is not a drop
    s.clear()
    assert s.peek() is None


def test_latest_slot_wait_take_wakes_on_publish_and_on_close():
    s = LatestSlot()
    out = []
    t = threading.Thread(target=lambda: out.append(s.wait_take(5.0)))
    t.start()
    time.sleep(0.05)
    s.publish("x")
    t.join(2)
    assert out == ["x"]
    t0 = time.monotonic()
    t2 = threading.Thread(target=lambda: out.append(s.wait_take(5.0)))
    t2.start()
    time.sleep(0.05)
    s.close()
    t2.join(2)
    assert out[-1] is None and time.monotonic() - t0 < 2


def test_rolling_rate_is_zero_without_two_samples_and_decays_to_zero():
    clock = [0.0]
    r = RollingRate(window_seconds=2.0, clock=lambda: clock[0])
    assert r.rate() == 0.0
    r.record(0.0)
    assert r.rate() == 0.0                                    # one sample is never a fabricated rate
    for t in (0.5, 1.0, 1.5, 2.0):
        r.record(t)
    assert r.rate(now=2.0) == pytest.approx(2.0)
    assert r.rate(now=10.0) == 0.0                            # a genuinely stalled stream reads zero, not history


def test_percentile_and_summarize():
    assert percentile([], 95) is None and percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert summarize([1, 2, 3])["max"] == 3 and summarize([])["n"] == 0


def test_trace_finalizes_and_hashes_the_file():
    d = work_dir("trace_ok")
    tw = TraceWriter(d / "t.jsonl", 100)
    tw.start()
    for i in range(20):
        assert tw.emit("x", i=i)
    st = tw.finish(10)
    assert st["state"] == STATE_FINALIZED and st["records"] == 20 and st["dropped"] == 0 and st["sha256"]
    lines = [json.loads(x) for x in (d / "t.jsonl").read_text().splitlines()]
    assert [r["i"] for r in lines] == list(range(20)) and [r["n"] for r in lines] == list(range(1, 21))
    assert tw.emit("late") is False                           # nothing is accepted after finalization


def test_trace_overflow_is_counted_and_the_terminal_state_is_partial_never_complete():
    d = work_dir("trace_drop")
    tw = TraceWriter(d / "t.jsonl", 2)
    tw.start()
    tw._fh.flush()
    # stall the writer thread by holding its file lock-free path: flood faster than it can drain
    ok = sum(tw.emit("x", i=i, pad="y" * 2000) for i in range(4000))
    st = tw.finish(30)
    assert st["dropped"] == 4000 - ok and st["dropped"] > 0
    assert st["state"] == STATE_PARTIAL and st["records"] == ok
