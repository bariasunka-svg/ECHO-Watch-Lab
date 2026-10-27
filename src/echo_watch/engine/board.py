# SPDX-License-Identifier: AGPL-3.0-only
"""The item board: the COMMITTED item state of one run (list, profile header, intervals, activity, last-supported).

Only ``ItemBoard.apply_*`` mutates it and only the controller calls those, under its commit lock, for a packet whose
control basis is current (see engine/session.py). Everything a list/profile/trace/annotated view shows about an item is
read from here or from the committed ``ResultPacket`` it produced - there is no second source of item state.

Review timing and episodes use the adopted ``item_policy.ItemTimeline`` (the preserved ECHO reference's review
timer): after first confirmed visibility, two consecutive eligible Not-visible samples start a gap; inside an
unrecovered gap only eligible Not-visible time is charged; uncertain / cannot-assess / offline / paused time freezes the
counter (never resets, never recovers); three supported samples spanning >= 1 s recover. Unknown gaps are never filled
with assumed absence. State intervals follow the reference watch's interval rule (same category, same epoch, <= 2 s apart,
same track for Visible).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..frames import Frame, new_id
from ..item_policy import ItemTimeline, Sample
from .contracts import (BOXED_STATES, LABEL, STATE_CANNOT_ASSESS, STATE_OFFLINE, STATE_PAUSED,
                        STATE_STOPPED, STATE_VISIBLE, ItemObservation, ItemRef, ItemRow)
from .evaluator import Evaluation

EVENT_LABEL = {
    "first_confirmed": "First confirmed live sighting",
    "gap_started": "Not visible (two consecutive eligible checks)",
    "recovered": "Visible again (three supported checks over at least 1 s)",
    "review_requested": "Review requested after the allowance of usable Not-visible checks. This is a review request, "
                        "not proof that the item was removed.",
    "episode_segmented": "Episode segmented (monitoring basis changed)",
}


@dataclass
class _ItemState:
    ref: ItemRef
    timeline: ItemTimeline
    obs: ItemObservation | None = None
    last_supported: dict | None = None
    interval: dict | None = None
    t_mono: float | None = None         # capture time of the packet behind ``obs``
    packet_id: str | None = None


class ItemBoard:
    def __init__(self, run_id: str, refs: list[ItemRef], allowance_s: float = 60.0, interval_cap: int = 20000):
        self.run_id = run_id
        self.items: dict[str, _ItemState] = {
            r.item_id: _ItemState(r, ItemTimeline(r.item_id, allowance_s)) for r in refs}
        self.intervals: list[dict] = []          # closed intervals (bounded)
        self.events: list[dict] = []             # activity events (bounded)
        self.interval_cap = interval_cap
        self.dropped_intervals = 0

    # ---- helpers ------------------------------------------------------------------------------------------
    def confirmed_ever(self, item_id: str) -> bool:
        return self.items[item_id].timeline.confirmed_ever

    def _event(self, item_id: str, ev: dict, s: Sample, packet_id: str | None) -> dict:
        k = ev["kind"]
        rec = {"event_id": new_id("evt"), "item_id": item_id, "kind": k, "label": EVENT_LABEL.get(k, k),
               "wall": s.t_wall, "t_mono": s.t_mono, "frame": [s.epoch, s.frame_seq], "packet_id": packet_id}
        g = ev.get("gap")
        if g:
            rec["gap"] = {x: (round(y, 2) if isinstance(y, float) else y) for x, y in g.items()
                          if x in ("charged_s", "unknown_s", "uncertain_s", "paused_s", "allowance_s", "interruptions",
                                   "onset_wall", "review", "recovered_wall")}
        if len(self.events) < 100000:
            self.events.append(rec)
        return rec

    def _interval(self, st: _ItemState, s: Sample, track: str | None, reason: str | None) -> None:
        iv = st.interval
        same = (iv is not None and iv["state"] == s.category and iv["epoch"] == s.epoch
                and 0 <= s.t_mono - iv["end_mono"] <= 2.0 and (s.category != STATE_VISIBLE or iv["track_id"] == track))
        if same:
            iv.update(end_mono=s.t_mono, end_wall=s.t_wall, end_ident=[s.epoch, s.frame_seq], samples=iv["samples"] + 1)
            if s.box is not None:
                iv["box"] = [round(v, 1) for v in s.box]
            return
        self._close(st)
        st.interval = {"interval_id": new_id("iv"), "run_id": self.run_id, "item_id": st.ref.item_id,
                       "state": s.category, "epoch": s.epoch, "start_mono": s.t_mono, "end_mono": s.t_mono,
                       "start_wall": s.t_wall, "end_wall": s.t_wall, "start_ident": [s.epoch, s.frame_seq],
                       "end_ident": [s.epoch, s.frame_seq], "samples": 1,
                       "box": [round(v, 1) for v in s.box] if s.box else None, "track_id": track, "reason": reason}

    def _close(self, st: _ItemState) -> None:
        iv = st.interval
        if iv is None:
            return
        iv["closed"] = 1
        if len(self.intervals) < self.interval_cap:
            self.intervals.append(iv)
        else:
            self.dropped_intervals += 1
        st.interval = None

    # ---- commit: an analysed frame ------------------------------------------------------------------------------
    def apply_evaluation(self, frame: Frame, ev: Evaluation, packet_id: str) -> tuple[ItemObservation, ...]:
        out = []
        for item_id, st in self.items.items():
            d = ev.decisions[item_id]
            cat, why = d.status, d.reason
            s = Sample(frame.t_mono, frame.t_wall, frame.epoch.epoch_id, cat, frame.seq, d.box, d.supports, why)
            tevents = st.timeline.sample(s)
            recs = tuple(self._event(item_id, e, s, packet_id) for e in tevents)
            self._interval(st, s, d.track_id, why)
            if cat == STATE_VISIBLE and d.box is not None:
                st.last_supported = {"ident": [frame.epoch.epoch_id, frame.seq], "t_mono": frame.t_mono,
                                     "t_wall": frame.t_wall, "box": [round(v, 1) for v in d.box],
                                     "size": [frame.width, frame.height], "packet_id": packet_id}
            neg = ev.negative.get(item_id, (None, None))
            gap = None
            g = st.timeline.gap
            if g is not None:
                gap = {x: (round(y, 2) if isinstance(y, float) else y) for x, y in g.items()
                       if x in ("charged_s", "unknown_s", "uncertain_s", "paused_s", "allowance_s", "interruptions",
                                "onset_wall", "review")}
            box = tuple(d.box) if (cat in BOXED_STATES and d.box is not None) else None
            obs = ItemObservation(item_id, st.ref.name, st.ref.ref_version, cat, why, box, d.track_id, d.score, d.app,
                                  d.conf, tuple(d.supports), d.confirmed_now, neg[0], neg[1],
                                  tuple(d.alternatives), st.last_supported, gap, recs, d.admission)
            st.obs, st.t_mono, st.packet_id = obs, frame.t_mono, packet_id
            out.append(obs)
        return tuple(out)

    # ---- commit: the session is not observing (paused / offline / stopped) ------------------------------------------
    def apply_not_observing(self, state: str, reason: str, t_mono: float, t_wall: float, epoch_id: str,
                            packet_id: str | None = None) -> tuple[ItemObservation, ...]:
        """Paused / offline are UNKNOWN time (frozen review counters), never absence. 'stopped' ends the live run."""
        out = []
        for item_id, st in self.items.items():
            recs = ()
            if state in (STATE_PAUSED, STATE_OFFLINE):
                s = Sample(t_mono, t_wall, epoch_id, state, None, None, [], reason)
                recs = tuple(self._event(item_id, e, s, packet_id) for e in st.timeline.sample(s))
                self._interval(st, s, None, reason)
            else:
                self._close(st)
            gap = None
            if st.timeline.gap is not None:
                gap = {x: (round(y, 2) if isinstance(y, float) else y) for x, y in st.timeline.gap.items()
                       if x in ("charged_s", "unknown_s", "uncertain_s", "paused_s", "allowance_s", "interruptions",
                                "onset_wall", "review")}
            st.obs = ItemObservation(item_id, st.ref.name, st.ref.ref_version, state, reason, None, None,
                                     last_supported=st.last_supported, gap=gap, events=recs)
            st.t_mono, st.packet_id = t_mono, packet_id
            out.append(st.obs)
        return tuple(out)

    def apply_resumed(self, reason: str, t_mono: float, t_wall: float, epoch_id: str) -> None:
        """After Resume nothing is known until a post-boundary frame is checked: Cannot assess, not absence."""
        for item_id, st in self.items.items():
            s = Sample(t_mono, t_wall, epoch_id, STATE_CANNOT_ASSESS, None, None, [], reason)
            st.timeline.sample(s)
            self._interval(st, s, None, reason)
            st.obs = ItemObservation(item_id, st.ref.name, st.ref.ref_version, STATE_CANNOT_ASSESS, reason, None, None,
                                     last_supported=st.last_supported)
            st.t_mono = t_mono

    def basis_changed(self, why: str, wall: float) -> None:
        for st in self.items.values():
            st.timeline.basis_changed(why, wall)

    def finish(self) -> None:
        for st in self.items.values():
            self._close(st)

    # ---- reads ---------------------------------------------------------------------------------------------------
    def rows(self, now_mono: float, live: bool, stale_after_s: float) -> tuple[ItemRow, ...]:
        rows = []
        for item_id, st in self.items.items():
            o = st.obs
            if o is None:
                rows.append(ItemRow(item_id, st.ref.name, st.ref.ref_version, STATE_CANNOT_ASSESS, LABEL[STATE_CANNOT_ASSESS],
                                    "no result yet", None, None, None, None, None, False))
                continue
            age = None if st.t_mono is None else now_mono - st.t_mono
            fresh = live and age is not None and age <= stale_after_s and o.state not in (STATE_PAUSED, STATE_OFFLINE, STATE_STOPPED)
            rows.append(ItemRow(item_id, o.name, o.ref_version, o.state, o.label, o.reason,
                                o.box if fresh else None, st.last_supported, o.gap, st.packet_id, age, fresh))
        return tuple(rows)

    def summary(self) -> dict:
        return {i: {"state": st.obs.state if st.obs else None, "confirmed_ever": st.timeline.confirmed_ever,
                    "last_supported": st.last_supported, "gap": st.timeline.gap} for i, st in self.items.items()}
