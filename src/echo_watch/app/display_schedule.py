# SPDX-License-Identifier: AGPL-3.0-only
"""Latest-completed paired display scheduling (Tk-free; EW-LAB-02 repair A).

Defect (independent review P1): the monitor submitted the newly arrived packet and then rejected every completed
preparation whose token was not the newest SUBMITTED one; when a newer packet arrived on every tick, every completed
image was discarded and the monitor never advanced.

Policy here:
* VALIDITY is a generation, not "the newest submission": monitor packets are valid for (run id, control basis) of the
  current session; setup images for (source generation, setup frame). Arrival of a newer frame never invalidates a
  completed, still-current preparation. Start/Pause/Resume/Stop/source change create a new generation, so pre-cutover
  results are rejected (result numbers restart per run, so packet ids alone are not identity).
* Each tick drains the ready preparations, keeps the newest eligible one (highest display order within the current
  generation; never regressing below what is already painted) and paints AT MOST that one, then coalesces the newest
  wanted preparation into the single-slot worker. The painted image always carries its own packet (pixels, boxes,
  observations, capture time) and its own prepared scale/offset.
* Bounded: one pending + one running preparation in the worker (LatestWorker) and only the completions of one tick
  are examined; nothing sleeps or prepares on the Tk thread.
* Measurable: every discarded completion is counted by reason; distinct painted packets are counted by the view.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PrepRequest:
    key: tuple                  # (gen, order, max_w, max_h): what was asked for
    gen: tuple                  # validity generation
    order: int                  # display order inside the generation (commit ordinal; 0 for a setup image)
    packet: object | None       # the immutable packet whose OWN pixels these are (None for a setup image)
    image: np.ndarray = field(repr=False, compare=False)
    max_w: int = 0
    max_h: int = 0


def packet_gen(run_id: str | None, basis_id: str | None) -> tuple:
    return ("packet", run_id, basis_id)


def setup_gen(source_generation: int, frame_identity: int) -> tuple:
    return ("setup", source_generation, frame_identity)


def packet_request(packet, max_w: int, max_h: int) -> PrepRequest:
    gen = packet_gen(packet.run_id, packet.basis.basis_id)
    return PrepRequest((gen, packet.seq, max_w, max_h), gen, int(packet.seq), packet, packet.image, max_w, max_h)


def setup_request(source_generation: int, frame, max_w: int, max_h: int) -> PrepRequest:
    gen = setup_gen(source_generation, id(frame))
    return PrepRequest((gen, 0, max_w, max_h), gen, 0, None, frame.image, max_w, max_h)


class DisplayScheduler:
    """Decides which completed preparation to paint and which request to coalesce next. ``worker`` needs
    ``submit(token, request)`` and its completions are passed to :meth:`tick` (each with ``key``, ``gen``, ``order``)."""

    def __init__(self, worker):
        self.worker = worker
        self.token = 0
        self.submitted_key: tuple | None = None
        self.painted_gen: tuple | None = None
        self.painted_order: int | None = None
        self.painted_key: tuple | None = None
        self.dropped: dict[str, int] = {}
        self.submitted = 0

    def _drop(self, why: str) -> None:
        self.dropped[why] = self.dropped.get(why, 0) + 1

    def eligible(self, res, current_gen: tuple | None) -> str | None:
        """None when ``res`` may be painted now, else the reason it may not."""
        if isinstance(res, Exception) or res is None:
            return "preparation_failed"
        if current_gen is None or res.gen != current_gen:
            return "outside_current_generation"
        if self.painted_gen == current_gen and self.painted_order is not None:
            if res.order < self.painted_order:
                return "older_than_painted"
            if res.order == self.painted_order and res.key == self.painted_key:
                return "already_painted"
        return None

    def tick(self, want: PrepRequest | None, current_gen: tuple | None, completions) -> object | None:
        chosen = None
        for _tok, res in completions:
            why = self.eligible(res, current_gen)
            if why is not None:
                self._drop(why)
                continue
            if chosen is not None:
                if res.order < chosen.order:
                    self._drop("superseded_by_newer_completion")
                    continue
                self._drop("superseded_by_newer_completion")
            chosen = res
        if chosen is not None:
            self.painted_gen, self.painted_order, self.painted_key = chosen.gen, chosen.order, chosen.key
        if want is not None and want.key != self.submitted_key and want.key != self.painted_key:
            self.token += 1
            self.submitted += 1
            self.worker.submit(self.token, want)
            self.submitted_key = want.key
        return chosen

    def forget_painted(self) -> None:
        """The canvas was cleared (e.g. a new source): the next valid completion may paint whatever its order."""
        self.painted_gen = self.painted_order = self.painted_key = None


def live_authority(packet, run_id: str | None, basis_id: str | None, phase: str, now_mono: float,
                   stale_s: float = 1.0) -> bool:
    """May this painted packet show LIVE boxes now? Only for the current run and control basis, while monitoring is
    active, and only while its capture is at most ``stale_s`` old (expired images never silently look live)."""
    return (packet is not None and packet.run_id == run_id and packet.basis.basis_id == basis_id and phase == "active"
            and now_mono - packet.frame_t_mono <= stale_s)
