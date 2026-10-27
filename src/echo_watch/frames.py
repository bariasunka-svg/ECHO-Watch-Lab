# SPDX-License-Identifier: AGPL-3.0-only
"""Frame identity and source-epoch clock mapping.

Adapted from the preserved ECHO Watch reference runtime/echowatch/frames.py (SHA-256
dd04645737185422fe35fd741e292a7cf66e51a9f5d43c65446a556aa440138a; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md) and shaped by the
frozen PatronSight ``FrameRead`` record (src/patronsight/vision/video_source.py, SHA-256
8dc0066e849525936b3c0fd2472e8fb8244953bb14fea80d77670d5f0ca4c90a: an immutable record carrying the frame index,
timestamp and image). Adaptation: image arrays are marked read-only so a published frame can never be mutated by a
consumer, and ``Frame.ident`` gives the (epoch, sequence) identity used by every result packet.

A source epoch begins whenever a source (re)starts. Within an epoch frames have a sequence number and a monotonic
ingress time; elapsed-time rules only compare monotonic times from the same epoch. Wall time is retained for
display/audit only.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

import numpy as np


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(frozen=True)
class Epoch:
    epoch_id: str
    source_id: str
    epoch_no: int
    started_mono: float
    started_wall: float
    # Input configuration version of the source when this epoch began, and the capture generation that produced it.
    config_version: int = 0
    generation: int = 0

    @staticmethod
    def begin(source_id: str, epoch_no: int, config_version: int = 0, generation: int = 0) -> "Epoch":
        return Epoch(new_id("ep"), source_id, epoch_no, time.monotonic(), time.time(), config_version, generation)


@dataclass(frozen=True)
class Frame:
    epoch: Epoch
    seq: int
    t_mono: float                 # ingress (capture) time on the monotonic clock
    t_wall: float
    image: np.ndarray = field(repr=False, compare=False)
    synthetic: bool = False
    # decode side channel (FFmpeg output ordinal/PTS; for files the original-file frame, replay loop and media
    # position). None when unavailable; never inferred from ingress time.
    media: dict | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        try:
            self.image.setflags(write=False)
        except ValueError:      # a non-owning view that cannot be locked; callers copy before mutating
            pass

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def ident(self) -> tuple[str, int]:
        return (self.epoch.epoch_id, self.seq)

    @property
    def ref(self) -> str:
        return f"{self.epoch.epoch_id}:{self.seq}"

    def age(self, now_mono: float | None = None) -> float:
        return (time.monotonic() if now_mono is None else now_mono) - self.t_mono
