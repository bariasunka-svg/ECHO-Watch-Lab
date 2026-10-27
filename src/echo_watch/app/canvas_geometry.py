# SPDX-License-Identifier: AGPL-3.0-only
# ADOPTED UNCHANGED (body) from the frozen PatronSight reference src/patronsight/app/canvas_geometry.py
# (archive Patsight_PS27D_C3_reference_source.zip; LICENSE: 'Original PatSight contributions: Copyright (c) 2026
# Bari Asunka', AGPL-3.0-only). Tk-free; used by the workbench canvas for aspect fit and click mapping.
"""Pure display-geometry helpers: fit a source frame into a canvas with
letterboxing, and map points both ways.

Kept separate from every Tk widget so the click-mapping behaviour
(PS-01 verification item D) is unit-testable without a display.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CanvasFit:
    """How a ``src_w`` x ``src_h`` frame is drawn inside a ``canvas_w`` x
    ``canvas_h`` canvas: uniform ``scale`` with ``offset`` letterbox
    margins (one axis has zero margin)."""

    canvas_w: int
    canvas_h: int
    src_w: int
    src_h: int
    scale: float
    offset_x: float
    offset_y: float

    @property
    def display_w(self) -> float:
        return self.src_w * self.scale

    @property
    def display_h(self) -> float:
        return self.src_h * self.scale


def fit(canvas_w: int, canvas_h: int, src_w: int, src_h: int) -> CanvasFit:
    canvas_w = max(1, int(canvas_w))
    canvas_h = max(1, int(canvas_h))
    src_w = max(1, int(src_w))
    src_h = max(1, int(src_h))
    scale = min(canvas_w / src_w, canvas_h / src_h)
    display_w = src_w * scale
    display_h = src_h * scale
    offset_x = (canvas_w - display_w) / 2.0
    offset_y = (canvas_h - display_h) / 2.0
    return CanvasFit(canvas_w, canvas_h, src_w, src_h, scale, offset_x, offset_y)


def canvas_to_source(fit_: CanvasFit, canvas_x: float, canvas_y: float) -> tuple[float, float] | None:
    """Map a canvas click to a source-pixel point, or ``None`` when the
    click landed on a letterbox margin (outside the drawn frame)."""
    if fit_.scale <= 0:
        return None
    sx = (canvas_x - fit_.offset_x) / fit_.scale
    sy = (canvas_y - fit_.offset_y) / fit_.scale
    # A tiny tolerance so a click exactly on the frame edge still counts.
    tol = 0.5 / fit_.scale
    if -tol <= sx <= fit_.src_w + tol and -tol <= sy <= fit_.src_h + tol:
        return (min(max(sx, 0.0), float(fit_.src_w)), min(max(sy, 0.0), float(fit_.src_h)))
    return None


def source_to_canvas(fit_: CanvasFit, src_x: float, src_y: float) -> tuple[float, float]:
    return (fit_.offset_x + src_x * fit_.scale, fit_.offset_y + src_y * fit_.scale)


def source_box_to_canvas(
    fit_: CanvasFit, box_xyxy: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    x1, y1 = source_to_canvas(fit_, box_xyxy[0], box_xyxy[1])
    x2, y2 = source_to_canvas(fit_, box_xyxy[2], box_xyxy[3])
    return (x1, y1, x2, y2)
