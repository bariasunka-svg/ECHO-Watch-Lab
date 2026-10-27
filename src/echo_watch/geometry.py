# SPDX-License-Identifier: AGPL-3.0-only
"""Source-coordinate geometry: boxes, polygons, letterbox mapping and region expansion.

Adapted from the preserved ECHO Watch reference runtime/echowatch/geometry.py (SHA-256
2d76474db306c473716eb8289d477aed0995bf293fe79ce46fb8a6a91d475e83; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md).
Adaptation: header only; function bodies are unchanged.

All persisted coordinates are source-frame pixels ``[x, y, w, h]`` of the exact retained frame. Display
scaling/letterboxing is undone before anything is stored.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[float, float, float, float]


def clamp_box(box, width: int, height: int) -> Box:
    x, y, w, h = (float(v) for v in box)
    x0, y0 = max(0.0, min(x, width)), max(0.0, min(y, height))
    x1, y1 = max(0.0, min(x + w, width)), max(0.0, min(y + h, height))
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


def valid_box(box, width: int, height: int, min_side: float = 8.0) -> bool:
    x, y, w, h = clamp_box(box, width, height)
    return w >= min_side and h >= min_side


def iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def expand(box, factor: float, width: int, height: int) -> Box:
    x, y, w, h = box
    dx, dy = w * factor, h * factor
    return clamp_box((x - dx, y - dy, w + 2 * dx, h + 2 * dy), width, height)


def int_box(box) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x0, y0 = int(round(x)), int(round(y))
    return x0, y0, max(1, int(round(x + w)) - x0), max(1, int(round(y + h)) - y0)


def crop(image: np.ndarray, box) -> np.ndarray:
    x, y, w, h = int_box(clamp_box(box, image.shape[1], image.shape[0]))
    return image[y:y + h, x:x + w]


@dataclass(frozen=True)
class Letterbox:
    """Mapping between a source frame and a fitted display/model canvas."""

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int

    @property
    def scale(self) -> float:
        return min(self.dst_w / self.src_w, self.dst_h / self.src_h)

    @property
    def pad(self) -> tuple[float, float]:
        s = self.scale
        return ((self.dst_w - self.src_w * s) / 2.0, (self.dst_h - self.src_h * s) / 2.0)

    def to_src(self, x: float, y: float) -> tuple[float, float]:
        px, py = self.pad
        s = self.scale
        return ((x - px) / s, (y - py) / s)

    def to_dst(self, x: float, y: float) -> tuple[float, float]:
        px, py = self.pad
        s = self.scale
        return (x * s + px, y * s + py)

    def box_to_src(self, box) -> Box:
        x0, y0 = self.to_src(box[0], box[1])
        x1, y1 = self.to_src(box[0] + box[2], box[1] + box[3])
        return clamp_box((x0, y0, x1 - x0, y1 - y0), self.src_w, self.src_h)


def offset_box(box, dx: float, dy: float) -> Box:
    return (box[0] + dx, box[1] + dy, box[2], box[3])


def offset_polygon(poly, dx: float, dy: float):
    return [[float(p[0]) + dx, float(p[1]) + dy] for p in poly]


def polygon_bbox(poly) -> Box:
    a = np.asarray(poly, dtype=float)
    x0, y0 = a.min(axis=0)
    x1, y1 = a.max(axis=0)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def simplify_polygon(points: np.ndarray, max_points: int = 160) -> list[list[float]]:
    import cv2

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        return []
    eps = 1.0
    out = cv2.approxPolyDP(pts, eps, True)
    while len(out) > max_points:
        eps *= 1.6
        out = cv2.approxPolyDP(pts, eps, True)
    return [[round(float(p[0][0]), 1), round(float(p[0][1]), 1)] for p in out]
