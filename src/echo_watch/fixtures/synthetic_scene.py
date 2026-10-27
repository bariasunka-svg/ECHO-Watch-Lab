# SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic synthetic display scene (ECHO-authored), always labelled SYNTHETIC.

Adapted (scene geometry and rendering) from the preserved ECHO Watch reference runtime/echowatch/sources/synthetic.py
(SHA-256 6efb3b0917bed76a0138c7518511520534615e0284ac71dcf260aca4c8e79ed7; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md). Adopted
unchanged: ``Item``, ``default_items``, ``extended_items``, ``SyntheticScene`` (background, item drawing, transformed
drawing, evaluator geometry). NOT adopted: ``SyntheticSource`` (the real-time frame source with threaded control
glides) and the optional foreground-panel feature of ``SyntheticScene`` (unused here) - the foundation replays an
ENCODED FILE through the real decoder instead.

It exists so the complete workflow and failure paths can be exercised without a camera. Results on it are integration
test material only: never evidence of real-display discovery or monitoring quality.

Layout is defined in 1280x720 design units and rendered at an explicit output size, so source geometry is exact at every
size: ``item_box_px`` gives an item's true pixel box including any scripted movement. The scripted geometry is
EVALUATOR TRUTH ONLY; it is never given to a model, the engine or the policy as input.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass

import cv2
import numpy as np

W, H = 1280, 720  # design units (and the default output size)


class SceneError(RuntimeError):
    pass


@dataclass
class Item:
    name: str
    kind: str
    box: tuple[int, int, int, int]
    color: tuple[int, int, int]
    present: bool = True
    obscured: bool = False
    dx: float = 0.0        # scripted translation, design units
    dy: float = 0.0
    angle: float = 0.0     # degrees, counter-clockwise
    scale: float = 1.0
    hue_x: float | None = None  # screen-hue phase source; set equal on two phones to make true look-alikes

    @property
    def moved(self) -> bool:
        return bool(self.dx or self.dy or self.angle or self.scale != 1.0)


def default_items() -> list[Item]:
    return [
        Item("phone_a", "phone", (150, 300, 92, 180), (40, 40, 46)),
        Item("phone_b", "phone", (300, 300, 92, 180), (40, 40, 46)),
        Item("speaker", "speaker", (470, 330, 120, 150), (120, 120, 128)),
        Item("watch_box", "box", (650, 360, 130, 120), (235, 235, 230)),
        Item("headphones", "headphones", (840, 320, 150, 160), (20, 20, 24)),
        Item("tablet", "tablet", (1030, 330, 190, 150), (60, 60, 70)),
    ]


def extended_items() -> list[Item]:
    """12-item layout for capacity probes: the 6 default items plus a second shelf row."""
    return default_items() + [
        Item("phone_c", "phone", (150, 575, 70, 120), (52, 48, 44)),
        Item("speaker_b", "speaker", (300, 570, 90, 125), (100, 110, 120)),
        Item("watch_box_b", "box", (470, 590, 110, 100), (225, 230, 235)),
        Item("tablet_b", "tablet", (640, 585, 150, 110), (70, 60, 60)),
        Item("headphones_b", "headphones", (850, 570, 130, 125), (30, 26, 20)),
        Item("speaker_c", "speaker", (1050, 575, 100, 120), (140, 120, 110)),
    ]


class SyntheticScene:
    def __init__(self, seed: int = 7, size: tuple[int, int] = (W, H), layout: str = "default"):
        self.width, self.height = size
        self.s = self.width / W
        if abs(self.height / H - self.s) > 1e-6:
            raise SceneError(f"Synthetic scene size must keep 16:9, got {size}")
        rng = np.random.default_rng(seed)
        bg = np.zeros((H, W, 3), np.float32)
        grad = np.linspace(170, 120, H, dtype=np.float32)[:, None]
        bg[:] = np.stack([grad * 0.95, grad * 0.9, grad * 0.85], axis=-1)
        bg[500:560] = (60, 90, 130)  # shelf front (BGR brownish)
        bg[490:500] = (90, 120, 160)
        bg += rng.normal(0, 6, bg.shape).astype(np.float32)
        for x in range(80, W, 260):  # wall panel seams = stable scene anchors
            bg[40:470, x:x + 3] -= 35
        cv2.putText(bg, "ECHO TEST WALL", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (70, 70, 70), 2)
        bg = np.clip(bg, 0, 255).astype(np.uint8)
        if self.s != 1.0:
            big = cv2.resize(bg, (self.width, self.height), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            big += np.random.default_rng(seed + 1).normal(0, 4, big.shape).astype(np.float32)
            bg = np.clip(big, 0, 255).astype(np.uint8)
        self.background = bg
        self.items = extended_items() if layout == "extended" else default_items()
        self.layout = layout
        self.lighting = 1.0
        self.shift = (0, 0)
        self.lock = threading.Lock()

    def item(self, name: str) -> Item:
        for it in self.items:
            if it.name == name:
                return it
        raise SceneError(f"No synthetic item {name}")

    # ---- evaluator geometry (never a model input) ------------------------------------
    def item_corners_px(self, it: Item) -> np.ndarray:
        x, y, w, h = it.box
        cx, cy = x + w / 2 + it.dx, y + h / 2 + it.dy
        a = math.radians(it.angle)
        hw, hh = w * it.scale / 2, h * it.scale / 2
        pts = []
        for px, py in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            rx = px * math.cos(a) + py * math.sin(a)
            ry = -px * math.sin(a) + py * math.cos(a)
            pts.append(((cx + rx) * self.s + self.shift[0], (cy + ry) * self.s + self.shift[1]))
        return np.array(pts, np.float32)

    def item_box_px(self, name: str) -> list[float]:
        """True pixel box [x, y, w, h] of the item as currently rendered (evaluator truth)."""
        c = self.item_corners_px(self.item(name))
        x0, y0 = c.min(axis=0)
        x1, y1 = c.max(axis=0)
        return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

    # ---- rendering -------------------------------------------------------------------
    def render(self, t: float) -> np.ndarray:
        with self.lock:
            img = self.background.copy()
            for it in self.items:
                if it.present:
                    if it.moved:
                        self._draw_transformed(img, it, t)
                    else:
                        self._draw_item(img, it, t, self.s, hue_x=it.hue_x)
            for it in self.items:
                if it.obscured:
                    x, y, w, h = it.box
                    x, y = x + it.dx, y + it.dy
                    s = self.s
                    cv2.ellipse(img, (int((x + w // 2) * s), int((y + h // 2) * s)), (int(w * 0.85 * s), int(h * 0.75 * s)),
                                15, 0, 360, (120, 150, 205), -1)
                    cv2.ellipse(img, (int((x + w // 2) * s), int((y + h + 60) * s)), (int(w * 0.5 * s), int(90 * s)),
                                0, 0, 360, (110, 140, 195), -1)
            if self.lighting != 1.0:
                img = np.clip(img.astype(np.float32) * self.lighting + (8 if self.lighting > 1 else 0), 0, 255).astype(np.uint8)
            dx, dy = self.shift
            if dx or dy:
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, m, (self.width, self.height), borderMode=cv2.BORDER_REPLICATE)
            return img

    def _draw_transformed(self, img: np.ndarray, it: Item, t: float) -> None:
        s = self.s
        x, y, w, h = it.box
        pw, ph = int(round(w * s)), int(round(h * s))
        patch = np.zeros((ph, pw, 3), np.uint8)
        local = Item(it.name, it.kind, (0, 0, w, h), it.color)
        self._draw_item(patch, local, t, s, hue_x=x if it.hue_x is None else it.hue_x)
        mask = np.zeros((ph, pw), np.uint8)
        if it.kind == "headphones":
            self._draw_item(mask_bgr := np.zeros((ph, pw, 3), np.uint8), Item("m", it.kind, (0, 0, w, h), (255, 255, 255)),
                            t, s, hue_x=x)
            mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
        else:
            mask[:] = 255
        cx, cy = (x + w / 2 + it.dx) * s, (y + h / 2 + it.dy) * s
        m = cv2.getRotationMatrix2D((pw / 2, ph / 2), it.angle, it.scale)
        m[0, 2] += cx - pw / 2
        m[1, 2] += cy - ph / 2
        warped = cv2.warpAffine(patch, m, (self.width, self.height), flags=cv2.INTER_LINEAR)
        wmask = cv2.warpAffine(mask, m, (self.width, self.height), flags=cv2.INTER_LINEAR)
        a = (wmask.astype(np.float32) / 255.0)[..., None]
        img[:] = (img.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8)

    @staticmethod
    def _draw_item(img: np.ndarray, it: Item, t: float, s: float = 1.0, hue_x: float | None = None) -> None:
        def p(v: float) -> int:
            return int(round(v * s))
        x, y, w, h = it.box
        if it.kind == "phone":
            cv2.rectangle(img, (p(x), p(y)), (p(x + w), p(y + h)), it.color, -1)
            hue = int((t * 20 + (x if hue_x is None else hue_x)) % 180)
            sh, sw = p(y + h - 15) - p(y + 15), p(x + w - 7) - p(x + 7)
            screen = np.full((sh, sw, 3), (hue, 200, 220), np.uint8)
            img[p(y + 15):p(y + h - 15), p(x + 7):p(x + w - 7)] = cv2.cvtColor(screen, cv2.COLOR_HSV2BGR)
            cv2.circle(img, (p(x + w // 2), p(y + h - 8)), max(1, p(4)), (90, 90, 90), -1)
        elif it.kind == "speaker":
            cv2.rectangle(img, (p(x), p(y)), (p(x + w), p(y + h)), it.color, -1)
            cv2.circle(img, (p(x + w // 2), p(y + 45)), p(32), (30, 30, 30), -1)
            cv2.circle(img, (p(x + w // 2), p(y + 110)), p(20), (30, 30, 30), -1)
        elif it.kind == "box":
            cv2.rectangle(img, (p(x), p(y)), (p(x + w), p(y + h)), it.color, -1)
            cv2.rectangle(img, (p(x), p(y + 40)), (p(x + w), p(y + 62)), (180, 90, 20), -1)
            cv2.putText(img, "WATCH", (p(x + 18), p(y + 100)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (40, 40, 40),
                        max(1, p(2)))
        elif it.kind == "headphones":
            cv2.ellipse(img, (p(x + w // 2), p(y + h // 2)), (p(w // 2 - 10), p(h // 2 - 10)), 0, 180, 360, it.color,
                        max(1, p(14)))
            cv2.rectangle(img, (p(x + 2), p(y + h // 2 - 5)), (p(x + 36), p(y + h - 10)), it.color, -1)
            cv2.rectangle(img, (p(x + w - 36), p(y + h // 2 - 5)), (p(x + w - 2), p(y + h - 10)), it.color, -1)
        else:
            cv2.rectangle(img, (p(x), p(y)), (p(x + w), p(y + h)), it.color, -1)
            cv2.rectangle(img, (p(x + 10), p(y + 10)), (p(x + w - 10), p(y + h - 10)), (200, 150, 60), -1)
