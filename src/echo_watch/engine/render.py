# SPDX-License-Identifier: AGPL-3.0-only
"""Annotation of a COMMITTED result packet onto its OWN analysed image (a projection of the packet; nothing else).

Boxes are drawn only for items whose committed state carries a live box (Visible / Checking) and only when the packet
still has live authority. A banner (PAUSED, SOURCE ENDED, STALE ...) means the packet is a HELD record: its boxes are
removed, because a rectangle is never painted on an image that is no longer the live view. States without a box (Not
visible, Match uncertain, Cannot assess, Paused, Offline) appear in the status panel only - an absent target gets no box.
The navy / cyan palette follows the product's ECHO Watch identity.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contracts import BOXED_STATES, LABEL, ResultPacket

# BGR
CYAN = (238, 211, 34)
AMBER = (60, 190, 245)
GREY = (170, 170, 170)
RED = (90, 90, 235)
NAVY = (58, 36, 14)
WHITE = (245, 245, 245)
STATE_COLOR = {"visible": CYAN, "checking": AMBER, "not_seen": RED, "uncertain": AMBER, "cannot_assess": GREY,
               "paused": GREY, "offline": GREY, "stopped": GREY}


def render_result(packet: ResultPacket, banner: str | None = None) -> np.ndarray:
    img = packet.image.copy()
    h, w = img.shape[:2]
    s = max(1.0, w / 1280.0)
    thick = max(2, int(round(2 * s)))
    live = banner is None
    if live:
        for o in packet.items:
            if o.state in BOXED_STATES and o.box is not None:
                x, y, bw, bh = (int(round(v)) for v in o.box)
                col = STATE_COLOR[o.state]
                cv2.rectangle(img, (x, y), (x + bw, y + bh), col, thick)
                label = f"{o.name} - {LABEL[o.state]}"
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, 1)
                ty = max(th + 4, y - 4)
                cv2.rectangle(img, (x, ty - th - 4), (x + tw + 8, ty + base), NAVY, -1)
                cv2.putText(img, label, (x + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, col, 1, cv2.LINE_AA)
    # status panel (navy, translucent): result identity and every member's committed state
    rows = [f"ECHO Watch  result {packet.seq}  frame {packet.frame_ident[1]}"
            + (f"  {banner}" if banner else "")]
    for o in packet.items:
        rows.append(f"{o.name}: {LABEL.get(o.state, o.state)}")
    pw = int(max(cv2.getTextSize(r, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, 1)[0][0] for r in rows) + 20 * s)
    ph = int((len(rows) * 22 + 10) * s)
    roi = img[8:8 + ph, 8:8 + pw]
    overlay = np.full_like(roi, NAVY)
    cv2.addWeighted(overlay, 0.72, roi, 0.28, 0, roi)
    y = int(24 * s)
    for k, r in enumerate(rows):
        col = WHITE
        if k:
            col = STATE_COLOR.get(packet.items[k - 1].state, WHITE)
        cv2.putText(img, r, (int(16 * s), 8 + y), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, col, 1, cv2.LINE_AA)
        y += int(22 * s)
    if banner:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), AMBER, int(4 * s))
    return img
