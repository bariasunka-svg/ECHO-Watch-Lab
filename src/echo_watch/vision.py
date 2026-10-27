# SPDX-License-Identifier: AGPL-3.0-only
"""Appearance features, ring context and background-geometry evidence (pure OpenCV/NumPy).

Adapted (SUBSET) from the preserved ECHO Watch reference runtime/echowatch/vision.py (SHA-256
2f6982e8a75de628816e7a118baf456cd34c9a03745b33ce320c7e763f9eebc2; AGPL-3.0-only, copyright holder not recorded in that project's NOTICE.md).
Adopted symbols, bodies unchanged: Features, features, similarity, ring_crops, ring_similarity, anchor_image,
background_geometry (+ its GEOM_* constants and GEOMETRY_METHOD). NOT adopted: the commissioned-region classifier
(AssetVision/Thresholds/alignment/derive_thresholds/load_*): the standalone engine has no commissioning stage and
uses no empty-position model, so absence evidence comes only from compatible-observation search plus the
foreground-excluded background-consensus geometry below.

This is appearance evidence for remembered items, never physical serial identity.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import clamp_box, expand, int_box

FEAT = 64
RING_FACTOR = 0.35
ANCHOR_W = 320


@dataclass
class Features:
    g: np.ndarray
    e: np.ndarray
    hist: np.ndarray
    mean: float
    std: float
    estd: float
    glare: float


def features(roi: np.ndarray) -> Features:
    if roi.size == 0:
        raise ValueError("empty region")
    small = cv2.resize(roi, (FEAT, FEAT), interpolation=cv2.INTER_AREA)
    g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    e = cv2.magnitude(gx, gy)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).astype(np.float32)
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    gray_full = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return Features(g=g, e=e, hist=hist, mean=float(g.mean()), std=float(g.std()), estd=float(e.std()),
                    glare=float((gray_full >= 250).mean()))


def _ncc(a: np.ndarray, sa: float, b: np.ndarray, sb: float, flat: float, mean_a: float, mean_b: float) -> float:
    a_flat, b_flat = sa < flat, sb < flat
    if a_flat and b_flat:
        return 1.0 - min(1.0, abs(mean_a - mean_b) / 40.0)
    if a_flat or b_flat:
        return -0.2
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def similarity(a: Features, b: Features) -> float:
    gs = _ncc(a.g, a.std, b.g, b.std, 4.0, a.mean, b.mean)
    es = _ncc(a.e, a.estd, b.e, b.estd, 3.0, float(a.e.mean()), float(b.e.mean()))
    hs = float(cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_CORREL))
    return 0.5 * gs + 0.3 * es + 0.2 * hs


def ring_crops(image: np.ndarray, box) -> list[np.ndarray]:
    h, w = image.shape[:2]
    outer = expand(box, RING_FACTOR, w, h)
    ox, oy, ow, oh = int_box(outer)
    bx, by, bw, bh = int_box(clamp_box(box, w, h))
    strips = [
        (ox, oy, ow, by - oy), (ox, by + bh, ow, oy + oh - (by + bh)),
        (ox, by, bx - ox, bh), (bx + bw, by, ox + ow - (bx + bw), bh),
    ]
    out = []
    for sx, sy, sw, sh in strips:
        if sw >= 6 and sh >= 6:
            out.append(image[sy:sy + sh, sx:sx + sw])
        else:
            out.append(None)
    return out


def ring_similarity(cur: list, ref: list) -> float | None:
    sims = [similarity(features(c), r) for c, r in zip(cur, ref) if c is not None and r is not None]
    return float(np.mean(sims)) if sims else None


def anchor_image(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (ANCHOR_W, max(8, int(round(h * ANCHOR_W / w)))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ---- camera/background geometry ---------------------------------------------------------------------------------
# A whole-frame phase correlation measures the dominant translation of the IMAGE; a large moving foreground object
# dominates it exactly like a camera shift does. Camera geometry is therefore qualified from spatially spread
# BACKGROUND patches only: current candidate regions, the remembered item boxes and recent target regions are
# excluded, and each remaining textured patch is located in the current frame by a bounded normalised template
# search. A patch votes "still" (best match at zero shift), "shifted" (best match elsewhere) or "changed" (no
# reliable match). Movement is confirmed only by a consistent, spatially spread multi-patch consensus on ONE shift;
# too little or inconsistent background is "unproven", which freezes negative evidence and never guesses.
GEOM_GRID = (16, 10)            # patch grid over the 320-px anchor (20 x 18 px patches at 16:9)
GEOM_MIN_UNMASKED = 0.85        # a patch is usable only if at least this fraction is not excluded
GEOM_MIN_STD = 4.0              # low-texture patches cannot vote
GEOM_MATCH = 0.6                # normalised correlation for a patch to match
GEOM_AMBIGUOUS = 0.85           # a patch matching itself this well off-position is ambiguous (aperture)
GEOM_SEARCH = 16                # template search radius in anchor px (64 px at 1280 width)
GEOM_MIN_VOTES = 6              # at least this many agreeing patches ...
GEOM_CONSENSUS = 0.6            # ... forming at least this fraction of the usable patches ...
GEOM_MIN_QUADRANTS = 2          # ... spread over at least this many image quadrants
GEOM_STILL_VETO = 0.15          # movement also needs (almost) no background patch to have stayed still
GEOM_EXCLUDE_PAD = 0.1          # exclusion boxes are grown by this fraction (capped at 6, plus 3 anchor px)
GEOMETRY_METHOD = {"name": "background-consensus-v1", "grid": list(GEOM_GRID), "min_unmasked": GEOM_MIN_UNMASKED,
                   "min_std": GEOM_MIN_STD, "match": GEOM_MATCH, "ambiguous": GEOM_AMBIGUOUS,
                   "search_anchor_px": GEOM_SEARCH, "min_votes": GEOM_MIN_VOTES, "consensus": GEOM_CONSENSUS,
                   "min_quadrants": GEOM_MIN_QUADRANTS, "still_veto": GEOM_STILL_VETO,
                   "exclude_pad": GEOM_EXCLUDE_PAD, "anchor_width": ANCHOR_W,
                   "tolerance": "max(6 px, 1% of source width)"}


def background_geometry(ref_anchor: np.ndarray, image: np.ndarray, exclude: list, tol_px: float | None = None) -> dict:
    """Camera/background geometry of ``image`` against a remembered scene anchor, from foreground-excluded,
    spatially spread background patches. ``exclude``: boxes [x, y, w, h] in source pixels (current candidates,
    remembered item boxes, recent target regions). Returns a dict with ``state`` in stable | moved | unproven,
    a human-readable ``why`` and every metric used (logged by the trace)."""
    h, w = image.shape[:2]
    tol = float(tol_px if tol_px is not None else max(6.0, 0.01 * w))
    cur = anchor_image(image)
    out: dict = {"method": GEOMETRY_METHOD["name"], "tolerance_px": round(tol, 2)}
    if cur.shape != ref_anchor.shape:
        out.update(state="unproven", why="the frame size differs from the remembered scene")
        return out
    s = w / ANCHOR_W
    tol_a = max(1.5, tol / s)
    ah, aw = cur.shape
    mask = np.ones((ah, aw), bool)
    for b in exclude or []:
        x, y, bw, bh = (float(v) / s for v in b)
        px, py = min(6.0, bw * GEOM_EXCLUDE_PAD) + 3, min(6.0, bh * GEOM_EXCLUDE_PAD) + 3
        x0, y0 = max(0, int(x - px)), max(0, int(y - py))
        x1, y1 = min(aw, int(np.ceil(x + bw + px))), min(ah, int(np.ceil(y + bh + py)))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = False
    out["excluded_fraction"] = round(1.0 - float(mask.mean()), 4)
    # whole-frame translation kept as a logged diagnostic only (it cannot separate camera from foreground)
    win = cv2.createHanningWindow((aw, ah), cv2.CV_32F)
    (wdx, wdy), wresp = cv2.phaseCorrelate(ref_anchor.copy(), cur.copy(), win)
    out["whole_frame"] = {"dx": round(wdx * s, 2), "dy": round(wdy * s, 2), "response": round(float(wresp), 4)}
    far = (int(round(wdx)), int(round(wdy))) if max(abs(wdx), abs(wdy)) > GEOM_SEARCH else None
    gx, gy = GEOM_GRID
    pw, ph = aw // gx, ah // gy
    still: list = []        # quadrants of patches matching at zero shift
    shifted: list = []      # (quadrant, dx, dy) in anchor px
    changed = excluded = low_texture = ambiguous = 0
    R = GEOM_SEARCH
    for j in range(gy):
        for i in range(gx):
            y0, x0 = j * ph, i * pw
            if mask[y0:y0 + ph, x0:x0 + pw].mean() < GEOM_MIN_UNMASKED:
                excluded += 1
                continue
            tpl = ref_anchor[y0:y0 + ph, x0:x0 + pw]
            if float(tpl.std()) < GEOM_MIN_STD:
                low_texture += 1
                continue
            ry0, rx0 = max(0, y0 - R), max(0, x0 - R)
            ry1, rx1 = min(ah, y0 + ph + R), min(aw, x0 + pw + R)
            # aperture check on the reference itself: a patch that also matches itself away from its own
            # position (a straight edge, a repeating pattern) cannot tell a shift from no shift
            own = cv2.matchTemplate(ref_anchor[ry0:ry1, rx0:rx1], tpl, cv2.TM_CCOEFF_NORMED)
            oy, ox = y0 - ry0, x0 - rx0
            r = int(np.ceil(tol_a))
            own[max(0, oy - r):oy + r + 1, max(0, ox - r):ox + r + 1] = -1.0
            if float(own.max()) >= GEOM_AMBIGUOUS:
                ambiguous += 1
                continue
            reg = cur[ry0:ry1, rx0:rx1]
            res = cv2.matchTemplate(reg, tpl, cv2.TM_CCOEFF_NORMED)
            z = float(res[y0 - ry0, x0 - rx0])
            _, best, _, loc = cv2.minMaxLoc(res)
            bdx, bdy = loc[0] + rx0 - x0, loc[1] + ry0 - y0
            if far is not None:   # a large whole-view shift beyond the search radius is tested directly
                fy, fx = y0 + far[1], x0 + far[0]
                if 0 <= fy and fy + ph <= ah and 0 <= fx and fx + pw <= aw:
                    fs = float(cv2.matchTemplate(cur[fy:fy + ph, fx:fx + pw], tpl, cv2.TM_CCOEFF_NORMED)[0, 0])
                    if fs > best:
                        best, bdx, bdy = fs, far[0], far[1]
            quad = (int(i >= gx / 2), int(j >= gy / 2))
            if z >= GEOM_MATCH and (z >= best - 0.05 or np.hypot(bdx, bdy) <= tol_a):
                still.append(quad)
            elif best >= GEOM_MATCH:
                shifted.append((quad, bdx, bdy))
            else:
                changed += 1
    usable = len(still) + len(shifted) + changed
    agree: list = []
    mdx = mdy = 0.0
    if shifted:
        mdx = float(np.median([x[1] for x in shifted]))
        mdy = float(np.median([x[2] for x in shifted]))
        agree = [x[0] for x in shifted if np.hypot(x[1] - mdx, x[2] - mdy) <= tol_a]
    out.update(patches=gx * gy, excluded_patches=excluded, low_texture_patches=low_texture,
               ambiguous_patches=ambiguous, usable=usable, still=len(still), shifted=len(shifted),
               shifted_agreeing=len(agree), changed=changed,
               consensus_shift={"dx": round(mdx * s, 1), "dy": round(mdy * s, 1)} if shifted else None)

    def consensus(quads: list) -> bool:
        return len(quads) >= GEOM_MIN_VOTES and len(quads) >= GEOM_CONSENSUS * usable and \
            len(set(quads)) >= GEOM_MIN_QUADRANTS
    shift_px = float(np.hypot(mdx, mdy) * s)
    if usable < GEOM_MIN_VOTES:
        out.update(state="unproven", why=f"too little visible textured background ({usable} usable patches) to tell "
                                         "camera movement from foreground movement")
    elif consensus(still):
        out.update(state="stable", why=None)
    elif consensus(agree) and shift_px > tol and len(still) <= max(2, GEOM_STILL_VETO * usable):
        out.update(state="moved", why=f"the camera view moved: the background shifted {shift_px:.0f} px consistently "
                                      f"({len(agree)} of {usable} background patches)")
    else:
        out.update(state="unproven", why=f"background evidence is inconsistent ({len(still)} still, {len(shifted)} "
                                         f"shifted, {changed} changed of {usable} patches); camera position unproven")
    return out
