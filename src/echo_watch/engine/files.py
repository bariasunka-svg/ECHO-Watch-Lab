# SPDX-License-Identifier: AGPL-3.0-only
"""Small durable-file helpers (atomic write, hashing, still-image encoding).

``write_atomic`` follows the preserved ECHO reference's evidence writer (runtime/echowatch/evidence.py: write a
temporary sibling, flush + fsync, replace) so a reader never sees a half-written file.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def encode_png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise OSError("could not encode PNG")
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError("could not decode image")
    return img
