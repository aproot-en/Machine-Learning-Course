"""Crop, center, and normalize characters for training and prediction."""

import cv2
import numpy as np
from scipy import ndimage

CANVAS_SIZE = 28
TARGET_EDGE = 20


def character_normalize(
    img: np.ndarray, canvas_size: int = CANVAS_SIZE, target_edge: int = TARGET_EDGE
) -> np.ndarray:
    """Crop and resize bright ink onto a centered square canvas. Empty input gives a blank canvas."""
    coords = cv2.findNonZero(img)
    if coords is None:
        return np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(coords)
    crop = img[y : y + h, x : x + w]

    scale = target_edge / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    if resized.sum() == 0:
        cy, cx = new_h / 2, new_w / 2
    else:
        cy, cx = ndimage.center_of_mass(resized)

    top = int(round(canvas_size / 2 - cy))
    left = int(round(canvas_size / 2 - cx))

    src_top = max(0, -top)
    src_left = max(0, -left)
    dst_top = max(0, top)
    dst_left = max(0, left)
    copy_h = min(new_h - src_top, canvas_size - dst_top)
    copy_w = min(new_w - src_left, canvas_size - dst_left)

    if copy_h > 0 and copy_w > 0:
        canvas[dst_top : dst_top + copy_h, dst_left : dst_left + copy_w] = resized[
            src_top : src_top + copy_h, src_left : src_left + copy_w
        ]

    return canvas


def character_normalize_batch(
    images: np.ndarray, canvas_size: int = CANVAS_SIZE, target_edge: int = TARGET_EDGE
) -> np.ndarray:
    """Apply character_normalize to a batch of shape (N, H, W)."""
    out = np.empty((images.shape[0], canvas_size, canvas_size), dtype=np.uint8)
    for i in range(images.shape[0]):
        out[i] = character_normalize(images[i], canvas_size, target_edge)
    return out


def normalize_pixels(X: np.ndarray) -> np.ndarray:
    """Convert uint8 pixels from [0, 255] to float32 values in [0, 1]."""
    return X.astype(np.float32) / 255.0
