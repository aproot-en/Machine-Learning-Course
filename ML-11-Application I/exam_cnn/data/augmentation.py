"""Vary training images at each epoch. Never flip letters."""

import cv2
import numpy as np

ROTATION_RANGE_DEG = 10
SHIFT_RANGE = 0.10
ZOOM_RANGE = 0.10
SHEAR_RANGE = 0.12
MORPH_PROB = 0.25
NOISE_PROB = 0.30
BLUR_PROB = 0.20


def _affine_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    angle = rng.uniform(-ROTATION_RANGE_DEG, ROTATION_RANGE_DEG)
    scale = 1.0 + rng.uniform(-ZOOM_RANGE, ZOOM_RANGE)
    shear = rng.uniform(-SHEAR_RANGE, SHEAR_RANGE)
    tx = rng.uniform(-SHIFT_RANGE, SHIFT_RANGE) * w
    ty = rng.uniform(-SHIFT_RANGE, SHIFT_RANGE) * h

    center = (w / 2, h / 2)
    rot_scale = cv2.getRotationMatrix2D(center, angle, scale)
    shear_mat = np.array([[1, shear, 0], [0, 1, 0]], dtype=np.float32)

    rot_scale_3x3 = np.vstack([rot_scale, [0, 0, 1]])
    shear_3x3 = np.vstack([shear_mat, [0, 0, 1]])
    combined = (shear_3x3 @ rot_scale_3x3)[:2, :]
    combined[:, 2] += (tx, ty)

    return cv2.warpAffine(
        img, combined, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def _morphology(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)
    if rng.random() < 0.5:
        return cv2.dilate(img, kernel, iterations=1)
    return cv2.erode(img, kernel, iterations=1)


def augment_image(
    img: np.ndarray, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Apply one random augmentation pass to a single uint8 image."""
    if rng is None:
        rng = np.random.default_rng()

    out = _affine_augment(img, rng)

    if rng.random() < MORPH_PROB:
        out = _morphology(out, rng)

    if rng.random() < BLUR_PROB:
        out = cv2.GaussianBlur(out, (3, 3), 0)

    if rng.random() < NOISE_PROB:
        noise = rng.normal(0, 8, size=out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return out
