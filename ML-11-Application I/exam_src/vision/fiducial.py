"""Find four circular corner markers and align the answer sheet."""

import cv2
import numpy as np

CANONICAL_SIZE = (850, 1200)  # (width, height) -- the blank template's own native size
MARKER_AREA_RANGE = (400, 1600)
MARKER_CIRCULARITY_MIN = 0.6
MARKER_ASPECT_TOL = 0.35
MORPH_CLOSE_KERNEL = 5
MORPH_CLOSE_ITERS = 2
BINARY_THRESHOLD = 128

# Canonical corner target positions, measured on the blank template
# (exam_input_files/test-data-image-org/ORG-Test-image.png) -- see build_template.py.
CANONICAL_CORNERS = {
    "tl": (50.5, 47.5),
    "tr": (806.0, 48.5),
    "br": (806.0, 1159.5),
    "bl": (50.5, 1159.5),
}


def find_marker_candidates(img_gray: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Returns (cx, cy, area, circularity) for every blob that looks like a
    fiducial marker, anywhere in the image."""
    _, binary = cv2.threshold(img_gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=MORPH_CLOSE_ITERS)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MARKER_AREA_RANGE[0] <= area <= MARKER_AREA_RANGE[1]):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        if abs(aspect - 1.0) > MARKER_ASPECT_TOL:
            continue
        circularity = 4 * np.pi * area / (cv2.arcLength(c, True) ** 2 + 1e-6)
        if circularity < MARKER_CIRCULARITY_MIN:
            continue
        found.append((x + w / 2, y + h / 2, area, circularity))
    return found


MIN_CORNER_SEPARATION_PX = 100  # two "different" corners closer than this are really one marker


def select_four_corners(
    candidates: list[tuple[float, float, float, float]], img_shape: tuple[int, int]
) -> dict[str, tuple[float, float]] | None:
    """Pick the nearest marker to each corner; reject missing or overlapping corners."""
    h, w = img_shape
    expected = {"tl": (0, 0), "tr": (w, 0), "br": (w, h), "bl": (0, h)}
    if not candidates:
        return None

    result = {}
    for name, (ex, ey) in expected.items():
        marker = min(candidates, key=lambda point: (point[0] - ex) ** 2 + (point[1] - ey) ** 2)
        result[name] = (marker[0], marker[1])

    points = list(result.values())
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dx, dy = points[i][0] - points[j][0], points[i][1] - points[j][1]
            if (dx * dx + dy * dy) ** 0.5 < MIN_CORNER_SEPARATION_PX:
                return None  # two corner slots collapsed onto the same real marker -- not 4 distinct corners

    return result


def warp_to_canonical(img: np.ndarray, corners: dict[str, tuple[float, float]]) -> np.ndarray:
    """Perspective-warp `img` so its detected corners land on
    CANONICAL_CORNERS, producing an image of CANONICAL_SIZE."""
    src = np.array([corners["tl"], corners["tr"], corners["br"], corners["bl"]], dtype=np.float32)
    dst = np.array(
        [
            CANONICAL_CORNERS["tl"],
            CANONICAL_CORNERS["tr"],
            CANONICAL_CORNERS["br"],
            CANONICAL_CORNERS["bl"],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, CANONICAL_SIZE)


def detect_and_warp(img_gray: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
    """End-to-end: find corners, warp to canonical space. Returns
    (warped_image, corners) or (None, None) if fewer than 4 corners found."""
    candidates = find_marker_candidates(img_gray)
    corners = select_four_corners(candidates, img_gray.shape)
    if corners is None:
        return None, None
    return warp_to_canonical(img_gray, corners), corners
