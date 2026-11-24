"""Check scan quality, align the sheet, and prepare it for OMR."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from exam_src.vision.fiducial import CANONICAL_SIZE, detect_and_warp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

MIN_SHORT_SIDE_PX = _CFG["answer_sheet"]["real_scan"]["min_short_side_px"]
BLUR_VARIANCE_MIN = 30.0  ## variance of Laplacian below this -> "too blurry"
## A sparse-ink white answer sheet legitimately averages ~230-240 (mostly white
## background) -- calibrated against the 5 real scans in exam_input_files/, all ~234-235.
BRIGHTNESS_RANGE = (100, 253)  ## mean pixel value outside this -> too dark/bright
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 8


@dataclass
class QualityReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    short_side_px: int = 0
    blur_variance: float = 0.0
    mean_brightness: float = 0.0
    corners_found: bool = False


def load_grayscale(path: str | Path) -> np.ndarray:
    """Read an image as grayscale. PDF files must be converted first."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    return img


def resize_for_processing(gray: np.ndarray) -> np.ndarray:
    """Resize to the template's 850x1200 coordinates so marker thresholds stay consistent."""
    target_width, target_height = CANONICAL_SIZE
    target_shape = (target_height, target_width)
    if gray.shape[:2] == target_shape:
        return gray
    return cv2.resize(gray, CANONICAL_SIZE, interpolation=cv2.INTER_AREA)


def enhance(gray: np.ndarray) -> np.ndarray:
    """Balance lighting and reduce noise before finding markers."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    evened = clahe.apply(gray)
    return cv2.bilateralFilter(evened, d=5, sigmaColor=50, sigmaSpace=50)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Convert the image to white ink on a black background."""
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C,
    )


def check_quality(gray: np.ndarray, corners_found: bool) -> QualityReport:
    """Check resolution, sharpness, brightness, and corner detection."""
    short_side = min(gray.shape)
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())

    reasons = []
    if short_side < MIN_SHORT_SIDE_PX:
        reasons.append(
            f"resolution too low: short side {short_side}px < {MIN_SHORT_SIDE_PX}px"
        )
    if blur_var < BLUR_VARIANCE_MIN:
        reasons.append(
            f"image too blurry: Laplacian variance {blur_var:.1f} < {BLUR_VARIANCE_MIN}"
        )
    if not (BRIGHTNESS_RANGE[0] <= brightness <= BRIGHTNESS_RANGE[1]):
        reasons.append(
            f"brightness out of range: mean {brightness:.1f} not in {BRIGHTNESS_RANGE}"
        )
    if not corners_found:
        reasons.append("could not find all 4 corner markers")

    return QualityReport(
        passed=len(reasons) == 0,
        reasons=reasons,
        short_side_px=short_side,
        blur_variance=blur_var,
        mean_brightness=brightness,
        corners_found=corners_found,
    )


@dataclass
class ProcessResult:
    quality: QualityReport
    warped_gray: np.ndarray | None = None
    warped_binary: np.ndarray | None = None


def process(path: str | Path) -> ProcessResult:
    """Align and binarize the scan; measure quality using the original image."""
    raw = load_grayscale(path)
    normalized = resize_for_processing(raw)
    enhanced = enhance(normalized)
    warped, corners = detect_and_warp(enhanced)

    quality = check_quality(raw, corners_found=warped is not None)
    if warped is None:
        return ProcessResult(quality=quality)

    binary = binarize(warped)
    return ProcessResult(quality=quality, warped_gray=warped, warped_binary=binary)
