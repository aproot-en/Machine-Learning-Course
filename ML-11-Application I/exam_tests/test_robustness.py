"""Check rotated, shadowed, and low-resolution scans."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from exam_src.vision.pipeline import process_sheet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_SCAN_PATH = PROJECT_ROOT / "exam_input_files/test-data-image-org/CCF_000001.png"
KNOWN_STUDENT_ID = "1493314x4541x"  # ground truth for this scan, confirmed against the
# handwritten subject-code field (updated after fixing
# ID-grid blank/"x" detection -- see answer_detection.py)

pytestmark = pytest.mark.skipif(not REAL_SCAN_PATH.exists(), reason="sample scan not found")


def _rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w, new_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h), borderValue=255)


@pytest.mark.parametrize("angle", [12, 15, 20, 25])
def test_handles_rotation_up_to_25_degrees(tmp_path, angle):
    """Handled correctly: perspective correction recovers a rotated scan
    and still extracts the correct student ID."""
    img = cv2.imread(str(REAL_SCAN_PATH), cv2.IMREAD_GRAYSCALE)
    rotated = _rotate(img, angle)
    path = tmp_path / f"rotated_{angle}.png"
    cv2.imwrite(str(path), rotated)

    result = process_sheet(str(path))
    assert result.quality.passed
    assert result.student_id.student_id == KNOWN_STUDENT_ID


def test_rejects_severe_shadow_instead_of_silently_corrupting(tmp_path):
    """Reject heavy shadows when four distinct corner markers cannot be found."""
    img = cv2.imread(str(REAL_SCAN_PATH), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    w = img.shape[1]
    gradient = np.linspace(0.4, 1.0, w)
    shadowed = (img * gradient[np.newaxis, :]).astype(np.uint8)
    path = tmp_path / "shadow.png"
    cv2.imwrite(str(path), shadowed)

    result = process_sheet(str(path))
    assert not result.quality.passed
    assert "corner" in result.quality.reasons[0]
    assert result.answers is None  # never silently produced a (wrong) answer set


def test_quality_failure_stops_detection_even_when_corners_are_found(tmp_path):
    """A readable image below the quality threshold must not be decoded."""
    img = cv2.imread(str(REAL_SCAN_PATH), cv2.IMREAD_GRAYSCALE)
    low_res = cv2.resize(img, (425, 600), interpolation=cv2.INTER_AREA)
    path = tmp_path / "low-resolution.png"
    cv2.imwrite(str(path), low_res)

    result = process_sheet(str(path))
    assert not result.quality.passed
    assert any("resolution too low" in reason for reason in result.quality.reasons)
    assert result.answers is None
    assert result.student_id is None
