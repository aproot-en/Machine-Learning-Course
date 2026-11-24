"""Check answer extraction using generated sheets with known marks."""

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from exam_src.vision.answer_detection import load_template
from exam_src.vision.pipeline import process_sheet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLANK_TEMPLATE_PATH = (
    PROJECT_ROOT / "exam_input_files/test-data-image-org/ORG-Test-image.png"
)
N_SYNTHETIC_SHEETS = 10
BLANK_PROBABILITY = (
    0.05  ## a few genuinely-unanswered questions per sheet, like real students
)

pytestmark = pytest.mark.skipif(
    not BLANK_TEMPLATE_PATH.exists(), reason="blank template image not found"
)


def draw_x_mark(
    img: np.ndarray, cx: float, cy: float, w: float, h: float, thickness: int = 2
):
    """Draws a clean X inside the box interior, simulating a student mark.
    Dark ink (low pixel value) on the light scanned page, matching real ink."""
    pad = 3
    x0, y0 = int(cx - w / 2 + pad), int(cy - h / 2 + pad)
    x1, y1 = int(cx + w / 2 - pad), int(cy + h / 2 - pad)
    cv2.line(img, (x0, y0), (x1, y1), color=40, thickness=thickness)
    cv2.line(img, (x0, y1), (x1, y0), color=40, thickness=thickness)


def generate_synthetic_sheet(
    template: dict, rng: random.Random
) -> tuple[np.ndarray, dict[int, str | None]]:
    img = cv2.imread(str(BLANK_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE).copy()
    grid = template["answer_grid"]
    bw, bh = grid["box_size"]
    ground_truth = {}

    for group_i, group_x in enumerate(grid["group_start_x"]):
        for row_i, y in enumerate(grid["row_y"]):
            q = group_i * grid["rows_per_group"] + row_i + 1
            if rng.random() < BLANK_PROBABILITY:
                ground_truth[q] = None
                continue
            choice_idx = rng.randrange(len(grid["choices"]))
            ground_truth[q] = grid["choices"][choice_idx]
            draw_x_mark(img, group_x + grid["choice_dx"][choice_idx], y, bw, bh)

    return img, ground_truth


def test_synthetic_sheets_end_to_end_accuracy(tmp_path):
    """Grade generated sheets with known answers and compare the detected results."""
    template = load_template()
    rng = random.Random(42)

    total_questions = 0
    total_correct = 0
    total_blank_correct = 0
    total_blank_expected = 0
    mismatches = []

    for i in range(N_SYNTHETIC_SHEETS):
        img, ground_truth = generate_synthetic_sheet(template, rng)
        path = tmp_path / f"synthetic_{i}.png"
        cv2.imwrite(str(path), img)

        result = process_sheet(str(path))
        assert (
            result.quality.passed
        ), f"sheet {i} failed quality gate: {result.quality.reasons}"
        assert result.answers is not None

        detected = {a.question: a.answer for a in result.answers}
        for q, expected in ground_truth.items():
            total_questions += 1
            if expected is None:
                total_blank_expected += 1
                if detected.get(q) is None:
                    total_blank_correct += 1
                    total_correct += 1
                else:
                    mismatches.append((i, q, expected, detected.get(q)))
            else:
                if detected.get(q) == expected:
                    total_correct += 1
                else:
                    mismatches.append((i, q, expected, detected.get(q)))

    accuracy = total_correct / total_questions
    print(
        f"\nSynthetic end-to-end accuracy: {accuracy:.4f} ({total_correct}/{total_questions})"
    )
    print(f"Blank detection: {total_blank_correct}/{total_blank_expected}")
    if mismatches:
        print(f"First 10 mismatches (sheet, q, expected, detected): {mismatches[:10]}")

    ## Clean synthetic X marks on a pristine template should be nearly
    ## perfectly recovered -- this is testing pipeline correctness, not
    ## real-world noise tolerance.
    assert (
        accuracy >= 0.98
    ), f"synthetic accuracy {accuracy:.4f} is lower than expected for clean synthetic marks"
