"""Measure the blank answer sheet and save its field coordinates."""

import json
from pathlib import Path

import cv2
import numpy as np

from exam_src.vision.fiducial import (
    CANONICAL_CORNERS,
    find_marker_candidates,
    select_four_corners,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLANK_TEMPLATE_PATH = (
    PROJECT_ROOT / "exam_input_files/test-data-image-org/ORG-Test-image.png"
)
OUTPUT_PATH = Path(__file__).resolve().parent / "templates/rmutt_answer_sheet_v1.json"

BOX_SIZE_RANGE = (12, 22)  ## px, both width and height
BOX_ASPECT_TOL = 0.25
BOX_MIN_AREA = 100
DEDUPE_TOLERANCE = 6  ## px, merges duplicate inner/outer contour hits on the same box

ID_REGION = {"x_max": 420, "y_min": 180, "y_max": 400}
ANSWER_REGION = {"y_min": 440, "y_max": 1110}
SUBJECT_CODE_SEARCH_REGION = {"x_min": 420, "x_max": 850, "y_min": 100, "y_max": 220}
SUBJECT_CODE_MIN_WIDTH = 200
SUBJECT_CODE_HEIGHT_RANGE = (20, 80)

EXPECTED_ID_COLS = 13
EXPECTED_ID_ROWS = 10
EXPECTED_ANSWER_GROUPS = 5
EXPECTED_ANSWER_ROWS = 20
CHOICES = ["ก", "ข", "ค", "ง"]


def find_box_candidates(binary: np.ndarray) -> list[tuple[float, float, float, float]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        if h:
            aspect = w / h
        else:
            aspect = 0
        if not (
            BOX_SIZE_RANGE[0] <= w <= BOX_SIZE_RANGE[1]
            and BOX_SIZE_RANGE[0] <= h <= BOX_SIZE_RANGE[1]
        ):
            continue
        if abs(aspect - 1.0) > BOX_ASPECT_TOL or area < BOX_MIN_AREA:
            continue
        boxes.append((x + w / 2, y + h / 2, w, h))
    return boxes


def dedupe(
    points: list[tuple[float, float, float, float]], tol: int = DEDUPE_TOLERANCE
):
    used = [False] * len(points)
    out = []
    for i, p in enumerate(points):
        if used[i]:
            continue
        group = [p]
        used[i] = True
        for j in range(i + 1, len(points)):
            if used[j]:
                continue
            if abs(points[j][0] - p[0]) < tol and abs(points[j][1] - p[1]) < tol:
                group.append(points[j])
                used[j] = True
        out.append(tuple(float(np.mean([g[k] for g in group])) for k in range(4)))
    return out


def cluster_1d(values: list[float], tol: float = 6) -> list[float]:
    values = sorted(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    response_items = []
    for g in groups:
        response_items.append(float(np.mean(g)))
    return response_items


def find_subject_code_box(binary: np.ndarray) -> tuple[int, int, int, int]:
    """Find the box below the subject-code label."""
    r = SUBJECT_CODE_SEARCH_REGION
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (
            r["x_min"] <= x <= r["x_max"]
            and r["y_min"] <= y <= r["y_max"]
            and w > SUBJECT_CODE_MIN_WIDTH
            and SUBJECT_CODE_HEIGHT_RANGE[0] < h < SUBJECT_CODE_HEIGHT_RANGE[1]
        ):
            candidates.append((x, y, w, h, cv2.contourArea(c)))
    assert candidates, "could not find the กระดาษคำตอบวิชา box on the blank template"
    x, y, w, h = max(candidates, key=lambda t: t[4])[:4]
    return int(x), int(y), int(w), int(h)


def main():
    img = cv2.imread(str(BLANK_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"could not read {BLANK_TEMPLATE_PATH}"
    print(f"Blank template size: {img.shape[1]}x{img.shape[0]}")

    ## --- corners ---
    candidates = find_marker_candidates(img)
    corners = select_four_corners(candidates, img.shape)
    assert (
        corners is not None
    ), "failed to find all 4 corner markers on the blank template"
    for name, (cx, cy) in corners.items():
        expected = CANONICAL_CORNERS[name]
        assert abs(cx - expected[0]) < 2 and abs(cy - expected[1]) < 2, (
            f"corner {name} at ({cx},{cy}) does not match fiducial.CANONICAL_CORNERS {expected} "
            "-- update CANONICAL_CORNERS if the template image changed"
        )
    print(f"Corners OK: {corners}")

    ## --- boxes ---
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    raw_boxes = find_box_candidates(binary)
    boxes = dedupe(raw_boxes)
    print(f"Found {len(raw_boxes)} raw box contours -> {len(boxes)} unique boxes")

    id_boxes = [
        b
        for b in boxes
        if b[0] < ID_REGION["x_max"]
        and ID_REGION["y_min"] <= b[1] <= ID_REGION["y_max"]
    ]
    ans_boxes = [
        b for b in boxes if ANSWER_REGION["y_min"] <= b[1] <= ANSWER_REGION["y_max"]
    ]
    print(f"ID region: {len(id_boxes)} boxes, Answer region: {len(ans_boxes)} boxes")

    id_cols = cluster_1d([b[0] for b in id_boxes])
    id_rows = cluster_1d([b[1] for b in id_boxes])
    ans_cols = cluster_1d([b[0] for b in ans_boxes])
    ans_rows = cluster_1d([b[1] for b in ans_boxes])

    assert (
        len(id_cols) == EXPECTED_ID_COLS
    ), f"expected {EXPECTED_ID_COLS} ID columns, got {len(id_cols)}: {id_cols}"
    assert (
        len(id_rows) == EXPECTED_ID_ROWS
    ), f"expected {EXPECTED_ID_ROWS} ID rows, got {len(id_rows)}: {id_rows}"
    assert (
        len(ans_rows) == EXPECTED_ANSWER_ROWS
    ), f"expected {EXPECTED_ANSWER_ROWS} answer rows, got {len(ans_rows)}: {ans_rows}"
    assert len(ans_cols) == EXPECTED_ANSWER_GROUPS * len(
        CHOICES
    ), f"expected {EXPECTED_ANSWER_GROUPS * len(CHOICES)} answer columns, got {len(ans_cols)}: {ans_cols}"
    print("Grid counts OK: ID 13x10, Answer 20 cols (5 groups x 4) x 20 rows")

    ## --- subject code box ---
    sc_x, sc_y, sc_w, sc_h = find_subject_code_box(binary)
    print(f"Subject code box: x={sc_x} y={sc_y} w={sc_w} h={sc_h}")

    group_starts = ans_cols[0 :: len(CHOICES)]
    choice_dx = [ans_cols[i] - group_starts[0] for i in range(len(CHOICES))]

    box_w = float(np.mean([b[2] for b in ans_boxes]))
    box_h = float(np.mean([b[3] for b in ans_boxes]))
    id_box_w = float(np.mean([b[2] for b in id_boxes]))
    id_box_h = float(np.mean([b[3] for b in id_boxes]))

    template = {
        "name": "rmutt_answer_sheet_v1",
        "source_image": str(BLANK_TEMPLATE_PATH.relative_to(PROJECT_ROOT)),
        "canonical_size": [int(img.shape[1]), int(img.shape[0])],
        "fiducial_corners": {k: list(v) for k, v in corners.items()},
        "student_id": {
            "num_digits": EXPECTED_ID_COLS,
            "digit_values": list(range(10)),
            "box_size": [id_box_w, id_box_h],
            "columns_x": id_cols,
            "rows_y": id_rows,
        },
        "subject_code_box": {
            "x": sc_x,
            "y": sc_y,
            "w": sc_w,
            "h": sc_h,
            "note": "handwritten (blue pen) exam/subject code -- cross-check against student_id, not OCR'd automatically",
        },
        "answer_grid": {
            "num_questions": EXPECTED_ANSWER_GROUPS * EXPECTED_ANSWER_ROWS,
            "groups": EXPECTED_ANSWER_GROUPS,
            "rows_per_group": EXPECTED_ANSWER_ROWS,
            "choices": CHOICES,
            "box_size": [box_w, box_h],
            "row_y": ans_rows,
            "group_start_x": group_starts,
            "choice_dx": choice_dx,
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
