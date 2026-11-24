"""Read answer marks and student-ID bubbles using the sheet template."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates/rmutt_answer_sheet_v1.json"

BOX_INSET_PX = 3  ## Exclude the printed border when measuring ink.
BLANK_THRESHOLD = 0.12
MARK_THRESHOLD = 0.22
SEPARATION_MARGIN = 0.10
AUTO_ACCEPT_THRESHOLD = 0.30  ## Accept a unique winner with at least 30% fill.

## ID bubbles contain printed digits, so they need different thresholds.
## A blank column represents "x". A close runner-up needs manual confirmation.
ID_BLANK_MAX = 0.40
ID_GAP_MIN = 0.30


def load_template(path: Path = TEMPLATE_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fill_ratio(
    binary: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    inset: int = BOX_INSET_PX,
) -> float:
    x0, y0 = int(cx - w / 2 + inset), int(cy - h / 2 + inset)
    x1, y1 = int(cx + w / 2 - inset), int(cy + h / 2 - inset)
    crop = binary[max(0, y0) : y1, max(0, x0) : x1]
    if crop.size == 0:
        return 0.0
    return float((crop > 0).mean())


@dataclass
class AnswerResult:
    question: int
    answer: str | None
    confidence: float | None
    status: str  ## "OK" | "REVIEW" | "BLANK" | "MULTIPLE"
    fill_ratios: dict


def decide(
    ratios: list[float], labels: list[str]
) -> tuple[str | None, float | None, str]:
    """Return a choice, confidence, and status. Keep uncertain choices unset for review."""
    order = np.argsort(ratios)[::-1]
    top, second = ratios[order[0]], ratios[order[1]]

    if top < BLANK_THRESHOLD:
        return None, top, "BLANK"
    ## User-facing confidence rule: a uniquely highest choice at 30% or
    ## above is final immediately, even when the runner-up is fairly close.
    if top >= AUTO_ACCEPT_THRESHOLD and top > second:
        return labels[order[0]], top, "OK"
    if (
        top >= MARK_THRESHOLD
        and second >= MARK_THRESHOLD
        and (top - second) < SEPARATION_MARGIN
    ):
        return None, top, "MULTIPLE"
    return None, top, "REVIEW"


def detect_answers(binary: np.ndarray, template: dict) -> list[AnswerResult]:
    """Read each column top to bottom: questions 1-20, then 21-40, and so on."""
    grid = template["answer_grid"]
    bw, bh = grid["box_size"]
    choices = grid["choices"]
    rows_per_group = grid["rows_per_group"]
    results = []
    for group_i, group_x in enumerate(grid["group_start_x"]):
        for row_i, y in enumerate(grid["row_y"]):
            q = group_i * rows_per_group + row_i + 1
            ratios = [
                fill_ratio(binary, group_x + dx, y, bw, bh) for dx in grid["choice_dx"]
            ]
            answer, conf, status = decide(ratios, choices)
            results.append(
                AnswerResult(
                    question=q,
                    answer=answer,
                    confidence=conf,
                    status=status,
                    fill_ratios=dict(zip(choices, ratios)),
                )
            )
    return sorted(results, key=lambda r: r.question)


def crop_subject_code_box(gray: np.ndarray, template: dict, pad: int = 5) -> np.ndarray:
    """Crop the subject-code field for display during review."""
    box = template["subject_code_box"]
    x0, y0 = max(0, box["x"] - pad), max(0, box["y"] - pad)
    x1, y1 = box["x"] + box["w"] + pad, box["y"] + box["h"] + pad
    return gray[y0:y1, x0:x1]


def crop_student_id_handwritten(
    gray: np.ndarray, template: dict, pad: int = 2
) -> np.ndarray:
    """Crop the handwritten ID above the bubbles for manual review."""
    box = template["student_id_handwritten_box"]
    x0 = max(0, int(box["x"] - pad))
    y0 = max(0, int(box["y"] - pad))
    x1 = int(box["x"] + box["w"] + pad)
    y1 = int(box["y"] + box["h"] + pad)
    return gray[y0:y1, x0:x1]


@dataclass
class StudentIdResult:
    student_id: str
    per_digit_confidence: list[float]
    status: str  ## "OK" | "REVIEW"
    best_guess_id: str  ## Suggested digits for positions marked "?".


def detect_student_id(binary: np.ndarray, template: dict) -> StudentIdResult:
    id_tpl = template["student_id"]
    bw, bh = id_tpl["box_size"]
    digits = []
    best_guess_digits = []
    confidences = []
    status = "OK"
    for x in id_tpl["columns_x"]:
        ratios = [fill_ratio(binary, x, y, bw, bh) for y in id_tpl["rows_y"]]
        order = np.argsort(ratios)[::-1]
        top, second = ratios[order[0]], ratios[order[1]]
        top_digit = str(id_tpl["digit_values"][order[0]])

        if top < ID_BLANK_MAX:
            ## No circle was actually bubbled -- this institution's convention
            ## for a literal "x" in the ID, not a detection failure.
            digits.append("x")
            best_guess_digits.append("x")
        elif (top - second) < ID_GAP_MIN:
            ## Cleared the blank threshold but without a confident single
            ## winner -- could be a faint mark or genuinely blank; don't guess
            ## in `student_id`, but keep the top candidate as a hint.
            digits.append("?")
            best_guess_digits.append(top_digit)
            status = "REVIEW"
        else:
            digits.append(top_digit)
            best_guess_digits.append(top_digit)
        confidences.append(float(top))
    return StudentIdResult(
        student_id="".join(digits),
        per_digit_confidence=confidences,
        status=status,
        best_guess_id="".join(best_guess_digits),
    )
