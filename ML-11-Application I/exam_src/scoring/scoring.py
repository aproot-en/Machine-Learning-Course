"""Compare answers with the key and calculate scores and grades."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from exam_src.db.models import AnswerKey, AnswerSheet, ExtractedAnswer, Result

CRITERION_BANDS = [  # (min_percentage, grade), checked highest-first
    (80, "A"),
    (75, "B+"),
    (70, "B"),
    (65, "C+"),
    (60, "C"),
    (55, "D+"),
    (50, "D"),
    (0, "F"),
]
NORM_T_SCORE_BANDS = [  # T-score = 50 + 10*(x-mean)/sd
    (63, "A"),
    (57, "B+"),
    (53, "B"),
    (47, "C+"),
    (43, "C"),
    (37, "D+"),
    (30, "D"),
    (float("-inf"), "F"),
]


def load_answer_key(
    session: Session,
    exam_id: int,
    key: dict[int, str | list[str]],
    scores: dict[int, float] | None = None,
    cancelled: set[int] | None = None,
) -> None:
    """Replace the key. Each question accepts a choice or a list of choices."""
    scores = scores or {}
    cancelled = cancelled or set()

    session.query(AnswerKey).filter_by(exam_id=exam_id).delete()
    for q, correct in key.items():
        choices = correct if isinstance(correct, list) else [correct]
        session.add(
            AnswerKey(
                exam_id=exam_id,
                question_no=q,
                correct_answers=",".join(choices),
                score=scores.get(q, 1.0),
                is_cancelled=q in cancelled,
            )
        )
    session.commit()


@dataclass
class ScoreBreakdown:
    student_id: str
    raw_score: float
    max_score: float
    percentage: float
    correct_count: int = 0
    incorrect_count: int = 0
    blank_count: int = 0
    cancelled_count: int = 0
    review_pending_count: int = 0
    per_question: list[dict] = field(default_factory=list)


def score_sheet(
    session: Session,
    sheet: AnswerSheet,
    negative_marking: bool = False,
    negative_penalty: float = 0.25,
) -> ScoreBreakdown:
    """Calculate one sheet's score, including provisional answers awaiting review."""
    key_rows = {
        k.question_no: k for k in session.query(AnswerKey).filter_by(exam_id=sheet.exam_id).all()
    }
    answers = {
        a.question_no: a
        for a in session.query(ExtractedAnswer).filter_by(answer_sheet_id=sheet.id).all()
    }

    raw_score = 0.0
    max_score = 0.0
    correct = incorrect = blank = cancelled_n = pending = 0
    per_question = []

    for q, key_row in sorted(key_rows.items()):
        if key_row.is_cancelled:
            # cancelled questions give everyone full credit -- add
            # to both raw and max so the percentage is unaffected either way.
            cancelled_n += 1
            raw_score += key_row.score
            max_score += key_row.score
            per_question.append({"q": q, "result": "cancelled"})
            continue

        extracted = answers.get(q)

        if extracted is None or extracted.status == "ERROR":
            max_score += key_row.score
            per_question.append({"q": q, "result": "error"})
            continue
        is_provisional = extracted.status in ("REVIEW", "MULTIPLE") and extracted.review is None
        if is_provisional:
            # Score the automatic best guess immediately. Human review is a
            # later verification/override and must not block initial results.
            pending += 1

        max_score += key_row.score
        final = extracted.final_answer
        correct_choices = set(key_row.correct_answers.split(","))

        if final is None:
            blank += 1
            per_question.append({"q": q, "result": "blank"})
        elif final in correct_choices:
            correct += 1
            raw_score += key_row.score
            per_question.append(
                {"q": q, "result": "correct", "answer": final, "provisional": is_provisional}
            )
        else:
            incorrect += 1
            if negative_marking:
                raw_score -= key_row.score * negative_penalty
            per_question.append(
                {"q": q, "result": "incorrect", "answer": final, "provisional": is_provisional}
            )

    raw_score = max(0.0, raw_score) if not negative_marking else raw_score
    percentage = (raw_score / max_score * 100) if max_score > 0 else 0.0

    return ScoreBreakdown(
        student_id=sheet.student_id or f"UNKNOWN-{sheet.id}",
        raw_score=raw_score,
        max_score=max_score,
        percentage=percentage,
        correct_count=correct,
        incorrect_count=incorrect,
        blank_count=blank,
        cancelled_count=cancelled_n,
        review_pending_count=pending,
        per_question=per_question,
    )


def assign_grade_criterion(percentage: float) -> str:
    for min_pct, grade in CRITERION_BANDS:
        if percentage >= min_pct:
            return grade
    return "F"


def assign_grade_norm(percentage: float, mean: float, sd: float) -> str:
    if sd == 0:
        return assign_grade_criterion(percentage)
    t_score = 50 + 10 * (percentage - mean) / sd
    for min_t, grade in NORM_T_SCORE_BANDS:
        if t_score >= min_t:
            return grade
    return "F"


def score_exam(
    session: Session, exam_id: int, grading: str = "criterion", negative_marking: bool = False
) -> list[ScoreBreakdown]:
    """Score all sheets and save results using criterion or norm-based grades."""
    sheets = session.query(AnswerSheet).filter_by(exam_id=exam_id).all()

    breakdowns = [score_sheet(session, s, negative_marking=negative_marking) for s in sheets]

    if grading == "norm":
        pcts = [b.percentage for b in breakdowns]
        mean, sd = float(np.mean(pcts)), float(np.std(pcts))

    for b in breakdowns:
        grade = (
            assign_grade_norm(b.percentage, mean, sd)
            if grading == "norm"
            else assign_grade_criterion(b.percentage)
        )
        # Unreadable IDs use a unique UNKNOWN-{sheet.id} value to avoid collisions.
        existing = (
            session.query(Result).filter_by(exam_id=exam_id, student_id=b.student_id).one_or_none()
        )
        if existing is None:
            existing = Result(exam_id=exam_id, student_id=b.student_id)
            session.add(existing)
        existing.raw_score = b.raw_score
        existing.max_score = b.max_score
        existing.percentage = b.percentage
        existing.grade = grade
        existing.correct_count = b.correct_count
        existing.incorrect_count = b.incorrect_count
        existing.blank_count = b.blank_count
        existing.cancelled_count = b.cancelled_count
        existing.review_pending_count = b.review_pending_count

    session.commit()
    return breakdowns


@dataclass
class SanityReport:
    ok: bool
    issues: list[str]


def sanity_check(
    session: Session, exam_id: int, expected_student_count: int | None = None
) -> SanityReport:
    """Report invalid scores, duplicate IDs, unresolved IDs, and missing sheets."""
    issues = []
    sheets = session.query(AnswerSheet).filter_by(exam_id=exam_id).all()
    results = session.query(Result).filter_by(exam_id=exam_id).all()

    for r in results:
        if r.raw_score < 0:
            issues.append(
                f"student {r.student_id}: negative raw_score {r.raw_score} (negative marking not expected here)"
            )
        if r.raw_score > r.max_score:
            issues.append(
                f"student {r.student_id}: raw_score {r.raw_score} exceeds max_score {r.max_score}"
            )

    ids = [s.student_id for s in sheets if s.student_id]
    dupes = [sid for sid, n in Counter(ids).items() if n > 1]
    if dupes:
        issues.append(f"duplicate student_id across sheets: {dupes}")

    unresolved = [s.source_file for s in sheets if s.student_id_status != "OK"]
    if unresolved:
        issues.append(f"{len(unresolved)} sheet(s) have an unresolved student_id: {unresolved}")

    if expected_student_count is not None and len(sheets) != expected_student_count:
        issues.append(f"expected {expected_student_count} sheets, found {len(sheets)}")

    return SanityReport(ok=len(issues) == 0, issues=issues)
