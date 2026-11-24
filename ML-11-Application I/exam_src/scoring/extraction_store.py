"""Save detected answers and record manual corrections."""

import json

from sqlalchemy.orm import Session

from exam_src.db.models import AnswerSheet, AuditLog, ExtractedAnswer, ReviewedAnswer
from exam_src.vision.pipeline import SheetResult


def save_sheet_result(
    session: Session, exam_id: int, source_file: str, result: SheetResult
) -> AnswerSheet:
    """Save one processed sheet and its detected answers."""
    sheet = AnswerSheet(
        exam_id=exam_id,
        source_file=source_file,
        student_id=result.student_id.student_id if result.student_id else None,
        student_id_best_guess=(
            result.student_id.best_guess_id if result.student_id else None
        ),
        student_id_confidence=(
            min(result.student_id.per_digit_confidence) if result.student_id else None
        ),
        student_id_status=result.student_id.status if result.student_id else "ERROR",
        quality_passed=result.quality.passed,
        quality_flags=json.dumps(result.quality.reasons, ensure_ascii=False),
    )
    session.add(sheet)
    session.flush()  ## assigns sheet.id

    if result.answers:
        for a in result.answers:
            session.add(
                ExtractedAnswer(
                    answer_sheet_id=sheet.id,
                    question_no=a.question,
                    predicted_answer=a.answer,
                    confidence=a.confidence,
                    status=a.status,
                    fill_ratios=json.dumps(a.fill_ratios, ensure_ascii=False),
                )
            )
    session.commit()
    return sheet


def get_review_queue(
    session: Session, exam_id: int | None = None
) -> list[ExtractedAnswer]:
    """List answers whose detection status is not OK, ordered by sheet and question."""
    query = (
        session.query(ExtractedAnswer)
        .join(AnswerSheet)
        .filter(ExtractedAnswer.status != "OK")
    )
    if exam_id is not None:
        query = query.filter(AnswerSheet.exam_id == exam_id)
    return query.order_by(AnswerSheet.id, ExtractedAnswer.question_no).all()


def submit_review(
    session: Session,
    extracted_answer_id: int,
    final_answer: str | None,
    reviewed_by: str,
    note: str = "",
) -> ReviewedAnswer:
    """Save a manual answer and audit entry. None confirms a blank answer."""
    extracted = session.get(ExtractedAnswer, extracted_answer_id)
    if extracted is None:
        raise ValueError(f"no ExtractedAnswer with id {extracted_answer_id}")

    old_value = extracted.predicted_answer
    review = (
        session.query(ReviewedAnswer)
        .filter_by(extracted_answer_id=extracted_answer_id)
        .one_or_none()
    )
    if review is None:
        review = ReviewedAnswer(
            extracted_answer_id=extracted_answer_id,
            final_answer=final_answer,
            reviewed_by=reviewed_by,
            note=note,
        )
        session.add(review)
    else:
        review.final_answer = final_answer
        review.reviewed_by = reviewed_by
        review.note = note

    session.add(
        AuditLog(
            user=reviewed_by,
            action="review_answer",
            entity="extracted_answer",
            entity_id=extracted_answer_id,
            old_value=old_value,
            new_value=final_answer,
        )
    )
    session.commit()
    return review
