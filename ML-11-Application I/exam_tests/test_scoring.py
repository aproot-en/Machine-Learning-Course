"""Check scores, grade boundaries, and manual answer corrections."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from exam_src.db.models import AnswerKey, AnswerSheet, Base, Exam, ExtractedAnswer
from exam_src.scoring.scoring import (
    assign_grade_criterion,
    assign_grade_norm,
    score_sheet,
)


@pytest.mark.parametrize(
    "pct,expected",
    [
        (80.00, "A"),
        (79.99, "B+"),
        (75.00, "B+"),
        (74.99, "B"),
        (70.00, "B"),
        (69.99, "C+"),
        (65.00, "C+"),
        (64.99, "C"),
        (60.00, "C"),
        (59.99, "D+"),
        (55.00, "D+"),
        (54.99, "D"),
        (50.00, "D"),
        (49.99, "F"),
        (100.0, "A"),
        (0.0, "F"),
    ],
)
def test_criterion_grade_boundaries(pct, expected):
    """Check values immediately below and at each grade boundary."""
    assert assign_grade_criterion(pct) == expected


@pytest.mark.parametrize(
    "pct,mean,sd,expected",
    [
        (80, 50, 15, "A"),  # T = 50 + 10*(80-50)/15 = 70 -> A (>=63)
        (50, 50, 15, "C+"),  # T = 50 -> C+ (>=47, <53)
        (20, 50, 15, "D"),  # T = 30 -> exactly the D/F boundary (>=30)
        (65, 50, 15, "B+"),  # T = 60 -> B+ (>=57, <63)
    ],
)
def test_norm_grade_known_values(pct, mean, sd, expected):
    assert assign_grade_norm(pct, mean, sd) == expected


def test_norm_grade_zero_sd_falls_back_to_criterion():
    """No spread in scores (everyone got the same %) -- T-score is
    undefined, must not divide by zero."""
    assert assign_grade_norm(72.0, mean=72.0, sd=0.0) == assign_grade_criterion(72.0)


def test_pending_review_best_guess_is_scored_provisionally():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        exam = Exam(title="pending review", year=2569, num_questions=2, choices="ก,ข,ค,ง")
        session.add(exam)
        session.flush()
        session.add_all(
            [
                AnswerKey(exam_id=exam.id, question_no=1, correct_answers="ก", score=1),
                AnswerKey(exam_id=exam.id, question_no=2, correct_answers="ข", score=1),
            ]
        )
        sheet = AnswerSheet(exam_id=exam.id, source_file="test.png", student_id="1")
        session.add(sheet)
        session.flush()
        session.add_all(
            [
                ExtractedAnswer(
                    answer_sheet_id=sheet.id,
                    question_no=1,
                    predicted_answer="ก",
                    confidence=0.9,
                    status="OK",
                ),
                ExtractedAnswer(
                    answer_sheet_id=sheet.id,
                    question_no=2,
                    predicted_answer="ข",
                    confidence=0.2,
                    status="REVIEW",
                ),
            ]
        )
        session.commit()

        score = score_sheet(session, sheet)
        assert score.raw_score == 2
        assert score.max_score == 2
        assert score.percentage == 100
        assert score.review_pending_count == 1
    finally:
        session.close()
