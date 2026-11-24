from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from exam_src.api.routers.exams import list_answer_key_sources, set_answer_key
from exam_src.api.routers.grading import review
from exam_src.api.schemas import (
    AnswerKeyEntry,
    AnswerKeySubmitRequest,
    ReviewSubmitRequest,
)
from exam_src.db.models import (
    AnswerKey,
    AnswerSheet,
    Base,
    Exam,
    ExtractedAnswer,
    Result,
)
from exam_src.scoring.scoring import score_exam


def test_only_exams_with_answer_keys_are_offered_as_reuse_sources():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        reusable = Exam(title="มีเฉลย", year=2569, num_questions=2, choices="ก,ข,ค,ง")
        empty = Exam(
            title="ยังไม่มีเฉลย", year=2569, num_questions=2, choices="ก,ข,ค,ง"
        )
        session.add_all([reusable, empty])
        session.flush()
        session.add_all(
            [
                AnswerKey(
                    exam_id=reusable.id, question_no=1, correct_answers="ก", score=1
                ),
                AnswerKey(
                    exam_id=reusable.id, question_no=2, correct_answers="ข", score=2
                ),
            ]
        )
        session.commit()

        sources = list_answer_key_sources(
            db=session, user={"username": "shared-user", "role": "admin"}
        )
        assert [source.id for source in sources] == [reusable.id]
        assert sources[0].answer_count == 2
        assert sources[0].choices == "ก,ข,ค,ง"
    finally:
        session.close()


def test_saving_key_scores_sheets_that_were_detected_earlier():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        exam = Exam(
            title="ตรวจแล้วรอเฉลย", year=2569, num_questions=1, choices="ก,ข,ค,ง"
        )
        session.add(exam)
        session.flush()
        sheet = AnswerSheet(exam_id=exam.id, source_file="sheet.png", student_id="123")
        session.add(sheet)
        session.flush()
        session.add(
            ExtractedAnswer(
                answer_sheet_id=sheet.id,
                question_no=1,
                predicted_answer="ก",
                confidence=0.9,
                status="OK",
            )
        )
        session.commit()

        response = set_answer_key(
            exam.id,
            AnswerKeySubmitRequest(
                entries=[
                    AnswerKeyEntry(question_no=1, correct_answers=["ก"], score=2),
                ]
            ),
            db=session,
            user={"username": "shared-user", "role": "admin"},
        )

        result = (
            session.query(Result).filter_by(exam_id=exam.id, student_id="123").one()
        )
        assert response == {"saved": 1, "scored_sheets": 1}
        assert result.raw_score == 2
        assert result.max_score == 2
        assert result.percentage == 100
    finally:
        session.close()


def test_manual_review_overrides_guess_and_rescores_immediately():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        exam = Exam(
            title="manual override", year=2569, num_questions=1, choices="ก,ข,ค,ง"
        )
        session.add(exam)
        session.flush()
        session.add(
            AnswerKey(exam_id=exam.id, question_no=1, correct_answers="ก", score=1)
        )
        sheet = AnswerSheet(exam_id=exam.id, source_file="sheet.png", student_id="123")
        session.add(sheet)
        session.flush()
        extracted = ExtractedAnswer(
            answer_sheet_id=sheet.id,
            question_no=1,
            predicted_answer="ข",
            confidence=0.2,
            status="REVIEW",
        )
        session.add(extracted)
        session.commit()
        score_exam(session, exam.id)
        assert session.query(Result).one().raw_score == 0

        response = review(
            extracted.id,
            ReviewSubmitRequest(final_answer="ก"),
            db=session,
            user={"username": "shared-user", "role": "admin"},
        )

        session.expire_all()
        assert response["score_recomputed"] is True
        assert session.query(Result).one().raw_score == 1
    finally:
        session.close()
