from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from exam_src.api.deps import get_current_user, get_db
from exam_src.api.routers import exams
from exam_src.db.models import AnswerKey, Base, Exam


def _test_app_and_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    app = FastAPI()
    app.include_router(exams.router)
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_current_user] = lambda: {"username": "shared"}
    return TestClient(app), session


def test_answer_key_templates_are_separate_from_exam_list():
    client, session = _test_app_and_session()
    try:
        exam = Exam(title="ข้อสอบจริง", year=2569, num_questions=2, choices="ก,ข")
        template = Exam(
            title="เฉลยกลาง",
            year=2569,
            num_questions=2,
            choices="ก,ข",
            is_answer_key_template=True,
        )
        session.add_all([exam, template])
        session.commit()

        exam_list = client.get("/api/exams")
        template_list = client.get("/api/exams/answer-key-templates")

        assert exam_list.status_code == 200
        assert [item["title"] for item in exam_list.json()] == ["ข้อสอบจริง"]
        assert template_list.status_code == 200
        assert [item["title"] for item in template_list.json()] == ["เฉลยกลาง"]
    finally:
        session.close()


def test_create_answer_key_template_records_its_type():
    client, session = _test_app_and_session()
    try:
        response = client.post(
            "/api/exams",
            json={
                "title": "เฉลยปลายภาค",
                "year": 2569,
                "num_questions": 2,
                "choices": ["ก", "ข"],
                "is_answer_key_template": True,
            },
        )

        assert response.status_code == 200
        assert response.json()["is_answer_key_template"] is True
        assert client.get("/api/exams").json() == []
        assert len(client.get("/api/exams/answer-key-templates").json()) == 1
    finally:
        session.close()


def test_answer_key_template_remains_available_as_reuse_source():
    client, session = _test_app_and_session()
    try:
        template = Exam(
            title="เฉลยสำหรับใช้ซ้ำ",
            year=2569,
            num_questions=1,
            choices="ก,ข",
            is_answer_key_template=True,
        )
        session.add(template)
        session.flush()
        session.add(
            AnswerKey(
                exam_id=template.id,
                question_no=1,
                correct_answers="ก",
                score=1,
            )
        )
        session.commit()

        response = client.get("/api/exams/answer-key-sources")

        assert response.status_code == 200
        assert response.json()[0]["id"] == template.id
        assert response.json()[0]["is_answer_key_template"] is True
    finally:
        session.close()


def test_answer_key_rejects_more_than_one_correct_choice_per_question():
    client, session = _test_app_and_session()
    try:
        exam = Exam(title="ข้อสอบ", year=2569, num_questions=1, choices="ก,ข,ค,ง")
        session.add(exam)
        session.commit()

        response = client.post(
            f"/api/exams/{exam.id}/answer-key",
            json={
                "entries": [
                    {
                        "question_no": 1,
                        "correct_answers": ["ก", "ข"],
                        "score": 1,
                        "is_cancelled": False,
                    }
                ],
            },
        )

        assert response.status_code == 422
        assert session.query(AnswerKey).filter_by(exam_id=exam.id).count() == 0
    finally:
        session.close()
