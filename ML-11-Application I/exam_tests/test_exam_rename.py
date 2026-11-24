from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from exam_src.api.deps import get_current_user, get_db
from exam_src.api.routers import exams
from exam_src.db.models import Base, Exam


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
    return app, session


def test_rename_exam_updates_only_the_title():
    app, session = _test_app_and_session()
    try:
        exam = Exam(
            title="ชื่อเดิม",
            year=2569,
            num_questions=100,
            choices="ก,ข,ค,ง",
            negative_marking=True,
        )
        session.add(exam)
        session.commit()
        exam_id = exam.id

        response = TestClient(app).patch(
            f"/api/exams/{exam_id}",
            json={"title": "  ชื่อใหม่  "},
        )

        assert response.status_code == 200
        assert response.json()["title"] == "ชื่อใหม่"
        session.expire_all()
        saved = session.get(Exam, exam_id)
        assert saved.title == "ชื่อใหม่"
        assert saved.year == 2569
        assert saved.num_questions == 100
        assert saved.choices == "ก,ข,ค,ง"
        assert saved.negative_marking is True
    finally:
        session.close()


def test_rename_exam_rejects_blank_title():
    app, session = _test_app_and_session()
    try:
        exam = Exam(title="ชื่อเดิม", year=2569, num_questions=1, choices="ก,ข")
        session.add(exam)
        session.commit()

        response = TestClient(app).patch(
            f"/api/exams/{exam.id}",
            json={"title": "   "},
        )

        assert response.status_code == 422
        session.refresh(exam)
        assert exam.title == "ชื่อเดิม"
    finally:
        session.close()


def test_rename_missing_exam_returns_not_found():
    app, session = _test_app_and_session()
    try:
        response = TestClient(app).patch(
            "/api/exams/999",
            json={"title": "ชื่อใหม่"},
        )
        assert response.status_code == 404
    finally:
        session.close()
