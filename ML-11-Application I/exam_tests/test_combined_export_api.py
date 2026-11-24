from io import BytesIO

from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from exam_src.api.deps import get_current_user, get_db
from exam_src.api.routers import results
from exam_src.db.models import Base, Exam, Result


def _test_app_and_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    app = FastAPI()
    app.include_router(results.router)
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_current_user] = lambda: {"username": "shared"}
    return app, session


def test_combined_export_endpoint_returns_named_multi_sheet_workbook():
    app, session = _test_app_and_session()
    try:
        exams = [
            Exam(title="section 1", year=2569, num_questions=1, choices="ก,ข,ค,ง"),
            Exam(title="section 2", year=2569, num_questions=1, choices="ก,ข,ค,ง"),
        ]
        session.add_all(exams)
        session.flush()
        session.add_all(
            [
                Result(
                    exam_id=exams[0].id,
                    student_id="1",
                    raw_score=1,
                    max_score=1,
                    percentage=100,
                    grade="A",
                    correct_count=1,
                    incorrect_count=0,
                    blank_count=0,
                    cancelled_count=0,
                    review_pending_count=0,
                ),
                Result(
                    exam_id=exams[1].id,
                    student_id="2",
                    raw_score=0.5,
                    max_score=1,
                    percentage=50,
                    grade="D",
                    correct_count=0,
                    incorrect_count=1,
                    blank_count=0,
                    cancelled_count=0,
                    review_pending_count=0,
                ),
            ]
        )
        session.commit()

        response = TestClient(app).post(
            "/api/exams/export/combined",
            json={
                "filename": "ผลรวม",
                "exams": [
                    {"exam_id": exams[0].id, "sheet_name": "sec1", "weight_percent": 30},
                    {"exam_id": exams[1].id, "sheet_name": "sec2", "weight_percent": 20},
                ],
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        assert "filename*=UTF-8''" in response.headers["content-disposition"]
        workbook = load_workbook(BytesIO(response.content), data_only=True)
        assert workbook.sheetnames == ["sec1", "sec2"]
        assert workbook["sec1"].cell(row=2, column=4).value == 30
        assert workbook["sec2"].cell(row=2, column=4).value == 10
    finally:
        session.close()


def test_combined_export_endpoint_rejects_duplicate_sheet_names():
    app, session = _test_app_and_session()
    try:
        response = TestClient(app).post(
            "/api/exams/export/combined",
            json={
                "filename": "result",
                "exams": [
                    {"exam_id": 1, "sheet_name": "sec1"},
                    {"exam_id": 2, "sheet_name": "SEC1"},
                ],
            },
        )
        assert response.status_code == 422
    finally:
        session.close()
