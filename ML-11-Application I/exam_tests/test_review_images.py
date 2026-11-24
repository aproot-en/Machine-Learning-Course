"""Check review images, crop selection, and upload path restrictions."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from exam_src.api.deps import get_db
from exam_src.api.routers import grading
from exam_src.db.models import AnswerSheet, Base, Exam


@pytest.fixture
def review_app(tmp_path, monkeypatch):
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr(grading, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(grading, "UPLOAD_DIR", tmp_path / "uploads")

    with Session(engine) as session:
        exam = Exam(title="Review images")
        session.add(exam)
        session.flush()
        folder = tmp_path / "uploads" / str(exam.id)
        folder.mkdir(parents=True)
        scan = np.arange(1200 * 850, dtype=np.uint8).reshape(1200, 850)
        image_path = folder / "sheet.png"
        assert cv2.imwrite(str(image_path), scan)
        sheet = AnswerSheet(exam_id=exam.id, source_file=str(image_path.relative_to(tmp_path)))
        session.add(sheet)
        session.commit()

        monkeypatch.setattr(
            grading,
            "preprocess_sheet",
            lambda path: SimpleNamespace(warped_gray=scan),
        )
        app = FastAPI()
        app.include_router(grading.router)
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            yield client, session, sheet, scan
    engine.dispose()


@pytest.mark.parametrize(
    "endpoint,crop_function",
    [
        ("subject-code-image", grading.crop_subject_code_box),
        ("handwritten-student-id-image", grading.crop_student_id_handwritten),
    ],
)
def test_review_shows_the_correct_field(review_app, endpoint, crop_function):
    client, session, sheet, scan = review_app
    response = client.get(f"/api/sheets/{sheet.id}/{endpoint}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    actual = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(actual, crop_function(scan, grading._TEMPLATE))


@pytest.mark.parametrize(
    "endpoint", ["image", "subject-code-image", "handwritten-student-id-image"]
)
def test_review_rejects_missing_sheet(review_app, endpoint):
    client, session, sheet, scan = review_app
    response = client.get(f"/api/sheets/99999/{endpoint}")
    assert response.status_code == 404
    assert response.json()["detail"] == "sheet not found"


@pytest.mark.parametrize(
    "endpoint", ["image", "subject-code-image", "handwritten-student-id-image"]
)
def test_review_cannot_read_files_outside_uploads(review_app, endpoint, tmp_path):
    client, session, sheet, scan = review_app
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not an uploaded sheet")
    sheet.source_file = "outside.png"
    session.commit()
    response = client.get(f"/api/sheets/{sheet.id}/{endpoint}")
    assert response.status_code == 404
    assert response.json()["detail"] == "image file not found"


@pytest.mark.parametrize("endpoint", ["subject-code-image", "handwritten-student-id-image"])
def test_review_reports_failed_alignment(review_app, endpoint, monkeypatch):
    client, session, sheet, scan = review_app
    monkeypatch.setattr(
        grading,
        "preprocess_sheet",
        lambda path: SimpleNamespace(warped_gray=None),
    )
    response = client.get(f"/api/sheets/{sheet.id}/{endpoint}")
    assert response.status_code == 422
