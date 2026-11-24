import pytest

from exam_src.api import deps
from exam_src.api.routers import grading


def test_shared_mode_requires_no_login():
    assert deps.AUTH_ENABLED is False
    assert deps.get_current_user() == {"username": "shared-user", "role": "admin"}


def test_grading_path_must_belong_to_requested_exam(tmp_path, monkeypatch):
    project = tmp_path / "project"
    exam_dir = project / "exam_reports" / "uploads" / "7"
    other_dir = project / "exam_reports" / "uploads" / "8"
    exam_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    valid = exam_dir / "sheet.png"
    other = other_dir / "sheet.png"
    valid.write_bytes(b"image")
    other.write_bytes(b"image")

    monkeypatch.setattr(grading, "PROJECT_ROOT", project)
    monkeypatch.setattr(grading, "UPLOAD_DIR", project / "exam_reports" / "uploads")

    relative = str(valid.relative_to(project))
    assert grading._resolve_exam_upload(7, relative) == valid.resolve()
    with pytest.raises(ValueError):
        grading._resolve_exam_upload(7, str(other.relative_to(project)))
    with pytest.raises(ValueError):
        grading._resolve_exam_upload(7, "../../outside.png")
