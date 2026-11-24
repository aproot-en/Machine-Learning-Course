"""FastAPI dependencies for the shared, login-free local application."""

from typing import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from exam_src.db.models import AnswerSheet, Exam
from exam_src.db.session import get_session

AUTH_ENABLED = False
DEV_USER = {"username": "shared-user", "role": "admin"}


def get_db() -> Generator[Session, None, None]:
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def get_current_user() -> dict:
    """Return one shared identity; the application intentionally has no login."""
    return DEV_USER


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="admin only")
    return user


def get_exam_or_404(db: Session, exam_id: int) -> Exam:
    exam = db.get(Exam, exam_id)
    if exam is None:
        raise HTTPException(404, "exam not found")
    return exam


def get_sheet_or_404(db: Session, sheet_id: int) -> AnswerSheet:
    sheet = db.get(AnswerSheet, sheet_id)
    if sheet is None:
        raise HTTPException(404, "sheet not found")
    return sheet
