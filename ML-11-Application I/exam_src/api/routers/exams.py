"""Create exams and manage reusable answer keys."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from exam_src.api.deps import get_current_user, get_db, get_exam_or_404
from exam_src.api.schemas import (
    AnswerKeyEntryOut,
    AnswerKeySourceOut,
    AnswerKeySubmitRequest,
    ExamCreateRequest,
    ExamOut,
    ExamRenameRequest,
)
from exam_src.db.models import AnswerKey, AnswerSheet, Exam

router = APIRouter(prefix="/api/exams", tags=["exams"])


@router.get("", response_model=list[ExamOut])
def list_exams(
    year: int | None = None, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    query = db.query(Exam).filter(Exam.is_answer_key_template.is_(False))
    if year is not None:
        query = query.filter(Exam.year == year)
    return query.order_by(Exam.id.desc()).all()


@router.get("/years", response_model=list[int])
def list_years(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    """Distinct years that have at least one exam, newest first -- powers
    the sidebar year selector."""
    rows = (
        db.query(Exam.year)
        .filter(Exam.is_answer_key_template.is_(False))
        .distinct()
        .order_by(Exam.year.desc())
        .all()
    )
    return [r[0] for r in rows]


@router.get("/answer-key-sources", response_model=list[AnswerKeySourceOut])
def list_answer_key_sources(db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    """Previous exams that contain at least one reusable answer-key row."""
    exams = db.query(Exam).order_by(Exam.id.desc()).all()
    sources = []
    for exam in exams:
        answer_count = db.query(AnswerKey).filter_by(exam_id=exam.id).count()
        if answer_count:
            sources.append(
                AnswerKeySourceOut(
                    id=exam.id,
                    title=exam.title,
                    year=exam.year,
                    num_questions=exam.num_questions,
                    choices=exam.choices,
                    answer_count=answer_count,
                    is_answer_key_template=exam.is_answer_key_template,
                )
            )
    return sources


@router.get("/answer-key-templates", response_model=list[ExamOut])
def list_answer_key_templates(
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    return (
        db.query(Exam).filter(Exam.is_answer_key_template.is_(True)).order_by(Exam.id.desc()).all()
    )


@router.post("", response_model=ExamOut)
def create_exam(
    body: ExamCreateRequest, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    exam = Exam(
        title=body.title,
        year=body.year,
        num_questions=body.num_questions,
        choices=",".join(body.choices),
        negative_marking=body.negative_marking,
        is_answer_key_template=body.is_answer_key_template,
    )
    db.add(exam)
    db.commit()
    return exam


@router.get("/{exam_id}", response_model=ExamOut)
def get_exam(exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)):
    exam = get_exam_or_404(db, exam_id)
    return exam


@router.patch("/{exam_id}", response_model=ExamOut)
def rename_exam(
    exam_id: int,
    body: ExamRenameRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    exam = get_exam_or_404(db, exam_id)
    exam.title = body.title
    db.commit()
    db.refresh(exam)
    return exam


@router.delete("/{exam_id}")
def delete_exam(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    """Delete related database rows; keep uploaded images on disk."""
    exam = get_exam_or_404(db, exam_id)
    db.delete(exam)
    db.commit()
    return {"deleted": exam_id}


@router.post("/{exam_id}/answer-key")
def set_answer_key(
    exam_id: int,
    body: AnswerKeySubmitRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Replace the answer key and rescore any existing sheets."""
    exam = get_exam_or_404(db, exam_id)

    seen_questions = {e.question_no for e in body.entries}
    if len(seen_questions) != len(body.entries):
        raise HTTPException(400, "duplicate question_no in submitted answer key")
    valid_choices = set(exam.choices.split(","))
    for entry in body.entries:
        if entry.question_no > exam.num_questions:
            raise HTTPException(400, f"question {entry.question_no} exceeds exam question count")
        if not set(entry.correct_answers).issubset(valid_choices):
            raise HTTPException(400, f"question {entry.question_no} contains an invalid choice")

    db.query(AnswerKey).filter_by(exam_id=exam_id).delete()
    for entry in body.entries:
        db.add(
            AnswerKey(
                exam_id=exam_id,
                question_no=entry.question_no,
                correct_answers=",".join(entry.correct_answers),
                score=entry.score,
                is_cancelled=entry.is_cancelled,
                clo_id=entry.clo_id,
            )
        )
    db.commit()

    # Sheets may have been detected before a key was entered. Saving the key
    # must make scores appear immediately without another manual action.
    scored_sheets = 0
    if db.query(AnswerSheet).filter_by(exam_id=exam_id).count():
        from exam_src.scoring.scoring import score_exam

        scored_sheets = len(
            score_exam(
                db,
                exam_id,
                grading="criterion",
                negative_marking=exam.negative_marking,
            )
        )
    return {"saved": len(body.entries), "scored_sheets": scored_sheets}


@router.get("/{exam_id}/answer-key", response_model=list[AnswerKeyEntryOut])
def get_answer_key(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    rows = db.query(AnswerKey).filter_by(exam_id=exam_id).order_by(AnswerKey.question_no).all()
    return [
        AnswerKeyEntryOut(
            question_no=r.question_no,
            correct_answers=r.correct_answers.split(","),
            score=r.score,
            is_cancelled=r.is_cancelled,
            clo_id=r.clo_id,
        )
        for r in rows
    ]
