"""Scores, statistics, and Excel downloads."""

import re
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.orm import Session

from exam_src.analytics.export_excel import export_combined_to_bytes, export_to_bytes
from exam_src.analytics.item_analysis import (
    analyze_clos,
    analyze_questions,
    exam_summary_stats,
)
from exam_src.api.deps import get_current_user, get_db, get_exam_or_404
from exam_src.api.schemas import CombinedExportRequest
from exam_src.db.models import AnswerKey, AnswerSheet, Exam, ExtractedAnswer, Result

router = APIRouter(prefix="/api/exams", tags=["results"])


@router.post("/export/combined")
def export_combined(
    body: CombinedExportRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    exam_ids = [item.exam_id for item in body.exams]
    existing_ids = {
        exam_id for (exam_id,) in db.query(Exam.id).filter(Exam.id.in_(exam_ids)).all()
    }
    missing_ids = [exam_id for exam_id in exam_ids if exam_id not in existing_ids]
    if missing_ids:
        raise HTTPException(404, f"exam not found: {missing_ids[0]}")

    selections = [
        (item.exam_id, item.sheet_name, item.weight_percent) for item in body.exams
    ]
    content = export_combined_to_bytes(db, selections)

    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", body.filename).rstrip(" .")
    safe_name = safe_name or "exam_results"
    download_name = f"{safe_name}.xlsx"
    encoded_name = quote(download_name)
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": (
                f"attachment; filename=exam_results.xlsx; filename*=UTF-8''{encoded_name}"
            )
        },
    )


@router.get("/{exam_id}/results")
def list_results(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    if db.query(AnswerKey).filter_by(exam_id=exam_id).count() == 0:
        return []
    results = (
        db.query(Result)
        .filter_by(exam_id=exam_id)
        .order_by(Result.percentage.desc())
        .all()
    )
    response_items = []
    for r in results:
        response_items.append(
            {
                "student_id": r.student_id,
                "raw_score": r.raw_score,
                "max_score": r.max_score,
                "percentage": round(r.percentage, 1),
                "grade": r.grade,
                "correct": r.correct_count,
                "incorrect": r.incorrect_count,
                "blank": r.blank_count,
                "review_pending": r.review_pending_count,
            }
        )
    return response_items


@router.get("/{exam_id}/results/{student_id}")
def result_detail(
    exam_id: int,
    student_id: str,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    result = (
        db.query(Result).filter_by(exam_id=exam_id, student_id=student_id).one_or_none()
    )
    if result is None:
        raise HTTPException(404, "no result for this student on this exam")
    sheet = (
        db.query(AnswerSheet)
        .filter_by(exam_id=exam_id, student_id=student_id)
        .one_or_none()
    )
    key_by_question = {
        k.question_no: k for k in db.query(AnswerKey).filter_by(exam_id=exam_id).all()
    }

    answers = []
    if sheet is not None:
        for a in (
            db.query(ExtractedAnswer)
            .filter_by(answer_sheet_id=sheet.id)
            .order_by(ExtractedAnswer.question_no)
        ):
            key = key_by_question.get(a.question_no)
            if key:
                correct_answer = key.correct_answers
            else:
                correct_answer = None
            final_answer = a.final_answer
            if key is None or key.is_cancelled or final_answer is None:
                is_correct = None
            else:
                is_correct = final_answer in key.correct_answers.split(",")
            answers.append(
                {
                    "question_no": a.question_no,
                    "answer": a.final_answer,
                    "status": a.status,
                    "correct_answer": correct_answer,
                    "is_cancelled": bool(key and key.is_cancelled),
                    "is_correct": is_correct,
                    "provisional": a.status in ("REVIEW", "MULTIPLE")
                    and a.review is None,
                }
            )
    return {
        "student_id": student_id,
        "raw_score": result.raw_score,
        "max_score": result.max_score,
        "percentage": round(result.percentage, 1),
        "grade": result.grade,
        "answers": answers,
    }


@router.get("/{exam_id}/analytics")
def analytics(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    get_exam_or_404(db, exam_id)
    if db.query(AnswerKey).filter_by(exam_id=exam_id).count() == 0:
        return {"n": 0}
    results = db.query(Result).filter_by(exam_id=exam_id).all()
    if not results:
        return {"n": 0}
    summary = exam_summary_stats([r.percentage for r in results])
    grade_distribution = {}
    for result in results:
        grade = result.grade
        if grade not in grade_distribution:
            grade_distribution[grade] = 0
        grade_distribution[grade] += 1
    grade_distribution = dict(sorted(grade_distribution.items()))

    return {
        "n": summary.n,
        "mean": round(summary.mean, 2),
        "median": round(summary.median, 2),
        "mode": round(summary.mode, 2) if summary.mode is not None else None,
        "sd": round(summary.sd, 2),
        "min": summary.minimum,
        "max": summary.maximum,
        "q1": summary.q1,
        "q2": summary.q2,
        "q3": summary.q3,
        "skewness": round(summary.skewness, 3),
        "kurtosis": round(summary.kurtosis, 3),
        "grade_distribution": grade_distribution,
    }


@router.get("/{exam_id}/question-stats")
def question_stats(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    response_items = []
    for q in analyze_questions(db, exam_id):
        response_items.append(
            {
                "question_no": q.question_no,
                "n_answered": q.n_answered,
                "p": round(q.difficulty_index, 3),
                "p_label": q.difficulty_label,
                "d": (
                    round(q.discrimination_index, 3)
                    if q.discrimination_index is not None
                    else None
                ),
                "d_label": q.discrimination_label,
                "distractor_counts": q.distractor_counts,
            }
        )
    return response_items


@router.get("/{exam_id}/clo-report")
def clo_report(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    response_items = []
    for c in analyze_clos(db, exam_id):
        response_items.append(
            {
                "clo_id": c.clo_id,
                "code": c.code,
                "description": c.description,
                "mean_pct": round(c.mean_pct, 1),
                "pass_rate": round(c.pass_rate * 100, 1),
                "n_students": c.n_students,
            }
        )
    return response_items


@router.get("/{exam_id}/export/excel")
def export_excel(
    exam_id: int,
    weight_percent: float = Query(30, ge=0, le=100),
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    get_exam_or_404(db, exam_id)
    content = export_to_bytes(db, exam_id, score_weight_percent=weight_percent)
    return Response(
        content=content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="exam_{exam_id}_results.xlsx"'
        },
    )
