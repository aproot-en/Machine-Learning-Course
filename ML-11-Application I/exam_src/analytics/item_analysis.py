"""Score summaries, question analysis, reliability, and CLO results."""

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import stats as sstats
from sqlalchemy.orm import Session

from exam_src.db.models import CLO, AnswerKey, AnswerSheet, ExtractedAnswer, Result

HIGH_LOW_GROUP_FRACTION = 0.27  ## STEP 14.3: top/bottom 27%


@dataclass
class ExamSummary:
    n: int
    mean: float
    median: float
    mode: float | None
    sd: float
    minimum: float
    maximum: float
    q1: float
    q2: float
    q3: float
    skewness: float
    kurtosis: float


def exam_summary_stats(percentages: list[float]) -> ExamSummary:
    """Summarize a nonempty list of percentage scores."""
    arr = np.asarray(percentages, dtype=float)
    if arr.size == 0:
        raise ValueError("no scores to summarize")
    if arr.size > 1:
        mode_result = sstats.mode(arr, keepdims=False)
    else:
        mode_result = None
    return ExamSummary(
        n=arr.size,
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        mode=float(mode_result.mode) if mode_result is not None else float(arr[0]),
        sd=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        minimum=float(arr.min()),
        maximum=float(arr.max()),
        q1=float(np.percentile(arr, 25)),
        q2=float(np.percentile(arr, 50)),
        q3=float(np.percentile(arr, 75)),
        skewness=float(sstats.skew(arr)) if arr.size > 2 else 0.0,
        kurtosis=float(sstats.kurtosis(arr)) if arr.size > 2 else 0.0,
    )


def difficulty_label(p: float) -> str:
    """Describe a question's proportion of correct answers."""
    if p > 0.80:
        return "ง่ายมาก"
    if p >= 0.60:
        return "ค่อนข้างง่าย"
    if p >= 0.40:
        return "เหมาะสม"
    if p >= 0.20:
        return "ค่อนข้างยาก"
    return "ยากมาก"


def discrimination_label(d: float) -> str:
    """Describe how well a question separates high and low scorers."""
    if d < 0:
        return "ผิดปกติ - ตรวจสอบเฉลย"
    if d >= 0.40:
        return "ดีมาก"
    if d >= 0.30:
        return "ดี"
    if d >= 0.20:
        return "พอใช้ ควรปรับปรุง"
    return "ต้องแก้ไข"


@dataclass
class QuestionAnalysis:
    question_no: int
    n_answered: int
    correct_rate: float
    difficulty_index: float
    difficulty_label: str
    discrimination_index: float | None
    discrimination_label: str | None
    distractor_counts: dict  ## {"high": {"ก": n, ...}, "low": {...}}


def _final_answers_by_sheet(
    db: Session, exam_id: int
) -> dict[int, dict[int, str | None]]:
    """sheet_id -> {question_no: final_answer_or_None}."""
    sheets = db.query(AnswerSheet).filter_by(exam_id=exam_id).all()
    out = {}
    for sheet in sheets:
        answers = db.query(ExtractedAnswer).filter_by(answer_sheet_id=sheet.id).all()
        out[sheet.id] = {a.question_no: a.final_answer for a in answers}
    return out


def analyze_questions(db: Session, exam_id: int) -> list[QuestionAnalysis]:
    """Calculate difficulty, discrimination, and choice counts for each active question."""
    results = db.query(Result).filter_by(exam_id=exam_id).all()
    if not results:
        return []

    ## total-score ranking -> high/low groups (STEP 14.3)
    ranked = sorted(results, key=lambda r: r.percentage, reverse=True)
    n_group = max(1, round(len(ranked) * HIGH_LOW_GROUP_FRACTION))
    high_ids = {r.student_id for r in ranked[:n_group]}
    low_ids = {r.student_id for r in ranked[-n_group:]}

    sheets = db.query(AnswerSheet).filter_by(exam_id=exam_id).all()
    sheet_by_student = {s.student_id: s for s in sheets if s.student_id}
    answers_by_sheet = _final_answers_by_sheet(db, exam_id)

    key_rows = (
        db.query(AnswerKey)
        .filter_by(exam_id=exam_id)
        .order_by(AnswerKey.question_no)
        .all()
    )
    out = []
    for key in key_rows:
        if key.is_cancelled:
            continue
        correct_choices = set(key.correct_answers.split(","))

        n_answered = 0
        n_correct = 0
        high_correct = low_correct = 0
        high_counts, low_counts = Counter(), Counter()

        for student_id, sheet in sheet_by_student.items():
            answer = answers_by_sheet.get(sheet.id, {}).get(key.question_no)
            if answer is None:
                continue
            n_answered += 1
            is_correct = answer in correct_choices
            n_correct += is_correct
            if student_id in high_ids:
                high_counts[answer] += 1
                high_correct += is_correct
            elif student_id in low_ids:
                low_counts[answer] += 1
                low_correct += is_correct

        if n_answered:
            p = n_correct / n_answered
        else:
            p = 0.0
        if n_group:
            d = (high_correct - low_correct) / n_group
        else:
            d = None

        out.append(
            QuestionAnalysis(
                question_no=key.question_no,
                n_answered=n_answered,
                correct_rate=p,
                difficulty_index=p,
                difficulty_label=difficulty_label(p),
                discrimination_index=d,
                discrimination_label=discrimination_label(d) if d is not None else None,
                distractor_counts={"high": dict(high_counts), "low": dict(low_counts)},
            )
        )
    return out


def kr20(item_correct_matrix: np.ndarray) -> float:
    """Calculate KR-20 from a student-by-question matrix of 0/1 scores."""
    k = item_correct_matrix.shape[1]
    if k < 2:
        return float("nan")
    p = item_correct_matrix.mean(axis=0)
    q = 1 - p
    total_scores = item_correct_matrix.sum(axis=1)
    variance_total = total_scores.var(ddof=1)
    if variance_total == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - (p * q).sum() / variance_total))


def reliability_label(r: float) -> str:
    if np.isnan(r):
        return "คำนวณไม่ได้ (ข้อมูลไม่พอ)"
    if r >= 0.7:
        return "ยอมรับได้ (>= 0.70)"
    else:
        return "ต่ำกว่ามาตรฐาน (< 0.70)"


@dataclass
class CloResult:
    clo_id: int
    code: str
    description: str
    mean_pct: float
    pass_rate: float
    n_students: int


def analyze_clos(db: Session, exam_id: int) -> list[CloResult]:
    """Calculate mean scores and pass rates for each learning outcome."""
    clos = db.query(CLO).filter_by(exam_id=exam_id).all()
    if not clos:
        return []

    sheets = db.query(AnswerSheet).filter_by(exam_id=exam_id).all()
    sheet_by_student = {s.student_id: s for s in sheets if s.student_id}
    answers_by_sheet = _final_answers_by_sheet(db, exam_id)
    results_by_student = {
        r.student_id: r for r in db.query(Result).filter_by(exam_id=exam_id).all()
    }

    out = []
    for clo in clos:
        clo_questions = (
            db.query(AnswerKey).filter_by(exam_id=exam_id, clo_id=clo.id).all()
        )
        if not clo_questions:
            continue
        max_score = sum(q.score for q in clo_questions if not q.is_cancelled)

        per_student_pct = []
        for student_id, sheet in sheet_by_student.items():
            if student_id not in results_by_student:
                continue
            earned = 0.0
            for q in clo_questions:
                if q.is_cancelled:
                    continue
                answer = answers_by_sheet.get(sheet.id, {}).get(q.question_no)
                if answer is not None and answer in q.correct_answers.split(","):
                    earned += q.score
            if max_score:
                pct = earned / max_score * 100
            else:
                pct = 0.0
            per_student_pct.append(pct)

        if not per_student_pct:
            continue
        arr = np.array(per_student_pct)
        out.append(
            CloResult(
                clo_id=clo.id,
                code=clo.code,
                description=clo.description,
                mean_pct=float(arr.mean()),
                pass_rate=float((arr >= clo.pass_threshold_pct).mean()),
                n_students=len(arr),
            )
        )
    return out
