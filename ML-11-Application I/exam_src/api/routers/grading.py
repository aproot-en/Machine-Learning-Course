"""Upload sheets, run saved grading jobs, and review results."""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy.orm import Session

from exam_src.api.deps import (
    get_current_user,
    get_db,
    get_exam_or_404,
    get_sheet_or_404,
)
from exam_src.api.schemas import (
    GradeRequest,
    ReviewSubmitRequest,
    StudentIdUpdateRequest,
)
from exam_src.db.models import (
    AnswerKey,
    AnswerSheet,
    AuditLog,
    Exam,
    ExtractedAnswer,
    GradingJob,
)
from exam_src.db.session import get_session
from exam_src.scoring.extraction_store import (
    get_review_queue,
    save_sheet_result,
    submit_review,
)
from exam_src.scoring.scoring import score_exam
from exam_src.vision.answer_detection import (
    crop_student_id_handwritten,
    crop_subject_code_box,
    load_template,
)
from exam_src.vision.pipeline import process_sheet
from exam_src.vision.preprocess import process as preprocess_sheet

_TEMPLATE = load_template()

router = APIRouter(prefix="/api", tags=["grading"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPLOAD_DIR = PROJECT_ROOT / "exam_reports/uploads"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  ## STEP 13.2: 50 MB per file
ALLOWED_MAGIC_PREFIXES = [b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff"]  ## PNG, JPEG

_GRADING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aegs-grading")


def _looks_like_allowed_image(head: bytes) -> bool:
    """Check the file signature rather than trusting its extension."""
    return any(head.startswith(sig) for sig in ALLOWED_MAGIC_PREFIXES)


def _resolve_exam_upload(exam_id: int, relative_path: str) -> Path:
    """Accept only an existing file directly inside this exam's upload folder."""
    exam_dir = (UPLOAD_DIR / str(exam_id)).resolve()
    candidate = (PROJECT_ROOT / relative_path).resolve()
    if candidate.parent != exam_dir or not candidate.is_file():
        raise ValueError("file path is not a valid upload for this exam")
    return candidate


@router.post("/exams/{exam_id}/upload")
async def upload_sheets(
    exam_id: int,
    files: list[UploadFile],
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    get_exam_or_404(db, exam_id)

    exam_dir = UPLOAD_DIR / str(exam_id)
    exam_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        head = await f.read(16)
        if not _looks_like_allowed_image(head):
            raise HTTPException(
                400,
                f"{f.filename}: not a recognized PNG/JPEG file (checked magic number)",
            )
        await f.seek(0)

        size = 0
        safe_name = Path(f.filename or "upload").name
        if not safe_name or safe_name in {".", ".."}:
            raise HTTPException(400, "invalid upload filename")
        dest = exam_dir / f"{uuid.uuid4().hex}_{safe_name}"
        with open(dest, "wb") as out:
            while chunk := await f.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        413,
                        f"{f.filename} exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit",
                    )
                out.write(chunk)
        saved.append(str(dest.relative_to(PROJECT_ROOT)))

    return {"exam_id": exam_id, "saved_files": saved}


def _run_grading_job(job_id: str):
    session = get_session()
    try:
        job = session.get(GradingJob, job_id)
        if job is None or job.status == "done":
            return
        job.status = "running"
        job.error = None
        session.commit()

        file_paths = json.loads(job.file_paths)
        for i, rel_path in enumerate(file_paths):
            if i < job.progress:
                continue
            abs_path = _resolve_exam_upload(job.exam_id, rel_path)

            ## Makes restart recovery idempotent if the process stopped after
            ## saving a sheet but just before updating the job progress.
            already_saved = (
                session.query(AnswerSheet)
                .filter_by(
                    exam_id=job.exam_id,
                    source_file=rel_path,
                )
                .one_or_none()
            )
            if already_saved is None:
                result = process_sheet(str(abs_path))
                save_sheet_result(session, job.exam_id, rel_path, result)

            job = session.get(GradingJob, job_id)
            job.progress = i + 1
            session.commit()

        ## Calculate scores as soon as all sheets are processed.
        exam = session.get(Exam, job.exam_id)
        if session.query(AnswerKey).filter_by(exam_id=job.exam_id).count():
            score_exam(
                session,
                job.exam_id,
                grading="criterion",
                negative_marking=exam.negative_marking,
            )

        job = session.get(GradingJob, job_id)
        job.status = "done"
        session.commit()
    except Exception as e:  ## noqa: BLE001 -- surface any failure to the polling client
        session.rollback()
        job = session.get(GradingJob, job_id)
        if job is not None:
            job.status = "error"
            job.error = str(e)
            session.commit()
    finally:
        session.close()


def resume_incomplete_jobs() -> int:
    """Queue unfinished jobs after application restart."""
    session = get_session()
    try:
        jobs = (
            session.query(GradingJob)
            .filter(GradingJob.status.in_(("queued", "running")))
            .all()
        )
        job_ids = [job.id for job in jobs]
        for job in jobs:
            job.status = "queued"
            job.error = None
        session.commit()
    finally:
        session.close()
    for job_id in job_ids:
        _GRADING_EXECUTOR.submit(_run_grading_job, job_id)
    return len(job_ids)


@router.post("/exams/{exam_id}/grade")
def start_grading(
    exam_id: int,
    file_paths: list[str],
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Queue the relative file paths returned by the upload endpoint."""
    get_exam_or_404(db, exam_id)
    if not file_paths:
        raise HTTPException(400, "no uploaded files supplied")
    try:
        for path in file_paths:
            _resolve_exam_upload(exam_id, path)
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc

    job_id = uuid.uuid4().hex
    db.add(
        GradingJob(
            id=job_id,
            exam_id=exam_id,
            file_paths=json.dumps(file_paths),
            status="queued",
            progress=0,
            total=len(file_paths),
        )
    )
    db.commit()
    _GRADING_EXECUTOR.submit(_run_grading_job, job_id)
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
def job_status(
    job_id: str, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    job = db.get(GradingJob, job_id)
    if job is None:
        raise HTTPException(404, "unknown job_id")
    payload = {"status": job.status, "progress": job.progress, "total": job.total}
    if job.error:
        payload["error"] = job.error
    if (
        job.status == "done"
        and db.query(AnswerKey).filter_by(exam_id=job.exam_id).count() == 0
    ):
        payload["warning"] = (
            "ตรวจคำตอบแล้ว แต่ยังคำนวณคะแนนไม่ได้เพราะข้อสอบนี้ยังไม่มีเฉลย"
        )
    return payload


def _sheet_image_path(sheet: AnswerSheet) -> Path:
    path = (PROJECT_ROOT / sheet.source_file).resolve()
    if UPLOAD_DIR.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "image file not found")
    return path


def _sheet_crop_response(
    sheet: AnswerSheet, crop_function, field_name: str
) -> Response:
    result = preprocess_sheet(str(_sheet_image_path(sheet)))
    if result.warped_gray is None:
        raise HTTPException(
            422, "could not perspective-correct this sheet (failed quality gate)"
        )

    crop = crop_function(result.warped_gray, _TEMPLATE)
    ok, image = cv2.imencode(".png", crop)
    if not ok:
        raise HTTPException(500, f"failed to encode {field_name} crop")
    return Response(content=image.tobytes(), media_type="image/png")


@router.get("/sheets/{sheet_id}/image")
def sheet_image(
    sheet_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    """Show the original scan for manual review."""
    sheet = get_sheet_or_404(db, sheet_id)
    return FileResponse(_sheet_image_path(sheet))


@router.get("/sheets/{sheet_id}/subject-code-image")
def sheet_subject_code_image(
    sheet_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    """Show the subject-code field as an image; no OCR is performed."""
    sheet = get_sheet_or_404(db, sheet_id)
    return _sheet_crop_response(sheet, crop_subject_code_box, "subject-code")


@router.get("/sheets/{sheet_id}/handwritten-student-id-image")
def sheet_handwritten_student_id_image(
    sheet_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Show the handwritten ID above the bubbles for manual confirmation."""
    sheet = get_sheet_or_404(db, sheet_id)
    return _sheet_crop_response(
        sheet, crop_student_id_handwritten, "handwritten student-id"
    )


@router.put("/sheets/{sheet_id}/student-id")
def update_student_id(
    sheet_id: int,
    body: StudentIdUpdateRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    """Save a corrected student ID and record the change in the audit log."""
    sheet = get_sheet_or_404(db, sheet_id)

    old_value = sheet.student_id
    sheet.student_id = body.student_id
    sheet.student_id_status = "OK"
    db.add(
        AuditLog(
            user=user["username"],
            action="correct_student_id",
            entity="answer_sheet",
            entity_id=sheet_id,
            old_value=old_value,
            new_value=body.student_id,
        )
    )

    ## Remove the old result, including the placeholder used for an unreadable ID.
    from exam_src.db.models import Result

    stale_key = old_value or f"UNKNOWN-{sheet.id}"
    stale_result = (
        db.query(Result)
        .filter_by(exam_id=sheet.exam_id, student_id=stale_key)
        .one_or_none()
    )
    if stale_result is not None:
        db.delete(stale_result)

    db.commit()
    return {"sheet_id": sheet_id, "student_id": sheet.student_id}


@router.get("/exams/{exam_id}/sheets")
def list_sheets(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    """List the sheets available for review."""
    sheets = (
        db.query(AnswerSheet).filter_by(exam_id=exam_id).order_by(AnswerSheet.id).all()
    )
    response_items = []
    for s in sheets:
        response_items.append(
            {
                "id": s.id,
                "source_file": s.source_file,
                "student_id": s.student_id,
                "student_id_best_guess": s.student_id_best_guess,
                "student_id_status": s.student_id_status,
            }
        )
    return response_items


@router.get("/sheets/{sheet_id}/answers")
def sheet_answers(
    sheet_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    """Return every answer on a sheet in question order, including unflagged answers."""
    get_sheet_or_404(db, sheet_id)
    answers = (
        db.query(ExtractedAnswer)
        .filter_by(answer_sheet_id=sheet_id)
        .order_by(ExtractedAnswer.question_no)
        .all()
    )
    response_items = []
    for a in answers:
        response_items.append(
            {
                "id": a.id,
                "question_no": a.question_no,
                "predicted_answer": a.predicted_answer,
                "final_answer": a.final_answer,
                "confidence": a.confidence,
                "status": a.status,
                "fill_ratios": a.fill_ratios,
                "already_reviewed": a.review is not None,
            }
        )
    return response_items


@router.get("/exams/{exam_id}/review-queue")
def review_queue(
    exam_id: int, db: Session = Depends(get_db), user: dict = Depends(get_current_user)
):
    items = get_review_queue(db, exam_id)
    response_items = []
    for item in items:
        response_items.append(
            {
                "id": item.id,
                "answer_sheet_id": item.answer_sheet_id,
                "source_file": item.sheet.source_file,
                "student_id": item.sheet.student_id,
                "question_no": item.question_no,
                "predicted_answer": item.predicted_answer,
                "confidence": item.confidence,
                "status": item.status,
                "fill_ratios": item.fill_ratios,
                "already_reviewed": item.review is not None,
            }
        )
    return response_items


@router.post("/review/{answer_id}")
def review(
    answer_id: int,
    body: ReviewSubmitRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    try:
        result = submit_review(
            db,
            answer_id,
            body.final_answer,
            reviewed_by=user["username"],
            note=body.note,
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    ## A manual decision overrides the automatic best guess. Refresh the
    ## exam result immediately so the score page never needs a separate click.
    exam_id = result.extracted.sheet.exam_id
    exam = db.get(Exam, exam_id)
    score_exam(db, exam_id, grading="criterion", negative_marking=exam.negative_marking)
    return {
        "extracted_answer_id": answer_id,
        "final_answer": result.final_answer,
        "score_recomputed": True,
    }


@router.post("/exams/{exam_id}/recompute")
def recompute(
    exam_id: int,
    body: GradeRequest,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    exam = get_exam_or_404(db, exam_id)
    if db.query(AnswerKey).filter_by(exam_id=exam_id).count() == 0:
        raise HTTPException(
            409, "ยังไม่มีเฉลยข้อสอบ กรุณากรอกหรือคัดลอกเฉลยก่อนคำนวณคะแนน"
        )
    if body.negative_marking is None:
        negative_marking = exam.negative_marking
    else:
        negative_marking = body.negative_marking
    breakdowns = score_exam(
        db, exam_id, grading=body.grading, negative_marking=negative_marking
    )
    return {"scored_sheets": len(breakdowns)}
