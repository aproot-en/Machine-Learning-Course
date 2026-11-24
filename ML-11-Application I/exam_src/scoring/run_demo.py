"""Run a sample extraction and scoring workflow."""

import json
from pathlib import Path

from exam_src.db.models import Exam
from exam_src.db.session import get_session, init_db
from exam_src.scoring.extraction_store import (
    get_review_queue,
    save_sheet_result,
    submit_review,
)
from exam_src.scoring.scoring import load_answer_key, sanity_check, score_exam
from exam_src.vision.pipeline import process_sheet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "exam_input_files/test-data-image-org"
OUTPUT_PATH = PROJECT_ROOT / "exam_reports/phase10_11_demo.json"
CHOICES = ["ก", "ข", "ค", "ง"]


def fabricated_answer_key(num_questions: int = 100) -> dict[int, str]:
    """NOT a real answer key -- cycles ก,ข,ค,ง so every choice is correct
    somewhere, purely to exercise the scoring arithmetic end to end."""
    return {q: CHOICES[(q - 1) % len(CHOICES)] for q in range(1, num_questions + 1)}


def main():
    init_db()
    session = get_session()

    exam = Exam(
        title="PHASE 10-11 validation (fabricated key, not a real exam)",
        num_questions=100,
        choices=",".join(CHOICES),
    )
    session.add(exam)
    session.commit()

    load_answer_key(session, exam.id, fabricated_answer_key(), cancelled={100})

    sheets = []
    for path in sorted(SAMPLES_DIR.glob("CCF_*.png")):
        result = process_sheet(str(path))
        sheet = save_sheet_result(session, exam.id, path.name, result)
        sheets.append(sheet)
        print(f"stored {path.name} -> sheet id {sheet.id}, student_id={sheet.student_id}")

    queue_before = get_review_queue(session, exam.id)
    print(f"\nReview queue: {len(queue_before)} items need a human decision")

    # demo: a teacher resolves the first queued item
    if queue_before:
        item = queue_before[0]
        chosen = CHOICES[0]  # arbitrary demo choice -- a real UI would let the teacher pick
        submit_review(
            session,
            item.id,
            chosen,
            reviewed_by="demo_teacher",
            note="PHASE 10-11 demo: simulated teacher review",
        )
        print(
            f"Reviewed item {item.id} (Q{item.question_no}, sheet {item.answer_sheet_id}): "
            f"model predicted {item.predicted_answer!r} ({item.status}) -> teacher set {chosen!r}"
        )

    breakdowns = score_exam(session, exam.id, grading="criterion")
    print("\nScores (SYNTHETIC -- fabricated answer key, not real grades):")
    for b in breakdowns:
        print(
            f"  {b.student_id}: {b.correct_count}/{b.max_score:.0f} correct "
            f"({b.percentage:.1f}%), {b.review_pending_count} still pending review"
        )

    report = sanity_check(session, exam.id, expected_student_count=len(sheets))
    print(f"\nSanity check: {'PASS' if report.ok else 'ISSUES FOUND'}")
    for issue in report.issues:
        print(f"  - {issue}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "note": "FABRICATED answer key -- scores here are a plumbing check, not real grades",
                "exam_id": exam.id,
                "review_queue_size_before_demo_review": len(queue_before),
                "scores": [
                    {
                        "student_id": b.student_id,
                        "correct": b.correct_count,
                        "max_score": b.max_score,
                        "percentage": round(b.percentage, 1),
                        "review_pending": b.review_pending_count,
                    }
                    for b in breakdowns
                ],
                "sanity_check": {"ok": report.ok, "issues": report.issues},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nWrote {OUTPUT_PATH}")

    session.close()


if __name__ == "__main__":
    main()
