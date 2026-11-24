"""Process a folder of answer-sheet images and save a report."""

import json
from pathlib import Path

from exam_src.vision.pipeline import process_sheet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DIR = PROJECT_ROOT / "exam_input_files/test-data-image-org"
OUTPUT_PATH = PROJECT_ROOT / "exam_reports/phase9_extraction_sample.json"


def main():
    sheets = []
    for path in sorted(SAMPLES_DIR.glob("CCF_*.png")):
        result = process_sheet(str(path))
        sheet = {
            "file": path.name,
            "quality_passed": result.quality.passed,
            "quality_reasons": result.quality.reasons,
        }
        if result.answers is not None:
            sheet["student_id"] = result.student_id.student_id
            sheet["student_id_status"] = result.student_id.status
            sheet["answers"] = [
                {
                    "q": a.question,
                    "answer": a.answer,
                    "confidence": a.confidence,
                    "status": a.status,
                }
                for a in result.answers
            ]
            statuses = [a.status for a in result.answers]
            sheet["status_counts"] = {s: statuses.count(s) for s in set(statuses)}
        sheets.append(sheet)
        print(
            f"{path.name}: quality_passed={sheet['quality_passed']} "
            f"student_id={sheet.get('student_id')} "
            f"status_counts={sheet.get('status_counts')}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sheets, f, indent=2, ensure_ascii=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
