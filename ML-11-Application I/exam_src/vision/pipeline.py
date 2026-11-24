"""Read a sheet only after it passes image quality checks."""

from dataclasses import dataclass

from exam_src.vision.answer_detection import (
    AnswerResult,
    StudentIdResult,
    detect_answers,
    detect_student_id,
    load_template,
)
from exam_src.vision.preprocess import ProcessResult, QualityReport, process

_TEMPLATE = load_template()


@dataclass
class SheetResult:
    quality: QualityReport
    student_id: StudentIdResult | None = None
    answers: list[AnswerResult] | None = None


def process_sheet(path: str) -> SheetResult:
    result: ProcessResult = process(path)
    # Do not read answers from a scan that failed quality checks.
    if not result.quality.passed or result.warped_binary is None:
        return SheetResult(quality=result.quality)
    return SheetResult(
        quality=result.quality,
        student_id=detect_student_id(result.warped_binary, _TEMPLATE),
        answers=detect_answers(result.warped_binary, _TEMPLATE),
    )
