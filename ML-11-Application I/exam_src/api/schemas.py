"""Request and response fields for the API."""

import datetime as _dt

from pydantic import BaseModel, Field, field_validator, model_validator


class ExamCreateRequest(BaseModel):
    title: str
    year: int = Field(
        default_factory=lambda: _dt.datetime.now().year + 543
    )  ## พ.ศ., matches the Thai UI
    num_questions: int = Field(default=100, gt=0, le=500)
    choices: list[str] = Field(default=["ก", "ข", "ค", "ง"])
    negative_marking: bool = False
    is_answer_key_template: bool = False


class ExamRenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def _clean_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("exam title cannot be blank")
        return value


class ExamOut(BaseModel):
    id: int
    title: str
    year: int
    num_questions: int
    choices: str
    negative_marking: bool
    is_answer_key_template: bool
    created_at: _dt.datetime

    @field_validator("created_at")
    @classmethod
    def _assume_utc(cls, v: _dt.datetime) -> _dt.datetime:
        ## SQLite drops timezone information. Restore UTC before sending dates to the browser.
        if v.tzinfo is None:
            return v.replace(tzinfo=_dt.timezone.utc)
        else:
            return v


class AnswerKeySourceOut(BaseModel):
    id: int
    title: str
    year: int
    num_questions: int
    choices: str
    answer_count: int
    is_answer_key_template: bool


class AnswerKeyEntry(BaseModel):
    """One row of the answer-key entry grid the teacher fills in."""

    question_no: int = Field(gt=0)
    correct_answers: list[str] = Field(min_length=1, max_length=1)
    score: float = Field(default=1.0, ge=0)
    is_cancelled: bool = False
    clo_id: int | None = None


class AnswerKeySubmitRequest(BaseModel):
    entries: list[AnswerKeyEntry]


class AnswerKeyEntryOut(BaseModel):
    question_no: int
    correct_answers: list[str]
    score: float
    is_cancelled: bool
    clo_id: int | None = None


class ReviewSubmitRequest(BaseModel):
    final_answer: str | None = None  ## None = confirmed blank
    note: str = ""


class GradeRequest(BaseModel):
    negative_marking: bool | None = None  ## None = use the exam's own setting
    grading: str = "criterion"  ## "criterion" | "norm"


class StudentIdUpdateRequest(BaseModel):
    student_id: str = Field(min_length=1, max_length=20)


class CombinedExportExam(BaseModel):
    exam_id: int = Field(gt=0)
    sheet_name: str = Field(min_length=1, max_length=31)
    weight_percent: float = Field(default=30, ge=0, le=100)

    @field_validator("sheet_name")
    @classmethod
    def _valid_excel_sheet_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("sheet name cannot be blank")
        if any(char in value for char in "[]:*?/\\"):
            raise ValueError("sheet name contains a character Excel does not allow")
        return value


class CombinedExportRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=150)
    exams: list[CombinedExportExam] = Field(min_length=1, max_length=100)

    @field_validator("filename")
    @classmethod
    def _clean_filename(cls, value: str) -> str:
        value = value.strip()
        if value.lower().endswith(".xlsx"):
            value = value[:-5].rstrip()
        if not value:
            raise ValueError("filename cannot be blank")
        return value

    @model_validator(mode="after")
    def _unique_exams_and_sheets(self):
        exam_ids = [item.exam_id for item in self.exams]
        if len(exam_ids) != len(set(exam_ids)):
            raise ValueError("an exam can only be exported once")
        sheet_names = [item.sheet_name.casefold() for item in self.exams]
        if len(sheet_names) != len(set(sheet_names)):
            raise ValueError("sheet names must be unique")
        return self
