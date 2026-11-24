"""Database tables for exams, sheets, reviews, scores, and grading jobs."""

import datetime as dt
import json

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Exam(Base):
    __tablename__ = "exams"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    year: Mapped[int] = mapped_column(
        Integer, default=lambda: _now().year + 543
    )  # พ.ศ., matches the Thai UI
    num_questions: Mapped[int] = mapped_column(Integer, default=100)
    choices: Mapped[str] = mapped_column(String(50), default="ก,ข,ค,ง")
    negative_marking: Mapped[bool] = mapped_column(Boolean, default=False)
    is_answer_key_template: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    answer_keys: Mapped[list["AnswerKey"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan"
    )
    sheets: Mapped[list["AnswerSheet"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan"
    )
    results: Mapped[list["Result"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan"
    )
    clos: Mapped[list["CLO"]] = relationship(back_populates="exam", cascade="all, delete-orphan")
    grading_jobs: Mapped[list["GradingJob"]] = relationship(
        back_populates="exam", cascade="all, delete-orphan"
    )


class GradingJob(Base):
    """Durable background-job state so grading survives process restarts."""

    __tablename__ = "grading_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"), index=True)
    file_paths: Mapped[str] = mapped_column(Text)  # JSON array of validated relative paths
    status: Mapped[str] = mapped_column(String(20), default="queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    exam: Mapped[Exam] = relationship(back_populates="grading_jobs")


class AnswerKey(Base):
    """One question's answers, points, cancellation status, and optional CLO."""

    __tablename__ = "answer_keys"
    __table_args__ = (UniqueConstraint("exam_id", "question_no"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    question_no: Mapped[int] = mapped_column(Integer)
    correct_answers: Mapped[str] = mapped_column(String(20))
    score: Mapped[float] = mapped_column(Float, default=1.0)
    is_cancelled: Mapped[bool] = mapped_column(Boolean, default=False)
    clo_id: Mapped[int] = mapped_column(ForeignKey("clos.id"), nullable=True)

    exam: Mapped[Exam] = relationship(back_populates="answer_keys")
    clo: Mapped["CLO"] = relationship(back_populates="questions")


class CLO(Base):
    """A learning outcome linked to an exam's questions."""

    __tablename__ = "clos"

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    code: Mapped[str] = mapped_column(String(20))
    description: Mapped[str] = mapped_column(Text, default="")
    pass_threshold_pct: Mapped[float] = mapped_column(Float, default=60.0)

    exam: Mapped[Exam] = relationship(back_populates="clos")
    questions: Mapped[list[AnswerKey]] = relationship(back_populates="clo")


class Student(Base):
    """Optional student names for the results roster."""

    __tablename__ = "students"

    id: Mapped[int] = mapped_column(primary_key=True)
    student_code: Mapped[str] = mapped_column(String(20), unique=True)
    name: Mapped[str] = mapped_column(String(200), default="")


class User(Base):
    """Stored user records; the local app currently uses a shared identity."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(20), default="teacher")  # admin | teacher | ta
    email: Mapped[str] = mapped_column(String(200), nullable=True)


class QuestionStats(Base):
    """Saved question statistics for an exam."""

    __tablename__ = "question_stats"
    __table_args__ = (UniqueConstraint("exam_id", "question_no"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    question_no: Mapped[int] = mapped_column(Integer)
    correct_rate: Mapped[float] = mapped_column(Float)
    difficulty_index: Mapped[float] = mapped_column(Float)
    discrimination_index: Mapped[float] = mapped_column(Float, nullable=True)
    distractor_analysis: Mapped[str] = mapped_column(Text, default="{}")  # JSON


class AnswerSheet(Base):
    """One uploaded sheet and its detected student ID and quality status."""

    __tablename__ = "answer_sheets"
    __table_args__ = (Index("ix_answer_sheets_exam_student", "exam_id", "student_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    source_file: Mapped[str] = mapped_column(String(500))
    student_id: Mapped[str] = mapped_column(String(20), nullable=True)
    student_id_best_guess: Mapped[str] = mapped_column(String(20), nullable=True)
    student_id_confidence: Mapped[float] = mapped_column(Float, nullable=True)
    student_id_status: Mapped[str] = mapped_column(String(20), default="OK")
    quality_passed: Mapped[bool] = mapped_column(Boolean, default=True)
    quality_flags: Mapped[str] = mapped_column(Text, default="")
    processed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    exam: Mapped[Exam] = relationship(back_populates="sheets")
    answers: Mapped[list["ExtractedAnswer"]] = relationship(
        back_populates="sheet", cascade="all, delete-orphan"
    )


class ExtractedAnswer(Base):
    """The detected answer and mark measurements for one question."""

    __tablename__ = "extracted_answers"
    __table_args__ = (UniqueConstraint("answer_sheet_id", "question_no"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    answer_sheet_id: Mapped[int] = mapped_column(ForeignKey("answer_sheets.id"))
    question_no: Mapped[int] = mapped_column(Integer)
    predicted_answer: Mapped[str] = mapped_column(String(10), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20))  # OK | REVIEW | BLANK | MULTIPLE | ERROR
    fill_ratios: Mapped[str] = mapped_column(Text, default="")  # JSON string, for the review UI

    sheet: Mapped[AnswerSheet] = relationship(back_populates="answers")
    review: Mapped["ReviewedAnswer"] = relationship(
        back_populates="extracted", uselist=False, cascade="all, delete-orphan"
    )

    @property
    def final_answer(self) -> str | None:
        """Use the manual correction first, then the detected answer or stored best guess."""
        if self.review is not None:
            return self.review.final_answer
        if self.predicted_answer is not None:
            return self.predicted_answer
        # Backward compatibility for rows extracted before ambiguous answers
        # retained their best guess: recover it from the stored fill ratios.
        if self.status in {"REVIEW", "MULTIPLE"} and self.fill_ratios:
            try:
                ratios = json.loads(self.fill_ratios)
                if ratios:
                    return max(ratios, key=ratios.get)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return None


class ReviewedAnswer(Base):
    """A teacher's answer correction. None means a confirmed blank."""

    __tablename__ = "reviewed_answers"

    id: Mapped[int] = mapped_column(primary_key=True)
    extracted_answer_id: Mapped[int] = mapped_column(
        ForeignKey("extracted_answers.id"), unique=True
    )
    final_answer: Mapped[str] = mapped_column(String(10), nullable=True)  # None = confirmed BLANK
    reviewed_by: Mapped[str] = mapped_column(String(100))
    reviewed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    note: Mapped[str] = mapped_column(Text, default="")

    extracted: Mapped[ExtractedAnswer] = relationship(back_populates="review")


class AuditLog(Base):
    """Who changed a value, when, and what it changed from and to."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user: Mapped[str] = mapped_column(String(100))
    action: Mapped[str] = mapped_column(String(50))
    entity: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[int] = mapped_column(Integer)
    old_value: Mapped[str] = mapped_column(Text, nullable=True)
    new_value: Mapped[str] = mapped_column(Text, nullable=True)
    timestamp: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class Result(Base):
    """One student's score for one exam."""

    __tablename__ = "results"
    __table_args__ = (UniqueConstraint("exam_id", "student_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id"))
    student_id: Mapped[str] = mapped_column(String(20))
    raw_score: Mapped[float] = mapped_column(Float)
    max_score: Mapped[float] = mapped_column(Float)
    percentage: Mapped[float] = mapped_column(Float)
    grade: Mapped[str] = mapped_column(String(5), nullable=True)
    correct_count: Mapped[int] = mapped_column(Integer)
    incorrect_count: Mapped[int] = mapped_column(Integer)
    blank_count: Mapped[int] = mapped_column(Integer)
    cancelled_count: Mapped[int] = mapped_column(Integer, default=0)
    review_pending_count: Mapped[int] = mapped_column(Integer, default=0)
    computed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)

    exam: Mapped[Exam] = relationship(back_populates="results")
