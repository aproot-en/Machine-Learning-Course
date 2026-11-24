from io import BytesIO

from openpyxl import load_workbook
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from exam_src.analytics.export_excel import build_workbook, export_combined_to_bytes
from exam_src.db.models import Base, Exam, Result


def test_score_export_uses_selected_weight_and_omits_incomplete_columns():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        exam = Exam(title="export", year=2569, num_questions=2, choices="ก,ข,ค,ง")
        session.add(exam)
        session.flush()
        session.add(
            Result(
                exam_id=exam.id,
                student_id="6500000001",
                raw_score=1,
                max_score=2,
                percentage=50,
                grade="D",
                correct_count=1,
                incorrect_count=1,
                blank_count=0,
                cancelled_count=0,
                review_pending_count=0,
            )
        )
        session.commit()

        workbook = build_workbook(session, exam.id, score_weight_percent=25)
        score_sheet = workbook["คะแนน"]
        headers = [cell.value for cell in score_sheet[1]]

        assert headers == [
            "รหัสนักศึกษา",
            "คะแนนเต็ม",
            "คะแนนดิบ",
            "คิดเป็น (25%)",
        ]
        assert "%" not in headers
        assert "เกรด" not in headers
        assert "ไม่ตอบ" not in headers
        assert "ยกเลิก" not in headers
        assert "รอตรวจ" not in headers
        assert score_sheet.cell(row=2, column=4).value == 12.5
    finally:
        session.close()


def test_combined_export_creates_one_custom_sheet_per_exam():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        exam_1 = Exam(
            title="section one", year=2569, num_questions=2, choices="ก,ข,ค,ง"
        )
        exam_2 = Exam(
            title="section two", year=2569, num_questions=2, choices="ก,ข,ค,ง"
        )
        session.add_all([exam_1, exam_2])
        session.flush()
        session.add_all(
            [
                Result(
                    exam_id=exam_1.id,
                    student_id="student-1",
                    raw_score=1,
                    max_score=2,
                    percentage=50,
                    grade="D",
                    correct_count=1,
                    incorrect_count=1,
                    blank_count=0,
                    cancelled_count=0,
                    review_pending_count=0,
                ),
                Result(
                    exam_id=exam_2.id,
                    student_id="student-2",
                    raw_score=2,
                    max_score=2,
                    percentage=100,
                    grade="A",
                    correct_count=2,
                    incorrect_count=0,
                    blank_count=0,
                    cancelled_count=0,
                    review_pending_count=0,
                ),
            ]
        )
        session.commit()

        content = export_combined_to_bytes(
            session,
            [
                (exam_1.id, "sec1", 30),
                (exam_2.id, "sec 2", 20),
            ],
        )
        workbook = load_workbook(BytesIO(content), data_only=True)

        assert workbook.sheetnames == ["sec1", "sec 2"]
        assert [cell.value for cell in workbook["sec1"][1]] == [
            "รหัสนักศึกษา",
            "คะแนนเต็ม",
            "คะแนนดิบ",
            "คิดเป็น (30%)",
        ]
        assert workbook["sec1"].cell(row=2, column=4).value == 15
        assert workbook["sec 2"].cell(row=2, column=4).value == 20
    finally:
        session.close()
