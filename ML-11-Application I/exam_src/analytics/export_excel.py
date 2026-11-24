"""Export scores, statistics, question analysis, and CLOs to Excel."""

from io import BytesIO

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter
from sqlalchemy.orm import Session

from exam_src.analytics.item_analysis import (
    analyze_clos,
    analyze_questions,
    exam_summary_stats,
    kr20,
    reliability_label,
)
from exam_src.db.models import AnswerKey, AnswerSheet, ExtractedAnswer, Result

HEADER_FILL = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
WARNING_FILL = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
HEADER_FONT = Font(bold=True)


def _style_header(ws, row=1):
    for cell in ws[row]:
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
    ws.freeze_panes = ws.cell(row=row + 1, column=1).coordinate
    ws.auto_filter.ref = ws.dimensions


def _autosize(ws):
    for col in ws.columns:
        width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(width + 2, 40)


def _populate_score_sheet(ws, results: list[Result], score_weight_percent: float) -> None:
    weight_label = f"{score_weight_percent:g}"
    ws.append(["รหัสนักศึกษา", "คะแนนเต็ม", "คะแนนดิบ", f"คิดเป็น ({weight_label}%)"])
    for result in results:
        weighted_score = round(result.percentage * score_weight_percent / 100, 2)
        ws.append([result.student_id, result.max_score, result.raw_score, weighted_score])
    _style_header(ws)
    _autosize(ws)


def build_workbook(db: Session, exam_id: int, score_weight_percent: float = 30) -> Workbook:
    if not 0 <= score_weight_percent <= 100:
        raise ValueError("score_weight_percent must be between 0 and 100")
    results = db.query(Result).filter_by(exam_id=exam_id).order_by(Result.percentage.desc()).all()
    key_rows = db.query(AnswerKey).filter_by(exam_id=exam_id).order_by(AnswerKey.question_no).all()

    wb = Workbook()

    # --- Sheet 1: คะแนน ---
    ws1 = wb.active
    ws1.title = "คะแนน"
    _populate_score_sheet(ws1, results, score_weight_percent)

    # --- Sheet 2: สรุปสถิติ ---
    ws2 = wb.create_sheet("สรุปสถิติ")
    if results:
        summary = exam_summary_stats([r.percentage for r in results])
        rows = [
            ("จำนวนผู้เข้าสอบ", summary.n),
            ("คะแนนเฉลี่ย (%)", round(summary.mean, 2)),
            ("มัธยฐาน (%)", round(summary.median, 2)),
            ("ฐานนิยม (%)", round(summary.mode, 2) if summary.mode is not None else "-"),
            ("ส่วนเบี่ยงเบนมาตรฐาน", round(summary.sd, 2)),
            ("สูงสุด (%)", round(summary.maximum, 2)),
            ("ต่ำสุด (%)", round(summary.minimum, 2)),
            ("Q1", round(summary.q1, 2)),
            ("Q2", round(summary.q2, 2)),
            ("Q3", round(summary.q3, 2)),
            ("ความเบ้ (skewness)", round(summary.skewness, 3)),
            ("ความโด่ง (kurtosis)", round(summary.kurtosis, 3)),
        ]
        ws2.append(["รายการ", "ค่า"])
        for label, value in rows:
            ws2.append([label, value])

        grade_counts = {}
        for r in results:
            grade_counts[r.grade] = grade_counts.get(r.grade, 0) + 1
        ws2.append([])
        ws2.append(["เกรด", "จำนวนคน"])
        for grade, n in sorted(grade_counts.items()):
            ws2.append([grade, n])
    _style_header(ws2)
    _autosize(ws2)

    # --- Sheet 3: วิเคราะห์ข้อ ---
    ws3 = wb.create_sheet("วิเคราะห์ข้อ")
    ws3.append(["ข้อที่", "ตอบแล้ว", "ค่าความยาก (p)", "แปลผล", "ค่าอำนาจจำแนก (D)", "แปลผล", "ธง"])
    for qa in analyze_questions(db, exam_id):
        flag = ""
        if qa.discrimination_index is not None and qa.discrimination_index < 0:
            flag = "ตรวจสอบเฉลย!"
        elif qa.difficulty_index > 0.95 or qa.difficulty_index < 0.05:
            flag = "พิจารณาแก้ข้อสอบ"
        row = ws3.max_row + 1
        ws3.append(
            [
                qa.question_no,
                qa.n_answered,
                round(qa.difficulty_index, 2),
                qa.difficulty_label,
                round(qa.discrimination_index, 2) if qa.discrimination_index is not None else "-",
                qa.discrimination_label or "-",
                flag,
            ]
        )
        if flag:
            for cell in ws3[row]:
                cell.fill = WARNING_FILL

    if key_rows:
        answers_by_sheet = {}
        sheets = db.query(AnswerSheet).filter_by(exam_id=exam_id).all()
        for s in sheets:
            answers_by_sheet[s.id] = {
                a.question_no: a.final_answer
                for a in db.query(ExtractedAnswer).filter_by(answer_sheet_id=s.id).all()
            }
        import numpy as np

        rows_matrix = []
        for s in sheets:
            row_vec = []
            for k in key_rows:
                if k.is_cancelled:
                    continue
                ans = answers_by_sheet.get(s.id, {}).get(k.question_no)
                row_vec.append(1 if ans is not None and ans in k.correct_answers.split(",") else 0)
            rows_matrix.append(row_vec)
        if rows_matrix and len(rows_matrix) > 1:
            r_kr20 = kr20(np.array(rows_matrix))
            ws3.append([])
            ws3.append(
                [
                    "ความเชื่อมั่นของข้อสอบทั้งฉบับ (KR-20)",
                    round(r_kr20, 3) if not np.isnan(r_kr20) else "N/A",
                    reliability_label(r_kr20),
                ]
            )
    _style_header(ws3)
    _autosize(ws3)

    # --- Sheet 4: CLO ---
    ws4 = wb.create_sheet("CLO")
    ws4.append(["CLO", "คำอธิบาย", "คะแนนเฉลี่ย (%)", "% ผ่านเกณฑ์", "จำนวนนักศึกษา"])
    for clo in analyze_clos(db, exam_id):
        ws4.append(
            [
                clo.code,
                clo.description,
                round(clo.mean_pct, 1),
                round(clo.pass_rate * 100, 1),
                clo.n_students,
            ]
        )
    _style_header(ws4)
    _autosize(ws4)

    return wb


def export_to_bytes(db: Session, exam_id: int, score_weight_percent: float = 30) -> bytes:
    wb = build_workbook(db, exam_id, score_weight_percent=score_weight_percent)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_combined_to_bytes(
    db: Session,
    selections: list[tuple[int, str, float]],
) -> bytes:
    """Export one exam's score table per custom-named worksheet."""
    wb = Workbook()
    wb.remove(wb.active)
    for exam_id, sheet_name, score_weight_percent in selections:
        if not 0 <= score_weight_percent <= 100:
            raise ValueError("score_weight_percent must be between 0 and 100")
        results = (
            db.query(Result).filter_by(exam_id=exam_id).order_by(Result.percentage.desc()).all()
        )
        ws = wb.create_sheet(title=sheet_name)
        _populate_score_sheet(ws, results, score_weight_percent)

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
