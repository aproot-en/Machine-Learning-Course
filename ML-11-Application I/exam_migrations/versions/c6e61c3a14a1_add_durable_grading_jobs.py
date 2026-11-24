"""add durable grading jobs

Revision ID: c6e61c3a14a1
Revises: af9ab19b0ec0
Create Date: 2026-10-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "c6e61c3a14a1"
down_revision: Union[str, Sequence[str], None] = "af9ab19b0ec0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "grading_jobs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("exam_id", sa.Integer(), nullable=False),
        sa.Column("file_paths", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["exam_id"], ["exams.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_grading_jobs_exam_id", "grading_jobs", ["exam_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_grading_jobs_exam_id", table_name="grading_jobs")
    op.drop_table("grading_jobs")
