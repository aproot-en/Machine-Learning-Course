"""add answer key template type

Revision ID: e8714e3266c2
Revises: c6e61c3a14a1
Create Date: 2026-10-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e8714e3266c2"
down_revision: Union[str, Sequence[str], None] = "c6e61c3a14a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Existing records were created and managed as exams, so keep them as
    # exams. Only records created through the dedicated answer-key flow from
    # this revision onward are marked as templates.
    with op.batch_alter_table("exams") as batch_op:
        batch_op.add_column(
            sa.Column(
                "is_answer_key_template",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("exams") as batch_op:
        batch_op.drop_column("is_answer_key_template")
