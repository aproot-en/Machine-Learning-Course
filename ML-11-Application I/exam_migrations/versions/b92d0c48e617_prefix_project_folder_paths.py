"""Update stored file paths after prefixing project folders with exam_."""

import json

import sqlalchemy as sa
from alembic import op

revision = "b92d0c48e617"
down_revision = "e8714e3266c2"
branch_labels = None
depends_on = None

# Keep both slash styles: existing databases may have been created on Windows.
FOLDERS = {
    "reports": "exam_reports",
    "inputFiles": "exam_input_files",
    "incorrectResult": "exam_incorrect_results",
    "data": "exam_scan_data",
    "compro 2596-1": "exam_compro_2596_1",
}


def _rewrite_path(path, folders):
    for old, new in folders.items():
        for separator in ("/", "\\"):
            prefix = old + separator
            if path.startswith(prefix):
                return new + path[len(old) :]
    return path


def _rewrite(folders):
    connection = op.get_bind()
    sheets = sa.table("answer_sheets", sa.column("id"), sa.column("source_file"))
    for sheet_id, path in connection.execute(sa.select(sheets.c.id, sheets.c.source_file)).all():
        rewritten = _rewrite_path(path, folders)
        if rewritten != path:
            connection.execute(
                sheets.update().where(sheets.c.id == sheet_id).values(source_file=rewritten)
            )

    jobs = sa.table("grading_jobs", sa.column("id"), sa.column("file_paths"))
    for job_id, encoded in connection.execute(sa.select(jobs.c.id, jobs.c.file_paths)).all():
        paths = json.loads(encoded)
        rewritten = [_rewrite_path(path, folders) for path in paths]
        if rewritten != paths:
            connection.execute(
                jobs.update().where(jobs.c.id == job_id).values(file_paths=json.dumps(rewritten))
            )


def upgrade():
    _rewrite(FOLDERS)


def downgrade():
    _rewrite({new: old for old, new in FOLDERS.items()})
