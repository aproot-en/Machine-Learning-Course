"""Existing sheets and queued jobs must survive the project-folder rename."""

import importlib
import json

import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations


def test_stored_paths_upgrade_and_downgrade(monkeypatch):
    migration = importlib.import_module(
        "exam_migrations.versions.b92d0c48e617_prefix_project_folder_paths"
    )
    engine = sa.create_engine("sqlite://")
    metadata = sa.MetaData()
    sheets = sa.Table(
        "answer_sheets",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source_file", sa.String),
    )
    jobs = sa.Table(
        "grading_jobs",
        metadata,
        sa.Column("id", sa.String, primary_key=True),
        sa.Column("file_paths", sa.Text),
    )
    original = [
        "reports/uploads/7/sheet.png",
        "reports\\uploads\\8\\sheet.png",
        "inputFiles/sample.png",
        "data/real-data/sample.jpg",
        "unrelated/reports/sample.png",
    ]
    expected = [
        "exam_reports/uploads/7/sheet.png",
        "exam_reports\\uploads\\8\\sheet.png",
        "exam_input_files/sample.png",
        "exam_scan_data/real-data/sample.jpg",
        "unrelated/reports/sample.png",
    ]
    try:
        with engine.begin() as connection:
            metadata.create_all(connection)
            connection.execute(
                sheets.insert(), [{"id": i, "source_file": path} for i, path in enumerate(original)]
            )
            connection.execute(jobs.insert(), {"id": "queued", "file_paths": json.dumps(original)})
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))

            def paths():
                stored = list(
                    connection.execute(
                        sa.select(sheets.c.source_file).order_by(sheets.c.id)
                    ).scalars()
                )
                queued = json.loads(connection.execute(sa.select(jobs.c.file_paths)).scalar_one())
                return stored, queued

            migration.upgrade()
            assert paths() == (expected, expected)
            migration.upgrade()
            assert paths() == (expected, expected)
            migration.downgrade()
            assert paths() == (original, original)
    finally:
        engine.dispose()
