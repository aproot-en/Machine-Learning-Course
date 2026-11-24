"""Open the configured SQLite database and apply migrations."""

from pathlib import Path

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as _f:
    _CFG = yaml.safe_load(_f)

if _CFG["database"]["driver"] != "sqlite":
    raise NotImplementedError(
        "only sqlite is wired up so far -- config.yaml requests something else"
    )

DB_PATH = PROJECT_ROOT / _CFG["database"]["sqlite_path"]
ENGINE = create_engine(f"sqlite:///{DB_PATH}")
SessionLocal = sessionmaker(bind=ENGINE)


def init_db():
    """Apply pending Alembic migrations without changing an up-to-date database."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(PROJECT_ROOT / "exam_migrations"))
    command.upgrade(alembic_cfg, "head")


def get_session() -> Session:
    return SessionLocal()
