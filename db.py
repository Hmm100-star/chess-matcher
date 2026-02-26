from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlsplit

from sqlalchemy import create_engine, inspect
from sqlalchemy import text
from sqlalchemy.orm import declarative_base, sessionmaker


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DB_URL = f"sqlite:///{DATA_DIR / 'chess_match.db'}"
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
ENVIRONMENT = os.getenv("FLASK_ENV", "").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

if IS_PRODUCTION and DATABASE_URL == DEFAULT_DB_URL:
    raise RuntimeError(
        "DATABASE_URL must be set to a persistent database (e.g., Supabase Postgres) "
        "when running in production."
    )

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, future=True, echo=False, connect_args=connect_args)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()

REQUIRED_SCHEMA: dict[str, set[str]] = {
    "teachers": {"id", "username", "password_hash", "created_at"},
    "classrooms": {"id", "teacher_id", "name", "created_at"},
    "students": {
        "id",
        "classroom_id",
        "name",
        "total_wins",
        "total_losses",
        "total_ties",
        "times_white",
        "times_black",
        "homework_correct",
        "homework_incorrect",
        "notes",
        "active",
    },
    "rounds": {
        "id",
        "classroom_id",
        "created_at",
        "win_weight",
        "homework_weight",
        "status",
        "homework_total_questions",
        "missing_homework_policy",
        "homework_metric_mode",
        "completion_override_reason",
    },
    "attendance": {"id", "round_id", "student_id", "status"},
    "matches": {
        "id",
        "round_id",
        "white_student_id",
        "black_student_id",
        "white_strength",
        "black_strength",
        "result",
        "notes",
        "notation_submitted_white",
        "notation_submitted_black",
        "updated_at",
    },
    "homework_entries": {
        "id",
        "match_id",
        "white_correct",
        "white_incorrect",
        "black_correct",
        "black_incorrect",
        "white_submitted",
        "black_submitted",
        "white_pct_wrong",
        "black_pct_wrong",
    },
    "audit_logs": {
        "id",
        "classroom_id",
        "round_id",
        "match_id",
        "actor_teacher_id",
        "field_name",
        "old_value",
        "new_value",
        "created_at",
    },
}

POSTGRES_COMPATIBILITY_PATCH_STATEMENTS: tuple[str, ...] = (
    "ALTER TABLE IF EXISTS students ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE",
    "ALTER TABLE IF EXISTS rounds ADD COLUMN IF NOT EXISTS status VARCHAR(50) NOT NULL DEFAULT 'open'",
    "ALTER TABLE IF EXISTS rounds ADD COLUMN IF NOT EXISTS homework_total_questions INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE IF EXISTS rounds ADD COLUMN IF NOT EXISTS missing_homework_policy VARCHAR(20) NOT NULL DEFAULT 'zero'",
    "ALTER TABLE IF EXISTS rounds ADD COLUMN IF NOT EXISTS homework_metric_mode VARCHAR(20) NOT NULL DEFAULT 'pct_correct'",
    "ALTER TABLE IF EXISTS rounds ADD COLUMN IF NOT EXISTS completion_override_reason TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE IF EXISTS matches ADD COLUMN IF NOT EXISTS notes TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE IF EXISTS matches ADD COLUMN IF NOT EXISTS notation_submitted_white BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE IF EXISTS matches ADD COLUMN IF NOT EXISTS notation_submitted_black BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE IF EXISTS matches ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW()",
    "ALTER TABLE IF EXISTS attendance ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'present'",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS white_correct INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS white_incorrect INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS black_correct INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS black_incorrect INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS white_submitted BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS black_submitted BOOLEAN NOT NULL DEFAULT FALSE",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS white_pct_wrong DOUBLE PRECISION NOT NULL DEFAULT 0.0",
    "ALTER TABLE IF EXISTS homework_entries ADD COLUMN IF NOT EXISTS black_pct_wrong DOUBLE PRECISION NOT NULL DEFAULT 0.0",
    "CREATE TABLE IF NOT EXISTS audit_logs (id SERIAL PRIMARY KEY, classroom_id INTEGER NOT NULL, round_id INTEGER NOT NULL, match_id INTEGER NOT NULL, actor_teacher_id INTEGER NOT NULL, field_name VARCHAR(100) NOT NULL, old_value TEXT NOT NULL DEFAULT '', new_value TEXT NOT NULL DEFAULT '', created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT NOW())",
    "ALTER TABLE IF EXISTS audit_logs ADD COLUMN IF NOT EXISTS actor_teacher_id INTEGER",
    "ALTER TABLE IF EXISTS audit_logs ADD COLUMN IF NOT EXISTS field_name VARCHAR(100) NOT NULL DEFAULT ''",
    "ALTER TABLE IF EXISTS audit_logs ADD COLUMN IF NOT EXISTS old_value TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE IF EXISTS audit_logs ADD COLUMN IF NOT EXISTS new_value TEXT NOT NULL DEFAULT ''",
    # Drop FK constraints that may have been added by create_all() — audit logs
    # should be a standalone table without cascading dependencies.
    "ALTER TABLE IF EXISTS audit_logs DROP CONSTRAINT IF EXISTS audit_logs_classroom_id_fkey",
    "ALTER TABLE IF EXISTS audit_logs DROP CONSTRAINT IF EXISTS audit_logs_round_id_fkey",
    "ALTER TABLE IF EXISTS audit_logs DROP CONSTRAINT IF EXISTS audit_logs_match_id_fkey",
    "ALTER TABLE IF EXISTS audit_logs DROP CONSTRAINT IF EXISTS audit_logs_actor_teacher_id_fkey",
)


def redacted_database_url(url: str | None = None) -> str:
    raw_url = url or DATABASE_URL
    try:
        parsed = urlsplit(raw_url)
    except Exception:
        return "<invalid database url>"

    if not parsed.scheme:
        return "<invalid database url>"

    hostname = parsed.hostname or ""
    username = parsed.username or ""
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""

    auth = ""
    if username:
        auth = f"{username}:***@"
    elif hostname:
        auth = ""

    return f"{parsed.scheme}://{auth}{hostname}{port}{path}{query}"


def database_url_warnings(url: str | None = None) -> list[str]:
    raw_url = url or DATABASE_URL
    warnings: list[str] = []

    try:
        parsed = urlsplit(raw_url)
    except Exception:
        return ["DATABASE_URL is not a valid URL."]

    scheme = parsed.scheme
    if not scheme:
        warnings.append("DATABASE_URL has no scheme.")
        return warnings

    if "postgresql" in scheme:
        if "+psycopg" not in scheme:
            warnings.append(
                "Postgres URL does not use the psycopg SQLAlchemy driver "
                "(expected postgresql+psycopg://...)."
            )
        params = parse_qs(parsed.query)
        if params.get("sslmode", [None])[0] != "require":
            warnings.append(
                "Postgres URL is missing sslmode=require; many cloud environments require SSL."
            )

    if raw_url == DEFAULT_DB_URL and IS_PRODUCTION:
        warnings.append("Production is using local SQLite fallback URL.")

    return warnings


def schema_compatibility_issues() -> list[str]:
    """Return human-readable issues for missing tables or columns."""
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    issues: list[str] = []

    for table_name, required_columns in REQUIRED_SCHEMA.items():
        if table_name not in existing_tables:
            issues.append(f"missing table: {table_name}")
            continue
        existing_columns = {col["name"] for col in inspector.get_columns(table_name)}
        missing_columns = sorted(required_columns - existing_columns)
        for column_name in missing_columns:
            issues.append(f"missing column: {table_name}.{column_name}")
    return issues


def verify_schema_compatibility(fail_fast: bool = False) -> list[str]:
    """Inspect schema compatibility and optionally raise in production."""
    issues = schema_compatibility_issues()
    if fail_fast and issues:
        joined = "; ".join(issues)
        raise RuntimeError(
            "Database schema is incompatible with the current app models. "
            f"Apply migrations before starting the app. Issues: {joined}"
        )
    return issues


def apply_postgres_schema_compatibility_patch() -> bool:
    """Apply idempotent schema compatibility updates for Postgres deployments.

    Returns True when running on Postgres and the patch statements were executed.
    """
    if engine.dialect.name != "postgresql":
        return False

    with engine.begin() as connection:
        for statement in POSTGRES_COMPATIBILITY_PATCH_STATEMENTS:
            connection.execute(text(statement))
    return True


@contextmanager
def session_scope() -> Iterator:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
