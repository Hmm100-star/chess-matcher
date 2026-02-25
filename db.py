from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from urllib.parse import parse_qs, urlsplit

from sqlalchemy import create_engine, inspect
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
        "missing_homework_penalty",
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
        "teacher_id",
        "classroom_id",
        "round_id",
        "match_id",
        "action",
        "payload",
        "created_at",
    },
}


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
