from __future__ import annotations

import csv
import logging
import os
import secrets
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask import has_request_context
from sqlalchemy import func, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import selectinload
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from db import (
    Base,
    IS_PRODUCTION,
    apply_postgres_schema_compatibility_patch,
    database_url_warnings,
    engine,
    redacted_database_url,
    session_scope,
    verify_schema_compatibility,
)
from models import (
    Attendance,
    AuditLog,
    AssignmentEntry,
    AssignmentType,
    Classroom,
    HomeworkEntry,
    Match,
    Round,
    RoundAssignmentType,
    Student,
    StudentAssignmentScore,
    Teacher,
)
from pairing_logic import normalize_weights
from services import (
    apply_homework_policy,
    canonicalize_homework_metric_mode,
    create_match_records,
    generate_matches_for_students,
    recalculate_totals,
    VALID_HOMEWORK_METRIC_MODES,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WIN_WEIGHT = 0.7
DEFAULT_HOMEWORK_WEIGHT = 0.3
INPUT_TEMPLATE_URL = (
    "https://docs.google.com/spreadsheets/d/1kJKOxY_5oYmAcgvMtz_e9llXeYifauULxCitCE9vAQM/edit?usp=sharing"
)
GITHUB_URL = "https://github.com/Hmm100-star/chess-match-selector"

_tables_initialized = False


def initialize_database() -> None:
    """Best-effort table initialization.

    In cloud deployments, database networking may be temporarily unavailable during
    process startup. Deferring table creation prevents the WSGI import step from
    crashing before the app can bind to a port.
    """

    global _tables_initialized
    if _tables_initialized:
        return

    try:
        logger.info(
            "Initializing database tables",
            extra={"database_url": redacted_database_url()},
        )
        Base.metadata.create_all(bind=engine)
        # Always apply the Postgres compatibility patch in production so that
        # idempotent fixes (e.g. dropping stale FK constraints) run even when
        # the column-level schema check reports no issues.
        if IS_PRODUCTION:
            patched = apply_postgres_schema_compatibility_patch()
            if patched:
                logger.info("Applied Postgres schema compatibility patch")
        schema_issues = verify_schema_compatibility(fail_fast=False)
        if schema_issues:
            logger.warning(
                "Database schema compatibility issues detected",
                extra={"database_url": redacted_database_url(), "issues": schema_issues},
            )
            if IS_PRODUCTION:
                joined = "; ".join(schema_issues)
                raise RuntimeError(
                    "Database schema is incompatible with the current app models. "
                    f"Apply migrations before starting the app. Issues: {joined}"
                )
        else:
            logger.info("Database schema compatibility check passed")
        _tables_initialized = True
        logger.info("Database initialization completed")
    except RuntimeError:
        # Preserve fail-fast behavior for production schema incompatibility checks.
        raise
    except Exception:
        logger.exception(
            "Database initialization failed; will retry on next request",
            extra={"database_url": redacted_database_url()},
        )

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

logger = logging.getLogger("chess_match_selector")
if not logger.handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

for warning in database_url_warnings():
    logger.warning(warning, extra={"database_url": redacted_database_url()})


@app.before_request
def ensure_database_initialized() -> None:
    initialize_database()


def allowed_file(filename: str) -> bool:
    return filename.lower().endswith(".csv")


def generate_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def require_csrf() -> None:
    token = session.get("csrf_token")
    form_token = request.form.get("csrf_token")
    if not token or not form_token or token != form_token:
        abort(400, description="Invalid CSRF token.")


def current_teacher_id() -> Optional[int]:
    return session.get("teacher_id")


def validate_and_create_teacher(
    username: str,
    password: str,
    confirm_password: str,
) -> Optional[str]:
    """Create a teacher account if input is valid.

    Returns an error message when validation fails, otherwise ``None``.
    """

    normalized_username = username.strip()

    if not normalized_username or not password:
        return "Username and password are required."
    if password != confirm_password:
        return "Passwords do not match."

    with session_scope() as db:
        existing_teacher = (
            db.query(Teacher)
            .filter(Teacher.username.ilike(normalized_username))
            .first()
        )
        if existing_teacher:
            return "Username is already taken."

        teacher = Teacher(
            username=normalized_username,
            password_hash=generate_password_hash(password),
        )
        db.add(teacher)
        try:
            db.flush()
        except IntegrityError:
            return "Username is already taken."

    return None


def require_login() -> Teacher:
    teacher_id = current_teacher_id()
    if not teacher_id:
        return redirect(url_for("login"))
    with session_scope() as db:
        teacher = db.get(Teacher, teacher_id)
        if not teacher:
            session.pop("teacher_id", None)
            return redirect(url_for("login"))
        return teacher


def get_classroom_or_404(classroom_id: int, teacher_id: int) -> Classroom:
    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher_id:
            abort(404)
        return classroom


def parse_non_negative_int(value: str, field_name: str) -> int:
    if value is None or value == "":
        return 0
    parsed = int(float(value))
    if parsed < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return parsed


def safe_strip(value: object, default: str = "") -> str:
    return str(value if value is not None else default).strip()


def validate_round_completion(
    matches: list[Match], round_record: Round, attendance_records: list[Attendance]
) -> list[str]:
    missing: list[str] = []
    for match in matches:
        if match.black_student_id is not None and not match.result:
            missing.append(f"Match {match.id} is missing result.")
        homework = match.homework_entry
        if not homework:
            missing.append(f"Match {match.id} is missing homework entry.")
            continue
        if match.white_student_id and homework.white_submitted and homework.white_correct < 0:
            missing.append(f"Match {match.id} has invalid white homework values.")
        if match.black_student_id and homework.black_submitted and homework.black_correct < 0:
            missing.append(f"Match {match.id} has invalid black homework values.")
    for record in attendance_records:
        if record.status not in {"present", "absent", "excused", "late"}:
            missing.append(f"Attendance record {record.id} has invalid status.")
    return missing


def log_field_change(
    db,
    teacher_id: int,
    classroom_id: int,
    round_id: int,
    match_id: int,
    field_name: str,
    old_value: str,
    new_value: str,
) -> None:
    if str(old_value) == str(new_value):
        return
    try:
        db.add(
            AuditLog(
                actor_teacher_id=teacher_id,
                classroom_id=classroom_id,
                round_id=round_id,
                match_id=match_id,
                field_name=field_name,
                old_value=str(old_value),
                new_value=str(new_value),
            )
        )
    except Exception:
        logger.warning(
            "Failed to create audit log entry",
            extra={
                "field_name": field_name,
                "match_id": match_id,
                "round_id": round_id,
            },
        )


def compute_attendance_metrics(attendance_records: list[Attendance]) -> dict[int, dict[str, float]]:
    per_student: dict[int, dict[str, float]] = {}
    streaks: dict[int, dict[str, int]] = {}
    for record in attendance_records:
        metrics = per_student.setdefault(
            record.student_id,
            {
                "present": 0,
                "absent": 0,
                "excused": 0,
                "late": 0,
                "total": 0,
            },
        )
        streak = streaks.setdefault(
            record.student_id,
            {"present_streak": 0, "absent_streak": 0, "max_present_streak": 0, "max_absent_streak": 0},
        )
        metrics["total"] += 1
        if record.status in metrics:
            metrics[record.status] += 1
        if record.status in {"present", "late"}:
            streak["present_streak"] += 1
            streak["absent_streak"] = 0
        elif record.status == "absent":
            streak["absent_streak"] += 1
            streak["present_streak"] = 0
        else:
            streak["present_streak"] = 0
            streak["absent_streak"] = 0
        streak["max_present_streak"] = max(streak["max_present_streak"], streak["present_streak"])
        streak["max_absent_streak"] = max(streak["max_absent_streak"], streak["absent_streak"])
    for student_id, metrics in per_student.items():
        attended = metrics["present"] + metrics["late"]
        total = metrics["total"] or 1
        metrics["absence_rate"] = round(metrics["absent"] / total, 3)
        metrics["attendance_rate"] = round(attended / total, 3)
        metrics["max_present_streak"] = streaks[student_id]["max_present_streak"]
        metrics["max_absent_streak"] = streaks[student_id]["max_absent_streak"]
        per_student[student_id] = metrics
    return per_student


def log_exception(error: Exception, support_id: str) -> None:
    extra = {"support_id": support_id, "error_type": type(error).__name__}
    if has_request_context():
        extra.update(
            {
                "path": request.path,
                "method": request.method,
                "teacher_id": session.get("teacher_id"),
            }
        )
    logger.exception("Unhandled application error", extra=extra)


@app.context_processor
def inject_globals() -> Dict[str, str]:
    return {
        "csrf_token": generate_csrf_token(),
        "github_url": GITHUB_URL,
        "input_template_url": INPUT_TEMPLATE_URL,
    }


@app.route("/")
def index() -> str:
    if current_teacher_id():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/setup", methods=["GET", "POST"])
def setup() -> str:
    with session_scope() as db:
        has_teacher = db.query(Teacher).count() > 0

    if has_teacher:
        return redirect(url_for("login"))

    error = None
    if request.method == "POST":
        require_csrf()
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        error = validate_and_create_teacher(
            username=username,
            password=password,
            confirm_password=confirm,
        )
        if not error:
            return redirect(url_for("login"))

    return render_template("setup.html", error=error, active_nav="")


@app.route("/signup", methods=["GET", "POST"])
def signup() -> str:
    error = None
    if request.method == "POST":
        require_csrf()
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        error = validate_and_create_teacher(
            username=username,
            password=password,
            confirm_password=confirm,
        )
        if not error:
            return redirect(url_for("login"))

    return render_template("signup.html", error=error, active_nav="")


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    try:
        with session_scope() as db:
            has_teacher = db.query(Teacher).count() > 0
    except SQLAlchemyError:
        logger.exception(
            "Database query failed while loading login page",
            extra={"database_url": redacted_database_url(), "path": request.path},
        )
        raise

    if not has_teacher:
        return redirect(url_for("setup"))

    error = None
    if request.method == "POST":
        require_csrf()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        try:
            with session_scope() as db:
                teacher = db.query(Teacher).filter(Teacher.username == username).first()
        except SQLAlchemyError:
            logger.exception(
                "Database query failed during login submit",
                extra={"database_url": redacted_database_url(), "path": request.path},
            )
            raise

        if not teacher or not check_password_hash(teacher.password_hash, password):
            error = "Invalid username or password."
        else:
            session["teacher_id"] = teacher.id
            return redirect(url_for("dashboard"))

    return render_template("login.html", error=error, active_nav="")


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    if isinstance(error, HTTPException):
        return error
    support_id = uuid.uuid4().hex[:12]
    log_exception(error, support_id)
    return render_template("500.html", support_id=support_id, active_nav=""), 500


@app.route("/health/db")
def health_db():
    try:
        schema_issues = verify_schema_compatibility(fail_fast=False)
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database_url": redacted_database_url(),
            "warnings": database_url_warnings(),
            "schema_compatibility": "issues" if schema_issues else "ok",
            "schema_issues": schema_issues,
        }
    except Exception as error:
        logger.exception(
            "Database health check failed",
            extra={"database_url": redacted_database_url()},
        )
        return {
            "status": "error",
            "database_url": redacted_database_url(),
            "warnings": database_url_warnings(),
            "schema_compatibility": "issues",
            "schema_issues": [],
            "error_type": type(error).__name__,
            "error": str(error),
        }, 500


@app.route("/logout", methods=["POST"])
def logout() -> str:
    require_csrf()
    session.pop("teacher_id", None)
    session.pop("classroom_id", None)
    return redirect(url_for("login"))


@app.route("/dashboard", methods=["GET", "POST"])
def dashboard() -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    error = None
    with session_scope() as db:
        classrooms = (
            db.query(Classroom)
            .filter(Classroom.teacher_id == teacher.id)
            .order_by(Classroom.created_at.desc())
            .all()
        )

        if request.method == "POST":
            require_csrf()
            name = request.form.get("classroom_name", "").strip()
            if not name:
                error = "Class name is required."
            else:
                classroom = Classroom(name=name, teacher_id=teacher.id)
                db.add(classroom)
                db.flush()  # get classroom.id
                # Auto-create a default "Homework" assignment type for every new classroom.
                db.add(AssignmentType(
                    classroom_id=classroom.id,
                    name="Homework",
                    metric_mode="pct_correct",
                    missing_policy="zero",
                ))
                db.commit()
                return redirect(url_for("dashboard"))

    return render_template(
        "dashboard.html",
        teacher=teacher,
        classrooms=classrooms,
        error=error,
        active_nav="dashboard",
        breadcrumbs=[{"label": "Dashboard"}],
    )


@app.route("/classrooms/<int:classroom_id>")
def classroom_overview(classroom_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        students = (
            db.query(Student)
            .filter(Student.classroom_id == classroom_id)
            .order_by(Student.name)
            .all()
        )
        rounds = (
            db.query(Round)
            .filter(Round.classroom_id == classroom_id)
            .order_by(Round.created_at.desc())
            .all()
        )
        assignment_types = (
            db.query(AssignmentType)
            .filter(AssignmentType.classroom_id == classroom_id)
            .order_by(AssignmentType.created_at)
            .all()
        )
        student_ids = [s.id for s in students]
        all_scores = (
            db.query(StudentAssignmentScore)
            .filter(StudentAssignmentScore.student_id.in_(student_ids))
            .all()
            if student_ids else []
        )
        scores_map: dict = {}
        for sc in all_scores:
            scores_map.setdefault(sc.student_id, {})[sc.assignment_type_id] = sc

    return render_template(
        "classroom.html",
        teacher=teacher,
        classroom=classroom,
        students=students,
        rounds=rounds,
        assignment_types=assignment_types,
        scores_map=scores_map,
        success=request.args.get("success"),
        error=request.args.get("error"),
        active_nav="dashboard",
        breadcrumbs=[
            {"label": "Dashboard", "url": url_for("dashboard")},
            {"label": classroom.name},
        ],
    )


@app.route("/classrooms/<int:classroom_id>/students", methods=["POST"])
def add_student(classroom_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    require_csrf()
    name = request.form.get("student_name", "").strip()
    if not name:
        return redirect(url_for("classroom_overview", classroom_id=classroom_id))

    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)
        db.add(Student(classroom_id=classroom_id, name=name))

    return redirect(url_for("classroom_overview", classroom_id=classroom_id))


@app.route("/students/<int:student_id>/update", methods=["POST"])
def update_student(student_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    require_csrf()
    name = request.form.get("student_name", "").strip()
    notes = request.form.get("notes", "").strip()
    active = request.form.get("active") == "on"
    classroom_id = int(request.form.get("classroom_id", "0"))

    with session_scope() as db:
        student = db.get(Student, student_id)
        if not student:
            abort(404)
        classroom = db.get(Classroom, student.classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        if name:
            student.name = name
        student.notes = notes
        student.active = active

    return redirect(url_for("classroom_overview", classroom_id=classroom_id))


# ── Assignment-type management ────────────────────────────────────────────────

@app.route("/classrooms/<int:classroom_id>/assignment_types", methods=["GET", "POST"])
def manage_assignment_types(classroom_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        error = None
        if request.method == "POST":
            require_csrf()
            at_name = request.form.get("name", "").strip()
            at_metric = request.form.get("metric_mode", "pct_correct").strip()
            at_policy = request.form.get("missing_policy", "zero").strip()
            if not at_name:
                error = "Assignment type name is required."
            elif at_metric not in VALID_HOMEWORK_METRIC_MODES:
                error = "Invalid metric mode."
            elif at_policy not in {"zero", "exclude", "penalty"}:
                error = "Invalid missing policy."
            else:
                at_metric = canonicalize_homework_metric_mode(at_metric)
                db.add(AssignmentType(
                    classroom_id=classroom_id,
                    name=at_name,
                    metric_mode=at_metric,
                    missing_policy=at_policy,
                ))

        assignment_types = (
            db.query(AssignmentType)
            .filter(AssignmentType.classroom_id == classroom_id)
            .order_by(AssignmentType.id)
            .all()
        )
        return render_template(
            "assignment_types.html",
            classroom=classroom,
            assignment_types=assignment_types,
            error=error,
            active_nav="dashboard",
            breadcrumbs=[
                {"label": "Dashboard", "url": url_for("dashboard")},
                {"label": classroom.name, "url": url_for("classroom_overview", classroom_id=classroom.id)},
                {"label": "Assignment Types"},
            ],
        )


@app.route("/classrooms/<int:classroom_id>/assignment_types/<int:type_id>/delete", methods=["POST"])
def delete_assignment_type(classroom_id: int, type_id: int):
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    require_csrf()
    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)
        at = db.get(AssignmentType, type_id)
        if not at or at.classroom_id != classroom_id:
            abort(404)
        # Refuse deletion if there are AssignmentEntry records for this type.
        has_entries = db.query(AssignmentEntry).filter(
            AssignmentEntry.assignment_type_id == type_id
        ).first()
        if has_entries:
            assignment_types = (
                db.query(AssignmentType)
                .filter(AssignmentType.classroom_id == classroom_id)
                .order_by(AssignmentType.id)
                .all()
            )
            return render_template(
                "assignment_types.html",
                classroom=classroom,
                assignment_types=assignment_types,
                error=f"Cannot delete '{at.name}' because it has score data. Remove score data first.",
                active_nav="dashboard",
                breadcrumbs=[
                    {"label": "Dashboard", "url": url_for("dashboard")},
                    {"label": classroom.name, "url": url_for("classroom_overview", classroom_id=classroom.id)},
                    {"label": "Assignment Types"},
                ],
            )
        db.delete(at)

    return redirect(url_for("manage_assignment_types", classroom_id=classroom_id))


@app.route("/classrooms/<int:classroom_id>/rounds/new", methods=["GET", "POST"])
def new_round(classroom_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        students = (
            db.query(Student)
            .filter(Student.classroom_id == classroom_id, Student.active.is_(True))
            .order_by(Student.name)
            .all()
        )
        assignment_types = (
            db.query(AssignmentType)
            .filter(AssignmentType.classroom_id == classroom_id)
            .order_by(AssignmentType.id)
            .all()
        )

        error = None
        if request.method == "POST":
            require_csrf()
            win_weight_raw = request.form.get("win_weight", "").strip()

            # ── Determine attendance ─────────────────────────────────────
            attendance_by_student: dict[int, str] = {}
            for student in students:
                status = request.form.get(f"attendance_status_{student.id}", "present")
                if status not in {"present", "absent", "excused", "late"}:
                    status = "present"
                attendance_by_student[student.id] = status
            absent_ids = {
                sid for sid, st in attendance_by_student.items() if st in {"absent", "excused"}
            }
            present_students = [s for s in students if s.id not in absent_ids]

            try:
                win_weight = float(win_weight_raw) if win_weight_raw else DEFAULT_WIN_WEIGHT
                if win_weight < 0:
                    raise ValueError("Win weight must be zero or greater.")
                if len(present_students) < 2:
                    raise ValueError("At least two present students are required to create matches.")
            except ValueError as exc:
                error = str(exc)
            else:
                # ── Parse per-assignment-type weights ────────────────────
                active_at_configs: list[dict] = []
                for at in assignment_types:
                    raw_w = request.form.get(f"rat_{at.id}_weight", "").strip()
                    raw_q = request.form.get(f"rat_{at.id}_total_questions", "").strip()
                    try:
                        at_weight = float(raw_w) if raw_w else 0.0
                        at_total_q = parse_non_negative_int(raw_q, f"{at.name} total questions") if raw_q else 0
                    except ValueError as exc:
                        error = f"{at.name}: {exc}"
                        break
                    if at_weight < 0:
                        error = f"{at.name} weight must be zero or greater."
                        break
                    if at_weight > 0:
                        active_at_configs.append({
                            "at": at,
                            "weight": at_weight,
                            "total_questions": at_total_q,
                        })

                if not error:
                    try:
                        if active_at_configs:
                            # ── New assignment-type path ─────────────────────
                            present_ids = [s.id for s in present_students]
                            assignment_type_data = []
                            for cfg in active_at_configs:
                                at = cfg["at"]
                                scores_qs = db.query(StudentAssignmentScore).filter(
                                    StudentAssignmentScore.student_id.in_(present_ids),
                                    StudentAssignmentScore.assignment_type_id == at.id,
                                ).all()
                                student_scores = {sc.student_id: (sc.correct, sc.incorrect) for sc in scores_qs}
                                assignment_type_data.append({
                                    "weight": cfg["weight"],
                                    "metric_mode": at.metric_mode,
                                    "student_scores": student_scores,
                                })
                            logger.info(
                                "Round generation started",
                                extra={
                                    "phase": "pairing",
                                    "classroom_id": classroom_id,
                                    "teacher_id": teacher.id,
                                    "present_student_count": len(present_students),
                                    "absent_student_count": len(absent_ids),
                                },
                            )
                            matches, unpaired, df, id_order = generate_matches_for_students(
                                present_students,
                                win_weight,
                                assignment_type_data=assignment_type_data,
                            )
                            # Determine next round number.
                            max_rn = (
                                db.query(func.max(Round.round_number))
                                .filter(Round.classroom_id == classroom_id)
                                .scalar()
                            )
                            next_round_number = (max_rn or 0) + 1
                            # Sum assignment weights for the legacy homework_weight field.
                            total_at_weight = sum(c["weight"] for c in active_at_configs)
                            all_weights = [win_weight] + [c["weight"] for c in active_at_configs]
                            norm = normalize_weights(all_weights)
                            round_record = Round(
                                classroom_id=classroom_id,
                                round_number=next_round_number,
                                win_weight=int(norm[0] * 100),
                                homework_weight=0,  # superseded by RoundAssignmentType
                                status="open",
                                homework_total_questions=0,
                                missing_homework_policy="zero",
                                homework_metric_mode="pct_correct",
                            )
                            db.add(round_record)
                            db.flush()

                            for student in students:
                                db.add(Attendance(
                                    round_id=round_record.id,
                                    student_id=student.id,
                                    status=attendance_by_student.get(student.id, "present"),
                                ))

                            # Create RoundAssignmentType records.
                            for cfg in active_at_configs:
                                db.add(RoundAssignmentType(
                                    round_id=round_record.id,
                                    assignment_type_id=cfg["at"].id,
                                    weight=int(cfg["weight"]),
                                    total_questions=cfg["total_questions"],
                                ))

                            # Create match records with AssignmentEntry per active type.
                            match_records = create_match_records(
                                present_students, matches, unpaired, df, id_order
                            )
                            for record in match_records:
                                record.round_id = round_record.id
                                db.add(record)
                                db.flush()  # get record.id
                                for cfg in active_at_configs:
                                    db.add(AssignmentEntry(
                                        match_id=record.id,
                                        assignment_type_id=cfg["at"].id,
                                    ))

                            logger.info(
                                "Round generation completed",
                                extra={
                                    "phase": "insert",
                                    "classroom_id": classroom_id,
                                    "round_id": round_record.id,
                                    "teacher_id": teacher.id,
                                    "match_count": len(match_records),
                                    "paired_match_count": len(matches),
                                    "bye_count": len(unpaired),
                                },
                            )
                            return redirect(
                                url_for("round_results", classroom_id=classroom_id, round_id=round_record.id)
                            )
                        else:
                            # ── Legacy homework path (no active assignment types) ──
                            homework_weight_raw = request.form.get("homework_weight", "").strip()
                            homework_total_raw = request.form.get("homework_total_questions", "").strip()
                            missing_homework_policy = request.form.get("missing_homework_policy", "zero").strip()
                            homework_metric_mode = request.form.get("homework_metric_mode", "pct_correct").strip()
                            homework_weight = float(homework_weight_raw) if homework_weight_raw else DEFAULT_HOMEWORK_WEIGHT
                            homework_total_questions = parse_non_negative_int(homework_total_raw, "Homework total questions") if homework_total_raw else 0
                            if missing_homework_policy not in {"zero", "exclude", "penalty"}:
                                missing_homework_policy = "zero"
                            if homework_metric_mode not in VALID_HOMEWORK_METRIC_MODES:
                                homework_metric_mode = "pct_correct"
                            homework_metric_mode = canonicalize_homework_metric_mode(homework_metric_mode)
                            if homework_weight < 0:
                                raise ValueError("Homework weight must be zero or greater.")
                            normalized_win, normalized_homework = normalize_weights(win_weight, homework_weight)
                            logger.info(
                                "Round generation started",
                                extra={
                                    "phase": "pairing",
                                    "classroom_id": classroom_id,
                                    "teacher_id": teacher.id,
                                    "present_student_count": len(present_students),
                                    "absent_student_count": len(absent_ids),
                                },
                            )
                            matches, unpaired, df, id_order = generate_matches_for_students(
                                present_students,
                                normalized_win,
                                normalized_homework,
                                homework_metric_mode,
                            )
                            max_rn = (
                                db.query(func.max(Round.round_number))
                                .filter(Round.classroom_id == classroom_id)
                                .scalar()
                            )
                            next_round_number = (max_rn or 0) + 1
                            round_record = Round(
                                classroom_id=classroom_id,
                                round_number=next_round_number,
                                win_weight=int(normalized_win * 100),
                                homework_weight=int(normalized_homework * 100),
                                status="open",
                                homework_total_questions=homework_total_questions,
                                missing_homework_policy=missing_homework_policy,
                                homework_metric_mode=homework_metric_mode,
                            )
                            db.add(round_record)
                            db.flush()

                            for student in students:
                                db.add(Attendance(
                                    round_id=round_record.id,
                                    student_id=student.id,
                                    status=attendance_by_student.get(student.id, "present"),
                                ))

                            match_records = create_match_records(
                                present_students, matches, unpaired, df, id_order
                            )
                            for record in match_records:
                                record.round_id = round_record.id
                                record.homework_entry = HomeworkEntry()
                                db.add(record)

                            logger.info(
                                "Round generation completed",
                                extra={
                                    "phase": "insert",
                                    "classroom_id": classroom_id,
                                    "round_id": round_record.id,
                                    "teacher_id": teacher.id,
                                    "match_count": len(match_records),
                                    "paired_match_count": len(matches),
                                    "bye_count": len(unpaired),
                                },
                            )
                            return redirect(
                                url_for("round_results", classroom_id=classroom_id, round_id=round_record.id)
                            )
                    except ValueError as exc:
                        error = str(exc)

    return render_template(
        "new_round.html",
        classroom=classroom,
        students=students,
        assignment_types=assignment_types,
        error=error,
        default_win_weight=DEFAULT_WIN_WEIGHT,
        default_homework_weight=DEFAULT_HOMEWORK_WEIGHT,
        default_homework_total_questions=10,
        default_missing_homework_policy="zero",
        default_homework_metric_mode="pct_correct",
        active_nav="dashboard",
        breadcrumbs=[
            {"label": "Dashboard", "url": url_for("dashboard")},
            {"label": classroom.name, "url": url_for("classroom_overview", classroom_id=classroom.id)},
            {"label": "New Round"},
        ],
    )


@app.route("/classrooms/<int:classroom_id>/rounds/<int:round_id>", methods=["GET", "POST"])
def round_results(classroom_id: int, round_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher
    error: Optional[str] = None
    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        round_record = db.get(Round, round_id)
        if (
            not classroom
            or classroom.teacher_id != teacher.id
            or not round_record
            or round_record.classroom_id != classroom_id
        ):
            abort(404)

        matches = (
            db.query(Match)
            .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
            .filter(Match.round_id == round_id)
            .order_by(Match.id)
            .all()
        )
        attendance_records = (
            db.query(Attendance)
            .filter(Attendance.round_id == round_id)
            .all()
        )
        attendance_by_student = {record.student_id: record for record in attendance_records}
        students = db.query(Student).filter(Student.classroom_id == classroom_id).all()
        student_map = {student.id: student for student in students}

        if request.method == "POST":
            require_csrf()
            action = safe_strip(request.form.get("action", "save"), "save") or "save"
            override_reason = safe_strip(request.form.get("override_reason", ""), "")

            try:
                total_questions = parse_non_negative_int(
                    request.form.get("homework_total_questions", str(round_record.homework_total_questions)),
                    "Homework total questions",
                )
                missing_policy = safe_strip(
                    request.form.get(
                        "missing_homework_policy",
                        round_record.missing_homework_policy or "zero",
                    ),
                    "zero",
                )
                metric_mode = safe_strip(
                    request.form.get(
                        "homework_metric_mode",
                        round_record.homework_metric_mode or "pct_correct",
                    ),
                    "pct_correct",
                )
                if missing_policy not in {"zero", "exclude", "penalty"}:
                    missing_policy = "zero"
                if metric_mode not in VALID_HOMEWORK_METRIC_MODES:
                    metric_mode = "pct_correct"
                metric_mode = canonicalize_homework_metric_mode(metric_mode)

                round_record.homework_total_questions = total_questions
                round_record.missing_homework_policy = missing_policy
                round_record.homework_metric_mode = metric_mode

                if action == "bulk_mark_all_absent":
                    for record in attendance_records:
                        record.status = "absent"
                else:
                    for record in attendance_records:
                        status = safe_strip(
                            request.form.get(
                                f"attendance_status_{record.student_id}",
                                record.status or "present",
                            ),
                            "present",
                        )
                        if status not in {"present", "absent", "excused", "late"}:
                            status = "present"
                        record.status = status

                for match in matches:
                    old_result = match.result or ""
                    old_notes = match.notes or ""
                    old_white_notation = match.notation_submitted_white
                    old_black_notation = match.notation_submitted_black

                    if action == "bulk_set_unresolved_tie" and not match.result and match.black_student_id:
                        result = "tie"
                    elif action == "bulk_clear_unresolved" and not match.result and match.black_student_id:
                        result = ""
                    else:
                        result = request.form.get(f"result_{match.id}", (match.result or "")).strip().lower()

                    notes = request.form.get(f"notes_{match.id}", match.notes or "").strip()
                    white_notation = request.form.get(f"notation_white_{match.id}") == "on"
                    black_notation = request.form.get(f"notation_black_{match.id}") == "on"

                    if match.black_student_id is None:
                        result = "bye"
                        black_notation = False
                    if result not in {"white", "black", "tie", "bye", ""}:
                        raise ValueError("Invalid result value.")

                    match.result = result or None
                    match.notes = notes
                    match.notation_submitted_white = white_notation
                    match.notation_submitted_black = black_notation
                    match.updated_at = datetime.utcnow()

                    log_field_change(
                        db,
                        teacher.id,
                        classroom_id,
                        round_id,
                        match.id,
                        "result",
                        old_result,
                        match.result or "",
                    )
                    log_field_change(
                        db,
                        teacher.id,
                        classroom_id,
                        round_id,
                        match.id,
                        "notes",
                        old_notes,
                        notes,
                    )
                    log_field_change(
                        db,
                        teacher.id,
                        classroom_id,
                        round_id,
                        match.id,
                        "notation_submitted_white",
                        str(old_white_notation),
                        str(white_notation),
                    )
                    log_field_change(
                        db,
                        teacher.id,
                        classroom_id,
                        round_id,
                        match.id,
                        "notation_submitted_black",
                        str(old_black_notation),
                        str(black_notation),
                    )

                    homework_entry = match.homework_entry
                    if not homework_entry:
                        homework_entry = HomeworkEntry(match_id=match.id)
                        db.add(homework_entry)
                        match.homework_entry = homework_entry

                    white_correct_raw = request.form.get(
                        f"white_entered_correct_{match.id}",
                        str(homework_entry.white_correct if homework_entry.white_submitted else ""),
                    )
                    black_correct_raw = request.form.get(
                        f"black_entered_correct_{match.id}",
                        str(homework_entry.black_correct if homework_entry.black_submitted else ""),
                    )

                    white_correct, white_wrong, white_submitted, white_pct_wrong = apply_homework_policy(
                        white_correct_raw,
                        total_questions,
                        missing_policy,
                    )
                    black_correct, black_wrong, black_submitted, black_pct_wrong = apply_homework_policy(
                        black_correct_raw,
                        total_questions,
                        missing_policy,
                    )
                    if match.black_student_id is None:
                        black_correct, black_wrong, black_submitted, black_pct_wrong = (0, 0, False, 0.0)

                    homework_entry.white_correct = white_correct
                    homework_entry.white_incorrect = white_wrong
                    homework_entry.white_submitted = white_submitted
                    homework_entry.white_pct_wrong = white_pct_wrong
                    homework_entry.black_correct = black_correct
                    homework_entry.black_incorrect = black_wrong
                    homework_entry.black_submitted = black_submitted
                    homework_entry.black_pct_wrong = black_pct_wrong

                if action == "reopen_round":
                    round_record.status = "open"
                    round_record.completion_override_reason = ""
                elif action == "complete_round":
                    missing = validate_round_completion(matches, round_record, attendance_records)
                    if missing and not override_reason:
                        raise ValueError(
                            f"Round has unresolved data: {missing[0]} Add override reason to complete anyway."
                        )
                    round_record.status = "completed"
                    round_record.completion_override_reason = override_reason

                all_matches = (
                    db.query(Match)
                    .join(Round)
                    .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
                    .filter(Round.classroom_id == classroom_id)
                    .all()
                )
                recalculate_totals(students, all_matches, db=db)
                # Flush inside this guarded block so DB write errors render
                # as recoverable form errors instead of bubbling as 500s at commit.
                db.flush()
                return redirect(
                    url_for("round_results", classroom_id=classroom_id, round_id=round_id)
                )
            except ValueError as exc:
                db.rollback()
                error = str(exc)
            except (SQLAlchemyError, TypeError, AttributeError, KeyError) as exc:
                db.rollback()
                logger.exception(
                    "Round results POST failed with recoverable error",
                    extra={
                        "classroom_id": classroom_id,
                        "round_id": round_id,
                        "teacher_id": teacher.id,
                        "action": action,
                        "error_type": type(exc).__name__,
                    },
                )
                error = "We couldn't save round changes due to a temporary data issue. Please retry."

            if error:
                # After rollback all ORM objects are expired.  Re-query so
                # the template can access relationships without a live session.
                classroom = db.get(Classroom, classroom_id)
                round_record = db.get(Round, round_id)
                matches = (
                    db.query(Match)
                    .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
                    .filter(Match.round_id == round_id)
                    .order_by(Match.id)
                    .all()
                )
                attendance_records = (
                    db.query(Attendance)
                    .filter(Attendance.round_id == round_id)
                    .all()
                )
                attendance_by_student = {record.student_id: record for record in attendance_records}
                students = db.query(Student).filter(Student.classroom_id == classroom_id).all()
                student_map = {student.id: student for student in students}

        opponent_counts: dict[tuple[int, int], int] = {}
        historical_matches = (
            db.query(Match)
            .join(Round)
            .filter(
                Round.classroom_id == classroom_id,
                Round.id < round_id,
                Match.black_student_id.isnot(None),
            )
            .all()
        )
        for prior_match in historical_matches:
            key = tuple(sorted([prior_match.white_student_id, prior_match.black_student_id]))
            opponent_counts[key] = opponent_counts.get(key, 0) + 1

        fairness_diagnostics = []
        for match in matches:
            if not match.black_student_id:
                continue
            white = student_map.get(match.white_student_id)
            black = student_map.get(match.black_student_id)
            if not white or not black:
                continue
            key = tuple(sorted([white.id, black.id]))
            repeat_count = opponent_counts.get(key, 0)
            fairness_diagnostics.append(
                {
                    "match_id": match.id,
                    "white_name": white.name,
                    "black_name": black.name,
                    "white_color_pressure": white.times_white - white.times_black,
                    "black_color_pressure": black.times_white - black.times_black,
                    "white_wlt": f"{white.total_wins}-{white.total_losses}-{white.total_ties}",
                    "black_wlt": f"{black.total_wins}-{black.total_losses}-{black.total_ties}",
                    "repeat_warning": repeat_count > 0,
                    "repeat_count": repeat_count,
                }
            )

        notation_summary = {
            "submitted": sum(
                int(match.notation_submitted_white) + int(match.notation_submitted_black)
                for match in matches
            ),
            "expected": sum(1 + (1 if match.black_student_id else 0) for match in matches),
        }
        attendance_metrics = compute_attendance_metrics(
            db.query(Attendance)
            .join(Round)
            .filter(Round.classroom_id == classroom_id)
            .order_by(Round.id)
            .all()
        )

        # Exceptions from the last completed round for the sidebar panel
        def _student_name(student_id: int | None) -> str:
            if student_id is None:
                return "Unknown student"
            s = student_map.get(student_id)
            return s.name if s else f"Student #{student_id}"

        eq_round = (
            db.query(Round)
            .filter(Round.classroom_id == classroom_id, Round.status == "completed")
            .order_by(Round.created_at.desc())
            .first()
        )
        eq_missing_homework: list[str] = []
        eq_missing_notation: list[str] = []
        eq_unresolved_results: list[int] = []
        eq_absences: list[str] = []
        if eq_round:
            eq_matches = (
                db.query(Match)
                .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
                .filter(Match.round_id == eq_round.id)
                .order_by(Match.id)
                .all()
            )
            eq_attendance = (
                db.query(Attendance).filter(Attendance.round_id == eq_round.id).all()
            )
            for m in eq_matches:
                if m.black_student_id is not None and not m.result:
                    eq_unresolved_results.append(m.id)
                hw = m.homework_entry
                if m.white_student_id and (not hw or not hw.white_submitted):
                    eq_missing_homework.append(f"Match {m.id}: {_student_name(m.white_student_id)}")
                if m.black_student_id and (not hw or not hw.black_submitted):
                    eq_missing_homework.append(f"Match {m.id}: {_student_name(m.black_student_id)}")
                if m.white_student_id and not m.notation_submitted_white:
                    eq_missing_notation.append(f"Match {m.id}: {_student_name(m.white_student_id)}")
                if m.black_student_id and not m.notation_submitted_black:
                    eq_missing_notation.append(f"Match {m.id}: {_student_name(m.black_student_id)}")
            eq_absences = [
                _student_name(r.student_id)
                for r in eq_attendance
                if r.status in {"absent", "excused"}
            ]

        # Previous round's absent student IDs — used by the bulk-exempt button
        prev_round = (
            db.query(Round)
            .filter(Round.classroom_id == classroom_id, Round.id < round_id)
            .order_by(Round.id.desc())
            .first()
        )
        prev_round_absent_ids: set[int] = set()
        prev_round_absent_names: list[str] = []
        if prev_round:
            for rec in db.query(Attendance).filter(
                Attendance.round_id == prev_round.id,
                Attendance.status == "absent",
            ).all():
                prev_round_absent_ids.add(rec.student_id)
                prev_round_absent_names.append(_student_name(rec.student_id))

        # Load assignment types active for this round.
        round_assignment_types = (
            db.query(RoundAssignmentType)
            .options(selectinload(RoundAssignmentType.assignment_type))
            .filter(RoundAssignmentType.round_id == round_id)
            .order_by(RoundAssignmentType.id)
            .all()
        )
        # Build a nested map: match_id -> {assignment_type_id -> AssignmentEntry}
        assignment_entries_map: dict[int, dict[int, AssignmentEntry]] = {}
        for match in matches:
            assignment_entries_map[match.id] = {e.assignment_type_id: e for e in match.assignment_entries}

    logger.info(
        "Round results rendered",
        extra={
            "phase": "render",
            "classroom_id": classroom_id,
            "round_id": round_id,
            "teacher_id": teacher.id,
            "match_count": len(matches),
        },
    )
    return render_template(
        "round_results.html",
        classroom=classroom,
        round_record=round_record,
        selected_homework_metric_mode=canonicalize_homework_metric_mode(
            round_record.homework_metric_mode
        ),
        matches=matches,
        student_map=student_map,
        attendance_by_student=attendance_by_student,
        notation_summary=notation_summary,
        attendance_metrics=attendance_metrics,
        fairness_diagnostics=fairness_diagnostics,
        error=error,
        eq_round=eq_round,
        eq_missing_homework=eq_missing_homework,
        eq_missing_notation=eq_missing_notation,
        eq_unresolved_results=eq_unresolved_results,
        eq_absences=eq_absences,
        prev_round_absent_ids=list(prev_round_absent_ids),
        prev_round_absent_names=prev_round_absent_names,
        round_assignment_types=round_assignment_types,
        assignment_entries_map=assignment_entries_map,
        active_nav="dashboard",
        breadcrumbs=[
            {"label": "Dashboard", "url": url_for("dashboard")},
            {"label": classroom.name, "url": url_for("classroom_overview", classroom_id=classroom.id)},
            {"label": f"Round {round_record.round_number if round_record.round_number is not None else round_record.id}"},
        ],
    )


@app.route("/classrooms/<int:classroom_id>/export/students")
def export_students(classroom_id: int):
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        students = (
            db.query(Student)
            .filter(Student.classroom_id == classroom_id)
            .order_by(Student.name)
            .all()
        )

    output_file = OUTPUTS_DIR / f"Student_Information_{classroom_id}.csv"
    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Student Name",
                "Total Wins",
                "Total Losses",
                "Total Ties",
                "# Times Played White",
                "# Times Played Black",
                "Correct Homework",
                "Incorrect Homework",
                "Notes",
            ]
        )
        for student in students:
            writer.writerow(
                [
                    student.name,
                    student.total_wins,
                    student.total_losses,
                    student.total_ties,
                    student.times_white,
                    student.times_black,
                    student.homework_correct,
                    student.homework_incorrect,
                    student.notes,
                ]
            )

    return send_file(output_file, as_attachment=True)


@app.route("/classrooms/<int:classroom_id>/rounds/<int:round_id>/autosave", methods=["POST"])
def autosave_round_field(classroom_id: int, round_id: int):
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    require_csrf()
    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        round_record = db.get(Round, round_id)
        if (
            not classroom
            or classroom.teacher_id != teacher.id
            or not round_record
            or round_record.classroom_id != classroom_id
        ):
            abort(404)

        field = request.form.get("field", "").strip()
        value = request.form.get("value", "")
        match_id_raw = request.form.get("match_id", "")
        assignment_type_id_raw = request.form.get("assignment_type_id", "").strip()

        try:
            if field in {"homework_total_questions"}:
                round_record.homework_total_questions = parse_non_negative_int(value, field)
            elif field in {"missing_homework_policy"}:
                if value not in {"zero", "exclude", "penalty"}:
                    raise ValueError("Invalid missing homework policy.")
                round_record.missing_homework_policy = value
            elif field in {"homework_metric_mode"}:
                if value not in VALID_HOMEWORK_METRIC_MODES:
                    raise ValueError("Invalid homework metric mode.")
                round_record.homework_metric_mode = canonicalize_homework_metric_mode(value)
            elif field.startswith("attendance_status_"):
                student_id = int(field.split("_")[-1])
                attendance_record = (
                    db.query(Attendance)
                    .filter(Attendance.round_id == round_id, Attendance.student_id == student_id)
                    .first()
                )
                if not attendance_record:
                    raise ValueError("Attendance record not found.")
                if value not in {"present", "absent", "excused", "late"}:
                    raise ValueError("Invalid attendance status.")
                attendance_record.status = value
            else:
                if not match_id_raw.isdigit():
                    raise ValueError("Match ID required.")
                match = (
                    db.query(Match)
                    .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
                    .filter(Match.id == int(match_id_raw), Match.round_id == round_id)
                    .first()
                )
                if not match:
                    raise ValueError("Match not found.")

                # ── Determine whether to use AssignmentEntry or legacy HomeworkEntry ──
                if assignment_type_id_raw and assignment_type_id_raw.isdigit():
                    # New AssignmentEntry path
                    at_id = int(assignment_type_id_raw)
                    rat = (
                        db.query(RoundAssignmentType)
                        .options(selectinload(RoundAssignmentType.assignment_type))
                        .filter(
                            RoundAssignmentType.round_id == round_id,
                            RoundAssignmentType.assignment_type_id == at_id,
                        )
                        .first()
                    )
                    if not rat:
                        raise ValueError("Assignment type not configured for this round.")
                    entry = next(
                        (e for e in match.assignment_entries if e.assignment_type_id == at_id),
                        None,
                    )
                    if not entry:
                        entry = AssignmentEntry(match_id=match.id, assignment_type_id=at_id)
                        db.add(entry)

                    if field == "result":
                        if value not in {"white", "black", "tie", "bye", ""}:
                            raise ValueError("Invalid result.")
                        if match.black_student_id is None:
                            value = "bye"
                        old = match.result or ""
                        match.result = value or None
                        log_field_change(db, teacher.id, classroom_id, round_id, match.id, "result", old, match.result or "")
                    elif field == "notes":
                        old = match.notes or ""
                        match.notes = value.strip()
                        log_field_change(db, teacher.id, classroom_id, round_id, match.id, "notes", old, match.notes)
                    elif field == "notation_submitted_white":
                        checked = value == "true"
                        old = str(match.notation_submitted_white)
                        match.notation_submitted_white = checked
                        log_field_change(db, teacher.id, classroom_id, round_id, match.id, "notation_submitted_white", old, str(checked))
                    elif field == "notation_submitted_black":
                        checked = value == "true" and match.black_student_id is not None
                        old = str(match.notation_submitted_black)
                        match.notation_submitted_black = checked
                        log_field_change(db, teacher.id, classroom_id, round_id, match.id, "notation_submitted_black", old, str(checked))
                    elif field in {"white_entered_correct", "black_entered_correct"}:
                        total_questions = int(rat.total_questions or 0)
                        missing_policy = (rat.assignment_type.missing_policy if rat.assignment_type else "zero")
                        if field == "white_entered_correct":
                            correct, wrong, submitted, pct_wrong = apply_homework_policy(value, total_questions, missing_policy)
                            entry.white_correct = correct
                            entry.white_incorrect = wrong
                            entry.white_submitted = submitted
                            entry.white_pct_wrong = pct_wrong
                        else:
                            correct, wrong, submitted, pct_wrong = apply_homework_policy(value, total_questions, missing_policy)
                            if match.black_student_id is None:
                                correct, wrong, submitted, pct_wrong = (0, 0, False, 0.0)
                            entry.black_correct = correct
                            entry.black_incorrect = wrong
                            entry.black_submitted = submitted
                            entry.black_pct_wrong = pct_wrong
                    elif field in {"white_exempt", "black_exempt"}:
                        exempt_val = value == "true"
                        if field == "white_exempt":
                            entry.white_exempt = exempt_val
                        else:
                            if match.black_student_id is not None:
                                entry.black_exempt = exempt_val
                    else:
                        raise ValueError("Unsupported autosave field.")
                else:
                    # Legacy HomeworkEntry path
                    homework_entry = match.homework_entry
                    if not homework_entry:
                        homework_entry = HomeworkEntry(match_id=match.id)
                        db.add(homework_entry)
                        match.homework_entry = homework_entry

                    if field == "result":
                        if value not in {"white", "black", "tie", "bye", ""}:
                            raise ValueError("Invalid result.")
                        if match.black_student_id is None:
                            value = "bye"
                        old = match.result or ""
                        match.result = value or None
                        log_field_change(
                            db, teacher.id, classroom_id, round_id, match.id, "result", old, match.result or ""
                        )
                    elif field == "notes":
                        old = match.notes or ""
                        match.notes = value.strip()
                        log_field_change(
                            db, teacher.id, classroom_id, round_id, match.id, "notes", old, match.notes
                        )
                    elif field == "notation_submitted_white":
                        checked = value == "true"
                        old = str(match.notation_submitted_white)
                        match.notation_submitted_white = checked
                        log_field_change(
                            db,
                            teacher.id,
                            classroom_id,
                            round_id,
                            match.id,
                            "notation_submitted_white",
                            old,
                            str(checked),
                        )
                    elif field == "notation_submitted_black":
                        checked = value == "true" and match.black_student_id is not None
                        old = str(match.notation_submitted_black)
                        match.notation_submitted_black = checked
                        log_field_change(
                            db,
                            teacher.id,
                            classroom_id,
                            round_id,
                            match.id,
                            "notation_submitted_black",
                            old,
                            str(checked),
                        )
                    elif field in {"white_entered_correct", "black_entered_correct"}:
                        total_questions = int(round_record.homework_total_questions or 0)
                        if field == "white_entered_correct":
                            correct, wrong, submitted, pct_wrong = apply_homework_policy(
                                value,
                                total_questions,
                                round_record.missing_homework_policy,
                            )
                            homework_entry.white_correct = correct
                            homework_entry.white_incorrect = wrong
                            homework_entry.white_submitted = submitted
                            homework_entry.white_pct_wrong = pct_wrong
                        else:
                            correct, wrong, submitted, pct_wrong = apply_homework_policy(
                                value,
                                total_questions,
                                round_record.missing_homework_policy,
                            )
                            if match.black_student_id is None:
                                correct, wrong, submitted, pct_wrong = (0, 0, False, 0.0)
                            homework_entry.black_correct = correct
                            homework_entry.black_incorrect = wrong
                            homework_entry.black_submitted = submitted
                            homework_entry.black_pct_wrong = pct_wrong
                    elif field in {"white_exempt", "black_exempt"}:
                        exempt_val = value == "true"
                        if field == "white_exempt":
                            homework_entry.white_exempt = exempt_val
                        else:
                            if match.black_student_id is not None:
                                homework_entry.black_exempt = exempt_val
                    else:
                        raise ValueError("Unsupported autosave field.")
                match.updated_at = datetime.utcnow()

            all_students = db.query(Student).filter(Student.classroom_id == classroom_id).all()
            all_matches = (
                db.query(Match)
                .join(Round)
                .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
                .filter(Round.classroom_id == classroom_id)
                .all()
            )
            recalculate_totals(all_students, all_matches, db=db)
            db.flush()
            return jsonify({"saved": True, "updated_at": datetime.utcnow().isoformat()})
        except ValueError as exc:
            db.rollback()
            return jsonify({"saved": False, "error": str(exc)}), 400
        except (SQLAlchemyError, TypeError, AttributeError, KeyError) as exc:
            db.rollback()
            logger.exception(
                "Autosave failed with recoverable error",
                extra={
                    "classroom_id": classroom_id,
                    "round_id": round_id,
                    "teacher_id": teacher.id,
                    "field": field,
                    "error_type": type(exc).__name__,
                },
            )
            return jsonify({"saved": False, "error": "Save failed due to a temporary data issue. Please retry."}), 500


@app.route("/classrooms/<int:classroom_id>/rounds/<int:round_id>/exempt_absent", methods=["POST"])
def exempt_absent_homework(classroom_id: int, round_id: int):
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    require_csrf()
    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        round_record = db.get(Round, round_id)
        if (
            not classroom
            or classroom.teacher_id != teacher.id
            or not round_record
            or round_record.classroom_id != classroom_id
        ):
            abort(404)

        # Exempt students who were absent in the PREVIOUS round
        prev_round = (
            db.query(Round)
            .filter(Round.classroom_id == classroom_id, Round.id < round_id)
            .order_by(Round.id.desc())
            .first()
        )
        absent_ids: set[int] = set()
        if prev_round:
            absent_ids = {
                rec.student_id
                for rec in db.query(Attendance)
                .filter(Attendance.round_id == prev_round.id, Attendance.status == "absent")
                .all()
            }

        exempted = 0
        matches = (
            db.query(Match)
            .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
            .filter(Match.round_id == round_id)
            .all()
        )
        round_assignment_types = (
            db.query(RoundAssignmentType)
            .filter(RoundAssignmentType.round_id == round_id)
            .all()
        )
        for match in matches:
            # Legacy HomeworkEntry exemption.
            hw = match.homework_entry
            if not hw:
                hw = HomeworkEntry(match_id=match.id)
                db.add(hw)
                match.homework_entry = hw

            newly_exempted_white = False
            newly_exempted_black = False
            if match.white_student_id in absent_ids and not hw.white_exempt:
                hw.white_exempt = True
                newly_exempted_white = True
                exempted += 1
            if match.black_student_id and match.black_student_id in absent_ids and not hw.black_exempt:
                hw.black_exempt = True
                newly_exempted_black = True
                exempted += 1

            # New AssignmentEntry exemptions (one per active type).
            entry_map = {e.assignment_type_id: e for e in match.assignment_entries}
            for rat in round_assignment_types:
                at_id = rat.assignment_type_id
                entry = entry_map.get(at_id)
                if not entry:
                    entry = AssignmentEntry(match_id=match.id, assignment_type_id=at_id)
                    db.add(entry)
                    entry_map[at_id] = entry
                if match.white_student_id in absent_ids and not entry.white_exempt:
                    entry.white_exempt = True
                if match.black_student_id and match.black_student_id in absent_ids and not entry.black_exempt:
                    entry.black_exempt = True

        all_students = db.query(Student).filter(Student.classroom_id == classroom_id).all()
        all_matches = (
            db.query(Match)
            .join(Round)
            .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
            .filter(Round.classroom_id == classroom_id)
            .all()
        )
        recalculate_totals(all_students, all_matches, db=db)
        db.flush()
        return jsonify({"ok": True, "exempted": exempted})


@app.route("/classrooms/<int:classroom_id>/exceptions")
def exceptions_queue(classroom_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        round_record = (
            db.query(Round)
            .filter(Round.classroom_id == classroom_id, Round.status == "completed")
            .order_by(Round.created_at.desc())
            .first()
        )
        if not round_record:
            return render_template(
                "exceptions_queue.html",
                classroom=classroom,
                round_record=None,
                missing_homework=[],
                missing_notation=[],
                unresolved_results=[],
                absences=[],
                active_nav="dashboard",
                breadcrumbs=[
                    {"label": "Dashboard", "url": url_for("dashboard")},
                    {"label": classroom.name, "url": url_for("classroom_overview", classroom_id=classroom.id)},
                    {"label": "Exceptions Queue"},
                ],
            )

        matches = (
            db.query(Match)
            .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
            .filter(Match.round_id == round_record.id)
            .order_by(Match.id)
            .all()
        )
        students = db.query(Student).filter(Student.classroom_id == classroom_id).all()
        student_map = {student.id: student for student in students}
        attendance_records = db.query(Attendance).filter(Attendance.round_id == round_record.id).all()

        def student_name(student_id: int | None) -> str:
            if student_id is None:
                return "Unknown student"
            student = student_map.get(student_id)
            return student.name if student else f"Student #{student_id}"

        missing_homework = []
        missing_notation = []
        unresolved_results = []
        for match in matches:
            if match.black_student_id is not None and not match.result:
                unresolved_results.append(match.id)
            homework = match.homework_entry
            if match.white_student_id and (not homework or not homework.white_submitted):
                missing_homework.append(f"Match {match.id}: {student_name(match.white_student_id)}")
            if match.black_student_id and (not homework or not homework.black_submitted):
                missing_homework.append(f"Match {match.id}: {student_name(match.black_student_id)}")
            if match.white_student_id and not match.notation_submitted_white:
                missing_notation.append(f"Match {match.id}: {student_name(match.white_student_id)}")
            if match.black_student_id and not match.notation_submitted_black:
                missing_notation.append(f"Match {match.id}: {student_name(match.black_student_id)}")

        absences = [
            student_name(record.student_id)
            for record in attendance_records
            if record.status in {"absent", "excused"}
        ]

    return render_template(
        "exceptions_queue.html",
        classroom=classroom,
        round_record=round_record,
        missing_homework=missing_homework,
        missing_notation=missing_notation,
        unresolved_results=unresolved_results,
        absences=absences,
        active_nav="dashboard",
        breadcrumbs=[
            {"label": "Dashboard", "url": url_for("dashboard")},
            {"label": classroom.name, "url": url_for("classroom_overview", classroom_id=classroom.id)},
            {"label": "Exceptions Queue"},
        ],
    )


@app.route("/classrooms/<int:classroom_id>/rounds/<int:round_id>/export")
def export_round(classroom_id: int, round_id: int):
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    with session_scope() as db:
        round_record = db.get(Round, round_id)
        if not round_record or round_record.classroom_id != classroom_id:
            abort(404)
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        matches = (
            db.query(Match)
            .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
            .filter(Match.round_id == round_id)
            .order_by(Match.id)
            .all()
        )
        students = (
            db.query(Student)
            .filter(Student.classroom_id == classroom_id)
            .all()
        )
        student_map = {student.id: student for student in students}

    output_file = OUTPUTS_DIR / f"next_matches_{classroom_id}_{round_id}.csv"
    display_round_number = round_record.round_number if round_record.round_number is not None else round_record.id
    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "Round Number",
                "White Player",
                "White Player Strength",
                "Black Player",
                "Black Player Strength",
                "Who Won",
                "White Homework Correct",
                "White Homework Incorrect",
                "Black Homework Correct",
                "Black Homework Incorrect",
                "White Homework Submitted",
                "Black Homework Submitted",
                "White Notation Submitted",
                "Black Notation Submitted",
                "Notes",
            ]
        )
        for match in matches:
            homework = match.homework_entry
            writer.writerow(
                [
                    display_round_number,
                    student_map.get(match.white_student_id).name
                    if match.white_student_id
                    else "",
                    match.white_strength or "",
                    student_map.get(match.black_student_id).name
                    if match.black_student_id
                    else "",
                    match.black_strength or "",
                    (match.result or "").title() if match.result else "",
                    homework.white_correct if homework else 0,
                    homework.white_incorrect if homework else 0,
                    homework.black_correct if homework else 0,
                    homework.black_incorrect if homework else 0,
                    homework.white_submitted if homework else False,
                    homework.black_submitted if homework else False,
                    match.notation_submitted_white,
                    match.notation_submitted_black,
                    match.notes,
                ]
            )

    return send_file(output_file, as_attachment=True)


@app.route("/classrooms/<int:classroom_id>/rounds/<int:round_id>/delete", methods=["POST"])
def delete_round(classroom_id: int, round_id: int):
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    require_csrf()

    with session_scope() as db:
        round_record = db.get(Round, round_id)
        if not round_record or round_record.classroom_id != classroom_id:
            abort(404)
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        students = (
            db.query(Student)
            .filter(Student.classroom_id == classroom_id)
            .all()
        )
        # All matches for this classroom *excluding* the round being deleted,
        # so recalculate_totals produces stats as if this round never happened.
        remaining_matches = (
            db.query(Match)
            .join(Round)
            .options(selectinload(Match.homework_entry), selectinload(Match.assignment_entries))
            .filter(Round.classroom_id == classroom_id, Match.round_id != round_id)
            .all()
        )
        recalculate_totals(students, remaining_matches, db=db)
        db.delete(round_record)

    return redirect(
        url_for("classroom_overview", classroom_id=classroom_id, success=f"Round {round_id} deleted.")
    )


@app.route("/classrooms/<int:classroom_id>/import", methods=["GET", "POST"])
def import_students(classroom_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    error = None
    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)

        if request.method == "POST":
            require_csrf()
            upload = request.files.get("file")
            replace_existing = request.form.get("replace_existing") == "on"

            if not upload or upload.filename == "":
                error = "Please upload a CSV file."
            else:
                filename = secure_filename(upload.filename)
                if not allowed_file(filename):
                    error = "Only .csv files are accepted."
                else:
                    path = UPLOADS_DIR / filename
                    upload.save(path)
                    df = pd.read_csv(path)
                    df.columns = df.columns.str.strip()
                    required = {
                        "Student Name",
                        "Total Wins",
                        "Total Losses",
                        "Total Ties",
                        "# Times Played White",
                        "# Times Played Black",
                        "Correct Homework",
                        "Incorrect Homework",
                        "Notes",
                    }
                    missing = required - set(df.columns)
                    if missing:
                        error = f"Missing columns: {', '.join(sorted(missing))}"
                    else:
                        def parse_int(value) -> int:
                            if pd.isna(value) or value == "":
                                return 0
                            parsed = int(float(value))
                            if parsed < 0:
                                raise ValueError("Numeric values cannot be negative.")
                            return parsed

                        students_to_add = []
                        try:
                            for _, row in df.iterrows():
                                name = str(row["Student Name"]).strip()
                                if not name:
                                    continue
                                students_to_add.append(
                                    Student(
                                        classroom_id=classroom_id,
                                        name=name,
                                        total_wins=parse_int(row["Total Wins"]),
                                        total_losses=parse_int(row["Total Losses"]),
                                        total_ties=parse_int(row["Total Ties"]),
                                        times_white=parse_int(row["# Times Played White"]),
                                        times_black=parse_int(row["# Times Played Black"]),
                                        homework_correct=parse_int(row["Correct Homework"]),
                                        homework_incorrect=parse_int(row["Incorrect Homework"]),
                                        notes=str(row["Notes"]) if not pd.isna(row["Notes"]) else "",
                                        active=True,
                                    )
                                )
                        except ValueError as exc:
                            error = str(exc)
                        else:
                            if replace_existing:
                                db.query(Student).filter(
                                    Student.classroom_id == classroom_id
                                ).delete()
                            db.add_all(students_to_add)
                            return redirect(
                                url_for("classroom_overview", classroom_id=classroom_id)
                            )

    return render_template(
        "import_students.html",
        classroom=classroom,
        error=error,
        active_nav="dashboard",
        breadcrumbs=[
            {"label": "Dashboard", "url": url_for("dashboard")},
            {"label": classroom.name, "url": url_for("classroom_overview", classroom_id=classroom.id)},
            {"label": "Import Students"},
        ],
    )


@app.route("/about")
def about_page() -> str:
    return render_template("about.html", active_nav="about")


if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", debug=debug_enabled)
