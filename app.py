from __future__ import annotations

import csv
import json
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
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import selectinload
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from db import (
    Base,
    IS_PRODUCTION,
    database_url_warnings,
    engine,
    redacted_database_url,
    session_scope,
    verify_schema_compatibility,
)
from models import (
    Attendance,
    AuditLog,
    Classroom,
    HomeworkEntry,
    Match,
    Round,
    Student,
    Teacher,
)
from pairing_logic import normalize_weights
from services import create_match_records, generate_matches_for_students, recalculate_totals


BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WIN_WEIGHT = 0.7
DEFAULT_HOMEWORK_WEIGHT = 0.3
DEFAULT_HOMEWORK_TOTAL_QUESTIONS = 10
DEFAULT_MISSING_HOMEWORK_POLICY = "zero"
DEFAULT_MISSING_HOMEWORK_PENALTY = 1
DEFAULT_HOMEWORK_METRIC_MODE = "pct_wrong"
HOMEWORK_POLICIES = {"zero", "exclude", "penalty"}
HOMEWORK_METRIC_MODES = {"raw_counts", "pct_wrong", "pct_correct"}
ATTENDANCE_STATUSES = {"present", "absent", "excused", "late"}
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
        schema_issues = verify_schema_compatibility(fail_fast=IS_PRODUCTION)
        if schema_issues:
            logger.warning(
                "Database schema compatibility issues detected",
                extra={"database_url": redacted_database_url(), "issues": schema_issues},
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


def parse_non_negative_int(raw_value: Optional[str], default: int = 0) -> int:
    if raw_value is None or raw_value == "":
        return default
    parsed = int(float(raw_value))
    if parsed < 0:
        raise ValueError("Values cannot be negative.")
    return parsed


def parse_bool(raw_value: Optional[str]) -> bool:
    return str(raw_value).lower() in {"1", "true", "on", "yes"}


def log_audit(
    db,
    teacher_id: int,
    classroom_id: int,
    action: str,
    round_id: Optional[int] = None,
    match_id: Optional[int] = None,
    payload: Optional[dict] = None,
) -> None:
    db.add(
        AuditLog(
            teacher_id=teacher_id,
            classroom_id=classroom_id,
            round_id=round_id,
            match_id=match_id,
            action=action,
            payload=json.dumps(payload or {}, sort_keys=True),
        )
    )


def apply_match_form_updates(
    db,
    match: Match,
    round_record: Round,
    form_data,
) -> None:
    result = form_data.get(f"result_{match.id}", "").strip().lower()
    notes = form_data.get(f"notes_{match.id}", "").strip()
    notation_white = parse_bool(form_data.get(f"notation_white_{match.id}", "0"))
    notation_black = parse_bool(form_data.get(f"notation_black_{match.id}", "0"))

    if match.black_student_id is None:
        result = "bye"
        notation_black = True

    if result not in {"white", "black", "tie", "bye", ""}:
        result = ""

    homework_entry = match.homework_entry
    if not homework_entry:
        homework_entry = HomeworkEntry(match_id=match.id)
        db.add(homework_entry)
        match.homework_entry = homework_entry

    total_questions = max(0, int(round_record.homework_total_questions or 0))

    def compute_homework(prefix: str) -> tuple[int, int, bool, float]:
        submitted = parse_bool(form_data.get(f"{prefix}_submitted_{match.id}", "0"))
        correct_raw = form_data.get(f"{prefix}_correct_{match.id}", "")
        if correct_raw == "" and not submitted:
            return 0, 0, False, 0.0
        if correct_raw != "" and not submitted:
            submitted = True
        correct = parse_non_negative_int(correct_raw, default=0)
        if total_questions > 0:
            correct = min(correct, total_questions)
            incorrect = max(0, total_questions - correct)
        else:
            incorrect = parse_non_negative_int(
                form_data.get(f"{prefix}_incorrect_{match.id}", "0"),
                default=0,
            )
        denominator = correct + incorrect
        pct_wrong = (incorrect / denominator) if denominator else 0.0
        return correct, incorrect, submitted, pct_wrong

    white_correct, white_incorrect, white_submitted, white_pct_wrong = compute_homework("white")
    black_correct, black_incorrect, black_submitted, black_pct_wrong = compute_homework("black")

    match.result = result or None
    match.notes = notes
    match.notation_submitted_white = notation_white
    match.notation_submitted_black = notation_black
    match.updated_at = datetime.utcnow()

    homework_entry.white_correct = white_correct
    homework_entry.white_incorrect = white_incorrect
    homework_entry.black_correct = black_correct
    homework_entry.black_incorrect = black_incorrect
    homework_entry.white_submitted = white_submitted
    homework_entry.black_submitted = black_submitted
    homework_entry.white_pct_wrong = white_pct_wrong
    homework_entry.black_pct_wrong = black_pct_wrong


def get_classroom_or_404(classroom_id: int, teacher_id: int) -> Classroom:
    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher_id:
            abort(404)
        return classroom


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

    return render_template("setup.html", error=error)


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

    return render_template("signup.html", error=error)


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

    return render_template("login.html", error=error)


@app.errorhandler(Exception)
def handle_unexpected_error(error: Exception):
    if isinstance(error, HTTPException):
        return error
    support_id = uuid.uuid4().hex[:12]
    log_exception(error, support_id)
    return render_template("500.html", support_id=support_id), 500


@app.route("/health/db")
def health_db():
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {
            "status": "ok",
            "database_url": redacted_database_url(),
            "warnings": database_url_warnings(),
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
                db.commit()
                return redirect(url_for("dashboard"))

    return render_template("dashboard.html", teacher=teacher, classrooms=classrooms, error=error)


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
        attendance_records = (
            db.query(Attendance)
            .join(Round, Attendance.round_id == Round.id)
            .filter(Round.classroom_id == classroom_id)
            .all()
        )

        attendance_by_student: dict[int, dict[str, float]] = {}
        for student in students:
            attendance_by_student[student.id] = {
                "present": 0,
                "absent": 0,
                "excused": 0,
                "late": 0,
                "total": 0,
                "absence_rate": 0.0,
            }
        for record in attendance_records:
            if record.student_id not in attendance_by_student:
                continue
            status = record.status if record.status in ATTENDANCE_STATUSES else "present"
            attendance_by_student[record.student_id][status] += 1
            attendance_by_student[record.student_id]["total"] += 1
        for student_id, stats in attendance_by_student.items():
            total = stats["total"]
            if total:
                stats["absence_rate"] = round((stats["absent"] / total) * 100, 1)
            stats["present_streak"] = 0
            stats["absence_streak"] = 0
        round_ids_desc = [round_obj.id for round_obj in rounds]
        statuses_by_student_round: dict[tuple[int, int], str] = {}
        for record in attendance_records:
            statuses_by_student_round[(record.student_id, record.round_id)] = record.status
        for student in students:
            present_streak = 0
            absence_streak = 0
            for round_id in round_ids_desc:
                status = statuses_by_student_round.get((student.id, round_id), "present")
                if status in {"present", "late"}:
                    if absence_streak == 0:
                        present_streak += 1
                    else:
                        break
                elif status in {"absent", "excused"}:
                    if present_streak == 0:
                        absence_streak += 1
                    else:
                        break
            attendance_by_student[student.id]["present_streak"] = present_streak
            attendance_by_student[student.id]["absence_streak"] = absence_streak

    return render_template(
        "classroom.html",
        teacher=teacher,
        classroom=classroom,
        students=students,
        rounds=rounds,
        attendance_by_student=attendance_by_student,
    )


@app.route("/classrooms/<int:classroom_id>/exceptions")
def classroom_exceptions(classroom_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher

    with session_scope() as db:
        classroom = db.get(Classroom, classroom_id)
        if not classroom or classroom.teacher_id != teacher.id:
            abort(404)
        active_round = (
            db.query(Round)
            .filter(Round.classroom_id == classroom_id, Round.status == "open")
            .order_by(Round.created_at.desc())
            .first()
        )
        if not active_round:
            return render_template(
                "exceptions.html",
                classroom=classroom,
                active_round=None,
                exceptions=[],
            )
        matches = (
            db.query(Match)
            .options(selectinload(Match.homework_entry))
            .filter(Match.round_id == active_round.id)
            .order_by(Match.id)
            .all()
        )
        students = (
            db.query(Student)
            .filter(Student.classroom_id == classroom_id)
            .all()
        )
        student_map = {student.id: student for student in students}
        attendance_records = (
            db.query(Attendance)
            .filter(Attendance.round_id == active_round.id)
            .all()
        )
        attendance_map = {record.student_id: record.status for record in attendance_records}

        exceptions = []
        for match in matches:
            white = student_map.get(match.white_student_id)
            black = student_map.get(match.black_student_id)
            if match.black_student_id is not None and not (match.result or "").strip():
                exceptions.append(
                    {
                        "type": "Missing result",
                        "detail": f"Match {match.id}: {white.name if white else 'TBD'} vs {black.name if black else 'TBD'}",
                    }
                )
            if not (match.notation_submitted_white and match.notation_submitted_black):
                exceptions.append(
                    {
                        "type": "Missing notation",
                        "detail": f"Match {match.id} notation incomplete",
                    }
                )
            homework = match.homework_entry
            if homework:
                if not homework.white_submitted and match.white_student_id:
                    exceptions.append(
                        {
                            "type": "Missing homework",
                            "detail": f"{white.name if white else 'White'} has not submitted homework",
                        }
                    )
                if not homework.black_submitted and match.black_student_id:
                    exceptions.append(
                        {
                            "type": "Missing homework",
                            "detail": f"{black.name if black else 'Black'} has not submitted homework",
                        }
                    )
        for student in students:
            status = attendance_map.get(student.id, "present")
            if status in {"absent", "excused"}:
                exceptions.append(
                    {
                        "type": "Absence",
                        "detail": f"{student.name}: {status}",
                    }
                )

    return render_template(
        "exceptions.html",
        classroom=classroom,
        active_round=active_round,
        exceptions=exceptions,
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

        error = None
        if request.method == "POST":
            require_csrf()
            win_weight_raw = request.form.get("win_weight", "").strip()
            homework_weight_raw = request.form.get("homework_weight", "").strip()
            homework_total_questions_raw = request.form.get(
                "homework_total_questions",
                str(DEFAULT_HOMEWORK_TOTAL_QUESTIONS),
            ).strip()
            missing_homework_policy = request.form.get(
                "missing_homework_policy",
                DEFAULT_MISSING_HOMEWORK_POLICY,
            ).strip()
            missing_homework_penalty_raw = request.form.get(
                "missing_homework_penalty",
                str(DEFAULT_MISSING_HOMEWORK_PENALTY),
            ).strip()
            homework_metric_mode = request.form.get(
                "homework_metric_mode",
                DEFAULT_HOMEWORK_METRIC_MODE,
            ).strip()

            try:
                win_weight = float(win_weight_raw) if win_weight_raw else DEFAULT_WIN_WEIGHT
                homework_weight = (
                    float(homework_weight_raw)
                    if homework_weight_raw
                    else DEFAULT_HOMEWORK_WEIGHT
                )
                homework_total_questions = parse_non_negative_int(
                    homework_total_questions_raw,
                    default=DEFAULT_HOMEWORK_TOTAL_QUESTIONS,
                )
                missing_homework_penalty = parse_non_negative_int(
                    missing_homework_penalty_raw,
                    default=DEFAULT_MISSING_HOMEWORK_PENALTY,
                )
            except ValueError:
                error = "Weights and homework settings must be valid numbers."
            else:
                if missing_homework_policy not in HOMEWORK_POLICIES:
                    missing_homework_policy = DEFAULT_MISSING_HOMEWORK_POLICY
                if homework_metric_mode not in HOMEWORK_METRIC_MODES:
                    homework_metric_mode = DEFAULT_HOMEWORK_METRIC_MODE

                attendance_map: dict[int, str] = {}
                for student in students:
                    status = request.form.get(
                        f"attendance_status_{student.id}",
                        "present",
                    ).strip()
                    if status not in ATTENDANCE_STATUSES:
                        status = "present"
                    attendance_map[student.id] = status

                absent_ids = {
                    student_id
                    for student_id, status in attendance_map.items()
                    if status in {"absent", "excused"}
                }
                present_students = [
                    student for student in students if student.id not in absent_ids
                ]

                if win_weight < 0 or homework_weight < 0:
                    error = "Weights must be zero or greater."
                elif len(present_students) < 2:
                    error = "At least two present students are required to create matches."
                else:
                    try:
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
                        normalized_win, normalized_homework = normalize_weights(
                            win_weight, homework_weight
                        )
                        matches, unpaired, df, id_order = generate_matches_for_students(
                            present_students,
                            normalized_win,
                            normalized_homework,
                            homework_metric_mode=homework_metric_mode,
                        )
                    except ValueError as exc:
                        error = str(exc)
                    else:
                        round_record = Round(
                            classroom_id=classroom_id,
                            win_weight=int(normalized_win * 100),
                            homework_weight=int(normalized_homework * 100),
                            status="open",
                            homework_total_questions=homework_total_questions,
                            missing_homework_policy=missing_homework_policy,
                            missing_homework_penalty=missing_homework_penalty,
                            homework_metric_mode=homework_metric_mode,
                        )
                        db.add(round_record)
                        db.flush()

                        for student in students:
                            status = attendance_map.get(student.id, "present")
                            db.add(
                                Attendance(
                                    round_id=round_record.id,
                                    student_id=student.id,
                                    status=status,
                                )
                            )

                        match_records = create_match_records(
                            present_students, matches, unpaired, df, id_order
                        )
                        for record in match_records:
                            record.round_id = round_record.id
                            record.homework_entry = HomeworkEntry(
                                white_submitted=False,
                                black_submitted=False,
                            )
                            db.add(record)
                        log_audit(
                            db,
                            teacher_id=teacher.id,
                            classroom_id=classroom_id,
                            round_id=round_record.id,
                            action="round.created",
                            payload={
                                "win_weight": normalized_win,
                                "homework_weight": normalized_homework,
                                "homework_total_questions": homework_total_questions,
                                "missing_homework_policy": missing_homework_policy,
                                "missing_homework_penalty": missing_homework_penalty,
                                "homework_metric_mode": homework_metric_mode,
                            },
                        )

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
                            url_for(
                                "round_results",
                                classroom_id=classroom_id,
                                round_id=round_record.id,
                            )
                        )

    return render_template(
        "new_round.html",
        classroom=classroom,
        students=students,
        error=error,
        default_win_weight=DEFAULT_WIN_WEIGHT,
        default_homework_weight=DEFAULT_HOMEWORK_WEIGHT,
        default_homework_total_questions=DEFAULT_HOMEWORK_TOTAL_QUESTIONS,
        default_missing_homework_policy=DEFAULT_MISSING_HOMEWORK_POLICY,
        default_missing_homework_penalty=DEFAULT_MISSING_HOMEWORK_PENALTY,
        default_homework_metric_mode=DEFAULT_HOMEWORK_METRIC_MODE,
    )


@app.route("/classrooms/<int:classroom_id>/rounds/<int:round_id>", methods=["GET", "POST"])
def round_results(classroom_id: int, round_id: int) -> str:
    teacher = require_login()
    if not isinstance(teacher, Teacher):
        return teacher
    error: Optional[str] = None
    success: Optional[str] = None
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
            .options(selectinload(Match.homework_entry))
            .filter(Match.round_id == round_id)
            .order_by(Match.id)
            .all()
        )
        attendance_records = (
            db.query(Attendance)
            .join(Round, Attendance.round_id == Round.id)
            .filter(Round.classroom_id == classroom_id, Round.id == round_id)
            .all()
        )

        students = (
            db.query(Student)
            .filter(Student.classroom_id == classroom_id)
            .all()
        )
        student_map = {student.id: student for student in students}
        attendance_map = {record.student_id: record for record in attendance_records}

        if request.method == "POST":
            require_csrf()
            try:
                action = request.form.get("action", "save").strip().lower()
                for match in matches:
                    apply_match_form_updates(db, match, round_record, request.form)
                    log_audit(
                        db,
                        teacher_id=teacher.id,
                        classroom_id=classroom_id,
                        round_id=round_id,
                        match_id=match.id,
                        action="match.edited",
                        payload={"via": "round_results"},
                    )

                for student in students:
                    attendance_status = request.form.get(
                        f"attendance_status_{student.id}",
                        attendance_map.get(student.id).status if student.id in attendance_map else "present",
                    ).strip()
                    if attendance_status not in ATTENDANCE_STATUSES:
                        attendance_status = "present"
                    record = attendance_map.get(student.id)
                    if record:
                        record.status = attendance_status

                all_matches = (
                    db.query(Match)
                    .join(Round)
                    .options(selectinload(Match.homework_entry), selectinload(Match.round))
                    .filter(Round.classroom_id == classroom_id)
                    .all()
                )
                recalculate_totals(students, all_matches)
                unresolved_matches = [
                    match.id
                    for match in matches
                    if match.black_student_id is not None and not (match.result or "").strip()
                ]
                missing_notation = [
                    match.id
                    for match in matches
                    if not (match.notation_submitted_white and match.notation_submitted_black)
                ]
                override = parse_bool(request.form.get("override_completion", "0"))
                override_reason = request.form.get("override_reason", "").strip()
                if action == "complete":
                    if (unresolved_matches or missing_notation) and not override:
                        error = (
                            "Round has unresolved results/notation. "
                            "Use override with a reason to complete anyway."
                        )
                        round_record.status = "open"
                    elif override and not override_reason:
                        error = "Override reason is required when override is enabled."
                        round_record.status = "open"
                    else:
                        round_record.status = "completed"
                        round_record.completion_override_reason = override_reason if override else ""
                        success = "Round marked complete."
                        log_audit(
                            db,
                            teacher_id=teacher.id,
                            classroom_id=classroom_id,
                            round_id=round_id,
                            action="round.completed",
                            payload={
                                "override": override,
                                "override_reason": override_reason,
                                "unresolved_matches": unresolved_matches,
                                "missing_notation": missing_notation,
                            },
                        )
                else:
                    round_record.status = "open"
                    success = "Draft saved."
                    log_audit(
                        db,
                        teacher_id=teacher.id,
                        classroom_id=classroom_id,
                        round_id=round_id,
                        action="round.saved",
                        payload={"status": "open"},
                    )
            except ValueError as exc:
                error = str(exc)

        opponent_counts: dict[tuple[int, int], int] = {}
        prior_matches = (
            db.query(Match)
            .join(Round, Match.round_id == Round.id)
            .filter(Round.classroom_id == classroom_id, Round.id < round_id)
            .all()
        )
        for prior in prior_matches:
            if prior.white_student_id and prior.black_student_id:
                key = tuple(sorted((prior.white_student_id, prior.black_student_id)))
                opponent_counts[key] = opponent_counts.get(key, 0) + 1

        diagnostics = []
        for match in matches:
            white = student_map.get(match.white_student_id)
            black = student_map.get(match.black_student_id)
            repeat_count = 0
            if white and black:
                repeat_count = opponent_counts.get(tuple(sorted((white.id, black.id))), 0)
            white_color_diff = (white.times_white - white.times_black) if white else 0
            black_color_diff = (black.times_white - black.times_black) if black else 0
            diagnostics.append(
                {
                    "match_id": match.id,
                    "repeat_count": repeat_count,
                    "white_color_diff": white_color_diff,
                    "black_color_diff": black_color_diff,
                }
            )

        unresolved_count = sum(
            1
            for match in matches
            if match.black_student_id is not None and not (match.result or "").strip()
        )
        missing_notation_count = sum(
            1
            for match in matches
            if not (match.notation_submitted_white and match.notation_submitted_black)
        )

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
        matches=matches,
        student_map=student_map,
        error=error,
        success=success,
        attendance_map=attendance_map,
        diagnostics=diagnostics,
        unresolved_count=unresolved_count,
        missing_notation_count=missing_notation_count,
    )


@app.route(
    "/classrooms/<int:classroom_id>/rounds/<int:round_id>/autosave",
    methods=["POST"],
)
def autosave_round_result(classroom_id: int, round_id: int):
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

        match_id = parse_non_negative_int(request.form.get("match_id", "0"), default=0)
        match = (
            db.query(Match)
            .options(selectinload(Match.homework_entry))
            .filter(Match.round_id == round_id, Match.id == match_id)
            .first()
        )
        if not match:
            abort(404)
        apply_match_form_updates(db, match, round_record, request.form)
        log_audit(
            db,
            teacher_id=teacher.id,
            classroom_id=classroom_id,
            round_id=round_id,
            match_id=match.id,
            action="match.autosaved",
            payload={"match_id": match.id},
        )
        return jsonify(
            {
                "ok": True,
                "match_id": match.id,
                "updated_at": match.updated_at.isoformat() if match.updated_at else None,
            }
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
            .options(selectinload(Match.homework_entry))
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
    with output_file.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "White Player",
                "White Player Strength",
                "Black Player",
                "Black Player Strength",
                "Who Won",
                "White Homework Correct",
                "White Homework Incorrect",
                "Black Homework Correct",
                "Black Homework Incorrect",
                "White Notation Submitted",
                "Black Notation Submitted",
                "Notes",
            ]
        )
        for match in matches:
            homework = match.homework_entry
            white_student = student_map.get(match.white_student_id) if match.white_student_id else None
            black_student = student_map.get(match.black_student_id) if match.black_student_id else None
            writer.writerow(
                [
                    white_student.name if white_student else "",
                    match.white_strength or "",
                    black_student.name if black_student else "",
                    match.black_strength or "",
                    (match.result or "").title() if match.result else "",
                    homework.white_correct if homework else 0,
                    homework.white_incorrect if homework else 0,
                    homework.black_correct if homework else 0,
                    homework.black_incorrect if homework else 0,
                    "yes" if match.notation_submitted_white else "no",
                    "yes" if match.notation_submitted_black else "no",
                    match.notes,
                ]
            )

    return send_file(output_file, as_attachment=True)


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

    return render_template("import_students.html", classroom=classroom, error=error)


@app.route("/about")
def about_page() -> str:
    return render_template("about.html")


if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", debug=debug_enabled)
