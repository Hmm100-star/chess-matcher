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
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from flask import has_request_context
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
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
from models import Attendance, Classroom, HomeworkEntry, Match, Round, Student, Teacher
from pairing_logic import normalize_weights
from services import create_match_records, generate_matches_for_students, recalculate_totals


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
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        if not username or not password:
            error = "Username and password are required."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            with session_scope() as db:
                teacher = Teacher(
                    username=username,
                    password_hash=generate_password_hash(password),
                )
                db.add(teacher)
            return redirect(url_for("login"))

    return render_template("setup.html", error=error)


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

    return render_template(
        "classroom.html",
        teacher=teacher,
        classroom=classroom,
        students=students,
        rounds=rounds,
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

            try:
                win_weight = float(win_weight_raw) if win_weight_raw else DEFAULT_WIN_WEIGHT
                homework_weight = (
                    float(homework_weight_raw)
                    if homework_weight_raw
                    else DEFAULT_HOMEWORK_WEIGHT
                )
            except ValueError:
                error = "Weights must be numeric."
            else:
                absent_ids = {
                    int(value)
                    for value in request.form.getlist("absent_students")
                    if value.isdigit()
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
                            present_students, normalized_win, normalized_homework
                        )
                    except ValueError as exc:
                        error = str(exc)
                    else:
                        round_record = Round(
                            classroom_id=classroom_id,
                            win_weight=int(normalized_win * 100),
                            homework_weight=int(normalized_homework * 100),
                            status="open",
                        )
                        db.add(round_record)
                        db.flush()

                        for student in students:
                            status = "absent" if student.id in absent_ids else "present"
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

        if request.method == "POST":
            require_csrf()
            try:
                for match in matches:
                    result = request.form.get(f"result_{match.id}", "").strip().lower()
                    notes = request.form.get(f"notes_{match.id}", "").strip()

                    if match.black_student_id is None:
                        result = "bye"

                    if result not in {"white", "black", "tie", "bye", ""}:
                        continue

                    match.result = result or None
                    match.notes = notes
                    match.updated_at = datetime.utcnow()

                    homework_entry = match.homework_entry
                    if not homework_entry:
                        homework_entry = HomeworkEntry(match_id=match.id)
                        db.add(homework_entry)
                        match.homework_entry = homework_entry

                    def parse_int(value: str) -> int:
                        if value is None or value == "":
                            return 0
                        parsed = int(float(value))
                        if parsed < 0:
                            raise ValueError("Homework counts cannot be negative.")
                        return parsed

                    homework_entry.white_correct = parse_int(
                        request.form.get(f"white_correct_{match.id}", "0")
                    )
                    homework_entry.white_incorrect = parse_int(
                        request.form.get(f"white_incorrect_{match.id}", "0")
                    )
                    homework_entry.black_correct = parse_int(
                        request.form.get(f"black_correct_{match.id}", "0")
                    )
                    homework_entry.black_incorrect = parse_int(
                        request.form.get(f"black_incorrect_{match.id}", "0")
                    )

                all_matches = (
                    db.query(Match)
                    .join(Round)
                    .filter(Round.classroom_id == classroom_id)
                    .all()
                )
                recalculate_totals(students, all_matches)
                round_record.status = "completed"
                return redirect(
                    url_for("round_results", classroom_id=classroom_id, round_id=round_id)
                )
            except ValueError as exc:
                error = str(exc)

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
                "Notes",
            ]
        )
        for match in matches:
            homework = match.homework_entry
            writer.writerow(
                [
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
