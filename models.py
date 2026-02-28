from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from db import Base


class Teacher(Base):
    __tablename__ = "teachers"

    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    classrooms = relationship("Classroom", back_populates="teacher", cascade="all, delete-orphan")


class Classroom(Base):
    __tablename__ = "classrooms"

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, ForeignKey("teachers.id"), nullable=False)
    name = Column(String(200), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    teacher = relationship("Teacher", back_populates="classrooms")
    students = relationship("Student", back_populates="classroom", cascade="all, delete-orphan")
    rounds = relationship("Round", back_populates="classroom", cascade="all, delete-orphan")
    assignment_types = relationship("AssignmentType", back_populates="classroom", cascade="all, delete-orphan", order_by="AssignmentType.id")


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"), nullable=False)
    name = Column(String(200), nullable=False)
    total_wins = Column(Integer, default=0, nullable=False)
    total_losses = Column(Integer, default=0, nullable=False)
    total_ties = Column(Integer, default=0, nullable=False)
    times_white = Column(Integer, default=0, nullable=False)
    times_black = Column(Integer, default=0, nullable=False)
    homework_correct = Column(Integer, default=0, nullable=False)
    homework_incorrect = Column(Integer, default=0, nullable=False)
    notes = Column(Text, default="", nullable=False)
    active = Column(Boolean, default=True, nullable=False)

    classroom = relationship("Classroom", back_populates="students")
    attendance = relationship("Attendance", back_populates="student", cascade="all, delete-orphan")
    assignment_scores = relationship("StudentAssignmentScore", back_populates="student", cascade="all, delete-orphan")


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"), nullable=False)
    round_number = Column(Integer, nullable=True)  # sequential per-classroom (NULL for legacy rows)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    win_weight = Column(Integer, default=70, nullable=False)
    homework_weight = Column(Integer, default=30, nullable=False)
    status = Column(String(50), default="open", nullable=False)
    homework_total_questions = Column(Integer, default=0, nullable=False)
    missing_homework_policy = Column(String(20), default="zero", nullable=False)  # legacy; superseded by missing_score_pct
    missing_score_pct = Column(Float, nullable=True)  # None=exclude, 0-100=treat blanks as that % correct
    homework_metric_mode = Column(String(20), default="pct_correct", nullable=False)
    completion_override_reason = Column(Text, default="", nullable=False)

    classroom = relationship("Classroom", back_populates="rounds")
    matches = relationship("Match", back_populates="round", cascade="all, delete-orphan")
    attendance_records = relationship(
        "Attendance", back_populates="round", cascade="all, delete-orphan"
    )
    round_assignment_types = relationship(
        "RoundAssignmentType", back_populates="round", cascade="all, delete-orphan"
    )


class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    status = Column(String(20), default="present", nullable=False)

    round = relationship("Round", back_populates="attendance_records")
    student = relationship("Student", back_populates="attendance")


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    white_student_id = Column(Integer, ForeignKey("students.id"))
    black_student_id = Column(Integer, ForeignKey("students.id"))
    white_strength = Column(String(20))
    black_strength = Column(String(20))
    result = Column(String(20))
    notes = Column(Text, default="", nullable=False)
    notation_submitted_white = Column(Boolean, default=False, nullable=False)
    notation_submitted_black = Column(Boolean, default=False, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    round = relationship("Round", back_populates="matches")
    homework_entry = relationship(
        "HomeworkEntry", back_populates="match", uselist=False, cascade="all, delete-orphan"
    )
    assignment_entries = relationship(
        "AssignmentEntry", back_populates="match", cascade="all, delete-orphan"
    )


class HomeworkEntry(Base):
    __tablename__ = "homework_entries"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    white_correct = Column(Integer, default=0, nullable=False)
    white_incorrect = Column(Integer, default=0, nullable=False)
    black_correct = Column(Integer, default=0, nullable=False)
    black_incorrect = Column(Integer, default=0, nullable=False)
    white_submitted = Column(Boolean, default=False, nullable=False)
    black_submitted = Column(Boolean, default=False, nullable=False)
    white_pct_wrong = Column(Float, default=0.0, nullable=False)
    black_pct_wrong = Column(Float, default=0.0, nullable=False)
    white_exempt = Column(Boolean, default=False, nullable=False)
    black_exempt = Column(Boolean, default=False, nullable=False)

    match = relationship("Match", back_populates="homework_entry")


class AssignmentType(Base):
    """A teacher-defined assignment category (e.g. Homework, Tactics Sheet) scoped to a classroom."""
    __tablename__ = "assignment_types"

    id = Column(Integer, primary_key=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"), nullable=False)
    name = Column(String(200), nullable=False)
    metric_mode = Column(String(20), default="pct_correct", nullable=False)
    missing_policy = Column(String(20), default="zero", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    classroom = relationship("Classroom", back_populates="assignment_types")
    round_assignment_types = relationship(
        "RoundAssignmentType", back_populates="assignment_type", cascade="all, delete-orphan"
    )
    assignment_entries = relationship(
        "AssignmentEntry", back_populates="assignment_type", cascade="all, delete-orphan"
    )
    student_scores = relationship(
        "StudentAssignmentScore", back_populates="assignment_type", cascade="all, delete-orphan"
    )


class RoundAssignmentType(Base):
    """Junction: which assignment types are active for a round, with per-round weight and config."""
    __tablename__ = "round_assignment_types"

    id = Column(Integer, primary_key=True)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=False)
    assignment_type_id = Column(Integer, ForeignKey("assignment_types.id"), nullable=False)
    weight = Column(Integer, default=30, nullable=False)  # integer 0-100
    total_questions = Column(Integer, default=0, nullable=False)

    round = relationship("Round", back_populates="round_assignment_types")
    assignment_type = relationship("AssignmentType", back_populates="round_assignment_types")


class AssignmentEntry(Base):
    """Per-match per-assignment-type score entry (generalisation of HomeworkEntry)."""
    __tablename__ = "assignment_entries"

    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    assignment_type_id = Column(Integer, ForeignKey("assignment_types.id"), nullable=False)
    white_correct = Column(Integer, default=0, nullable=False)
    white_incorrect = Column(Integer, default=0, nullable=False)
    black_correct = Column(Integer, default=0, nullable=False)
    black_incorrect = Column(Integer, default=0, nullable=False)
    white_submitted = Column(Boolean, default=False, nullable=False)
    black_submitted = Column(Boolean, default=False, nullable=False)
    white_pct_wrong = Column(Float, default=0.0, nullable=False)
    black_pct_wrong = Column(Float, default=0.0, nullable=False)
    white_exempt = Column(Boolean, default=False, nullable=False)
    black_exempt = Column(Boolean, default=False, nullable=False)

    match = relationship("Match", back_populates="assignment_entries")
    assignment_type = relationship("AssignmentType", back_populates="assignment_entries")


class StudentAssignmentScore(Base):
    """Cumulative correct/incorrect totals per student per assignment type (replaces Student.homework_correct/incorrect)."""
    __tablename__ = "student_assignment_scores"

    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    assignment_type_id = Column(Integer, ForeignKey("assignment_types.id"), nullable=False)
    correct = Column(Integer, default=0, nullable=False)
    incorrect = Column(Integer, default=0, nullable=False)

    student = relationship("Student", back_populates="assignment_scores")
    assignment_type = relationship("AssignmentType", back_populates="student_scores")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    classroom_id = Column(Integer, nullable=False)
    round_id = Column(Integer, nullable=False)
    match_id = Column(Integer, nullable=False)
    actor_teacher_id = Column(Integer, nullable=False)
    field_name = Column(String(100), nullable=False)
    old_value = Column(Text, default="", nullable=False)
    new_value = Column(Text, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
