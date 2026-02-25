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


class Round(Base):
    __tablename__ = "rounds"

    id = Column(Integer, primary_key=True)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    win_weight = Column(Integer, default=70, nullable=False)
    homework_weight = Column(Integer, default=30, nullable=False)
    status = Column(String(50), default="open", nullable=False)
    homework_total_questions = Column(Integer, default=0, nullable=False)
    missing_homework_policy = Column(String(20), default="zero", nullable=False)
    missing_homework_penalty = Column(Integer, default=1, nullable=False)
    homework_metric_mode = Column(String(20), default="pct_wrong", nullable=False)
    completion_override_reason = Column(Text, default="", nullable=False)

    classroom = relationship("Classroom", back_populates="rounds")
    matches = relationship("Match", back_populates="round", cascade="all, delete-orphan")
    attendance_records = relationship(
        "Attendance", back_populates="round", cascade="all, delete-orphan"
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

    match = relationship("Match", back_populates="homework_entry")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    teacher_id = Column(Integer, ForeignKey("teachers.id"), nullable=False)
    classroom_id = Column(Integer, ForeignKey("classrooms.id"), nullable=False)
    round_id = Column(Integer, ForeignKey("rounds.id"), nullable=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=True)
    action = Column(String(80), nullable=False)
    payload = Column(Text, default="", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
