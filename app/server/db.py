"""Database tables (SQLAlchemy). SQLite by default; any SQLAlchemy URL (e.g. PostgreSQL) via REHABNET_DB_URL.

Personal health data of a session (per-repetition scores, feedback, angle traces) is stored encrypted in
`Session.data_enc`; only the summary numbers needed for lists and charts are stored in plain columns.
"""
from __future__ import annotations

import datetime as dt
import os

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, create_engine)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import DATA_DIR, DB_URL

os.makedirs(DATA_DIR, exist_ok=True)
engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def now():
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(10))                 # 'doctor' | 'patient'
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    pw_hash: Mapped[str] = mapped_column(String(100))
    must_change_pw: Mapped[bool] = mapped_column(Boolean, default=False)
    totp_secret: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    doctor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # patient profile
    condition: Mapped[str | None] = mapped_column(String(200), nullable=True)
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    assignments = relationship("Assignment", back_populates="patient", foreign_keys="Assignment.patient_id")


class Assignment(Base):
    """An exercise prescribed by a doctor to a patient."""
    __tablename__ = "assignments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    doctor_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    exercise: Mapped[str] = mapped_column(String(40))
    side: Mapped[str | None] = mapped_column(String(1), nullable=True)
    sets: Mapped[int] = mapped_column(Integer, default=2)
    reps: Mapped[int] = mapped_column(Integer, default=10)
    per_week: Mapped[int] = mapped_column(Integer, default=5)
    notes: Mapped[str] = mapped_column(Text, default="")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    patient = relationship("User", back_populates="assignments", foreign_keys=[patient_id])


class Session(Base):
    """One exercise session performed by a patient."""
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    assignment_id: Mapped[int | None] = mapped_column(ForeignKey("assignments.id"), nullable=True)
    exercise: Mapped[str] = mapped_column(String(40))
    side: Mapped[str | None] = mapped_column(String(1), nullable=True)
    started: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    ended: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    target_reps: Mapped[int] = mapped_column(Integer, default=0)
    reps: Mapped[int] = mapped_column(Integer, default=0)
    mean_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    consistency: Mapped[float | None] = mapped_column(Float, nullable=True)
    rom_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    stability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    weakest: Mapped[str | None] = mapped_column(String(20), nullable=True)
    camera_view: Mapped[str | None] = mapped_column(String(20), nullable=True)
    source: Mapped[str] = mapped_column(String(10), default="camera")   # 'camera' | 'video' (demo / upload)
    data_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


class Baseline(Base):
    """Per-patient, per-exercise personal baseline (Milestone 4): the patient's own reference repetitions."""
    __tablename__ = "baselines"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    exercise: Mapped[str] = mapped_column(String(40))
    side: Mapped[str | None] = mapped_column(String(1), nullable=True)
    created: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    data_enc: Mapped[bytes] = mapped_column(LargeBinary)


class AuditLog(Base):
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[dt.datetime] = mapped_column(DateTime, default=now)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    action: Mapped[str] = mapped_column(String(60))
    detail: Mapped[str] = mapped_column(Text, default="")


def init_db():
    Base.metadata.create_all(engine)
