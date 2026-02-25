"""
BioAesthetic — ORM Models
SQLAlchemy mapped tables for all user data.
"""

from sqlalchemy import (
    String, Integer, Float, Boolean, ForeignKey,
    Text, Date, UniqueConstraint
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base
import datetime


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(256), unique=True, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String(64), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    role: Mapped[str] = mapped_column(String(32), default="user")

    # Relationships
    skin_reports: Mapped[list["SkinReport"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    physique_profiles: Mapped[list["PhysiqueProfile"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    daily_metrics: Mapped[list["DailyMetrics"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    youth_indices: Mapped[list["YouthIndexRecord"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class SkinReport(Base):
    __tablename__ = "skin_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    analysis_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    acne_score: Mapped[float] = mapped_column(Float, nullable=False)
    pigmentation_score: Mapped[float] = mapped_column(Float, nullable=False)
    pore_score: Mapped[float] = mapped_column(Float, nullable=False)
    inflammation_score: Mapped[float] = mapped_column(Float, nullable=False)
    under_eye_score: Mapped[float] = mapped_column(Float, nullable=False)
    skin_age_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    composite_skin_score: Mapped[float] = mapped_column(Float, nullable=False)

    # S3 path for encrypted image
    image_s3_key: Mapped[str | None] = mapped_column(String(512), nullable=True)

    user: Mapped["User"] = relationship(back_populates="skin_reports")


class PhysiqueProfile(Base):
    __tablename__ = "physique_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)

    height_cm: Mapped[float] = mapped_column(Float, nullable=False)
    weight_kg: Mapped[float] = mapped_column(Float, nullable=False)
    body_fat_pct: Mapped[float] = mapped_column(Float, nullable=False)
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    activity_level: Mapped[str] = mapped_column(String(32), nullable=False)
    goal: Mapped[str] = mapped_column(String(32), nullable=False)
    training_days: Mapped[int] = mapped_column(Integer, default=4)
    preferred_split: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Computed
    bmr: Mapped[float | None] = mapped_column(Float, nullable=True)
    tdee: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_calories: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lean_body_mass_kg: Mapped[float | None] = mapped_column(Float, nullable=True)

    user: Mapped["User"] = relationship(back_populates="physique_profiles")


class DailyMetrics(Base):
    __tablename__ = "daily_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    recorded_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    sleep_hours: Mapped[float] = mapped_column(Float, nullable=False)
    morning_sunlight_minutes: Mapped[float] = mapped_column(Float, nullable=False)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False)
    zinc_mg: Mapped[float] = mapped_column(Float, nullable=False)
    stress_rating: Mapped[int] = mapped_column(Integer, nullable=False)

    # Computed hormone score
    hormone_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("user_id", "recorded_date", name="uq_user_date_metrics"),
    )

    user: Mapped["User"] = relationship(back_populates="daily_metrics")


class YouthIndexRecord(Base):
    __tablename__ = "youth_index_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    recorded_week: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    youth_index: Mapped[int] = mapped_column(Integer, nullable=False)
    skin_component: Mapped[float] = mapped_column(Float, nullable=False)
    physique_component: Mapped[float] = mapped_column(Float, nullable=False)
    hormone_component: Mapped[float] = mapped_column(Float, nullable=False)
    recovery_component: Mapped[float] = mapped_column(Float, nullable=False)
    weekly_delta: Mapped[float | None] = mapped_column(Float, nullable=True)
    grade: Mapped[str] = mapped_column(String(4), nullable=False)

    __table_args__ = (
        UniqueConstraint("user_id", "recorded_week", name="uq_user_week_youth"),
    )

    user: Mapped["User"] = relationship(back_populates="youth_indices")
