"""
BioAesthetic — Pydantic Schemas
Request/response models for all API endpoints.
"""

from pydantic import BaseModel, Field, EmailStr, field_validator, model_validator
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

class UserRegisterSchema(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    display_name: str = Field(..., min_length=2, max_length=64)

class UserLoginSchema(BaseModel):
    email: EmailStr
    password: str

class TokenResponseSchema(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds

class UserPublicSchema(BaseModel):
    id: int
    email: str
    display_name: str
    created_at: datetime

    class Config:
        from_attributes = True


# ══════════════════════════════════════════════════════════════════════════════
# SKIN
# ══════════════════════════════════════════════════════════════════════════════

class SkinReportSchema(BaseModel):
    """Output of the Skin Analysis Engine."""
    acne_score: float = Field(..., ge=0.0, le=1.0, description="Acne probability 0–1")
    pigmentation_score: float = Field(..., ge=0.0, le=1.0, description="Hyperpigmentation density")
    pore_score: float = Field(..., ge=0.0, le=1.0, description="Pore density normalized")
    inflammation_score: float = Field(..., ge=0.0, le=1.0, description="Skin redness / inflammation")
    under_eye_score: float = Field(..., ge=0.0, le=1.0, description="Under-eye darkness")
    skin_age_estimate: int = Field(..., ge=10, le=90, description="Estimated skin age in years")
    composite_skin_score: float = Field(..., ge=0.0, le=100.0, description="Aggregate 0–100")
    analysis_id: Optional[str] = None
    analyzed_at: Optional[datetime] = None
    recommendations: Optional[List[str]] = None


# ══════════════════════════════════════════════════════════════════════════════
# PHYSIQUE
# ══════════════════════════════════════════════════════════════════════════════

class ActivityLevel(str, Enum):
    sedentary = "sedentary"          # desk job, no exercise
    light = "light"                  # 1–3 days/week
    moderate = "moderate"            # 3–5 days/week
    active = "active"                # 6–7 days/week
    very_active = "very_active"      # athlete / manual labor


class PhysiqueGoal(str, Enum):
    fat_loss = "fat_loss"
    muscle_gain = "muscle_gain"
    maintenance = "maintenance"
    recomposition = "recomposition"


class TrainingSplit(str, Enum):
    full_body = "full_body"
    upper_lower = "upper_lower"
    ppl = "ppl"                      # Push / Pull / Legs
    bro_split = "bro_split"


class PhysiqueInputSchema(BaseModel):
    height_cm: float = Field(..., ge=100, le=250)
    weight_kg: float = Field(..., ge=30, le=300)
    body_fat_pct: float = Field(..., ge=2, le=60, description="Body fat as percentage (e.g. 18.5)")
    age: int = Field(..., ge=13, le=100)
    activity_level: ActivityLevel
    goal: PhysiqueGoal
    training_days_per_week: int = Field(default=4, ge=1, le=7)
    preferred_split: Optional[TrainingSplit] = None

    # Optional performance data
    bench_1rm_kg: Optional[float] = Field(None, ge=0, le=400)
    squat_1rm_kg: Optional[float] = Field(None, ge=0, le=500)
    deadlift_1rm_kg: Optional[float] = Field(None, ge=0, le=600)

    @field_validator("body_fat_pct")
    @classmethod
    def validate_bf(cls, v: float) -> float:
        if v < 3:
            raise ValueError("Body fat below 3% is physiologically unsafe.")
        return round(v, 2)


class MacroSchema(BaseModel):
    calories: int
    protein_g: int
    fat_g: int
    carbs_g: int


class CaloriePlanSchema(BaseModel):
    bmr: float
    tdee: float
    target_calories: int
    macros: MacroSchema
    goal: PhysiqueGoal
    lean_body_mass_kg: float
    notes: List[str] = []


class ExerciseSchema(BaseModel):
    name: str
    sets: int
    reps: str          # e.g. "8–12" or "5"
    rest_seconds: int
    muscle_group: str
    progressive_overload_note: Optional[str] = None


class WorkoutDaySchema(BaseModel):
    day_label: str     # e.g. "Day 1 – Push"
    exercises: List[ExerciseSchema]


class WorkoutPlanSchema(BaseModel):
    split: str
    training_days: int
    weekly_plan: List[WorkoutDaySchema]
    overload_protocol: str
    notes: List[str] = []


# ══════════════════════════════════════════════════════════════════════════════
# HORMONE
# ══════════════════════════════════════════════════════════════════════════════

class DailyMetricsInputSchema(BaseModel):
    """User self-reported daily lifestyle metrics."""
    sleep_hours: float = Field(..., ge=0, le=24, description="Total sleep hours last night")
    morning_sunlight_minutes: float = Field(..., ge=0, le=180)
    step_count: int = Field(..., ge=0, le=100_000)
    zinc_mg: float = Field(..., ge=0, le=200, description="Dietary / supplemental zinc intake in mg")
    stress_rating: int = Field(..., ge=0, le=10, description="Subjective stress level 0 (none) to 10 (extreme)")
    recorded_date: Optional[str] = Field(None, description="ISO date string; defaults to today")


class HormoneScoreSchema(BaseModel):
    """Proxy hormone optimization score — NOT a medical result."""
    hormone_score: float = Field(..., ge=0, le=100)
    sleep_score: float
    sun_score: float
    activity_score: float
    micronutrient_score: float
    stress_penalty: float
    breakdown: Dict[str, float]
    recommendations: List[str]
    disclaimer: str = (
        "This score is a lifestyle proxy model and does not constitute "
        "medical advice or diagnosis. Consult a licensed healthcare provider "
        "for hormonal health concerns."
    )


# ══════════════════════════════════════════════════════════════════════════════
# YOUTH INDEX
# ══════════════════════════════════════════════════════════════════════════════

class YouthIndexSchema(BaseModel):
    youth_index: int = Field(..., ge=0, le=100)
    skin_component: float
    physique_component: float
    hormone_component: float
    recovery_component: float
    weekly_delta: Optional[float] = Field(None, description="Change from last week's score")
    grade: str   # S / A / B / C / D
    breakdown: Dict[str, float]
    computed_at: datetime


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC
# ══════════════════════════════════════════════════════════════════════════════

class MessageSchema(BaseModel):
    message: str
    detail: Optional[str] = None
