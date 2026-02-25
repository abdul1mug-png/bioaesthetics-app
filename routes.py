"""
BioAesthetic — API Routes
All endpoint implementations.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from datetime import date, datetime, timezone
import logging

from database import get_db
from models import User, SkinReport, PhysiqueProfile, DailyMetrics, YouthIndexRecord
from schemas import (
    UserRegisterSchema, UserLoginSchema, TokenResponseSchema, UserPublicSchema,
    PhysiqueInputSchema, CaloriePlanSchema, WorkoutPlanSchema, MacroSchema,
    WorkoutDaySchema, ExerciseSchema,
    DailyMetricsInputSchema, HormoneScoreSchema,
    SkinReportSchema, YouthIndexSchema, MessageSchema,
)
from auth_service import (
    create_user, authenticate_user, create_access_token,
    create_refresh_token, decode_token, get_user_by_id,
)
from skin_engine import skin_engine
from physique_engine import physique_engine
from hormone_engine import hormone_engine
from youth_index import youth_index_engine, derive_physique_score
from config import settings

log = logging.getLogger(__name__)
bearer_scheme = HTTPBearer()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH DEPENDENCY
# ══════════════════════════════════════════════════════════════════════════════

async def current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """FastAPI dependency — validate JWT and return authenticated user."""
    try:
        payload = decode_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Refresh tokens cannot access resources.")

    user = await get_user_by_id(db, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or deactivated.")
    return user


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTER
# ══════════════════════════════════════════════════════════════════════════════

auth_router = APIRouter()


@auth_router.post("/register", response_model=UserPublicSchema, status_code=201)
async def register(data: UserRegisterSchema, db: AsyncSession = Depends(get_db)):
    try:
        user = await create_user(db, data)
        return user
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@auth_router.post("/login", response_model=TokenResponseSchema)
async def login(data: UserLoginSchema, db: AsyncSession = Depends(get_db)):
    try:
        user = await authenticate_user(db, data.email, data.password)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    access = create_access_token(user.id, user.email)
    refresh = create_refresh_token(user.id)
    return TokenResponseSchema(
        access_token=access,
        refresh_token=refresh,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SKIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

skin_router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


@skin_router.post("/upload_face", response_model=SkinReportSchema)
async def upload_face(
    file: UploadFile = File(...),
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Accepts a facial image and runs the skin analysis pipeline.
    Returns all skin metric scores.
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, or WebP."
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    try:
        result = skin_engine.analyse(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        log.error(f"Skin analysis failed for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Skin analysis failed. Please try again.")

    # Persist report
    report = SkinReport(
        user_id=user.id,
        analysis_id=result.analysis_id,
        acne_score=result.acne_score,
        pigmentation_score=result.pigmentation_score,
        pore_score=result.pore_score,
        inflammation_score=result.inflammation_score,
        under_eye_score=result.under_eye_score,
        skin_age_estimate=result.skin_age_estimate,
        composite_skin_score=result.composite_skin_score,
    )
    db.add(report)
    await db.flush()

    return SkinReportSchema(
        acne_score=result.acne_score,
        pigmentation_score=result.pigmentation_score,
        pore_score=result.pore_score,
        inflammation_score=result.inflammation_score,
        under_eye_score=result.under_eye_score,
        skin_age_estimate=result.skin_age_estimate,
        composite_skin_score=result.composite_skin_score,
        analysis_id=result.analysis_id,
        analyzed_at=result.analyzed_at,
        recommendations=result.recommendations,
    )


@skin_router.get("/skin_report", response_model=SkinReportSchema)
async def get_skin_report(
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Retrieve the user's most recent skin analysis report."""
    result = await db.execute(
        select(SkinReport)
        .where(SkinReport.user_id == user.id)
        .order_by(desc(SkinReport.created_at))
        .limit(1)
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="No skin report found. Upload a face image first.")

    return SkinReportSchema(
        acne_score=report.acne_score,
        pigmentation_score=report.pigmentation_score,
        pore_score=report.pore_score,
        inflammation_score=report.inflammation_score,
        under_eye_score=report.under_eye_score,
        skin_age_estimate=report.skin_age_estimate,
        composite_skin_score=report.composite_skin_score,
        analysis_id=report.analysis_id,
        analyzed_at=report.created_at,
        recommendations=None,  # Recs are returned at analysis time only
    )


# ══════════════════════════════════════════════════════════════════════════════
# PHYSIQUE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

physique_router = APIRouter()


@physique_router.post("/physique_input", response_model=MessageSchema, status_code=201)
async def physique_input(
    data: PhysiqueInputSchema,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Store user physique data and compute a calorie plan."""
    plan = physique_engine.calorie_plan(
        weight_kg=data.weight_kg,
        body_fat_pct=data.body_fat_pct,
        activity_level=data.activity_level.value,
        goal=data.goal.value,
    )

    profile = PhysiqueProfile(
        user_id=user.id,
        height_cm=data.height_cm,
        weight_kg=data.weight_kg,
        body_fat_pct=data.body_fat_pct,
        age=data.age,
        activity_level=data.activity_level.value,
        goal=data.goal.value,
        training_days=data.training_days_per_week,
        preferred_split=data.preferred_split.value if data.preferred_split else None,
        bmr=plan.bmr,
        tdee=plan.tdee,
        target_calories=plan.target_calories,
        lean_body_mass_kg=plan.lean_body_mass_kg,
    )
    db.add(profile)
    await db.flush()

    return MessageSchema(message="Physique profile saved.", detail=f"Target: {plan.target_calories} kcal/day")


@physique_router.get("/calorie_plan", response_model=CaloriePlanSchema)
async def get_calorie_plan(
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the calorie and macro plan from the latest physique profile."""
    result = await db.execute(
        select(PhysiqueProfile)
        .where(PhysiqueProfile.user_id == user.id)
        .order_by(desc(PhysiqueProfile.created_at))
        .limit(1)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="No physique profile found. POST /physique/physique_input first.")

    plan = physique_engine.calorie_plan(
        weight_kg=profile.weight_kg,
        body_fat_pct=profile.body_fat_pct,
        activity_level=profile.activity_level,
        goal=profile.goal,
    )

    return CaloriePlanSchema(
        bmr=plan.bmr,
        tdee=plan.tdee,
        target_calories=plan.target_calories,
        macros=MacroSchema(
            calories=plan.macros.calories,
            protein_g=plan.macros.protein_g,
            fat_g=plan.macros.fat_g,
            carbs_g=plan.macros.carbs_g,
        ),
        goal=plan.goal,
        lean_body_mass_kg=plan.lean_body_mass_kg,
        notes=plan.notes,
    )


@physique_router.get("/workout_plan", response_model=WorkoutPlanSchema)
async def get_workout_plan(
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate a personalised workout plan from the latest physique profile."""
    result = await db.execute(
        select(PhysiqueProfile)
        .where(PhysiqueProfile.user_id == user.id)
        .order_by(desc(PhysiqueProfile.created_at))
        .limit(1)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="No physique profile found. POST /physique/physique_input first.")

    plan = physique_engine.workout_plan(
        training_days=profile.training_days,
        preferred_split=profile.preferred_split,
        goal=profile.goal,
    )

    days_out = []
    for d in plan.weekly_plan:
        exs = [
            ExerciseSchema(
                name=e.name, sets=e.sets, reps=e.reps,
                rest_seconds=e.rest_seconds, muscle_group=e.muscle_group,
                progressive_overload_note=e.progressive_overload_note,
            )
            for e in d.exercises
        ]
        days_out.append(WorkoutDaySchema(day_label=d.day_label, exercises=exs))

    return WorkoutPlanSchema(
        split=plan.split,
        training_days=plan.training_days,
        weekly_plan=days_out,
        overload_protocol=plan.overload_protocol,
        notes=plan.notes,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HORMONE ROUTER
# ══════════════════════════════════════════════════════════════════════════════

hormone_router = APIRouter()


@hormone_router.post("/daily_metrics", response_model=HormoneScoreSchema, status_code=201)
async def post_daily_metrics(
    data: DailyMetricsInputSchema,
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Store daily lifestyle metrics and return the hormone optimization score."""
    recorded = date.fromisoformat(data.recorded_date) if data.recorded_date else date.today()

    score_result = hormone_engine.score(
        sleep_hours=data.sleep_hours,
        sunlight_minutes=data.morning_sunlight_minutes,
        step_count=data.step_count,
        zinc_mg=data.zinc_mg,
        stress_rating=data.stress_rating,
    )

    # Upsert (overwrite if same date already exists)
    existing = await db.execute(
        select(DailyMetrics)
        .where(DailyMetrics.user_id == user.id, DailyMetrics.recorded_date == recorded)
    )
    row = existing.scalar_one_or_none()
    if row:
        row.sleep_hours = data.sleep_hours
        row.morning_sunlight_minutes = data.morning_sunlight_minutes
        row.step_count = data.step_count
        row.zinc_mg = data.zinc_mg
        row.stress_rating = data.stress_rating
        row.hormone_score = score_result.hormone_score
    else:
        row = DailyMetrics(
            user_id=user.id,
            recorded_date=recorded,
            sleep_hours=data.sleep_hours,
            morning_sunlight_minutes=data.morning_sunlight_minutes,
            step_count=data.step_count,
            zinc_mg=data.zinc_mg,
            stress_rating=data.stress_rating,
            hormone_score=score_result.hormone_score,
        )
        db.add(row)
    await db.flush()

    return HormoneScoreSchema(
        hormone_score=score_result.hormone_score,
        sleep_score=score_result.sleep_score,
        sun_score=score_result.sun_score,
        activity_score=score_result.activity_score,
        micronutrient_score=score_result.micronutrient_score,
        stress_penalty=score_result.stress_penalty,
        breakdown=score_result.breakdown,
        recommendations=score_result.recommendations,
    )


@hormone_router.get("/hormone_score", response_model=HormoneScoreSchema)
async def get_hormone_score(
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the hormone score from the most recent daily metrics entry."""
    result = await db.execute(
        select(DailyMetrics)
        .where(DailyMetrics.user_id == user.id)
        .order_by(desc(DailyMetrics.recorded_date))
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="No daily metrics found. POST /hormone/daily_metrics first.")

    score_result = hormone_engine.score(
        sleep_hours=row.sleep_hours,
        sunlight_minutes=row.morning_sunlight_minutes,
        step_count=row.step_count,
        zinc_mg=row.zinc_mg,
        stress_rating=row.stress_rating,
    )

    return HormoneScoreSchema(
        hormone_score=score_result.hormone_score,
        sleep_score=score_result.sleep_score,
        sun_score=score_result.sun_score,
        activity_score=score_result.activity_score,
        micronutrient_score=score_result.micronutrient_score,
        stress_penalty=score_result.stress_penalty,
        breakdown=score_result.breakdown,
        recommendations=score_result.recommendations,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD ROUTER
# ══════════════════════════════════════════════════════════════════════════════

dashboard_router = APIRouter()


@dashboard_router.get("/youth_index", response_model=YouthIndexSchema)
async def get_youth_index(
    user: User = Depends(current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Compute and return the composite Youth Index score.
    Pulls the latest skin, physique, and hormone records.
    """
    # Fetch latest skin report
    skin_row = (await db.execute(
        select(SkinReport).where(SkinReport.user_id == user.id)
        .order_by(desc(SkinReport.created_at)).limit(1)
    )).scalar_one_or_none()

    # Fetch latest physique profile
    physique_row = (await db.execute(
        select(PhysiqueProfile).where(PhysiqueProfile.user_id == user.id)
        .order_by(desc(PhysiqueProfile.created_at)).limit(1)
    )).scalar_one_or_none()

    # Fetch latest daily metrics
    metrics_row = (await db.execute(
        select(DailyMetrics).where(DailyMetrics.user_id == user.id)
        .order_by(desc(DailyMetrics.recorded_date)).limit(1)
    )).scalar_one_or_none()

    if not any([skin_row, physique_row, metrics_row]):
        raise HTTPException(
            status_code=404,
            detail="Insufficient data. Please upload a face image, submit physique data, and log daily metrics."
        )

    # Defaults for missing data
    skin_score    = skin_row.composite_skin_score if skin_row else 50.0
    hormone_score = metrics_row.hormone_score if (metrics_row and metrics_row.hormone_score) else 50.0

    physique_score = 50.0
    sleep_score = activity_score = 0.5
    if physique_row:
        physique_score = derive_physique_score(
            body_fat_pct=physique_row.body_fat_pct,
            lbm_kg=physique_row.lean_body_mass_kg or (physique_row.weight_kg * 0.8),
            height_cm=physique_row.height_cm,
            goal=physique_row.goal,
        )
    if metrics_row:
        hr = hormone_engine.score(
            sleep_hours=metrics_row.sleep_hours,
            sunlight_minutes=metrics_row.morning_sunlight_minutes,
            step_count=metrics_row.step_count,
            zinc_mg=metrics_row.zinc_mg,
            stress_rating=metrics_row.stress_rating,
        )
        sleep_score = hr.sleep_score
        activity_score = hr.activity_score

    # Previous youth index for delta
    prev = (await db.execute(
        select(YouthIndexRecord).where(YouthIndexRecord.user_id == user.id)
        .order_by(desc(YouthIndexRecord.recorded_week)).limit(1)
    )).scalar_one_or_none()
    previous_score = prev.youth_index if prev else None

    yi = youth_index_engine.compute(
        skin_composite_score=skin_score,
        physique_score=physique_score,
        hormone_score=hormone_score,
        sleep_score=sleep_score,
        activity_score=activity_score,
        previous_youth_index=previous_score,
    )

    # Persist weekly record
    today_monday = date.today()
    existing_yi = (await db.execute(
        select(YouthIndexRecord).where(
            YouthIndexRecord.user_id == user.id,
            YouthIndexRecord.recorded_week == today_monday
        )
    )).scalar_one_or_none()

    if not existing_yi:
        record = YouthIndexRecord(
            user_id=user.id,
            recorded_week=today_monday,
            youth_index=yi.youth_index,
            skin_component=yi.skin_component,
            physique_component=yi.physique_component,
            hormone_component=yi.hormone_component,
            recovery_component=yi.recovery_component,
            weekly_delta=yi.weekly_delta,
            grade=yi.grade,
        )
        db.add(record)
        await db.flush()

    return YouthIndexSchema(
        youth_index=yi.youth_index,
        skin_component=yi.skin_component,
        physique_component=yi.physique_component,
        hormone_component=yi.hormone_component,
        recovery_component=yi.recovery_component,
        weekly_delta=yi.weekly_delta,
        grade=yi.grade,
        breakdown=yi.breakdown,
        computed_at=yi.computed_at,
    )
