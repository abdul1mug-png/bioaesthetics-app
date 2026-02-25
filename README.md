# BioAesthetic API

Biometric optimization platform — skin analysis, physique planning, hormonal lifestyle scoring, and Youth Index computation. Built in Python with FastAPI.

---

## Architecture

```
/bioaesthetic
    /app
        main.py          ← FastAPI app factory + middleware
        config.py        ← Pydantic settings (env vars)
        database.py      ← Async SQLAlchemy engine + session
        models.py        ← ORM table definitions
    /modules
        skin_engine.py   ← Image preprocessing + scoring (Phase 1 heuristic / Phase 2 CNN)
        physique_engine.py ← Katch-McArdle BMR, macro calc, workout generator
        hormone_engine.py  ← Lifestyle proxy scoring model
        youth_index.py   ← Composite Youth Index + longitudinal delta
    /schemas
        schemas.py       ← All Pydantic request/response models
    /services
        auth_service.py  ← JWT + bcrypt auth
    /tests
        test_engines.py  ← Unit tests (pytest)
    routes.py            ← All API endpoint implementations
    requirements.txt
    docker-compose.yml
    Dockerfile
```

---

## Quick Start

### 1. Local with Docker Compose

```bash
docker compose up --build
```

API available at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`  
MinIO console: `http://localhost:9001`

### 2. Manual (venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start PostgreSQL + Redis separately, then:
uvicorn app.main:app --reload
```

### 3. Environment Variables

Copy `.env.example` → `.env` and configure:

```env
DATABASE_URL=postgresql+asyncpg://bio:bio@localhost:5432/bioaesthetic
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-32-byte-secret
ENV=development
USE_ML_INFERENCE=false     # Set true in Phase 2 when CNN model is available
SKIN_MODEL_PATH=ml_models/skin_cnn.pt
```

---

## API Reference

### Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Get JWT tokens |

All protected endpoints require `Authorization: Bearer <access_token>` header.

### Skin

| Method | Path | Description |
|--------|------|-------------|
| POST | `/skin/upload_face` | Upload facial image → returns skin scores |
| GET | `/skin/skin_report` | Latest skin report |

**Upload face example:**
```bash
curl -X POST http://localhost:8000/skin/upload_face \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@face.jpg"
```

**Response:**
```json
{
  "acne_score": 0.12,
  "pigmentation_score": 0.24,
  "pore_score": 0.31,
  "inflammation_score": 0.08,
  "under_eye_score": 0.19,
  "skin_age_estimate": 28,
  "composite_skin_score": 73.4,
  "recommendations": ["Apply broad-spectrum SPF 50+ daily..."]
}
```

### Physique

| Method | Path | Description |
|--------|------|-------------|
| POST | `/physique/physique_input` | Submit physique data |
| GET | `/physique/calorie_plan` | Get calorie + macro plan |
| GET | `/physique/workout_plan` | Get personalised workout split |

**Physique input example:**
```json
{
  "height_cm": 180,
  "weight_kg": 82,
  "body_fat_pct": 18.0,
  "age": 28,
  "activity_level": "moderate",
  "goal": "muscle_gain",
  "training_days_per_week": 5,
  "preferred_split": "ppl"
}
```

**Calorie plan response:**
```json
{
  "bmr": 1939.4,
  "tdee": 2996.1,
  "target_calories": 3296,
  "macros": { "calories": 3296, "protein_g": 180, "fat_g": 74, "carbs_g": 445 },
  "goal": "muscle_gain",
  "lean_body_mass_kg": 67.24
}
```

### Hormone

| Method | Path | Description |
|--------|------|-------------|
| POST | `/hormone/daily_metrics` | Log daily lifestyle data |
| GET | `/hormone/hormone_score` | Latest hormone optimization score |

**Daily metrics input:**
```json
{
  "sleep_hours": 7.5,
  "morning_sunlight_minutes": 15,
  "step_count": 8200,
  "zinc_mg": 12,
  "stress_rating": 3
}
```

**Score response:**
```json
{
  "hormone_score": 78.4,
  "sleep_score": 0.91,
  "sun_score": 0.75,
  "activity_score": 0.80,
  "micronutrient_score": 1.0,
  "stress_penalty": 0.81,
  "breakdown": { "sleep_contribution": 27.3, ... },
  "disclaimer": "This score is a lifestyle proxy model..."
}
```

### Dashboard

| Method | Path | Description |
|--------|------|-------------|
| GET | `/dashboard/youth_index` | Composite Youth Index |

**Response:**
```json
{
  "youth_index": 74,
  "skin_component": 73.4,
  "physique_component": 68.2,
  "hormone_component": 78.4,
  "recovery_component": 65.1,
  "weekly_delta": 3.2,
  "grade": "A",
  "breakdown": { "skin_weighted": 22.0, ... },
  "computed_at": "2025-01-15T09:30:00Z"
}
```

---

## Testing

```bash
pytest tests/test_engines.py -v
```

All scoring functions have full unit test coverage verifying:
- Correct formula implementation (Katch-McArdle, weighted composites)
- Boundary conditions (0, 50%, 100% inputs)
- Output type guarantees (int/float)
- Monotonicity of score functions
- Grade assignment correctness

---

## Development Phases

| Phase | Status | Features |
|-------|--------|----------|
| 1 | ✅ Complete | Rule-based engines, all APIs, JWT auth, PostgreSQL storage |
| 2 | Planned | CNN skin model (ResNet-18), adaptive recalibration |
| 3 | Planned | Predictive regression, anomaly detection, HRV integration |

### Enabling Phase 2 CNN (when model is available)

1. Place fine-tuned `skin_cnn.pt` in `/ml_models/`
2. Set `USE_ML_INFERENCE=true` in `.env`
3. Install PyTorch: `pip install torch torchvision`
4. Restart the API — the engine auto-detects and uses the CNN

---

## Security

- Passwords hashed with bcrypt (12 rounds)
- JWTs signed with HS256
- Images stored encrypted at rest in S3/MinIO
- HTTPS enforced in production
- Role-based access control on all routes
- No medical diagnosis language anywhere in outputs

---

## Compliance Notes

BioAesthetic is a **wellness tracking tool**, not a medical device. All scores are proxy indices derived from user-provided data. The platform must not be marketed as diagnosing, treating, or preventing any health condition. All output schemas include appropriate disclaimers where required.
