"""
BioAesthetic — Youth Index Model
Composite score (0–100) from skin, physique, hormone, and recovery signals.
Stores weekly deltas for longitudinal trend analysis.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class YouthIndexResult:
    youth_index: int                # 0–100 integer
    skin_component: float
    physique_component: float
    hormone_component: float
    recovery_component: float
    weekly_delta: Optional[float]   # +/- change from last week; None if first calc
    grade: str                      # S / A / B / C / D
    breakdown: dict[str, float]
    computed_at: datetime


# ══════════════════════════════════════════════════════════════════════════════
# GRADE MAP
# ══════════════════════════════════════════════════════════════════════════════

def _grade(score: int) -> str:
    if score >= 90: return "S"
    if score >= 75: return "A"
    if score >= 60: return "B"
    if score >= 45: return "C"
    return "D"


# ══════════════════════════════════════════════════════════════════════════════
# RECOVERY SCORE ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════

class RecoveryEstimator:
    """
    Recovery is a derived signal — estimated from sleep and activity scores
    until the user integrates HRV / wearable data (Phase 3 extension).
    """

    def estimate(self, sleep_score: float, activity_score: float) -> float:
        """
        Weighted combination: sleep is the dominant recovery input.
        Returns [0, 1].
        """
        return round(0.65 * sleep_score + 0.35 * activity_score, 4)


# ══════════════════════════════════════════════════════════════════════════════
# YOUTH INDEX ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class YouthIndexEngine:
    """
    Weights:
        skin_score      0.30
        physique_score  0.30
        hormone_score   0.20
        recovery_score  0.20
    """

    WEIGHTS = {
        "skin":      0.30,
        "physique":  0.30,
        "hormone":   0.20,
        "recovery":  0.20,
    }

    def __init__(self):
        self._recovery = RecoveryEstimator()

    def compute(
        self,
        skin_composite_score: float,    # 0–100
        physique_score: float,          # 0–100  (derived externally from BMI + goal proximity)
        hormone_score: float,           # 0–100
        sleep_score: float,             # 0–1    (from hormone engine)
        activity_score: float,          # 0–1    (from hormone engine)
        previous_youth_index: Optional[float] = None,
    ) -> YouthIndexResult:
        """
        Compute Youth Index.

        Args:
            skin_composite_score: Output of SkinEngine (0–100).
            physique_score: Caller-derived score (0–100).
            hormone_score: Output of HormoneEngine (0–100).
            sleep_score: Raw sleep component from HormoneEngine (0–1).
            activity_score: Raw activity component from HormoneEngine (0–1).
            previous_youth_index: Last week's score for delta computation.
        """
        # Normalize all inputs to [0, 1]
        skin_n     = skin_composite_score / 100.0
        physique_n = physique_score / 100.0
        hormone_n  = hormone_score / 100.0
        recovery_n = self._recovery.estimate(sleep_score, activity_score)

        # Weighted composite
        raw = (
            self.WEIGHTS["skin"]     * skin_n
            + self.WEIGHTS["physique"] * physique_n
            + self.WEIGHTS["hormone"]  * hormone_n
            + self.WEIGHTS["recovery"] * recovery_n
        )

        youth_index = int(round(raw * 100))

        # Delta
        delta: Optional[float] = None
        if previous_youth_index is not None:
            delta = round(youth_index - previous_youth_index, 2)

        breakdown = {
            "skin_weighted":      round(self.WEIGHTS["skin"] * skin_n * 100, 2),
            "physique_weighted":  round(self.WEIGHTS["physique"] * physique_n * 100, 2),
            "hormone_weighted":   round(self.WEIGHTS["hormone"] * hormone_n * 100, 2),
            "recovery_weighted":  round(self.WEIGHTS["recovery"] * recovery_n * 100, 2),
        }

        return YouthIndexResult(
            youth_index=youth_index,
            skin_component=round(skin_n * 100, 2),
            physique_component=round(physique_n * 100, 2),
            hormone_component=round(hormone_n * 100, 2),
            recovery_component=round(recovery_n * 100, 2),
            weekly_delta=delta,
            grade=_grade(youth_index),
            breakdown=breakdown,
            computed_at=datetime.now(timezone.utc),
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHYSIQUE SCORE DERIVER
# ══════════════════════════════════════════════════════════════════════════════

def derive_physique_score(
    body_fat_pct: float,
    lbm_kg: float,
    height_cm: float,
    goal: str,
) -> float:
    """
    Heuristic physique score (0–100) based on body composition aesthetics.
    Uses fat-free mass index (FFMI) and body fat percentage proximity to ideal.

    FFMI = LBM_kg / (height_m)^2
    Natural elite FFMI ceiling: ~25 for males
    """
    height_m = height_cm / 100.0
    ffmi = lbm_kg / (height_m ** 2)

    # FFMI score: 0 at <14, 100 at 25+
    ffmi_score = min(max((ffmi - 14.0) / (25.0 - 14.0), 0.0), 1.0)

    # Body fat score: ideal range 10–18% (approximate)
    if body_fat_pct <= 12:
        bf_score = 1.0
    elif body_fat_pct <= 20:
        bf_score = 1.0 - (body_fat_pct - 12) / 30.0
    else:
        bf_score = max(0.0, 1.0 - (body_fat_pct - 12) / 45.0)

    raw = 0.6 * ffmi_score + 0.4 * bf_score
    return round(raw * 100.0, 2)


# Module-level singleton
youth_index_engine = YouthIndexEngine()
