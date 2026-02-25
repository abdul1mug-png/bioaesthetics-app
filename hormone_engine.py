"""
BioAesthetic — Hormonal Optimization Engine
Proxy lifestyle scoring model. Not a medical device.
No diagnosis language. All outputs are wellness indices only.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

DISCLAIMER = (
    "This score is a lifestyle proxy model based on self-reported data and does not "
    "constitute medical advice, diagnosis, or treatment. Consult a licensed healthcare "
    "provider for any concerns related to hormonal health."
)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HormoneScoreResult:
    hormone_score: float            # 0–100 composite
    sleep_score: float              # 0–1 component
    sun_score: float                # 0–1 component
    activity_score: float           # 0–1 component
    micronutrient_score: float      # 0–1 component
    stress_penalty: float           # 0–1 component
    breakdown: dict[str, float]
    recommendations: list[str]
    disclaimer: str = DISCLAIMER


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL SCORERS
# ══════════════════════════════════════════════════════════════════════════════

class SleepScorer:
    """
    Optimal sleep ≈ 8 hours.
    Score saturates at 8h; below 5h applies nonlinear penalty.
    """
    OPTIMAL_HOURS = 8.0
    FLOOR_HOURS = 4.0

    def score(self, hours: float) -> float:
        if hours >= self.OPTIMAL_HOURS:
            return 1.0
        if hours < self.FLOOR_HOURS:
            # Steep penalty below 4h
            return max(0.0, hours / (self.FLOOR_HOURS * 2))
        # Linear between floor and optimal
        return (hours - self.FLOOR_HOURS) / (self.OPTIMAL_HOURS - self.FLOOR_HOURS) * 0.85 + 0.05


class SunlightScorer:
    """
    Morning sunlight for circadian entrainment.
    Target: 20 minutes of direct outdoor light.
    """
    TARGET_MINUTES = 20.0

    def score(self, minutes: float) -> float:
        return float(min(minutes / self.TARGET_MINUTES, 1.0))


class ActivityScorer:
    """
    Daily step count as activity proxy.
    8,000–10,000 steps is a commonly cited target.
    """
    TARGET_STEPS = 9_000.0
    MIN_STEPS = 1_000.0

    def score(self, steps: int) -> float:
        if steps >= self.TARGET_STEPS:
            return 1.0
        if steps < self.MIN_STEPS:
            return 0.05
        return (steps - self.MIN_STEPS) / (self.TARGET_STEPS - self.MIN_STEPS)


class MicronutrientScorer:
    """
    Zinc intake proxy.
    RDA: ~11 mg/day for adult males, ~8 mg/day for females.
    Model uses 11 mg as target; scores 0 below 2 mg, 1.0 at ≥14 mg.
    """
    TARGET_MG = 11.0
    MIN_MG = 2.0
    EXCESS_CUTOFF = 40.0  # UL for zinc

    def score(self, zinc_mg: float) -> float:
        if zinc_mg > self.EXCESS_CUTOFF:
            # Very high zinc can inhibit copper absorption — penalty
            return max(0.3, 1.0 - (zinc_mg - self.EXCESS_CUTOFF) / 100.0)
        if zinc_mg < self.MIN_MG:
            return 0.0
        return min((zinc_mg - self.MIN_MG) / (self.TARGET_MG - self.MIN_MG), 1.0)


class StressScorer:
    """
    Stress_rating: 0 (no stress) → 10 (extreme stress).
    Converts to a penalty: 0 stress = 1.0, max stress = 0.0.
    Nonlinear: moderate stress has less penalty.
    """

    def score(self, stress_rating: int) -> float:
        # Quadratic penalty for higher stress
        normalised = stress_rating / 10.0
        return float(max(0.0, 1.0 - normalised ** 1.4))


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SCORER
# ══════════════════════════════════════════════════════════════════════════════

class HormoneEngine:
    """
    Combines individual lifestyle signals into a composite proxy score.

    Weights (must sum to 1.0):
        sleep       0.30
        sunlight    0.20
        activity    0.20
        micronutrient 0.20
        stress      0.10
    """

    WEIGHTS = {
        "sleep":         0.30,
        "sun":           0.20,
        "activity":      0.20,
        "micronutrient": 0.20,
        "stress":        0.10,
    }

    def __init__(self):
        self._sleep   = SleepScorer()
        self._sun     = SunlightScorer()
        self._activity = ActivityScorer()
        self._micro   = MicronutrientScorer()
        self._stress  = StressScorer()

    def score(
        self,
        sleep_hours: float,
        sunlight_minutes: float,
        step_count: int,
        zinc_mg: float,
        stress_rating: int,
    ) -> HormoneScoreResult:

        # Component scores [0, 1]
        s_sleep  = self._sleep.score(sleep_hours)
        s_sun    = self._sun.score(sunlight_minutes)
        s_act    = self._activity.score(step_count)
        s_micro  = self._micro.score(zinc_mg)
        s_stress = self._stress.score(stress_rating)

        # Weighted composite → scale to 0–100
        raw = (
            self.WEIGHTS["sleep"]         * s_sleep
            + self.WEIGHTS["sun"]         * s_sun
            + self.WEIGHTS["activity"]    * s_act
            + self.WEIGHTS["micronutrient"] * s_micro
            + self.WEIGHTS["stress"]      * s_stress
        )
        composite = round(raw * 100.0, 2)

        breakdown = {
            "sleep_contribution":         round(self.WEIGHTS["sleep"] * s_sleep * 100, 2),
            "sunlight_contribution":      round(self.WEIGHTS["sun"] * s_sun * 100, 2),
            "activity_contribution":      round(self.WEIGHTS["activity"] * s_act * 100, 2),
            "micronutrient_contribution": round(self.WEIGHTS["micronutrient"] * s_micro * 100, 2),
            "stress_contribution":        round(self.WEIGHTS["stress"] * s_stress * 100, 2),
        }

        recs = self._generate_recommendations(
            s_sleep, s_sun, s_act, s_micro, s_stress, stress_rating, zinc_mg
        )

        return HormoneScoreResult(
            hormone_score=composite,
            sleep_score=round(s_sleep, 4),
            sun_score=round(s_sun, 4),
            activity_score=round(s_act, 4),
            micronutrient_score=round(s_micro, 4),
            stress_penalty=round(s_stress, 4),
            breakdown=breakdown,
            recommendations=recs,
        )

    def _generate_recommendations(
        self,
        sleep: float,
        sun: float,
        activity: float,
        micro: float,
        stress: float,
        stress_rating: int,
        zinc_mg: float,
    ) -> list[str]:
        recs = []

        if sleep < 0.6:
            recs.append(
                "Sleep is your highest-leverage variable. Target 7.5–9 hours. "
                "Maintain a consistent wake time and limit screen exposure 60 min before bed."
            )
        if sun < 0.5:
            recs.append(
                "Get 10–20 minutes of outdoor morning light within 30 minutes of waking. "
                "This is one of the most impactful inputs for circadian rhythm regulation."
            )
        if activity < 0.5:
            recs.append(
                "Increase daily movement. A 30-minute walk adds ~3,000–4,000 steps and improves "
                "insulin sensitivity and mood."
            )
        if micro < 0.5:
            if zinc_mg < 5:
                recs.append(
                    "Zinc intake appears low. Consider zinc-rich foods (oysters, red meat, pumpkin seeds) "
                    "or a 25–30 mg zinc bisglycinate supplement with dinner."
                )
            elif zinc_mg > 40:
                recs.append(
                    "Zinc intake exceeds the tolerable upper limit. Excess zinc inhibits copper absorption. "
                    "Reduce to 11–25 mg/day."
                )
        if stress_rating >= 7:
            recs.append(
                "High stress rating detected. Consider structured breathwork (4-7-8 technique), "
                "10-minute daily meditation, or reducing caffeine intake after noon."
            )
        if stress_rating >= 5 and stress_rating < 7:
            recs.append(
                "Moderate stress. Prioritise a brief daily decompression routine — a 20-minute walk "
                "outdoors significantly attenuates cortisol."
            )
        if not recs:
            recs.append("All lifestyle metrics are in a good range. Maintain consistency for best results.")

        return recs


# Module-level singleton
hormone_engine = HormoneEngine()
