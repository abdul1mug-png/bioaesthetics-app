"""
BioAesthetic — Physique Optimization Engine
Katch-McArdle BMR → TDEE → calorie/macro targets → workout plan generator.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Activity multipliers for TDEE
ACTIVITY_MULTIPLIERS = {
    "sedentary":   1.2,
    "light":       1.375,
    "moderate":    1.55,
    "active":      1.725,
    "very_active": 1.9,
}

# Calorie adjustments per goal
GOAL_ADJUSTMENTS = {
    "fat_loss":      -400,
    "muscle_gain":   +300,
    "maintenance":   0,
    "recomposition": -150,  # mild deficit + high protein
}


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MacroTarget:
    calories: int
    protein_g: int
    fat_g: int
    carbs_g: int


@dataclass
class CaloriePlan:
    bmr: float
    tdee: float
    target_calories: int
    macros: MacroTarget
    goal: str
    lean_body_mass_kg: float
    notes: list[str] = field(default_factory=list)


@dataclass
class Exercise:
    name: str
    sets: int
    reps: str
    rest_seconds: int
    muscle_group: str
    progressive_overload_note: Optional[str] = None


@dataclass
class WorkoutDay:
    day_label: str
    exercises: list[Exercise]


@dataclass
class WorkoutPlan:
    split: str
    training_days: int
    weekly_plan: list[WorkoutDay]
    overload_protocol: str
    notes: list[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# MACRO CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class MacroCalculator:
    """
    Computes LBM → BMR (Katch-McArdle) → TDEE → macro split.
    """

    # Protein constants
    PROTEIN_MIN_G_PER_KG = 1.8
    PROTEIN_MAX_G_PER_KG = 2.2
    FAT_G_PER_KG = 0.9

    def lean_body_mass(self, weight_kg: float, body_fat_pct: float) -> float:
        """LBM = weight × (1 − body_fat_fraction)"""
        return weight_kg * (1.0 - body_fat_pct / 100.0)

    def bmr(self, lbm_kg: float) -> float:
        """Katch-McArdle formula: BMR = 370 + 21.6 × LBM(kg)"""
        return 370.0 + 21.6 * lbm_kg

    def tdee(self, bmr: float, activity_level: str) -> float:
        multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
        return bmr * multiplier

    def target_calories(self, tdee: float, goal: str) -> int:
        adjustment = GOAL_ADJUSTMENTS.get(goal, 0)
        return max(1200, round(tdee + adjustment))

    def macros(self, calories: int, weight_kg: float, goal: str) -> MacroTarget:
        """
        Protein: 1.8–2.2 g/kg (higher end for muscle gain / recomp)
        Fat: 0.9 g/kg floor
        Carbs: fill remaining calories
        """
        if goal in ("muscle_gain", "recomposition"):
            protein_g = round(weight_kg * self.PROTEIN_MAX_G_PER_KG)
        else:
            protein_g = round(weight_kg * self.PROTEIN_MIN_G_PER_KG)

        fat_g = round(weight_kg * self.FAT_G_PER_KG)
        protein_cals = protein_g * 4
        fat_cals = fat_g * 9
        carb_cals = max(0, calories - protein_cals - fat_cals)
        carb_g = round(carb_cals / 4)

        return MacroTarget(
            calories=calories,
            protein_g=protein_g,
            fat_g=fat_g,
            carbs_g=carb_g,
        )

    def build_plan(
        self,
        weight_kg: float,
        body_fat_pct: float,
        activity_level: str,
        goal: str,
    ) -> CaloriePlan:
        lbm = self.lean_body_mass(weight_kg, body_fat_pct)
        bmr_val = self.bmr(lbm)
        tdee_val = self.tdee(bmr_val, activity_level)
        cals = self.target_calories(tdee_val, goal)
        macros = self.macros(cals, weight_kg, goal)

        notes = []
        if goal == "fat_loss" and cals < 1400:
            notes.append("Calorie target is quite aggressive. Consider a larger deficit spread over more weeks.")
        if goal == "muscle_gain":
            notes.append("Prioritise progressive overload. Surplus without training will accumulate fat.")
        if goal == "recomposition":
            notes.append("Body recomposition is slower than dedicated phases. Expect 0.5–1% BF reduction per month.")

        return CaloriePlan(
            bmr=round(bmr_val, 1),
            tdee=round(tdee_val, 1),
            target_calories=cals,
            macros=macros,
            goal=goal,
            lean_body_mass_kg=round(lbm, 2),
            notes=notes,
        )


# ══════════════════════════════════════════════════════════════════════════════
# WORKOUT LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

# Each exercise: (name, sets, reps, rest_s, muscle_group)
EXERCISE_DB: dict[str, list[tuple]] = {
    "chest": [
        ("Barbell Bench Press",    4, "5–8",  180, "Chest"),
        ("Incline Dumbbell Press", 3, "8–12", 120, "Chest"),
        ("Cable Fly",              3, "12–15", 90, "Chest"),
        ("Push-Up",                3, "12–20", 60, "Chest"),
    ],
    "back": [
        ("Barbell Deadlift",       4, "4–6",  180, "Back"),
        ("Pull-Up",                4, "6–10", 120, "Back"),
        ("Seated Cable Row",       3, "10–12", 90, "Back"),
        ("Single-Arm DB Row",      3, "10–12", 90, "Back"),
    ],
    "shoulders": [
        ("Overhead Press",         4, "6–10", 150, "Shoulders"),
        ("Lateral Raise",          3, "12–15", 75, "Shoulders"),
        ("Face Pull",              3, "15–20", 75, "Shoulders"),
    ],
    "legs": [
        ("Barbell Squat",          4, "5–8",  180, "Quads/Glutes"),
        ("Romanian Deadlift",      3, "8–12", 120, "Hamstrings"),
        ("Leg Press",              3, "10–12",120, "Quads"),
        ("Walking Lunge",          3, "12 each",90,"Glutes/Quads"),
        ("Standing Calf Raise",    4, "15–20", 60, "Calves"),
    ],
    "biceps": [
        ("Barbell Curl",           3, "8–12",  75, "Biceps"),
        ("Incline DB Curl",        3, "10–12", 75, "Biceps"),
    ],
    "triceps": [
        ("Close-Grip Bench Press", 3, "8–10", 100, "Triceps"),
        ("Overhead Tricep Extension",3,"10–12",75, "Triceps"),
        ("Tricep Pushdown",        3, "12–15", 60, "Triceps"),
    ],
    "core": [
        ("Plank",                  3, "45–60s", 60, "Core"),
        ("Hanging Leg Raise",      3, "12–15",  60, "Core"),
        ("Cable Crunch",           3, "15–20",  60, "Core"),
    ],
}

OVERLOAD_PROTOCOL = (
    "Progressive Overload Rule: If you complete all prescribed reps across all sets "
    "for 2 consecutive sessions, increase the load by 2.5 kg (upper body) or 5 kg (lower body). "
    "If you cannot complete minimum reps, reduce load by 10% and rebuild."
)


def _make_exercise(t: tuple) -> Exercise:
    name, sets, reps, rest, muscle = t
    return Exercise(
        name=name, sets=sets, reps=reps, rest_seconds=rest, muscle_group=muscle,
        progressive_overload_note="Add 2.5–5 kg when all sets completed at top of rep range for 2 sessions."
    )


# ══════════════════════════════════════════════════════════════════════════════
# WORKOUT PLAN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

class WorkoutGenerator:

    def _full_body(self, days: int) -> list[WorkoutDay]:
        """3-day full body — suitable for beginners."""
        day_template = WorkoutDay(
            day_label="Full Body",
            exercises=[
                _make_exercise(EXERCISE_DB["legs"][0]),       # Squat
                _make_exercise(EXERCISE_DB["chest"][0]),      # Bench
                _make_exercise(EXERCISE_DB["back"][1]),       # Pull-Up
                _make_exercise(EXERCISE_DB["shoulders"][0]),  # OHP
                _make_exercise(EXERCISE_DB["core"][0]),       # Plank
            ]
        )
        return [
            WorkoutDay(day_label=f"Day {i+1} – Full Body", exercises=day_template.exercises)
            for i in range(min(days, 3))
        ]

    def _upper_lower(self, days: int) -> list[WorkoutDay]:
        upper = WorkoutDay(
            day_label="Upper",
            exercises=[
                _make_exercise(EXERCISE_DB["chest"][0]),
                _make_exercise(EXERCISE_DB["back"][1]),
                _make_exercise(EXERCISE_DB["shoulders"][0]),
                _make_exercise(EXERCISE_DB["biceps"][0]),
                _make_exercise(EXERCISE_DB["triceps"][0]),
            ]
        )
        lower = WorkoutDay(
            day_label="Lower",
            exercises=[
                _make_exercise(EXERCISE_DB["legs"][0]),
                _make_exercise(EXERCISE_DB["legs"][1]),
                _make_exercise(EXERCISE_DB["legs"][4]),
                _make_exercise(EXERCISE_DB["core"][1]),
            ]
        )
        pattern = [upper, lower, upper, lower]
        return [
            WorkoutDay(day_label=f"Day {i+1} – {pattern[i % 4].day_label}", exercises=pattern[i % 4].exercises)
            for i in range(min(days, 4))
        ]

    def _ppl(self, days: int) -> list[WorkoutDay]:
        push = WorkoutDay("Push", [
            _make_exercise(EXERCISE_DB["chest"][0]),
            _make_exercise(EXERCISE_DB["chest"][1]),
            _make_exercise(EXERCISE_DB["shoulders"][0]),
            _make_exercise(EXERCISE_DB["shoulders"][1]),
            _make_exercise(EXERCISE_DB["triceps"][1]),
        ])
        pull = WorkoutDay("Pull", [
            _make_exercise(EXERCISE_DB["back"][0]),
            _make_exercise(EXERCISE_DB["back"][1]),
            _make_exercise(EXERCISE_DB["back"][2]),
            _make_exercise(EXERCISE_DB["biceps"][0]),
            _make_exercise(EXERCISE_DB["biceps"][1]),
        ])
        legs = WorkoutDay("Legs", [
            _make_exercise(EXERCISE_DB["legs"][0]),
            _make_exercise(EXERCISE_DB["legs"][1]),
            _make_exercise(EXERCISE_DB["legs"][2]),
            _make_exercise(EXERCISE_DB["legs"][4]),
            _make_exercise(EXERCISE_DB["core"][1]),
        ])
        pattern = [push, pull, legs]
        return [
            WorkoutDay(day_label=f"Day {i+1} – {pattern[i % 3].day_label}", exercises=pattern[i % 3].exercises)
            for i in range(min(days, 6))
        ]

    def _bro_split(self, days: int) -> list[WorkoutDay]:
        splits = [
            WorkoutDay("Chest", [_make_exercise(e) for e in EXERCISE_DB["chest"]]),
            WorkoutDay("Back", [_make_exercise(e) for e in EXERCISE_DB["back"]]),
            WorkoutDay("Shoulders", [_make_exercise(e) for e in EXERCISE_DB["shoulders"]] + [_make_exercise(EXERCISE_DB["core"][0])]),
            WorkoutDay("Arms", [_make_exercise(e) for e in EXERCISE_DB["biceps"]] + [_make_exercise(e) for e in EXERCISE_DB["triceps"]]),
            WorkoutDay("Legs", [_make_exercise(e) for e in EXERCISE_DB["legs"]]),
        ]
        return [
            WorkoutDay(day_label=f"Day {i+1} – {splits[i].day_label}", exercises=splits[i].exercises)
            for i in range(min(days, 5))
        ]

    def generate(
        self,
        training_days: int,
        preferred_split: Optional[str],
        goal: str,
    ) -> WorkoutPlan:
        """
        Select an appropriate split and generate the weekly plan.
        """
        # Auto-select split based on training days if not specified
        if not preferred_split:
            if training_days <= 3:
                preferred_split = "full_body"
            elif training_days == 4:
                preferred_split = "upper_lower"
            elif training_days <= 6:
                preferred_split = "ppl"
            else:
                preferred_split = "ppl"

        split_fn = {
            "full_body":   self._full_body,
            "upper_lower": self._upper_lower,
            "ppl":         self._ppl,
            "bro_split":   self._bro_split,
        }.get(preferred_split, self._upper_lower)

        weekly = split_fn(training_days)
        notes = [
            "Ensure each muscle group is trained at least twice per week for hypertrophy.",
            "Warm up with 5–10 minutes of low-intensity cardio before lifting.",
        ]
        if goal == "fat_loss":
            notes.append("Add 2–3 sessions of 30-min low-intensity cardio on rest days to increase energy expenditure.")
        if goal == "muscle_gain":
            notes.append("Prioritise sleep (8+ hours) and calorie surplus on training days.")

        return WorkoutPlan(
            split=preferred_split.replace("_", " ").title(),
            training_days=training_days,
            weekly_plan=weekly,
            overload_protocol=OVERLOAD_PROTOCOL,
            notes=notes,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PHYSIQUE ENGINE FACADE
# ══════════════════════════════════════════════════════════════════════════════

class PhysiqueEngine:
    def __init__(self):
        self.macro_calc = MacroCalculator()
        self.workout_gen = WorkoutGenerator()

    def calorie_plan(
        self,
        weight_kg: float,
        body_fat_pct: float,
        activity_level: str,
        goal: str,
    ) -> CaloriePlan:
        return self.macro_calc.build_plan(weight_kg, body_fat_pct, activity_level, goal)

    def workout_plan(
        self,
        training_days: int,
        preferred_split: Optional[str],
        goal: str,
    ) -> WorkoutPlan:
        return self.workout_gen.generate(training_days, preferred_split, goal)


physique_engine = PhysiqueEngine()
