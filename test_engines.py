"""
BioAesthetic — Unit Tests
All scoring functions must be numerically verified.
Run with: pytest tests/test_engines.py -v
"""

import pytest
import math


# ══════════════════════════════════════════════════════════════════════════════
# PHYSIQUE ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestMacroCalculator:

    def setup_method(self):
        from modules.physique_engine import MacroCalculator
        self.calc = MacroCalculator()

    def test_lean_body_mass(self):
        lbm = self.calc.lean_body_mass(weight_kg=80, body_fat_pct=15)
        assert lbm == pytest.approx(68.0, abs=0.01)

    def test_lean_body_mass_zero_fat(self):
        lbm = self.calc.lean_body_mass(weight_kg=70, body_fat_pct=0)
        assert lbm == 70.0

    def test_bmr_katch_mcardle(self):
        """BMR = 370 + 21.6 × LBM"""
        bmr = self.calc.bmr(lbm_kg=68.0)
        assert bmr == pytest.approx(370 + 21.6 * 68.0, abs=0.01)

    def test_tdee_sedentary(self):
        tdee = self.calc.tdee(bmr=1840, activity_level="sedentary")
        assert tdee == pytest.approx(1840 * 1.2, abs=1)

    def test_tdee_active(self):
        tdee = self.calc.tdee(bmr=1840, activity_level="active")
        assert tdee == pytest.approx(1840 * 1.725, abs=1)

    def test_target_calories_fat_loss(self):
        cals = self.calc.target_calories(tdee=2500, goal="fat_loss")
        assert cals == 2100  # 2500 - 400

    def test_target_calories_muscle_gain(self):
        cals = self.calc.target_calories(tdee=2500, goal="muscle_gain")
        assert cals == 2800  # 2500 + 300

    def test_target_calories_maintenance(self):
        cals = self.calc.target_calories(tdee=2500, goal="maintenance")
        assert cals == 2500

    def test_calories_floor_enforced(self):
        """Should never go below 1200 kcal."""
        cals = self.calc.target_calories(tdee=1500, goal="fat_loss")
        assert cals >= 1200

    def test_macros_protein_floor(self):
        macros = self.calc.macros(calories=2000, weight_kg=80, goal="fat_loss")
        assert macros.protein_g >= 80 * 1.8
        assert macros.fat_g >= 80 * 0.9

    def test_macros_carbs_not_negative(self):
        macros = self.calc.macros(calories=1500, weight_kg=90, goal="fat_loss")
        assert macros.carbs_g >= 0

    def test_full_plan_output_types(self):
        plan = self.calc.build_plan(
            weight_kg=80, body_fat_pct=18,
            activity_level="moderate", goal="muscle_gain"
        )
        assert isinstance(plan.bmr, float)
        assert isinstance(plan.target_calories, int)
        assert isinstance(plan.lean_body_mass_kg, float)


class TestWorkoutGenerator:

    def setup_method(self):
        from modules.physique_engine import WorkoutGenerator
        self.gen = WorkoutGenerator()

    def test_full_body_max_3_days(self):
        plan = self.gen.generate(training_days=3, preferred_split="full_body", goal="muscle_gain")
        assert len(plan.weekly_plan) <= 3

    def test_ppl_6_days(self):
        plan = self.gen.generate(training_days=6, preferred_split="ppl", goal="muscle_gain")
        assert len(plan.weekly_plan) == 6
        labels = [d.day_label for d in plan.weekly_plan]
        assert any("Push" in l for l in labels)
        assert any("Pull" in l for l in labels)
        assert any("Legs" in l for l in labels)

    def test_upper_lower_4_days(self):
        plan = self.gen.generate(training_days=4, preferred_split="upper_lower", goal="fat_loss")
        assert len(plan.weekly_plan) == 4

    def test_auto_split_selection_1_day(self):
        plan = self.gen.generate(training_days=1, preferred_split=None, goal="maintenance")
        # Should auto-select full_body
        assert "Full Body" in plan.split or "Body" in plan.split

    def test_overload_protocol_present(self):
        plan = self.gen.generate(training_days=4, preferred_split=None, goal="muscle_gain")
        assert len(plan.overload_protocol) > 20

    def test_exercises_have_required_fields(self):
        plan = self.gen.generate(training_days=3, preferred_split="full_body", goal="maintenance")
        for day in plan.weekly_plan:
            assert len(day.exercises) > 0
            for ex in day.exercises:
                assert ex.name
                assert ex.sets > 0
                assert ex.rest_seconds > 0
                assert ex.muscle_group


# ══════════════════════════════════════════════════════════════════════════════
# HORMONE ENGINE TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestSleepScorer:

    def setup_method(self):
        from modules.hormone_engine import SleepScorer
        self.scorer = SleepScorer()

    def test_optimal_sleep(self):
        assert self.scorer.score(8.0) == 1.0

    def test_oversleeping_still_scores_1(self):
        assert self.scorer.score(10.0) == 1.0

    def test_zero_sleep(self):
        assert self.scorer.score(0.0) == 0.0

    def test_midpoint_is_intermediate(self):
        s = self.scorer.score(6.0)
        assert 0.0 < s < 1.0

    def test_monotonic_increase(self):
        scores = [self.scorer.score(h) for h in [0, 2, 4, 6, 7, 8]]
        assert scores == sorted(scores)


class TestSunlightScorer:

    def setup_method(self):
        from modules.hormone_engine import SunlightScorer
        self.scorer = SunlightScorer()

    def test_zero_sunlight(self):
        assert self.scorer.score(0) == 0.0

    def test_target_sunlight(self):
        assert self.scorer.score(20) == 1.0

    def test_excess_capped_at_1(self):
        assert self.scorer.score(60) == 1.0

    def test_half_target(self):
        assert self.scorer.score(10) == pytest.approx(0.5, abs=0.01)


class TestActivityScorer:

    def setup_method(self):
        from modules.hormone_engine import ActivityScorer
        self.scorer = ActivityScorer()

    def test_target_steps(self):
        assert self.scorer.score(9000) == 1.0

    def test_zero_steps_low_score(self):
        assert self.scorer.score(0) == pytest.approx(0.05, abs=0.01)

    def test_monotonic(self):
        scores = [self.scorer.score(s) for s in [0, 2000, 5000, 7000, 9000]]
        assert scores == sorted(scores)


class TestMicronutrientScorer:

    def setup_method(self):
        from modules.hormone_engine import MicronutrientScorer
        self.scorer = MicronutrientScorer()

    def test_optimal_zinc(self):
        assert self.scorer.score(11) == 1.0

    def test_zero_zinc(self):
        assert self.scorer.score(0) == 0.0

    def test_excess_zinc_penalised(self):
        high = self.scorer.score(100)
        normal = self.scorer.score(11)
        assert high < normal

    def test_sub_floor_is_zero(self):
        assert self.scorer.score(1.0) == 0.0


class TestStressScorer:

    def setup_method(self):
        from modules.hormone_engine import StressScorer
        self.scorer = StressScorer()

    def test_no_stress(self):
        assert self.scorer.score(0) == 1.0

    def test_max_stress(self):
        assert self.scorer.score(10) == 0.0

    def test_moderate_stress_intermediate(self):
        s = self.scorer.score(5)
        assert 0.0 < s < 1.0

    def test_monotonic_decrease(self):
        scores = [self.scorer.score(r) for r in range(11)]
        assert scores == sorted(scores, reverse=True)


class TestHormoneEngine:

    def setup_method(self):
        from modules.hormone_engine import HormoneEngine
        self.engine = HormoneEngine()

    def test_perfect_metrics_near_100(self):
        result = self.engine.score(
            sleep_hours=8, sunlight_minutes=20,
            step_count=9000, zinc_mg=11, stress_rating=0
        )
        assert result.hormone_score == pytest.approx(100.0, abs=0.1)

    def test_terrible_metrics_near_0(self):
        result = self.engine.score(
            sleep_hours=0, sunlight_minutes=0,
            step_count=0, zinc_mg=0, stress_rating=10
        )
        assert result.hormone_score < 10.0

    def test_score_in_range(self):
        result = self.engine.score(
            sleep_hours=6, sunlight_minutes=10,
            step_count=5000, zinc_mg=8, stress_rating=4
        )
        assert 0.0 <= result.hormone_score <= 100.0

    def test_breakdown_sums_to_total(self):
        result = self.engine.score(
            sleep_hours=7, sunlight_minutes=15,
            step_count=7000, zinc_mg=10, stress_rating=3
        )
        total_from_breakdown = sum(result.breakdown.values())
        assert total_from_breakdown == pytest.approx(result.hormone_score, abs=0.5)

    def test_recommendations_not_empty(self):
        result = self.engine.score(
            sleep_hours=4, sunlight_minutes=0,
            step_count=1000, zinc_mg=1, stress_rating=9
        )
        assert len(result.recommendations) > 0

    def test_disclaimer_present(self):
        result = self.engine.score(8, 20, 9000, 11, 0)
        assert "medical" in result.disclaimer.lower()


# ══════════════════════════════════════════════════════════════════════════════
# YOUTH INDEX TESTS
# ══════════════════════════════════════════════════════════════════════════════

class TestYouthIndexEngine:

    def setup_method(self):
        from modules.youth_index import YouthIndexEngine
        self.engine = YouthIndexEngine()

    def test_perfect_inputs_near_100(self):
        result = self.engine.compute(
            skin_composite_score=95.0,
            physique_score=90.0,
            hormone_score=95.0,
            sleep_score=1.0,
            activity_score=1.0,
        )
        assert result.youth_index >= 90

    def test_low_inputs_near_0(self):
        result = self.engine.compute(
            skin_composite_score=10.0,
            physique_score=5.0,
            hormone_score=8.0,
            sleep_score=0.05,
            activity_score=0.05,
        )
        assert result.youth_index < 20

    def test_output_integer(self):
        result = self.engine.compute(70, 65, 72, 0.7, 0.7)
        assert isinstance(result.youth_index, int)

    def test_delta_computed_correctly(self):
        result = self.engine.compute(70, 70, 70, 0.7, 0.7, previous_youth_index=60.0)
        assert result.weekly_delta is not None
        # Delta should be positive (improved)
        assert result.weekly_delta > 0

    def test_no_delta_without_previous(self):
        result = self.engine.compute(70, 70, 70, 0.7, 0.7, previous_youth_index=None)
        assert result.weekly_delta is None

    def test_grade_s_above_90(self):
        result = self.engine.compute(95, 95, 95, 1.0, 1.0)
        assert result.grade == "S"

    def test_grade_d_below_45(self):
        result = self.engine.compute(10, 5, 8, 0.05, 0.05)
        assert result.grade == "D"

    def test_bounds_always_valid(self):
        for skin in [0, 50, 100]:
            for hormone in [0, 50, 100]:
                r = self.engine.compute(skin, 50, hormone, 0.5, 0.5)
                assert 0 <= r.youth_index <= 100


class TestDerivePhhysiqueScore:

    def setup_method(self):
        from modules.youth_index import derive_physique_score
        self.fn = derive_physique_score

    def test_lean_athlete_scores_high(self):
        # FFMI ~23, low BF
        score = self.fn(body_fat_pct=10, lbm_kg=75, height_cm=180, goal="maintenance")
        assert score > 70

    def test_obese_scores_low(self):
        score = self.fn(body_fat_pct=40, lbm_kg=60, height_cm=170, goal="fat_loss")
        assert score < 50

    def test_score_bounded_0_100(self):
        for bf in [5, 15, 25, 40]:
            for lbm in [40, 60, 80]:
                s = self.fn(bf, lbm, 175, "maintenance")
                assert 0 <= s <= 100


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE SKIN SCORE TEST
# ══════════════════════════════════════════════════════════════════════════════

class TestCompositeSkinScore:

    def setup_method(self):
        from modules.skin_engine import compute_composite_skin_score
        self.fn = compute_composite_skin_score

    def test_perfect_skin(self):
        score = self.fn(0, 0, 0, 0, 0)
        assert score == 100.0

    def test_worst_skin(self):
        score = self.fn(1, 1, 1, 1, 1)
        assert score == pytest.approx(0.0, abs=0.1)

    def test_bounded_0_100(self):
        for v in [0.0, 0.3, 0.6, 1.0]:
            s = self.fn(v, v, v, v, v)
            assert 0 <= s <= 100

    def test_partial_issues(self):
        score = self.fn(0.5, 0.3, 0.2, 0.4, 0.1)
        assert 0 < score < 100
