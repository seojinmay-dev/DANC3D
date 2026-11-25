"""RAW 가중 점수와 표시 EMA 및 분리한 상태 객체의 원본 UI 동등성을 검증한다."""

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

from app.core import score
from app.core.feedback import FeedbackStabilizer, final_feedback_metrics


RESEARCH_PATH = Path("/Users/may/Desktop/dance_test/06_2_realtime_webcam_feedback.py")


def display_trace():
    """최초 NONE, 최초 확인, hold, NONE 복귀와 후보 변경을 포함한 프레임을 만든다."""
    messages_and_times = (
        [("NONE", step / 128) for step in range(5)]
        + [("팔을 더 올리세요", .0625 + step / 64) for step in range(5)]
        + [("무릎을 더 펴세요", .25 + step / 32) for step in range(5)]
        + [("무릎을 더 펴세요", .625)]
        + [("NONE", .75 + step / 32) for step in range(5)]
        + [("NONE", 1.125)]
        + [("왼팔을 더 내리세요" if step % 2 else "오른팔을 더 올리세요", 1.25 + step / 32)
           for step in range(6)]
        + [("몸을 더 왼쪽으로 돌리세요", 1.5 + step / 32) for step in range(5)]
    )
    raw_scores = [np.nan, .9, .7, .4, 1., .125, .625, np.nan, .8]
    return [(raw_scores[step % len(raw_scores)], message, playback)
            for step, (message, playback) in enumerate(messages_and_times)]


class RawScoreTests(unittest.TestCase):
    def test_four_part_mean_worst_two_and_weights(self):
        metrics = score.final_score_metrics(.9, [1., .9, .7, .5], 0)
        self.assertAlmostEqual(metrics["score_mean_part"], .775)
        self.assertAlmostEqual(metrics["score_worst_two"], .6)
        self.assertAlmostEqual(metrics["pose_score"], .7225)
        self.assertEqual(metrics["final_score"], metrics["pose_score"])
        self.assertEqual(metrics["score_old_overall"], .9)

    def test_invalid_parts_are_excluded_and_single_part(self):
        metrics = score.final_score_metrics(.7, [np.nan, .6, np.inf, np.nan], -.8)
        self.assertEqual(metrics["score_mean_part"], .6)
        self.assertEqual(metrics["score_worst_two"], .6)
        self.assertEqual(metrics["pose_score"], .6)
        metrics = score.final_score_metrics(.7, [.9, np.nan, .5, np.nan], 0)
        self.assertAlmostEqual(metrics["score_mean_part"], .7)
        self.assertAlmostEqual(metrics["score_worst_two"], .7)

    def test_no_valid_parts_stay_nan(self):
        metrics = score.final_score_metrics(np.nan, [np.nan, np.inf, -np.inf, np.nan], 0)
        for key in ("score_mean_part", "score_worst_two", "pose_score", "final_score", "display_score"):
            self.assertTrue(math.isnan(metrics[key]), key)

    def test_worst_two_are_sorted_and_overall_is_diagnostic_only(self):
        first = score.final_score_metrics(0, [.2, 1., .8, .1], 0)
        second = score.final_score_metrics(1, [.1, .8, .2, 1.], 0)
        self.assertAlmostEqual(first["score_worst_two"], .15)
        for key in first:
            if key != "score_old_overall":
                self.assertEqual(first[key], second[key])

    def test_timing_factor_boundaries(self):
        for late, expected in ((0, 1), (.3, 1), (np.nextafter(.3, 0), 1),
                               (.65, .85), (1., .7), (1.5, .7)):
            with self.subTest(late=late):
                metrics = score.final_score_metrics(1, [1] * 4, -late)
                self.assertEqual(metrics["score_late_sec"], late)
                self.assertAlmostEqual(metrics["score_timing_factor"], expected)
        after_free = score.final_score_metrics(1, [1] * 4, -.300001)
        self.assertLess(after_free["score_timing_factor"], 1)
        before_cap = score.final_score_metrics(1, [1] * 4, -.999999)
        self.assertGreater(before_cap["score_timing_factor"], .7)

    def test_pose_gate_preserves_original_floating_point_order(self):
        # 부동소수 계산 후 PoseScore가 정확히 0.8인 입력을 사용해 gate 자체를 검사한다.
        at_gate = score.final_score_metrics(1, [np.nextafter(.8, 1)] * 4, -1)
        self.assertEqual(at_gate["pose_score"], .8)
        self.assertEqual(at_gate["final_score"], at_gate["pose_score"] * .7)
        below_gate = score.final_score_metrics(1, [np.nextafter(.8, 0)] * 4, -1)
        self.assertLess(below_gate["pose_score"], .8)
        self.assertEqual(below_gate["final_score"], below_gate["pose_score"])
        # 원본도 [0.8]*4 가중 합은 0.8보다 작게 표현된다. 임의 반올림하지 않는다.
        original_boundary = score.final_score_metrics(1, [.8] * 4, -1)
        self.assertLess(original_boundary["pose_score"], .8)
        self.assertEqual(original_boundary["final_score"], original_boundary["pose_score"])

    def test_fast_movement_has_no_extra_penalty(self):
        metrics = score.final_score_metrics(.9, [.9] * 4, 2)
        self.assertEqual(metrics["score_late_sec"], 0)
        self.assertEqual(metrics["score_timing_factor"], 1)
        self.assertEqual(metrics["final_score"], metrics["pose_score"])

    def test_invalid_timing_respects_pose_gate(self):
        for offset in (np.nan, np.inf, -np.inf):
            high = score.final_score_metrics(1, [1] * 4, offset)
            self.assertTrue(math.isnan(high["final_score"]))
            self.assertTrue(math.isnan(high["display_score"]))
            low = score.final_score_metrics(.5, [.5] * 4, offset)
            self.assertEqual(low["final_score"], .5)
            self.assertTrue(math.isnan(low["score_timing_factor"]))

    def test_score_clipping_and_python_round(self):
        for parts, expected in (([2] * 4, 1), ([-1] * 4, 0)):
            self.assertEqual(score.final_score_metrics(1, parts, 0)["pose_score"], expected)
        for value, rounded in ((.125, 12), (.625, 62), (.875, 88)):
            metrics = score.final_score_metrics(value, [value] * 4, 0)
            self.assertEqual(metrics["display_score"], rounded)
            self.assertIsInstance(metrics["display_score"], int)

    def test_input_contract_and_preservation(self):
        for parts in ([1, 1, 1], [[1, 1, 1, 1]], [1j] * 4, ["bad"] * 4):
            with self.assertRaises(ValueError):
                score.final_score_metrics(1, parts, 0)
        for overall, offset in ((None, 0), ([1], 0), (1, None), (1, [0]), (1, 1j)):
            with self.assertRaises(ValueError):
                score.final_score_metrics(overall, [1] * 4, offset)
        parts = np.array([1., .9, .7, np.nan])
        before = parts.copy()
        score.final_score_metrics(.9, parts, 0)
        np.testing.assert_array_equal(parts, before)


class ScoreEMAStabilizerTests(unittest.TestCase):
    def test_initial_nan_and_first_valid_score(self):
        obj = score.ScoreEMAStabilizer()
        obj.update(np.nan)
        self.assertIsNone(obj.smoothed_score)
        self.assertTrue(math.isnan(obj.snapshot(np.nan)["displayed_score"]))
        obj.update(.9)
        self.assertEqual(obj.smoothed_score, 90)
        self.assertEqual(obj.snapshot(.9)["displayed_score"], 90)

    def test_alpha_point_two_and_raw_preservation(self):
        obj = score.ScoreEMAStabilizer()
        obj.update(.9)
        obj.update(.4)
        self.assertEqual(obj.smoothed_score, 80)
        self.assertEqual(obj.snapshot(.4), {"raw_final_score": .4, "displayed_score": 80})
        obj.update(1)
        self.assertEqual(obj.smoothed_score, 84)

    def test_nan_and_infinity_keep_previous_display_score(self):
        obj = score.ScoreEMAStabilizer()
        obj.update(.9)
        for invalid in (np.nan, np.inf, -np.inf):
            obj.update(invalid)
            self.assertEqual(obj.smoothed_score, 90)
            self.assertEqual(obj.snapshot(invalid)["displayed_score"], 90)

    def test_ema_round_and_invalid_update_is_atomic(self):
        obj = score.ScoreEMAStabilizer()
        obj.update(.125)
        self.assertEqual(obj.snapshot(.125)["displayed_score"], 12)
        obj.update(.625)
        self.assertEqual(obj.smoothed_score, 22.5)
        self.assertEqual(obj.snapshot(.625)["displayed_score"], 22)
        for invalid in (None, [1], "bad", 1j):
            with self.assertRaises(ValueError):
                obj.update(invalid)
            self.assertEqual(obj.smoothed_score, 22.5)

    def test_separate_display_objects_do_not_overwrite_raw_metrics(self):
        raw_score = score.final_score_metrics(.4, [.4] * 4, -.6)
        raw_feedback = final_feedback_metrics(None, .9, 0, .6, 0)
        saved_score, saved_feedback = raw_score.copy(), raw_feedback.copy()
        score_display, feedback_display = score.ScoreEMAStabilizer(), FeedbackStabilizer()
        score_display.update(1)
        score_display.update(raw_score["final_score"])
        feedback_display.update(raw_feedback["final_feedback"], .6)
        self.assertEqual(score_display.snapshot(raw_score["final_score"])["displayed_score"], 88)
        self.assertEqual(feedback_display.snapshot(raw_feedback["final_feedback"])["displayed_feedback"], "NONE")
        self.assertEqual(raw_score, saved_score)
        self.assertEqual(raw_feedback, saved_feedback)


@unittest.skipUnless(RESEARCH_PATH.is_file(), "로컬 6-2 연구 파일이 없습니다.")
class ResearchScoreParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("danc3d_score_research", RESEARCH_PATH)
        cls.research = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.research)
        cls.research.np = np

    def assert_metrics_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            with self.subTest(field=key):
                if isinstance(expected[key], str):
                    self.assertEqual(actual[key], expected[key])
                else:
                    np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)

    def test_constants_and_raw_score_metrics(self):
        for name in ("POSE_MEAN_WEIGHT", "POSE_WORST_WEIGHT", "POSE_TIMING_GATE", "TIMING_FREE_SEC",
                     "TIMING_MAX_PENALTY_SEC", "TIMING_MIN_FACTOR", "SCORE_EMA_ALPHA"):
            self.assertEqual(getattr(score, name), getattr(self.research, name))
        part_cases = ([1, .9, .7, .5], [.8] * 4, [np.nextafter(.8, 1)] * 4,
                      [.9] * 4, [.5] * 4, [np.nan, .6, np.inf, np.nan], [np.nan] * 4,
                      [-1, -.5, .5, 1], [.125] * 4, [.625] * 4)
        offsets = (-2, -1, -.999999, -.65, -.3, -.299999, 0, 1, np.nan, np.inf)
        for parts in part_cases:
            for offset in offsets:
                with self.subTest(parts=parts, offset=offset):
                    self.assert_metrics_equal(score.final_score_metrics(.87, parts, offset),
                                              self.research.final_score_metrics(.87, parts, offset))

    def test_split_stabilizers_match_original_every_frame(self):
        for start_time in (0., 10.):
            original = self.research.UIStabilizer(start_time)
            score_display = score.ScoreEMAStabilizer()
            feedback_display = FeedbackStabilizer(start_time)
            for frame, (raw_score, raw_feedback, playback) in enumerate(display_trace()):
                with self.subTest(start_time=start_time, frame=frame):
                    now = start_time + playback
                    original.update(raw_score, raw_feedback, now)
                    score_display.update(raw_score)
                    feedback_display.update(raw_feedback, now)
                    actual = dict(score_display.snapshot(raw_score), **feedback_display.snapshot(raw_feedback))
                    expected = original.snapshot(raw_score, raw_feedback)
                    self.assert_metrics_equal(actual, expected)
                    self.assertEqual(score_display.smoothed_score, original.smoothed_score)
                    self.assertEqual(feedback_display.has_displayed_confirmation, original.has_displayed_confirmation)
                    self.assertEqual(feedback_display.displayed_feedback_since_time, original.displayed_feedback_since_time)

    def test_raw_metrics_feed_split_stabilizers_without_mutation(self):
        original = self.research.UIStabilizer()
        score_display, feedback_display = score.ScoreEMAStabilizer(), FeedbackStabilizer()
        for frame in range(24):
            playback = frame / 16
            overall = .9 if frame < 12 else .5
            parts = [overall] * 4 if frame != 20 else [np.nan] * 4
            aligned_time = max(0, playback - .5)
            pose = {"text": "팔을 더 올리세요"} if 5 <= frame < 12 else None
            raw_feedback = final_feedback_metrics(pose, overall, aligned_time, playback, 0)
            raw_score = score.final_score_metrics(overall, parts, raw_feedback["timing_offset_sec"])
            saved_score, saved_feedback = raw_score.copy(), raw_feedback.copy()
            original.update(raw_score["final_score"], raw_feedback["final_feedback"], playback)
            score_display.update(raw_score["final_score"])
            feedback_display.update(raw_feedback["final_feedback"], playback)
            actual = dict(score_display.snapshot(raw_score["final_score"]),
                          **feedback_display.snapshot(raw_feedback["final_feedback"]))
            self.assert_metrics_equal(actual, original.snapshot(raw_score["final_score"], raw_feedback["final_feedback"]))
            self.assert_metrics_equal(raw_score, saved_score)
            self.assert_metrics_equal(raw_feedback, saved_feedback)


if __name__ == "__main__":
    unittest.main()
