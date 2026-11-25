"""해석 가능한 합성 Pose로 교정 방향, RAW 우선순위와 표시 안정화를 검증한다."""

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

from app.core import feedback
from app.core.pose_features import extract_bone_directions
from app.core.pose_similarity import compare


RESEARCH_PATH = Path("/Users/may/Desktop/dance_test/06_2_realtime_webcam_feedback.py")


def neutral_world():
    """실제 인체 LEFT는 +x, torso-up은 +y인 곧게 편 양팔·양다리를 만든다."""
    world = np.zeros((33, 3), dtype=np.float64)
    world[11], world[12] = [.5, 1, 0], [-.5, 1, 0]
    world[13], world[14] = [1.5, 1, 0], [-1.5, 1, 0]
    world[15], world[16] = [2.5, 1, 0], [-2.5, 1, 0]
    world[23], world[24] = [.5, 0, 0], [-.5, 0, 0]
    world[25], world[26] = [.5, -1, 0], [-.5, -1, 0]
    world[27], world[28] = [.5, -2, 0], [-.5, -2, 0]
    return world


def set_arm(world, side, elevation=0, elbow_angle=180):
    """팔 높이와 팔꿈치 각도를 독립적으로 조절한 World를 복사하여 반환한다."""
    world = world.copy()
    shoulder, elbow, wrist, sign = (11, 13, 15, 1) if side == "left" else (12, 14, 16, -1)
    elevation, flexion = np.deg2rad([elevation, 180 - elbow_angle])
    direction = np.array([sign * np.cos(elevation), np.sin(elevation), 0])
    forearm = np.array([sign * np.cos(elevation + flexion), np.sin(elevation + flexion), 0])
    world[elbow] = world[shoulder] + direction
    world[wrist] = world[elbow] + forearm
    return world


def set_leg(world, side, outward=0, knee_angle=180):
    """대퇴 벌림과 무릎 굽힘을 독립적으로 조절한 World 복사본을 반환한다."""
    world = world.copy()
    hip, knee, ankle, sign = (23, 25, 27, 1) if side == "left" else (24, 26, 28, -1)
    outward, flexion = np.deg2rad([outward, 180 - knee_angle])
    direction = np.array([sign * np.sin(outward), -np.cos(outward), 0])
    shin = direction * np.cos(flexion) + np.array([0, 0, np.sin(flexion)])
    world[knee] = world[hip] + direction
    world[ankle] = world[knee] + shin
    return world


def rotate_yaw(world, yaw):
    """원본의 signed torso yaw가 지정한 도만큼 증가하도록 전체 Pose를 회전한다."""
    radians = np.deg2rad(yaw)
    cosine, sine = np.cos(radians), np.sin(radians)
    rotation = np.array([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]])
    return world @ rotation.T


def pose_cosines(reference, user):
    """검증된 1단계 모듈에서 실제 DIR8 cosine을 계산한다."""
    reference_vectors, reference_mask = extract_bone_directions(reference)
    user_vectors, user_mask = extract_bone_directions(user)
    return compare(reference_vectors, user_vectors, reference_mask, user_mask)[0]


def meaningful_cases():
    """양측의 예상 가능한 교정 및 복합/무효 Pose 입력을 제공한다."""
    base = neutral_world()
    cases = [(base, base.copy())]
    for side in ("left", "right"):
        cases += [(base, set_arm(base, side, elevation=elevation)) for elevation in (-60, 60)]
        bent_arm = set_arm(base, side, elbow_angle=90)
        wide_leg = set_leg(base, side, outward=45)
        bent_knee = set_leg(base, side, knee_angle=90)
        cases += [(base, bent_arm), (bent_arm, base), (base, wide_leg), (wide_leg, base),
                  (base, bent_knee), (bent_knee, base)]
    cases += [(base, rotate_yaw(base, yaw)) for yaw in (-45, 45, 179)]
    bilateral = set_arm(set_arm(base, "left", elevation=-60), "right", elevation=-60)
    conflicting = set_arm(set_arm(base, "left", elevation=-45), "right", elevation=60)
    multiple = set_leg(set_arm(base, "left", elevation=-45), "right", knee_angle=90)
    partial = set_arm(base, "left", elevation=-60)
    partial[15] = np.nan
    cases += [(base, bilateral), (base, conflicting), (base, multiple), (base, partial),
              (base, np.full((33, 3), np.nan)), (base, np.zeros((33, 3)))]
    return cases


class PoseFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.base = neutral_world()

    def evaluate(self, reference, user):
        return feedback.pose_feedback(reference, user, pose_cosines(reference, user))

    def test_normal_pose_has_no_candidate(self):
        metrics, candidate = self.evaluate(self.base, self.base)
        self.assertIsNone(candidate)
        self.assertEqual(metrics["priority_feedback"], "NONE")
        self.assertEqual(metrics["priority_feedback_type"], "NONE")
        self.assertTrue(math.isnan(metrics["priority_feedback_error_deg"]))

    def test_left_right_arm_raise_lower(self):
        for side, bone, subject in (("left", 0, "왼팔을"), ("right", 2, "오른팔을")):
            for elevation, action, verb in ((-60, "RAISE", "올리세요"), (60, "LOWER", "내리세요")):
                with self.subTest(side=side, elevation=elevation):
                    metrics, candidate = self.evaluate(self.base, set_arm(self.base, side, elevation=elevation))
                    self.assertEqual(metrics[side + "_arm_action"], action)
                    self.assertAlmostEqual(metrics[side + "_arm_direction_error_deg"], elevation)
                    self.assertEqual(candidate["text"], subject + " 더 " + verb)
                    self.assertEqual(candidate["side"], side)
                    self.assertEqual(candidate["bones"], [bone])

    def test_elbow_extend_bend(self):
        for side in ("left", "right"):
            bent = set_arm(self.base, side, elbow_angle=90)
            for reference, user, action in ((self.base, bent, "EXTEND"), (bent, self.base, "BEND")):
                metrics, candidate = self.evaluate(reference, user)
                self.assertEqual(metrics[side + "_elbow_action"], action)
                self.assertEqual(candidate["type"], "elbow_angle")
                self.assertEqual(candidate["side"], side)
                self.assertAlmostEqual(candidate["error_deg"], 90)

    def test_leg_out_in(self):
        for side in ("left", "right"):
            wide = set_leg(self.base, side, outward=45)
            for reference, user, action in ((wide, self.base, "OUT"), (self.base, wide, "IN")):
                metrics, candidate = feedback.leg_direction_feedback(reference, user, pose_cosines(reference, user))
                self.assertEqual(metrics[side + "_leg_action"], action)
                self.assertEqual(candidate["type"], "leg_direction")
                self.assertEqual(candidate["side"], side)
                self.assertAlmostEqual(candidate["error_deg"], 45)
                self.assertEqual(candidate["text"],
                                 ("왼쪽" if side == "left" else "오른쪽") + " 다리를 "
                                 + ("더 밖으로 보내세요" if action == "OUT" else "몸쪽으로 가져오세요"))

    def test_knee_extend_bend(self):
        for side in ("left", "right"):
            bent = set_leg(self.base, side, knee_angle=90)
            for reference, user, action in ((self.base, bent, "EXTEND"), (bent, self.base, "BEND")):
                metrics, candidate = feedback.knee_feedback(reference, user, pose_cosines(reference, user))
                self.assertEqual(metrics[side + "_knee_action"], action)
                self.assertEqual(candidate["type"], "knee_angle")
                self.assertEqual(candidate["side"], side)
                self.assertAlmostEqual(candidate["error_deg"], 90)

    def test_torso_left_right_and_yaw_wrapping(self):
        for yaw, action, text in ((-45, "TURN_LEFT", "몸을 더 왼쪽으로 돌리세요"),
                                  (45, "TURN_RIGHT", "몸을 더 오른쪽으로 돌리세요")):
            metrics, candidate = self.evaluate(self.base, rotate_yaw(self.base, yaw))
            self.assertEqual(metrics["torso_rotation_action"], action)
            self.assertEqual(candidate["text"], text)
            self.assertEqual(candidate["type"], "torso_rotation")
            self.assertAlmostEqual(metrics["torso_yaw_error_deg"], -yaw)
        metrics, candidate = feedback.torso_feedback(rotate_yaw(self.base, -179), rotate_yaw(self.base, 179))
        self.assertAlmostEqual(metrics["torso_yaw_error_deg"], 2)
        self.assertIsNone(candidate)

    def test_same_action_merges_messages(self):
        user = set_arm(set_arm(self.base, "left", elevation=-60), "right", elevation=-60)
        _, candidate = self.evaluate(self.base, user)
        self.assertEqual(candidate["text"], "팔을 더 올리세요")
        self.assertEqual(candidate["side"], "both")
        self.assertEqual(candidate["bones"], [0, 2])
        wide = set_leg(set_leg(self.base, "left", outward=45), "right", outward=45)
        _, candidate = feedback.leg_direction_feedback(wide, self.base, pose_cosines(wide, self.base))
        self.assertEqual(candidate["text"], "다리를 더 밖으로 보내세요")
        self.assertEqual(candidate["side"], "both")

    def test_conflicting_actions_choose_larger_error_and_first_tie(self):
        user = set_arm(set_arm(self.base, "left", elevation=-45), "right", elevation=60)
        _, candidate = self.evaluate(self.base, user)
        self.assertEqual(candidate["text"], "오른팔을 더 내리세요")
        self.assertEqual(candidate["representative_side"], "right")
        user = set_arm(set_arm(self.base, "left", elevation=-60), "right", elevation=60)
        _, candidate = self.evaluate(self.base, user)
        self.assertEqual(candidate["side"], "left")

    def test_priority_is_largest_error_with_original_tie_order(self):
        user = set_leg(set_arm(self.base, "left", elevation=-45), "right", knee_angle=90)
        _, candidate = self.evaluate(self.base, user)
        self.assertEqual(candidate["type"], "knee_angle")
        self.assertEqual(candidate["side"], "right")
        user = set_leg(set_arm(self.base, "left", elevation=-90, elbow_angle=90), "right", knee_angle=90)
        _, candidate = self.evaluate(self.base, user)
        self.assertEqual(candidate["type"], "arm_direction")
        self.assertAlmostEqual(candidate["error_deg"], 90)

    def test_cosine_gate_and_angle_threshold(self):
        user = set_arm(self.base, "left", elevation=-60)
        metrics, candidate = feedback.pose_feedback(self.base, user, np.full(8, .8))
        self.assertEqual(metrics["left_arm_action"], "NONE")
        self.assertIsNone(candidate)
        values = np.full(8, .8)
        values[0] = np.nextafter(.8, 0)
        self.assertEqual(feedback.pose_feedback(self.base, user, values)[0]["left_arm_action"], "RAISE")
        for error, expected in ((-15, "RAISE"), (15, "LOWER"), (np.nextafter(15., 0), "NONE"), (np.nan, "NONE")):
            self.assertEqual(feedback.feedback_action(error, 15, True, "RAISE", "LOWER"), expected)
        self.assertEqual(feedback.feedback_action(60, 15, False, "RAISE", "LOWER"), "NONE")
        self.assertIsNone(feedback.pose_feedback(self.base, user, np.full(8, np.nan))[1])
        # 몸통은 유효 DIR8 오류가 없어도 독립적인 yaw 판정으로 교정한다.
        self.assertEqual(feedback.pose_feedback(self.base, rotate_yaw(self.base, 45), np.ones(8))[1]["type"], "torso_rotation")

    def test_body_up_ignores_global_y_and_translation(self):
        user = set_arm(self.base, "left", elevation=-60)
        transform = np.diag([-1., -1., 1.])
        shift = np.array([4., 2., 3.])
        reference_rotated = self.base @ transform.T + shift
        user_rotated = user @ transform.T + shift
        metrics, candidate = self.evaluate(reference_rotated, user_rotated)
        self.assertEqual(metrics["left_arm_action"], "RAISE")
        self.assertAlmostEqual(metrics["left_arm_direction_error_deg"], -60)
        self.assertEqual(candidate["side"], "left")

    def test_invalid_landmarks_and_degenerate_pose(self):
        for user in (np.full((33, 3), np.nan), np.zeros((33, 3))):
            metrics, candidate = self.evaluate(self.base, user)
            self.assertIsNone(candidate)
            self.assertEqual(metrics["torso_rotation_error_flag"], 0)
            self.assertTrue(math.isnan(metrics["torso_yaw_error_deg"]))
        user = set_arm(self.base, "left", elevation=-60)
        user[15] = np.nan
        metrics, candidate = self.evaluate(self.base, user)
        self.assertEqual(metrics["left_elbow_action"], "NONE")
        self.assertEqual(metrics["left_arm_action"], "RAISE")
        self.assertEqual(candidate["type"], "arm_direction")

    def test_input_validation_and_no_input_mutation(self):
        for reference, user, values in ((np.zeros((32, 3)), self.base, np.ones(8)),
                                        (self.base, self.base, np.ones(7)),
                                        (self.base, self.base, np.ones(8, complex))):
            with self.assertRaises(ValueError):
                feedback.pose_feedback(reference, user, values)
        before = self.base.copy()
        self.evaluate(self.base, set_arm(self.base, "left", elevation=-60))
        np.testing.assert_array_equal(self.base, before)


class TimingFeedbackTests(unittest.TestCase):
    def test_late_threshold_similarity_gate_and_no_fast_message(self):
        for overall, playback, expected in ((.8, .3, "LATE"), (.8, np.nextafter(.3, 0), "NONE"),
                                            (np.nextafter(.8, 0), .5, "NONE"), (np.nan, .5, "NONE"),
                                            (1., 0., "NONE")):
            result = feedback.final_feedback_metrics(None, overall, 0, playback, 0)
            self.assertEqual(result["timing_feedback"], expected)
            self.assertEqual(result["timing_offset_sec"], -playback)
        fast = feedback.final_feedback_metrics(None, 1, 1, 0, 1)
        self.assertEqual(fast["final_feedback"], "NONE")

    def test_pose_precedes_timing(self):
        base = neutral_world()
        user = set_arm(base, "left", elevation=-60)
        pose = feedback.pose_feedback(base, user, pose_cosines(base, user))[1]
        result = feedback.final_feedback_metrics(pose, 1, 0, 1, 0)
        self.assertEqual(result["final_feedback"], "왼팔을 더 올리세요")
        self.assertEqual(result["priority_pose_feedback"], result["final_feedback"])
        self.assertEqual(result["timing_feedback"], "NONE")

    def test_missing_alignment_and_invalid_inputs(self):
        result = feedback.final_feedback_metrics(None, np.nan, None, .5)
        self.assertEqual(result["aligned_reference_index"], -1)
        self.assertTrue(math.isnan(result["timing_offset_sec"]))
        self.assertEqual(result["final_feedback"], "NONE")
        for index in (-1, True, .5):
            with self.assertRaises(ValueError):
                feedback.final_feedback_metrics(None, 1, 0, 0, index)
        with self.assertRaises(ValueError):
            feedback.final_feedback_metrics(None, 1, [0], 0)


class FeedbackStabilizerTests(unittest.TestCase):
    def test_five_confirmations_first_message_without_initial_hold(self):
        obj = feedback.FeedbackStabilizer()
        for step in range(4):
            obj.update("팔을 더 올리세요", step / 100)
            self.assertEqual(obj.displayed_feedback, "NONE")
        obj.update("팔을 더 올리세요", .04)
        self.assertEqual(obj.displayed_feedback, "팔을 더 올리세요")
        self.assertTrue(obj.has_displayed_confirmation)
        self.assertEqual(obj.feedback_candidate_count, 5)

    def test_initial_none_does_not_start_confirmed_hold(self):
        obj = feedback.FeedbackStabilizer()
        for _ in range(5):
            obj.update("NONE", 0)
        self.assertFalse(obj.has_displayed_confirmation)
        for _ in range(5):
            obj.update("팔을 더 올리세요", .125)
        self.assertEqual(obj.displayed_feedback, "팔을 더 올리세요")

    def test_hold_boundary_and_candidate_reset(self):
        obj = feedback.FeedbackStabilizer()
        for _ in range(5):
            obj.update("A", .125)
        for step in range(5):
            obj.update("B", .25 + step / 32)
        self.assertEqual(obj.feedback_candidate_count, 5)
        self.assertEqual(obj.displayed_feedback, "A")
        obj.update("B", .625)
        self.assertEqual(obj.displayed_feedback, "B")
        self.assertEqual(obj.displayed_feedback_since_time, .625)
        obj.update("C", .625)
        self.assertEqual(obj.feedback_candidate_count, 1)
        obj.update("B", .625)
        self.assertEqual(obj.feedback_candidate_count, 1)

    def test_none_requires_confirmation_and_hold(self):
        obj = feedback.FeedbackStabilizer()
        for _ in range(5):
            obj.update("A", .125)
        for _ in range(5):
            obj.update("NONE", .25)
        self.assertEqual(obj.displayed_feedback, "A")
        obj.update("NONE", .625)
        self.assertEqual(obj.displayed_feedback, "NONE")

    def test_snapshot_preserves_raw_and_invalid_update_is_atomic(self):
        obj = feedback.FeedbackStabilizer()
        obj.update("A", 0)
        snapshot = obj.snapshot("A")
        self.assertEqual(snapshot["raw_final_feedback"], "A")
        self.assertEqual(snapshot["displayed_feedback"], "NONE")
        snapshot["feedback_candidate"] = "changed"
        self.assertEqual(obj.feedback_candidate, "A")
        for text, now in ((None, 0), ("B", np.nan), ("B", -1), ("B", [1])):
            with self.assertRaises(ValueError):
                obj.update(text, now)
            self.assertEqual(obj.feedback_candidate_count, 1)
            self.assertEqual(obj.feedback_candidate, "A")


@unittest.skipUnless(RESEARCH_PATH.is_file(), "로컬 6-2 연구 파일이 없습니다.")
class ResearchFeedbackParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("danc3d_feedback_research", RESEARCH_PATH)
        cls.research = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.research)
        cls.research.np = np

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                with self.subTest(field=key):
                    self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_equal(a, b)
        elif isinstance(expected, (float, np.floating)):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
        else:
            self.assertEqual(actual, expected)

    def test_constants_and_raw_direction_candidates(self):
        for name in ("BONE_ERROR_THRESHOLD", "ARM_DIRECTION_THRESHOLD_DEG", "ELBOW_ANGLE_THRESHOLD_DEG",
                     "KNEE_ANGLE_THRESHOLD_DEG", "LEG_OUTWARD_THRESHOLD_DEG", "TORSO_ROTATION_THRESHOLD_DEG",
                     "POSE_GATE", "TIMING_THRESHOLD_SEC", "FEEDBACK_CONFIRM_FRAMES", "FEEDBACK_MIN_HOLD_SEC"):
            self.assertEqual(getattr(feedback, name), getattr(self.research, name))
        for case, (reference, user) in enumerate(meaningful_cases()):
            values = pose_cosines(reference, user)
            sequence, user_sequence = dict(world=reference[None]), dict(world=user[None])
            with self.subTest(case=case):
                self.assert_nested_equal(feedback.pose_feedback(reference, user, values),
                                         self.research.pose_feedback(sequence, 0, 0, values, user_sequence))
                self.assert_nested_equal(feedback.knee_feedback(reference, user, values),
                                         self.research.knee_feedback(sequence, 0, 0, values, user_sequence))
                self.assert_nested_equal(feedback.leg_direction_feedback(reference, user, values),
                                         self.research.leg_direction_feedback(sequence, 0, 0, values, user_sequence))
                self.assert_nested_equal(feedback.torso_feedback(reference, user),
                                         self.research.torso_feedback(sequence, 0, 0, user_sequence))
                for shoulder, elbow, wrist in ((11, 13, 15), (12, 14, 16)):
                    self.assert_nested_equal(feedback.arm_measurements(user, shoulder, elbow, wrist),
                                             self.research.arm_measurements(user, shoulder, elbow, wrist))

    def test_timing_and_final_feedback(self):
        base = neutral_world()
        pose = feedback.pose_feedback(base, set_arm(base, "left", elevation=-60), np.zeros(8))[1]
        times = np.array([0., .125, .5])
        for candidate in (None, pose):
            for aligned in (None, 0, 1, 2):
                for overall in (np.nan, .79, .8, 1):
                    for playback in (0., .3, .5, 1.):
                        actual = feedback.final_feedback_metrics(candidate, overall,
                            times[aligned] if aligned is not None else None, playback, aligned, 2.5)
                        expected = self.research.final_feedback_metrics(candidate, overall, aligned, times, playback, 2.5)
                        self.assert_nested_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
