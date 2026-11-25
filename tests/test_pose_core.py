"""DIR8 계약, 오류 판정 및 읽기 전용 연구 코드 동등성을 검증한다."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from app.core.pose_features import (
    BONES, PARTS, TORSO_EDGES, extract_bone_directions,
    validate_world_landmarks, pelvis_center, center_world_landmarks,
    angle_deg, angle_diff_deg,
)
from app.core.pose_similarity import compare, status, priority_error, valid_mean

RESEARCH_PATH = Path('/Users/may/Desktop/dance_test/06_2_realtime_webcam_feedback.py')


class PoseFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.world = np.random.default_rng(42).normal(size=(33, 3))

    def test_input_validation(self):
        for invalid in [None, np.zeros((32, 3)), np.zeros((33, 2)), [['bad']], np.zeros((33, 3), complex)]:
            with self.subTest(shape=np.shape(invalid)), self.assertRaises(ValueError):
                validate_world_landmarks(invalid)
        self.world[0, 0] = np.inf
        world = validate_world_landmarks(self.world)
        self.assertEqual(world.dtype, np.float64)
        self.assertTrue(np.isnan(world[0]).all())
        self.assertTrue(np.isinf(self.world[0, 0]))

    def test_shape_norm_and_indices(self):
        expected_indices = [(11, 13), (13, 15), (12, 14), (14, 16),
                            (23, 25), (25, 27), (24, 26), (26, 28)]
        self.assertEqual([(a, b) for _, a, b in BONES], expected_indices)
        self.assertEqual(PARTS, ('left_arm', 'right_arm', 'left_leg', 'right_leg'))
        self.assertEqual(TORSO_EDGES, ((11, 12), (12, 24), (24, 23), (23, 11)))
        directions, mask = extract_bone_directions(self.world)
        self.assertEqual(directions.shape, (8, 3))
        self.assertEqual(directions.dtype, np.float64)
        self.assertEqual(mask.shape, (8,))
        self.assertEqual(mask.dtype, np.bool_)
        np.testing.assert_allclose(np.linalg.norm(directions[mask], axis=1), 1)
        for index, (parent, child) in enumerate(expected_indices):
            delta = self.world[child] - self.world[parent]
            np.testing.assert_allclose(directions[index], delta / np.linalg.norm(delta))

    def test_zero_nan_and_length_boundary(self):
        self.world[13] = self.world[11]
        self.world[16, 2] = np.nan
        directions, mask = extract_bone_directions(self.world)
        self.assertFalse(mask[0])
        self.assertFalse(mask[3])
        self.assertTrue(np.isnan(directions[[0, 3]]).all())
        for length, expected in [(1e-8, False), (1.0001e-8, True), (0, False)]:
            world = np.zeros((33, 3))
            world[13, 0] = length
            self.assertEqual(extract_bone_directions(world)[1][0], expected)

    def test_pelvis_and_translation(self):
        original = self.world.copy()
        np.testing.assert_allclose(pelvis_center(self.world), (self.world[23] + self.world[24]) / 2)
        centered = center_world_landmarks(self.world)
        np.testing.assert_allclose(pelvis_center(centered), 0, atol=1e-15)
        np.testing.assert_allclose(extract_bone_directions(centered)[0], extract_bone_directions(self.world)[0])
        np.testing.assert_array_equal(self.world, original)
        self.world[23, 0] = np.nan
        self.assertTrue(np.isnan(pelvis_center(self.world)).all())
        self.assertTrue(np.isnan(center_world_landmarks(self.world)).all())

    def test_angles(self):
        self.assertEqual(angle_deg([1, 0, 0], [0, 1, 0]), 90)
        self.assertTrue(math.isnan(angle_deg([0, 0, 0], [1, 0, 0])))
        self.assertTrue(math.isnan(angle_deg([np.inf, 0, 0], [1, 0, 0])))
        with self.assertRaises(ValueError):
            angle_deg([1, 0], [1, 0, 0])
        self.assertEqual(angle_diff_deg(-179, 179), 2)
        self.assertTrue(math.isnan(angle_diff_deg(np.inf, 0)))


class PoseSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.tile([1., 0., 0.], (8, 1))
        self.mask = np.ones(8, bool)
        self.cosines = np.array([1, .9, .8, .7, .6, .5, -.5, -1])
        self.user = np.column_stack([self.cosines, np.sqrt(1 - self.cosines ** 2), np.zeros(8)])

    def test_identical(self):
        values, overall, parts = compare(self.reference, self.reference, self.mask, self.mask)
        np.testing.assert_array_equal(values, np.ones(8))
        self.assertEqual(overall, 1)
        self.assertEqual(parts, [1] * 4)

    def test_means_and_priority(self):
        values, overall, parts = compare(self.reference, self.user, self.mask, self.mask)
        np.testing.assert_allclose(values, self.cosines)
        self.assertAlmostEqual(overall, np.mean(self.cosines))
        np.testing.assert_allclose(parts, [.95, .75, .55, -.75])
        self.assertEqual(priority_error(values), 7)
        self.assertEqual([status(value) for value in values], ['OK'] * 3 + ['ERROR'] * 5)

    def test_invalid_intersection(self):
        reference_mask = self.mask.copy()
        user_mask = self.mask.copy()
        reference_mask[0] = False
        user_mask[1] = False
        self.user[2] = np.nan  # 잘못된 True mask여도 측정 불가 좌표는 제외한다.
        self.reference[3] = np.inf
        values, overall, parts = compare(self.reference, self.user, reference_mask, user_mask)
        self.assertTrue(np.isnan(values[:4]).all())
        self.assertTrue(np.isnan(parts[:2]).all())
        self.assertAlmostEqual(overall, np.mean(self.cosines[4:]))
        _, empty_overall, empty_parts = compare(self.reference, self.user, np.zeros(8, bool), user_mask)
        self.assertTrue(math.isnan(empty_overall))
        self.assertTrue(np.isnan(empty_parts).all())

    def test_threshold_tie_and_no_error(self):
        self.assertEqual(status(.8), 'OK')
        self.assertEqual(status(np.nextafter(.8, 0)), 'ERROR')
        for value in [np.nan, np.inf, -np.inf]:
            self.assertEqual(status(value), 'INVALID')
        self.assertIsNone(priority_error(np.full(8, np.nan)))
        self.assertIsNone(priority_error(np.ones(8)))
        self.assertEqual(priority_error([.7, .7, np.nan, .9, .9, .9, .9, .9]), 0)
        self.assertTrue(math.isnan(valid_mean([np.nan, np.inf])))
        self.assertEqual(valid_mean([1, np.nan, -1]), 0)

    def test_validation_and_clip(self):
        for vectors, mask in [(np.zeros((7, 3)), self.mask), (self.reference, np.ones(7)),
                              (self.reference, np.full(8, 2)), (self.reference * 2, self.mask)]:
            with self.assertRaises(ValueError):
                compare(vectors, self.user, mask, self.mask)
        with self.assertRaises(ValueError):
            priority_error([1, 2])
        vectors = self.reference * (1 + 1e-10)
        values, _, _ = compare(vectors, vectors, self.mask, self.mask)
        np.testing.assert_array_equal(values, np.ones(8))


@unittest.skipUnless(RESEARCH_PATH.is_file(), '로컬 6-2 연구 파일이 없습니다.')
class ResearchParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('danc3d_readonly_research', RESEARCH_PATH)
        cls.research = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.research)
        cls.research.np = np

    def test_world_dir8_compare_and_decisions(self):
        rng = np.random.default_rng(6)
        for index in range(30):
            world = rng.normal(size=(33, 3))
            if index % 3 == 0:
                world[13] = world[11]
            if index % 5 == 0:
                world[24] = np.nan
            points = [SimpleNamespace(x=x, y=y, z=z) for x, y, z in world]
            result = SimpleNamespace(pose_world_landmarks=[points], pose_landmarks=[])
            old, _ = self.research.webcam_pose(result)
            vectors, mask = extract_bone_directions(world)
            np.testing.assert_array_equal(mask, old['mask'][0])
            np.testing.assert_allclose(vectors, old['vectors'][0], rtol=0, atol=0, equal_nan=True)
            user_vectors, user_mask = extract_bone_directions(rng.normal(size=(33, 3)))
            user = dict(vectors=user_vectors[None], mask=user_mask[None])
            expected = self.research.compare(old, 0, 0, user)
            actual = compare(vectors, user_vectors, mask, user_mask)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=0, atol=0, equal_nan=True)
            self.assertEqual(priority_error(actual[0]), self.research.priority_error(expected[0]))
            for value in actual[0]:
                self.assertEqual(status(value), self.research.status(value)[0])

    def test_angles_and_valid_mean(self):
        rng = np.random.default_rng(8)
        for _ in range(30):
            first, second = rng.normal(size=(2, 3))
            self.assertEqual(angle_deg(first, second), self.research.angle_deg(first, second))
        self.assertEqual(angle_diff_deg(179, -179), self.research.angle_diff_deg(179, -179))
        self.assertEqual(valid_mean([np.nan, .9, .7]), self.research.valid_mean([np.nan, .9, .7]))
