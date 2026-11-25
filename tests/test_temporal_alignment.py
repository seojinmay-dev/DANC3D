"""누적 DTW, Recovery 경계 조건 및 실제 6-2 연구 구현과의 동등성을 검증한다."""

import importlib.util
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from app.core import temporal_alignment as alignment


RESEARCH_PATH = Path("/Users/may/Desktop/dance_test/06_2_realtime_webcam_feedback.py")
REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "references"


def make_reference(count=12):
    """순서를 구별할 수 있는 단위 DIR8와 0.125초 간격 시간축을 만든다."""
    angles = np.arange(count, dtype=np.float64) * 0.5
    directions = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(count)))
    vectors = np.repeat(directions[:, None, :], 8, axis=1)
    return vectors, np.ones((count, 8), dtype=bool), np.arange(count) * 0.125


def recovery_trace():
    """빠르게 진행한 뒤 이전 자세로 돌아와 두 차례 Recovery를 유발한다."""
    return (
        [(row, row / 32) for row in range(5)]
        + [(1, time) for time in (0.125, 0.15625, 0.1875)]
        + [(row, 0.21875 + (row - 2) / 32) for row in range(2, 9)]
        + [(4, time) for time in (0.5, 0.53125, 0.5625, 0.625, 0.6875)]
    )


class InputValidationTests(unittest.TestCase):
    def setUp(self):
        self.vectors, self.mask, self.times = make_reference(4)

    def test_reference_shapes(self):
        invalid_inputs = [
            (self.vectors[0], self.mask, self.times),
            (self.vectors[:, :7], self.mask, self.times),
            (self.vectors, self.mask[:, :7], self.times),
            (self.vectors, self.mask, self.times[:-1]),
            (self.vectors, self.mask, self.times[:, None]),
            (self.vectors[:0], self.mask[:0], self.times[:0]),
        ]
        for cls in (alignment.WindowedOnlineDTW, alignment.LocalRecoveryDTW):
            for values in invalid_inputs:
                with self.subTest(cls=cls.__name__, shapes=[np.shape(v) for v in values]):
                    with self.assertRaises(ValueError):
                        cls(*values)

    def test_reference_values(self):
        for times in ([0, 0.1, 0.1, 0.3], [0, 0.2, 0.1, 0.3],
                      [0, 0.1, np.nan, 0.3], [0, 0.1, np.inf, 0.3],
                      [-1, 0, 1, 2], [0j, 1j, 2j, 3j]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                alignment.WindowedOnlineDTW(self.vectors, self.mask, times)
        for vectors in (self.vectors * 2, self.vectors.astype(complex),
                        np.full_like(self.vectors, np.nan), np.full_like(self.vectors, np.inf)):
            with self.assertRaises(ValueError):
                alignment.WindowedOnlineDTW(vectors, self.mask, self.times)
        for mask in (np.full((4, 8), 2), np.full((4, 8), np.nan), np.full((4, 8), "True")):
            with self.assertRaises(ValueError):
                alignment.WindowedOnlineDTW(self.vectors, mask, self.times)

    def test_single_frame_and_reference_ownership(self):
        dtw = alignment.WindowedOnlineDTW(self.vectors[:1], self.mask[:1].astype(int), self.times[:1])
        self.vectors[:] = 0
        self.mask[:] = False
        self.times[:] = 10
        self.assertEqual(dtw.update(dtw.reference[0], dtw.valid[0], 0), 0)
        self.assertEqual(dtw.reference.dtype, np.float64)
        self.assertEqual(dtw.valid.dtype, np.bool_)
        self.assertEqual(dtw.times[0], 0)
        self.assertEqual(dtw.previous_length.dtype, np.int64)

    def test_invalid_updates_leave_all_state_unchanged(self):
        obj = alignment.LocalRecoveryDTW(self.vectors, self.mask, self.times)
        obj.update(self.vectors[0], self.mask[0], 0)
        cost = obj.dtw.previous_cost.copy()
        length = obj.dtw.previous_length.copy()
        metrics = obj.metrics
        invalid_inputs = [
            (self.vectors[0, :7], self.mask[0], 0),
            (self.vectors[0], self.mask[0, :7], 0),
            (self.vectors[0] * 2, self.mask[0], 0),
            (np.full((8, 3), np.nan), self.mask[0], 0),
            (self.vectors[0], np.full(8, 2), 0),
        ] + [(self.vectors[0], self.mask[0], time)
             for time in (-1, np.nan, np.inf, [0], "bad", 1j)]
        for values in invalid_inputs:
            with self.assertRaises(ValueError):
                obj.update(*values)
            np.testing.assert_array_equal(obj.dtw.previous_cost, cost)
            np.testing.assert_array_equal(obj.dtw.previous_length, length)
            self.assertEqual(obj.dtw.predicted_index, 0)
            self.assertIs(obj.metrics, metrics)
            self.assertEqual((obj.counter, obj.last_trigger, obj.trigger_count), (0, -math.inf, 0))

    def test_milliseconds_origin_conversion(self):
        timestamps = np.array([12000, 12033, 12067], dtype=np.uint64)
        saved = timestamps.copy()
        times = alignment.reference_times_from_timestamps(timestamps)
        np.testing.assert_allclose(times, [0, 0.033, 0.067], rtol=0, atol=0)
        np.testing.assert_array_equal(timestamps, saved)
        self.assertEqual(times.dtype, np.float64)
        for values in ([], [1, 1], [2, 1], [np.nan], [[1]], [-1, 0]):
            with self.assertRaises(ValueError):
                alignment.reference_times_from_timestamps(values)


class WindowedOnlineDTWTests(unittest.TestCase):
    def setUp(self):
        self.vectors, self.mask, self.times = make_reference()
        self.dtw = alignment.WindowedOnlineDTW(self.vectors, self.mask, self.times)

    def test_identical_sequence(self):
        endpoints = [self.dtw.update(vectors, mask, time)
                     for vectors, mask, time in zip(self.vectors, self.mask, self.times)]
        self.assertEqual(endpoints, list(range(len(self.times))))
        self.assertTrue(all(isinstance(index, int) for index in endpoints))

    def test_slower_and_faster_sequence(self):
        for rows in ([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 2, 4, 6, 8]):
            with self.subTest(rows=rows):
                dtw = alignment.WindowedOnlineDTW(self.vectors, self.mask, self.times)
                endpoints = [dtw.update(self.vectors[row], self.mask[row], step * 0.1)
                             for step, row in enumerate(rows)]
                self.assertEqual(endpoints, rows)

    def test_inclusive_warping_window(self):
        vectors = np.tile([1., 0., 0.], (13, 8, 1))
        times = np.arange(13) * 0.25
        dtw = alignment.WindowedOnlineDTW(vectors, np.ones((13, 8), bool), times)
        self.assertEqual(dtw.update(vectors[0], self.mask[0], 1.0), 0)
        # playback=1의 양 끝 0초와 2초를 포함하고 2.25초는 제외한다.
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(dtw.previous_cost)), np.arange(1, 10))
        self.assertEqual(dtw.update(vectors[0], self.mask[0], 1.25), 1)
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(dtw.previous_cost)), np.arange(2, 11))

    def test_normalized_cumulative_cost_is_not_nearest_neighbor(self):
        dtw = alignment.WindowedOnlineDTW(self.vectors[:5], self.mask[:5], self.times[:5])
        endpoint = dtw.update(self.vectors[2], self.mask[2], 0.25)
        self.assertEqual(endpoint, 3)  # 가장 닮은 단일 row 2와 달라야 한다.
        self.assertEqual(int(np.argmin(dtw.previous_cost[1:])), 0)
        np.testing.assert_array_equal(dtw.previous_length, [0, 1, 2, 3, 4, 5])
        normalized = dtw.previous_cost[1:] / dtw.previous_length[1:]
        self.assertEqual(int(np.argmin(normalized)), endpoint)

    def test_non_decreasing_endpoint(self):
        rows = [0, 1, 2, 3, 4, 1, 0, 2, 3]
        endpoints = [self.dtw.update(self.vectors[row], self.mask[row], step / 32)
                     for step, row in enumerate(rows)]
        self.assertTrue((np.diff(endpoints) >= 0).all())
        self.assertEqual(endpoints[-1], 4)

    def test_diagonal_ties_and_state_accumulation(self):
        vectors = np.tile([1., 0., 0.], (4, 8, 1))
        dtw = alignment.WindowedOnlineDTW(vectors, self.mask[:4], self.times[:4])
        self.assertEqual(dtw.update(vectors[0], self.mask[0], 0), 0)
        np.testing.assert_array_equal(dtw.previous_length, [0, 1, 2, 3, 4])
        self.assertEqual(dtw.update(vectors[0], self.mask[0], 0), 0)
        np.testing.assert_array_equal(dtw.previous_length, [0, 2, 2, 3, 4])
        np.testing.assert_array_equal(dtw.previous_cost, [np.inf, 0, 0, 0, 0])

    def test_invalid_and_unreachable_updates_preserve_state(self):
        self.assertIsNone(self.dtw.update(np.full((8, 3), np.nan), np.zeros(8, bool), 0))
        self.assertEqual(self.dtw.previous_cost[0], 0)
        self.assertEqual(self.dtw.predicted_index, -1)
        self.dtw.update(self.vectors[0], self.mask[0], 0)
        cost, length = self.dtw.previous_cost.copy(), self.dtw.previous_length.copy()
        for user, mask, time in [(np.full((8, 3), np.nan), np.zeros(8, bool), 0.1),
                                 (self.vectors[0], self.mask[0], 100)]:
            self.assertIsNone(self.dtw.update(user, mask, time))
            np.testing.assert_array_equal(self.dtw.previous_cost, cost)
            np.testing.assert_array_equal(self.dtw.previous_length, length)
            self.assertEqual(self.dtw.predicted_index, 0)
        self.assertEqual(self.dtw.update(self.vectors[1], self.mask[1], 0.125), 1)

    def test_all_invalid_reference_and_disconnected_window(self):
        invalid = alignment.WindowedOnlineDTW(np.full_like(self.vectors, np.nan),
                                              np.zeros_like(self.mask), self.times)
        self.assertIsNone(invalid.update(self.vectors[0], self.mask[0], 0))
        # window에 후보가 있어도 시작 경계와 연결된 누적 경로가 없으면 None이다.
        self.assertIsNone(self.dtw.update(self.vectors[-1], self.mask[-1], 1.375))
        self.assertEqual(self.dtw.previous_cost[0], 0)
        self.assertEqual(self.dtw.predicted_index, -1)


class RecoverySimilaritiesTests(unittest.TestCase):
    def test_mask_intersection_and_empty_candidates(self):
        vectors = np.tile([1., 0., 0.], (3, 8, 1))
        reference_mask = np.zeros((3, 8), bool)
        reference_mask[0, :2] = True
        reference_mask[1, 1:3] = True
        vectors[0, 1] = [-1, 0, 0]
        vectors[~reference_mask] = np.nan
        user = np.tile([1., 0., 0.], (8, 1))
        user_mask = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        scores = alignment.recovery_similarities(vectors, reference_mask, user, user_mask)
        np.testing.assert_allclose(scores, [0, 1, np.nan], equal_nan=True)
        self.assertEqual(scores.dtype, np.float64)
        empty = alignment.recovery_similarities(vectors[:0], reference_mask[:0], user, user_mask)
        self.assertEqual(empty.shape, (0,))

    def test_cosine_clipping(self):
        vectors = np.tile([1. + 1e-10, 0., 0.], (2, 8, 1))
        vectors[1] *= -1
        scores = alignment.recovery_similarities(vectors, np.ones((2, 8), bool),
                                                 vectors[0], np.ones(8, bool))
        np.testing.assert_array_equal(scores, [1, -1])


class LocalRecoveryDTWTests(unittest.TestCase):
    def setUp(self):
        self.vectors, self.mask, self.times = make_reference()
        self.obj = alignment.LocalRecoveryDTW(self.vectors, self.mask, self.times)

    def advance(self, trace):
        for row, time in trace:
            self.obj.update(self.vectors[row], self.mask[row], time)

    def test_initial_and_invalid_metrics(self):
        expected_keys = {
            "best_local_reference_index", "best_local_reference_time_sec", "best_local_similarity",
            "current_dtw_similarity", "pre_recovery_aligned_index", "pre_recovery_aligned_time_sec",
            "recovery_similarity_gain", "recovery_candidate", "recovery_counter", "recovery_triggered",
            "recovery_trigger_count", "recovery_anchor_index", "recovery_anchor_time_sec",
        }
        self.assertEqual(set(self.obj.metrics), expected_keys)
        self.assertIsNone(self.obj.update(np.full((8, 3), np.nan), np.zeros(8, bool), 0))
        for key, value in self.obj.metrics.items():
            if key.endswith("_index"):
                self.assertEqual(value, -1)
            elif key in {"recovery_candidate", "recovery_counter", "recovery_triggered", "recovery_trigger_count"}:
                self.assertEqual(value, 0)
            else:
                self.assertTrue(math.isnan(value), key)

    def test_three_consecutive_candidates_reset_entire_anchor_state(self):
        self.advance(recovery_trace()[:5])
        for count, time in enumerate((0.125, 0.15625, 0.1875), 1):
            endpoint = self.obj.update(self.vectors[1], self.mask[1], time)
            metrics = self.obj.metrics
            self.assertEqual(metrics["recovery_candidate"], 1)
            self.assertEqual(metrics["recovery_counter"], count)
            self.assertEqual(metrics["pre_recovery_aligned_index"], 4)
            self.assertEqual(metrics["recovery_triggered"], int(count == 3))
            self.assertEqual(endpoint, 1 if count == 3 else 4)
        self.assertEqual(self.obj.counter, 0)
        self.assertEqual(self.obj.last_trigger, 0.1875)
        self.assertEqual(self.obj.trigger_count, 1)
        self.assertEqual(metrics["recovery_anchor_index"], 1)
        self.assertEqual(metrics["recovery_anchor_time_sec"], 0.125)
        self.assertEqual(metrics["best_local_reference_index"], 1)
        self.assertEqual(self.obj.dtw.predicted_index, 1)
        np.testing.assert_array_equal(np.flatnonzero(np.isfinite(self.obj.dtw.previous_cost)), [2])
        self.assertEqual(self.obj.dtw.previous_cost[2], 1 - metrics["best_local_similarity"])
        expected_length = np.zeros(len(self.times) + 1, dtype=np.int64)
        expected_length[2] = 1
        np.testing.assert_array_equal(self.obj.dtw.previous_length, expected_length)
        # 새 anchor로부터 다음 update가 실제 누적 경로를 이어가는지 확인한다.
        self.assertEqual(self.obj.update(self.vectors[2], self.mask[2], 0.21875), 2)
        self.assertEqual(self.obj.dtw.previous_length[3], 2)
        self.assertEqual(self.obj.metrics["recovery_anchor_index"], -1)
        self.assertTrue(math.isnan(self.obj.metrics["recovery_anchor_time_sec"]))
        self.assertEqual(self.obj.metrics["recovery_trigger_count"], 1)

    def test_half_second_cooldown_preserves_evidence(self):
        self.advance(recovery_trace()[:15])
        for count, time in enumerate((0.5, 0.53125, 0.5625, 0.625), 1):
            self.assertEqual(self.obj.update(self.vectors[4], self.mask[4], time), 8)
            self.assertEqual(self.obj.metrics["recovery_counter"], count)
            self.assertEqual(self.obj.metrics["recovery_triggered"], 0)
            self.assertEqual(self.obj.trigger_count, 1)
        self.assertEqual(self.obj.update(self.vectors[4], self.mask[4], 0.6875), 4)
        self.assertEqual(self.obj.metrics["recovery_counter"], 5)
        self.assertEqual(self.obj.metrics["recovery_triggered"], 1)
        self.assertEqual(self.obj.trigger_count, 2)
        self.assertEqual(self.obj.last_trigger, 0.6875)
        self.assertEqual(self.obj.metrics["recovery_anchor_index"], 4)

    def test_interrupted_evidence_restarts_counter(self):
        self.advance(recovery_trace()[:7])
        self.assertEqual(self.obj.counter, 2)
        self.obj.update(np.full((8, 3), np.nan), np.zeros(8, bool), 0.16)
        self.assertEqual(self.obj.counter, 0)
        for count in (1, 2):
            self.obj.update(self.vectors[1], self.mask[1], 0.1875)
            self.assertEqual(self.obj.metrics["recovery_counter"], count)
            self.assertEqual(self.obj.trigger_count, 0)
        self.obj.update(self.vectors[1], self.mask[1], 0.1875)
        self.assertEqual(self.obj.trigger_count, 1)

    def test_no_unnecessary_recovery(self):
        for row, time in enumerate(self.times):
            self.assertEqual(self.obj.update(self.vectors[row], self.mask[row], time), row)
            self.assertEqual(self.obj.metrics["recovery_candidate"], 0)
            self.assertEqual(self.obj.metrics["recovery_triggered"], 0)
            self.assertEqual(self.obj.metrics["recovery_similarity_gain"], 0)
        self.assertEqual(self.obj.trigger_count, 0)

    def test_candidate_threshold_boundaries(self):
        # 후보 판정만 분리하기 위해 endpoint를 고정하고 cosine은 실제 계산한다.
        cases = [
            (0.95, 0.5, 0, 1),
            (np.nextafter(0.95, 0), 0.5, 0, 0),
            (1., 0.95, 0, 1),
            (1., np.nextafter(0.95, 1), 0, 0),
            (1., 0.5, 0.20, 1),
            (1., 0.5, np.nextafter(0.20, 1), 0),
        ]
        user = np.tile([1., 0., 0.], (8, 1))
        for local, current, playback, expected in cases:
            directions = np.array([[local, np.sqrt(1 - local ** 2), 0],
                                   [current, np.sqrt(1 - current ** 2), 0]])
            vectors = np.repeat(directions[:, None, :], 8, axis=1)
            obj = alignment.LocalRecoveryDTW(vectors, self.mask[:2], [0., 0.125])
            with self.subTest(local=local, current=current, playback=playback):
                with patch.object(obj.dtw, "update", return_value=1):
                    obj.update(user, self.mask[0], playback)
                self.assertEqual(obj.metrics["recovery_candidate"], expected)

    def test_no_recovery_without_valid_dtw_endpoint(self):
        # 시작 경계에 연결되지 않은 window에도 local 후보는 있지만 gain은 NaN이다.
        self.assertIsNone(self.obj.update(self.vectors[-1], self.mask[-1], 1.375))
        self.assertEqual(self.obj.metrics["best_local_reference_index"], 11)
        self.assertEqual(self.obj.metrics["pre_recovery_aligned_index"], -1)
        self.assertTrue(math.isnan(self.obj.metrics["recovery_similarity_gain"]))
        self.assertEqual(self.obj.metrics["recovery_candidate"], 0)
        self.assertEqual(self.obj.trigger_count, 0)

    def test_empty_window_and_first_local_tie(self):
        vectors = np.tile([1., 0., 0.], (3, 8, 1))
        obj = alignment.LocalRecoveryDTW(vectors, self.mask[:3], [0., 0.5, 1.])
        obj.update(vectors[0], self.mask[0], 0.5)
        self.assertEqual(obj.metrics["best_local_reference_index"], 0)
        self.assertIsNone(obj.update(vectors[0], self.mask[0], 3.))
        self.assertEqual(obj.metrics["best_local_reference_index"], -1)
        self.assertEqual(obj.metrics["recovery_counter"], 0)


@unittest.skipUnless(RESEARCH_PATH.is_file(), "로컬 6-2 연구 파일이 없습니다.")
class ResearchParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("danc3d_temporal_research", RESEARCH_PATH)
        cls.research = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.research)
        cls.research.np = np  # 원본의 지연 import 전역만 주입하며 알고리즘은 변경하지 않는다.

    def assert_state_equal(self, actual, expected):
        np.testing.assert_allclose(actual.previous_cost, expected.previous_cost, rtol=0, atol=0)
        np.testing.assert_array_equal(actual.previous_length, expected.previous_length)
        self.assertEqual(actual.predicted_index, expected.predicted_index)

    def assert_metrics_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in expected:
            with self.subTest(metric=key):
                np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)

    def compare_trace(self, vectors, mask, times, observations, with_recovery):
        cls_name = "LocalRecoveryDTW" if with_recovery else "WindowedOnlineDTW"
        actual = getattr(alignment, cls_name)(vectors, mask, times)
        expected = getattr(self.research, cls_name)(vectors, mask, times)
        triggers = []
        for step, (user, user_mask, playback) in enumerate(observations):
            with self.subTest(cls=cls_name, step=step):
                actual_index = actual.update(user, user_mask, playback)
                self.assertEqual(actual_index, expected.update(user, user_mask, playback))
                if with_recovery:
                    self.assert_state_equal(actual.dtw, expected.dtw)
                    self.assert_metrics_equal(actual.metrics, expected.metrics)
                    self.assertEqual(actual.counter, expected.counter)
                    self.assertEqual(actual.last_trigger, expected.last_trigger)
                    self.assertEqual(actual.trigger_count, expected.trigger_count)
                    if actual.metrics["recovery_triggered"]:
                        triggers.append((step, playback, actual.metrics["recovery_anchor_index"]))
                else:
                    self.assert_state_equal(actual, expected)
        return triggers

    def test_constants_match_research(self):
        for name in ("WARPING_WINDOW_SEC", "RECOVERY_LOCAL_SIM_THRESHOLD", "RECOVERY_SIM_GAIN_THRESHOLD",
                     "RECOVERY_PLAYBACK_TOLERANCE_SEC", "RECOVERY_CONSECUTIVE_FRAMES", "RECOVERY_COOLDOWN_SEC"):
            self.assertEqual(getattr(alignment, name), getattr(self.research, name))

    def test_synthetic_speed_and_recovery_trace(self):
        vectors, mask, times = make_reference()
        observations = [(vectors[row], mask[row], playback) for row, playback in recovery_trace()]
        self.compare_trace(vectors, mask, times, observations, with_recovery=False)
        triggers = self.compare_trace(vectors, mask, times, observations, with_recovery=True)
        self.assertEqual(triggers, [(7, 0.1875, 1), (19, 0.6875, 4)])
        for rows in ([0, 0, 1, 1, 2, 2, 3, 3], [0, 2, 4, 6, 8]):
            observations = [(vectors[row], mask[row], step * 0.1) for step, row in enumerate(rows)]
            for with_recovery in (False, True):
                self.compare_trace(vectors, mask, times, observations, with_recovery)

    def test_random_masks_invalid_frames_and_empty_windows(self):
        rng = np.random.default_rng(62)
        vectors = rng.normal(size=(32, 8, 3))
        vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
        mask = rng.random((32, 8)) > 0.2
        vectors[~mask] = np.nan
        times = np.arange(32) / 16
        observations = []
        for step in range(64):
            row = min(step // 2, 31)
            user = vectors[row].copy()
            user_mask = mask[row].copy()
            if step % 9 == 8:
                user[:] = np.nan
                user_mask[:] = False
            observations.append((user, user_mask, step / 32))
        observations += [(vectors[0], mask[0], 100), (vectors[0], mask[0], 0)]
        for with_recovery in (False, True):
            self.compare_trace(vectors, mask, times, observations, with_recovery)

    def test_tied_cost_paths_and_local_candidates(self):
        vectors = np.tile([1., 0., 0.], (10, 8, 1))
        mask = np.ones((10, 8), bool)
        times = np.arange(10) / 4
        observations = [(vectors[0], mask[0], time) for time in (0, 0, 0.25, 0.5, 1, 1.25, 0)]
        for with_recovery in (False, True):
            self.compare_trace(vectors, mask, times, observations, with_recovery)

    def test_existing_npz_prefixes(self):
        paths = sorted(REFERENCE_DIR.glob("*.npz"))
        if not paths:
            self.skipTest("로컬 Reference NPZ가 없습니다.")
        for path in paths:
            with self.subTest(reference=path.name), np.load(path, allow_pickle=False) as data:
                vectors = data["normalized_bone_vectors"].astype(np.float64)
                mask = data["bone_valid_mask"].astype(bool)
                times = alignment.reference_times_from_timestamps(data["timestamps_ms"])
                observations = [(vectors[row], mask[row], times[row]) for row in range(min(64, len(times)))]
                for with_recovery in (False, True):
                    self.compare_trace(vectors, mask, times, observations, with_recovery)


if __name__ == "__main__":
    unittest.main()
