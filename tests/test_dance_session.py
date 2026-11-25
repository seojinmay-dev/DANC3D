"""세션 수명주기, 로컬 Reference 검증 및 실제 6-2 평가 파이프라인 동등성 테스트."""

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from app.core.pose_features import BONES, PARTS, extract_bone_directions
from app.core.pose_similarity import compare
from app.core.score import final_score_metrics
from app.services.dance_session import DanceSession


PROJECT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESEARCH_PATH = Path("/Users/may/Desktop/dance_test/06_2_realtime_webcam_feedback.py")
SYNTHETIC_SONG_ID = "synthetic_session_F"


def make_world(yaw=0):
    """곧게 편 수평 양팔·수직 양다리를 원본 좌표에서 yaw만 회전한 합성 Pose다."""
    world = np.zeros((33, 3), dtype=np.float64)
    for left, right, x, y in ((11, 12, .5, 1), (13, 14, 1.5, 1), (15, 16, 2.5, 1),
                              (23, 24, .5, 0), (25, 26, .5, -1), (27, 28, .5, -2)):
        world[left], world[right] = [x, y, 0], [-x, y, 0]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, 0, -sine], [0, 1, 0], [sine, 0, cosine]])
    return world @ rotation.T


def write_synthetic_reference(data_dir, worlds=None):
    """임시 디렉터리에만 생성하는 catalog/metadata/NPZ fixture다."""
    directory = data_dir / "references"
    directory.mkdir(exist_ok=True)
    worlds = np.array([make_world(row * .5) for row in range(12)]) if worlds is None else np.asarray(worlds)
    features = [extract_bone_directions(world) for world in worlds]
    arrays = dict(frame_indices=60 + np.arange(len(worlds)) * 2,
                  timestamps_ms=12000 + np.arange(len(worlds)) * 125,
                  raw_world=worlds,
                  normalized_bone_vectors=np.stack([vectors for vectors, _ in features]),
                  bone_valid_mask=np.stack([mask for _, mask in features]))
    np.savez(directory / (SYNTHETIC_SONG_ID + ".npz"), **arrays)
    metadata = dict(sequence=SYNTHETIC_SONG_ID,
                    bone_mapping={name: dict(index=index, parent=parent, child=child)
                                  for index, (name, parent, child) in enumerate(BONES)})
    (directory / (SYNTHETIC_SONG_ID + ".json")).write_text(json.dumps(metadata), encoding="utf-8")
    song = dict(id=SYNTHETIC_SONG_ID, title="Synthetic test", artist="Test fixture",
                reference_npz="references/" + SYNTHETIC_SONG_ID + ".npz",
                reference_metadata="references/" + SYNTHETIC_SONG_ID + ".json")
    (data_dir / "songs.json").write_text(json.dumps([song]), encoding="utf-8")
    return arrays


def recovery_observations():
    """일반 진행, 뒤로 복구, cooldown 중 연속 후보와 두 번째 복구를 포함한다."""
    return ([(row, row / 32) for row in range(5)]
            + [(1, time) for time in (.125, .15625, .1875)]
            + [(row, .21875 + (row - 2) / 32) for row in range(2, 9)]
            + [(4, time) for time in (.5, .53125, .5625, .625, .6875)])


class SessionFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="danc3d_session_test_")
        self.addCleanup(self.temporary.cleanup)
        self.data_dir = Path(self.temporary.name)
        self.arrays = write_synthetic_reference(self.data_dir)

    def session(self):
        return DanceSession(SYNTHETIC_SONG_ID, data_dir=self.data_dir)

    def replace_array(self, name, value):
        arrays = dict(self.arrays)
        if value is None:
            arrays.pop(name)
        else:
            arrays[name] = value
        np.savez(self.data_dir / "references" / (SYNTHETIC_SONG_ID + ".npz"), **arrays)

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                with self.subTest(field=key):
                    self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (float, np.floating, np.ndarray)):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)
        else:
            self.assertEqual(actual, expected)


class SyntheticSessionTests(SessionFixture):
    def test_song_loading_time_origin_and_frame_row_distinction(self):
        session = self.session()
        self.assertEqual(session.song_id, SYNTHETIC_SONG_ID)
        self.assertEqual(session.reference.times[0], 0)
        self.assertEqual(session.reference.timestamps_ms[0], 12000)
        np.testing.assert_array_equal(session.reference.times, np.arange(12) * .125)
        result = session.update(0, self.arrays["raw_world"][0])
        self.assertEqual(result["aligned_reference_index"], 0)
        self.assertEqual(result["aligned_reference_frame_index"], 60)
        self.assertEqual(result["aligned_reference_time_sec"], 0)
        self.assertEqual(result["raw_final_score"], 1)
        self.assertFalse(session.reference.raw_world.flags.writeable)

    def test_unknown_and_duplicate_song_id(self):
        with self.assertRaisesRegex(ValueError, "Unknown song_id"):
            DanceSession("missing", data_dir=self.data_dir)
        catalog_path = self.data_dir / "songs.json"
        song = json.loads(catalog_path.read_text())[0]
        catalog_path.write_text(json.dumps([song, song]))
        with self.assertRaisesRegex(ValueError, "Duplicate song_id"):
            self.session()

    def test_missing_paths_and_arrays(self):
        for name in self.arrays:
            with self.subTest(array=name):
                self.replace_array(name, None)
                with self.assertRaisesRegex(ValueError, "Missing NPZ arrays"):
                    self.session()
        write_synthetic_reference(self.data_dir)
        (self.data_dir / "references" / (SYNTHETIC_SONG_ID + ".npz")).unlink()
        with self.assertRaises(FileNotFoundError):
            self.session()

    def test_missing_metadata_and_catalog_path_validation(self):
        metadata_path = self.data_dir / "references" / (SYNTHETIC_SONG_ID + ".json")
        metadata_path.unlink()
        with self.assertRaises(FileNotFoundError):
            self.session()
        write_synthetic_reference(self.data_dir)
        catalog_path = self.data_dir / "songs.json"
        song = json.loads(catalog_path.read_text())[0]
        song["reference_npz"] = "../outside.npz"
        catalog_path.write_text(json.dumps([song]))
        with self.assertRaisesRegex(ValueError, "inside data_dir"):
            self.session()
        catalog_path.write_text("{broken json")
        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            self.session()

    def test_npz_shape_validation(self):
        for name, invalid in (("raw_world", self.arrays["raw_world"][:, :32]),
                              ("normalized_bone_vectors", self.arrays["normalized_bone_vectors"][:, :7]),
                              ("bone_valid_mask", self.arrays["bone_valid_mask"][:, :7]),
                              ("timestamps_ms", self.arrays["timestamps_ms"][:-1])):
            with self.subTest(array=name):
                self.replace_array(name, invalid)
                with self.assertRaises(ValueError):
                    self.session()

    def test_npz_time_axis_frame_indices_and_mask_validation(self):
        cases = [("timestamps_ms", np.zeros(12)), ("timestamps_ms", np.arange(12)[::-1]),
                 ("timestamps_ms", np.full(12, np.nan)), ("frame_indices", np.arange(12, dtype=float)),
                 ("frame_indices", np.zeros(12, int)), ("frame_indices", np.arange(-1, 11)),
                 ("bone_valid_mask", np.full((12, 8), 2))]
        for name, invalid in cases:
            with self.subTest(array=name):
                self.replace_array(name, invalid)
                with self.assertRaises(ValueError):
                    self.session()

    def test_dir8_metadata_order_and_actual_vector_order(self):
        path = self.data_dir / "references" / (SYNTHETIC_SONG_ID + ".json")
        metadata = json.loads(path.read_text())
        metadata["bone_mapping"]["left_upper_arm"]["index"] = 2
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "DIR8 order"):
            self.session()
        write_synthetic_reference(self.data_dir)
        swapped = self.arrays["normalized_bone_vectors"].copy()
        swapped[:, [0, 2]] = swapped[:, [2, 0]]
        self.replace_array("normalized_bone_vectors", swapped)
        with self.assertRaisesRegex(ValueError, "DIR8 order/directions"):
            self.session()
        self.replace_array("bone_valid_mask", np.zeros((12, 8), bool))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.session()

    def test_reference_is_not_reloaded_on_update_or_reset(self):
        with patch("app.services.dance_session.np.load", wraps=np.load) as loader:
            session = self.session()
            for row in range(3):
                session.update(row * .125, self.arrays["raw_world"][row])
            reference = session.reference
            session.reset()
            session.update(0, self.arrays["raw_world"][0])
            session.end()
            self.assertEqual(loader.call_count, 1)
            self.assertFalse(loader.call_args.kwargs["allow_pickle"])
            self.assertIs(session.reference, reference)

    def test_normal_world_dir8_and_result_contract(self):
        session = self.session()
        world = self.arrays["raw_world"][0].tolist()
        result = session.update(playback_time=0, world_landmarks=world)
        required = {"song_id", "playback_time_sec", "aligned_reference_index", "aligned_reference_time_sec",
                    "timing_offset_sec", "pose_detected", "valid_user_bone_count", "bone_similarities",
                    "overall_similarity", "part_similarities", "bone_status", "priority_error_bone",
                    "error_bone_count", "priority_pose_feedback", "timing_feedback", "raw_final_feedback",
                    "displayed_feedback", "torso_yaw_error_deg", "pose_score", "score_timing_factor",
                    "raw_final_score", "displayed_score", "recovery_triggered", "recovery_trigger_count",
                    "recovery_anchor_index"}
        self.assertTrue(required.issubset(result))
        expected_vectors, expected_mask = extract_bone_directions(world)
        np.testing.assert_array_equal(result["user_dir8"], expected_vectors)
        np.testing.assert_array_equal(result["user_bone_valid_mask"], expected_mask)
        self.assertEqual(result["valid_user_bone_count"], 8)
        self.assertTrue(result["pose_detected"])
        self.assertEqual(list(result["bone_similarities"]), [name for name, _, _ in BONES])
        self.assertEqual(list(result["part_similarities"]), list(PARTS))
        self.assertEqual(set(result["bone_status"].values()), {"OK"})

    def test_invalid_landmark_shape_and_playback_are_atomic(self):
        session = self.session()
        session.update(0, self.arrays["raw_world"][0])
        saved = session.last_result
        cost = session.alignment.dtw.previous_cost.copy()
        for value in (np.zeros((32, 3)), np.zeros((33, 2)), [["bad"]], np.zeros((33, 3), complex)):
            with self.assertRaises(ValueError):
                session.update(.1, value)
        for time in (None, -1, np.nan, np.inf, [0], 1j):
            with self.assertRaises(ValueError):
                session.update(time, self.arrays["raw_world"][0])
        self.assertEqual(session.evaluation_sample_count, 1)
        self.assert_nested_equal(session.last_result, saved)
        np.testing.assert_array_equal(session.alignment.dtw.previous_cost, cost)

    def test_dtw_and_stabilizer_instances_persist_between_frames(self):
        session = self.session()
        objects = (session.alignment, session.score_stabilizer, session.feedback_stabilizer)
        for row in range(4):
            result = session.update(row * .125, self.arrays["raw_world"][row])
            self.assertEqual(result["aligned_reference_index"], row)
            self.assertGreaterEqual(session.alignment.dtw.previous_length[row + 1], row + 1)
            self.assertEqual(objects, (session.alignment, session.score_stabilizer, session.feedback_stabilizer))

    def test_aligned_reference_is_distinct_from_content_playback(self):
        session = self.session()
        session.update(0, self.arrays["raw_world"][0])
        result = session.update(.6, self.arrays["raw_world"][1])
        self.assertEqual(result["aligned_reference_index"], 1)
        self.assertEqual(result["aligned_reference_time_sec"], .125)
        self.assertEqual(result["playback_time_sec"], .6)
        self.assertAlmostEqual(result["overall_similarity"], 1)
        self.assertEqual(result["priority_pose_feedback"], "NONE")
        self.assertEqual(result["timing_feedback"], "LATE")
        self.assertEqual(result["raw_final_feedback"], "동작이 조금 늦습니다.")
        self.assertEqual(result["timing_offset_sec"], .125 - .6)

    def test_pose_priority_and_raw_display_separation(self):
        session = self.session()
        world = self.arrays["raw_world"][0].copy()
        world[15] = world[13] + [0, 1, 0]
        with patch.object(session.alignment, "update", return_value=0):
            session.update(0, self.arrays["raw_world"][0])
            result = session.update(.6, world)
        self.assertEqual(result["priority_pose_feedback"], "왼쪽 팔꿈치를 더 펴세요")
        self.assertEqual(result["raw_final_feedback"], result["priority_pose_feedback"])
        self.assertEqual(result["timing_feedback"], "NONE")
        self.assertGreater(result["displayed_score"], result["display_score"])
        user_vectors, user_mask = extract_bone_directions(world)
        values, overall, parts = compare(session.reference.normalized_bone_vectors[0], user_vectors,
                                         session.reference.bone_valid_mask[0], user_mask)
        expected = final_score_metrics(overall, parts, -.6)
        self.assertEqual(result["raw_final_score"], expected["final_score"])
        self.assertEqual(result["priority_error_bone"], "left_forearm")
        self.assertEqual(result["error_bone_count"], 1)
        self.assertEqual(result["bone_status"]["left_forearm"], "ERROR")

    def test_recovery_trace_returns_anchor_and_count(self):
        session = self.session()
        triggers = []
        for step, (row, time) in enumerate(recovery_observations()):
            result = session.update(time, self.arrays["raw_world"][row])
            if result["recovery_triggered"]:
                triggers.append((step, time, result["recovery_anchor_index"], result["recovery_trigger_count"]))
        self.assertEqual(triggers, [(7, .1875, 1, 1), (19, .6875, 4, 2)])

    def test_missing_pose_skips_dtw_discards_stale_alignment_and_preserves_ema(self):
        session = self.session()
        valid = session.update(0, self.arrays["raw_world"][0])
        cost, length = session.alignment.dtw.previous_cost.copy(), session.alignment.dtw.previous_length.copy()
        for missing in (None, np.full((33, 3), np.nan), np.zeros((33, 3))):
            with patch.object(session.alignment, "update", wraps=session.alignment.update) as updater:
                result = session.update(.125, missing)
                updater.assert_not_called()
            self.assertFalse(result["pose_detected"])
            self.assertEqual(result["aligned_reference_index"], -1)
            self.assertIsNone(session.last_aligned_index)
            self.assertTrue(math.isnan(result["overall_similarity"]))
            self.assertTrue(math.isnan(result["raw_final_score"]))
            self.assertEqual(result["displayed_score"], valid["displayed_score"])
            self.assertFalse(result["sample_recorded"])
            np.testing.assert_array_equal(session.alignment.dtw.previous_cost, cost)
            np.testing.assert_array_equal(session.alignment.dtw.previous_length, length)
        self.assertEqual(session.valid_sample_count, 1)

    def test_missing_pose_breaks_only_recovery_evidence(self):
        session = self.session()
        for row, time in recovery_observations()[:7]:
            session.update(time, self.arrays["raw_world"][row])
        self.assertEqual(session.alignment.counter, 2)
        cost = session.alignment.dtw.previous_cost.copy()
        result = session.update(.171875, None)
        self.assertEqual(session.alignment.counter, 0)
        self.assertTrue(math.isnan(result["recovery_counter"]))  # 원본 run의 무효 진단 규칙이다.
        self.assertEqual(result["recovery_trigger_count"], 0)
        np.testing.assert_array_equal(session.alignment.dtw.previous_cost, cost)
        for count in (1, 2, 3):
            result = session.update(.1875, self.arrays["raw_world"][1])
            self.assertEqual(result["recovery_triggered"], int(count == 3))

    def test_unreachable_alignment_has_no_fake_score(self):
        session = self.session()
        result = session.update(100, self.arrays["raw_world"][0])
        self.assertTrue(result["pose_detected"])
        self.assertEqual(result["aligned_reference_index"], -1)
        self.assertTrue(math.isnan(result["raw_final_score"]))
        self.assertFalse(result["sample_recorded"])
        self.assertEqual(session.valid_sample_count, 0)

    def test_pause_resume_preserve_all_accumulated_state(self):
        session = self.session()
        for row, time in recovery_observations()[:7]:
            session.update(time, self.arrays["raw_world"][row])
        saved_result = session.last_result
        cost, length = session.alignment.dtw.previous_cost.copy(), session.alignment.dtw.previous_length.copy()
        score_state = session.score_stabilizer.smoothed_score
        feedback_state = deepcopy(session.feedback_stabilizer.__dict__)
        counters = (session.valid_sample_count, session.evaluation_sample_count)
        alignment = session.alignment
        session.pause()
        session.pause()
        with patch.object(session.alignment, "update", wraps=session.alignment.update) as updater:
            result = session.update(.1875, None)
            updater.assert_not_called()
        self.assertEqual(result["state"], "paused")
        self.assertFalse(result["evaluation_performed"])
        self.assertFalse(result["sample_recorded"])
        self.assertTrue(math.isnan(result["raw_final_score"]))
        self.assertEqual(result["aligned_reference_index"], -1)
        self.assertEqual(session.last_aligned_index, 4)
        self.assertEqual(session.alignment.counter, 2)
        np.testing.assert_array_equal(session.alignment.dtw.previous_cost, cost)
        np.testing.assert_array_equal(session.alignment.dtw.previous_length, length)
        self.assertEqual(session.score_stabilizer.smoothed_score, score_state)
        self.assertEqual(session.feedback_stabilizer.__dict__, feedback_state)
        self.assertEqual((session.valid_sample_count, session.evaluation_sample_count), counters)
        self.assert_nested_equal(session.last_result, saved_result)
        session.resume()
        self.assertIs(session.alignment, alignment)
        result = session.update(.1875, self.arrays["raw_world"][1])
        self.assertEqual(result["recovery_triggered"], 1)

    def test_reset_reuses_reference_clears_every_practice_state(self):
        session = self.session()
        for row, time in recovery_observations()[:8]:
            session.update(time, self.arrays["raw_world"][row])
        reference, old_alignment = session.reference, session.alignment
        session.end()
        session.reset()
        self.assertIs(session.reference, reference)
        self.assertIsNot(session.alignment, old_alignment)
        self.assertEqual(session.state, "running")
        self.assertEqual((session.valid_sample_count, session.evaluation_sample_count), (0, 0))
        self.assertIsNone(session.last_aligned_index)
        self.assertIsNone(session.last_result)
        self.assertEqual(session.alignment.trigger_count, 0)
        self.assertEqual(session.alignment.counter, 0)
        self.assertEqual(session.alignment.last_trigger, -math.inf)
        self.assertEqual(session.alignment.dtw.predicted_index, -1)
        self.assertEqual(session.alignment.dtw.previous_cost[0], 0)
        self.assertTrue(np.isinf(session.alignment.dtw.previous_cost[1:]).all())
        self.assertFalse(session.alignment.dtw.previous_length.any())
        self.assertIsNone(session.score_stabilizer.smoothed_score)
        self.assertEqual(session.feedback_stabilizer.displayed_feedback, "NONE")
        self.assertEqual(session.feedback_stabilizer.feedback_candidate_count, 0)
        self.assertEqual(session.update(0, self.arrays["raw_world"][0])["aligned_reference_index"], 0)

    def test_two_sessions_are_independent(self):
        first, second = self.session(), self.session()
        for row, time in recovery_observations()[:8]:
            first.update(time, self.arrays["raw_world"][row])
        first.pause()
        self.assertEqual(second.state, "running")
        self.assertEqual(second.alignment.dtw.predicted_index, -1)
        self.assertEqual(second.alignment.trigger_count, 0)
        self.assertIsNone(second.score_stabilizer.smoothed_score)
        self.assertEqual(second.valid_sample_count, 0)
        self.assertFalse(np.shares_memory(first.alignment.dtw.previous_cost, second.alignment.dtw.previous_cost))
        self.assertEqual(second.update(0, self.arrays["raw_world"][0])["aligned_reference_index"], 0)

    def test_session_mean_uses_valid_raw_scores_not_ema(self):
        session = self.session()
        user = self.arrays["raw_world"][0].copy()
        user[15] = user[13] + [0, 1, 0]
        with patch.object(session.alignment, "update", return_value=0):
            first = session.update(0, self.arrays["raw_world"][0])
            second = session.update(.6, user)
        session.update(.7, None)
        summary = session.end()
        self.assertEqual(summary["valid_sample_count"], 2)
        self.assertEqual(summary["evaluation_sample_count"], 3)
        self.assertEqual(summary["mean_raw_final_score"], (first["raw_final_score"] + second["raw_final_score"]) / 2)
        self.assertNotAlmostEqual(summary["mean_raw_final_score"] * 100,
                                  (first["displayed_score"] + second["displayed_score"]) / 2)
        self.assertEqual(summary["displayed_score"], second["displayed_score"])
        self.assertTrue(summary["evaluation_available"])

    def test_empty_end_is_unavailable_and_end_blocks_new_input(self):
        session = self.session()
        session.update(0, None)
        session.pause()
        summary = session.end()
        self.assertEqual(summary["valid_sample_count"], 0)
        self.assertFalse(summary["evaluation_available"])
        self.assertTrue(math.isnan(summary["mean_raw_final_score"]))
        self.assertTrue(math.isnan(summary["displayed_score"]))
        self.assert_nested_equal(summary, session.end())
        for action in (lambda: session.update(0, self.arrays["raw_world"][0]), session.pause, session.resume):
            with self.assertRaises(RuntimeError):
                action()
        self.assertEqual(session.evaluation_sample_count, 1)

    def test_result_and_latest_snapshot_cannot_change_statistics(self):
        session = self.session()
        result = session.update(0, self.arrays["raw_world"][0])
        result["raw_final_score"] = 0
        result["user_dir8"][:] = np.nan
        snapshot = session.last_result
        self.assertEqual(snapshot["raw_final_score"], 1)
        self.assertTrue(np.isfinite(snapshot["user_dir8"]).all())
        snapshot["bone_similarities"]["left_upper_arm"] = -1
        self.assertEqual(session.last_result["bone_similarities"]["left_upper_arm"], 1)
        self.assertEqual(session.end()["mean_raw_final_score"], 1)


@unittest.skipUnless(RESEARCH_PATH.is_file(), "로컬 6-2 연구 파일이 없습니다.")
class ResearchSessionParityTests(SessionFixture):
    # 일반 기능 테스트의 중복 실행 없이 fixture/비교 도우미만 재사용한다.
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("danc3d_session_research", RESEARCH_PATH)
        cls.research = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.research)
        cls.research.np = np

    def evaluate_original(self, sequence, alignment, ui, playback, world):
        """카메라 없이 실제 연구 함수를 원본 run의 running 평가 순서로 호출한다."""
        result = None if world is None else SimpleNamespace(
            pose_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in world]], pose_landmarks=[])
        user, _ = self.research.webcam_pose(result)
        detected = bool(user["mask"][0].any())
        aligned = alignment.update(user["vectors"][0], user["mask"][0], playback) if detected else None
        if not detected:
            alignment.counter = 0
        invalid = dict(vectors=np.full((1, 8, 3), np.nan), mask=np.zeros((1, 8), bool),
                       world=np.full((1, 33, 3), np.nan))
        calculation = sequence if aligned is not None else invalid
        index = aligned if aligned is not None else 0
        values, overall, parts = self.research.compare(calculation, index, 0, user)
        metrics, candidate = self.research.pose_feedback(calculation, index, 0, values, user)
        metrics.update(self.research.final_feedback_metrics(candidate, overall, aligned, sequence["times"], playback, np.nan))
        metrics.update(self.research.final_score_metrics(overall, parts, metrics["timing_offset_sec"]))
        recovery = dict(alignment.metrics)
        if not detected:
            for key in recovery:
                if key != "recovery_trigger_count":
                    recovery[key] = (-1 if key.endswith("_index") else
                                     0 if key in ("recovery_candidate", "recovery_triggered") else float("nan"))
        metrics.update(recovery)
        ui.update(metrics["final_score"], metrics["final_feedback"], playback)
        metrics.update(ui.snapshot(metrics["final_score"], metrics["final_feedback"]))
        metrics.update(overall_similarity=overall,
                       bone_similarities={name: float(value) for (name, _, _), value in zip(BONES, values)},
                       part_similarities={name: value for name, value in zip(PARTS, parts)},
                       bone_status={name: self.research.status(value)[0] for (name, _, _), value in zip(BONES, values)})
        priority = self.research.priority_error(values)
        metrics["priority_error_bone"] = BONES[priority][0] if priority is not None else "NONE"
        metrics["pose_detected"] = detected
        metrics["valid_user_bone_count"] = int(user["mask"][0].sum())
        return metrics

    def compare_observations(self, session, observations):
        reference = session.reference
        sequence = dict(world=reference.raw_world, vectors=reference.normalized_bone_vectors,
                        mask=reference.bone_valid_mask, times=reference.times)
        alignment = self.research.LocalRecoveryDTW(sequence["vectors"], sequence["mask"], sequence["times"])
        ui = self.research.UIStabilizer(0)
        triggers = []
        for frame, (playback, world) in enumerate(observations):
            actual = session.update(playback, world)
            expected = self.evaluate_original(sequence, alignment, ui, playback, world)
            with self.subTest(frame=frame):
                for key in actual.keys() & expected.keys():
                    with self.subTest(field=key):
                        self.assert_nested_equal(actual[key], expected[key])
                self.assert_nested_equal(session.alignment.dtw.previous_cost, alignment.dtw.previous_cost)
                self.assert_nested_equal(session.alignment.dtw.previous_length, alignment.dtw.previous_length)
                self.assertEqual(session.alignment.dtw.predicted_index, alignment.dtw.predicted_index)
                self.assertEqual(session.score_stabilizer.smoothed_score, ui.smoothed_score)
            if actual["recovery_triggered"]:
                triggers.append((frame, playback, actual["recovery_anchor_index"]))
        return triggers

    def test_original_run_pipeline_with_recovery_missing_and_partial_poses(self):
        session = self.session()
        observations = [(time, self.arrays["raw_world"][row]) for row, time in recovery_observations()]
        partial = self.arrays["raw_world"][7].copy()
        partial[15] = np.nan
        observations += [(.7, None), (.75, self.arrays["raw_world"][5]), (.8, partial),
                         (.85, np.zeros((33, 3))), (.9, self.arrays["raw_world"][6]),
                         (100, self.arrays["raw_world"][11])]
        self.assertEqual(self.compare_observations(session, observations), [(7, .1875, 1), (19, .6875, 4)])

    def test_original_run_pipeline_with_late_timing_and_pose_priority(self):
        worlds = np.repeat(make_world()[None], 24, axis=0)
        write_synthetic_reference(self.data_dir, worlds)
        session = self.session()
        bent = make_world()
        bent[15] = bent[13] + [0, 1, 0]
        observations = [(0, make_world())] + [(.5 + step / 32, make_world()) for step in range(6)]
        observations += [(1 + step / 32, bent) for step in range(6)] + [(1.25, None)]
        self.compare_observations(session, observations)


class RealReferenceSessionTests(unittest.TestCase):
    def test_all_ten_songs_load_and_evaluate_local_frames(self):
        songs = json.loads((PROJECT_DATA_DIR / "songs.json").read_text(encoding="utf-8"))
        self.assertEqual(len(songs), 10)
        for song in songs:
            with self.subTest(song_id=song["id"]):
                session = DanceSession(song["id"])
                self.assertEqual(len(session.reference.times), song["frame_count"])
                self.assertEqual(session.reference.times[0], 0)
                for row in range(8):
                    result = session.update(session.reference.times[row], session.reference.raw_world[row])
                    self.assertTrue(result["pose_detected"])
                    self.assertNotEqual(result["aligned_reference_index"], -1)
                    self.assertTrue(math.isfinite(result["raw_final_score"]))
                self.assertEqual(session.end()["valid_sample_count"], 8)


if __name__ == "__main__":
    unittest.main()
