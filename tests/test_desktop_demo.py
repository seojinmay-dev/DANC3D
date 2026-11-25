"""카메라를 열지 않고 CLI, JPG 대응, 입력 분리, 그리기와 재생 제어를 검증한다."""

import ast
from contextlib import redirect_stdout, redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unicodedata
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from demo import desktop_demo as demo
from app.core.pose_features import BONES, PARTS, TORSO_EDGES, extract_bone_directions
from app.services.dance_session import DanceSession


HAS_CV2 = importlib.util.find_spec("cv2") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None


def sample_result():
    """RAW/표시 값이 서로 다른 서비스 반환 fixture다. 평가 알고리즘은 구현하지 않는다."""
    return dict(pose_detected=True, evaluation_performed=True, sample_recorded=True,
                bone_status={name: "OK" for name, _, _ in BONES},
                bone_similarities={name: .9 for name, _, _ in BONES},
                part_similarities={name: .9 for name in PARTS},
                overall_similarity=.9, priority_error_bone="NONE", error_bone_count=0,
                valid_user_bone_count=8, aligned_reference_index=0, aligned_reference_time_sec=0.,
                timing_offset_sec=0., torso_yaw_error_deg=0., priority_feedback_type="NONE",
                raw_final_score=.12, pose_score=.8, score_timing_factor=.9,
                displayed_score=88, displayed_feedback="팔을 더 올리세요", raw_final_feedback="동작이 조금 늦습니다.",
                recovery_triggered=0, recovery_trigger_count=0, recovery_anchor_index=-1)


def image_coords():
    """얼굴과 12개 Body Joint 위치를 구별할 수 있는 원본 image landmark fixture다."""
    coords = np.full((33, 2), [.5, .05], dtype=np.float64)
    for joint, xy in {
        11: (.65, .2), 12: (.35, .2), 13: (.75, .35), 14: (.25, .35),
        15: (.85, .5), 16: (.15, .5), 23: (.6, .55), 24: (.4, .55),
        25: (.65, .7), 26: (.35, .7), 27: (.7, .9), 28: (.3, .9),
    }.items():
        coords[joint] = xy
    return coords


def readiness_landmarks():
    """visibility를 포함한 정면 T-Pose의 합성 원본 좌표다."""
    points = np.column_stack((image_coords(), np.full(33, .9)))
    points[[11, 12, 13, 14, 15, 16], :2] = [(.6,.2),(.4,.2),(.75,.2),(.25,.2),(.92,.2),(.08,.2)]
    points[31,:2], points[32,:2] = (.7,.92), (.3,.92)
    return points


def prepare_countdown(controller, now):
    """Space로 대기 후 실제 판정 통과로 카운트다운을 자동 시작한다."""
    controller.handle_key(32, now)
    controller.update_readiness(readiness_landmarks(), 640, 480)


class StubSession:
    """재생 제어가 공개 Session 메서드만 호출하는지 확인하는 상태 spy다."""

    def __init__(self):
        self.reference = SimpleNamespace(times=np.array([0., 1.]))
        self.state = "running"
        self.calls = []
        self.reset_count = 0
        self.end_count = 0
        self.sample_count = 0

    def pause(self):
        self.calls.append("pause")
        self.state = "paused"

    def resume(self):
        self.calls.append("resume")
        self.state = "running"

    def reset(self):
        self.calls.append("reset")
        self.state = "running"
        self.reset_count += 1
        self.sample_count = 0

    def update(self, playback_time, world_landmarks):
        self.calls.append(("update", playback_time, world_landmarks))
        if self.state == "running":
            self.sample_count += 1
        return dict(sample_result(), playback_time_sec=playback_time)

    def end(self):
        self.calls.append("end")
        self.state = "ended"
        self.end_count += 1
        return dict(song_id="test", evaluation_sample_count=self.sample_count, valid_sample_count=self.sample_count,
                    mean_raw_final_score=.5 if self.sample_count else float("nan"), recovery_trigger_count=0)


class CatalogAndReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="danc3d_demo_jpg_")
        self.addCleanup(self.temporary.cleanup)
        self.data_dir = Path(self.temporary.name)
        self.song = dict(id="test_F", title="테스트 곡", artist="테스트",
                         reference_metadata="references/test_F.json", raw_source_dir="raw/테스트/test_F")
        self.reference = SimpleNamespace(frame_indices=np.array([60, 64, 70]), times=np.array([0., .5, 1.]))
        self.raw_dir = self.data_dir / "raw" / unicodedata.normalize("NFD", "테스트") / "test_F"
        self.raw_dir.mkdir(parents=True)
        (self.data_dir / "references").mkdir()
        self.metadata_path = self.data_dir / "references/test_F.json"
        self.metadata = dict(sequence="test_F", raw_source_dir="data/raw/테스트/test_F")
        self.metadata_path.write_text(json.dumps(self.metadata), encoding="utf-8")
        (self.data_dir / "songs.json").write_text(json.dumps([self.song]), encoding="utf-8")
        for frame in (60, 64, 70, 72):
            (self.raw_dir / "test_F_{:07d}.jpg".format(frame)).write_bytes(b"synthetic path fixture")

    def test_catalog_real_ten_and_local_fixture(self):
        self.assertEqual(len(demo.load_songs()), 10)
        self.assertEqual(demo.load_songs(self.data_dir)[0]["id"], "test_F")

    def test_select_song_and_list_without_loading_runtime(self):
        songs = demo.load_songs(self.data_dir)
        self.assertEqual(demo.select_song(songs, "test_F"), self.song)
        with self.assertRaises(ValueError):
            demo.select_song(songs, "unknown")
        with patch("builtins.input", return_value="1"):
            self.assertEqual(demo.select_song(songs), self.song)
        with patch.object(demo, "load_runtime", side_effect=AssertionError("No GUI deps for listing")):
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(demo.main(["--list-songs"], data_dir=self.data_dir), 0)
            self.assertIn("test_F", output.getvalue())

    def test_direct_script_cli_works_outside_project(self):
        result = subprocess.run([sys.executable, str(Path(demo.__file__)), "--list-songs"],
                                cwd=self.data_dir, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("033_D00_001_F", result.stdout)
        self.assertEqual(len(result.stdout.strip().splitlines()), 10)

    def test_unknown_cli_song_does_not_open_camera(self):
        with patch.object(demo, "run_desktop", side_effect=AssertionError("No camera")):
            with redirect_stderr(io.StringIO()):
                self.assertEqual(demo.main(["--song", "unknown"], data_dir=self.data_dir), 1)

    def test_frame_index_correspondence_and_nfc_nfd(self):
        images = demo.ReferenceImages(self.song, self.reference, self.data_dir)
        row, frame, path = images.at_playback(.5)
        self.assertEqual((row, frame), (1, 64))
        self.assertEqual(path, self.raw_dir / "test_F_0000064.jpg")
        self.assertNotIn(1, images.paths)
        self.assertEqual(images.excluded_frame_indices, (72,))
        self.assertEqual(images.at_playback(.25)[0], 0)  # 표시 시간 동점은 이전 row다.

    def test_missing_jpg_is_not_replaced_with_adjacent_frame(self):
        (self.raw_dir / "test_F_0000064.jpg").unlink()
        images = demo.ReferenceImages(self.song, self.reference, self.data_dir)
        self.assertEqual(images.at_playback(.5), (1, 64, None))
        self.assertEqual(images.missing_frame_indices, (64,))

    def test_catalog_raw_path_fallback_and_missing_source(self):
        self.metadata["raw_source_dir"] = "data/raw/missing/test_F"
        self.metadata_path.write_text(json.dumps(self.metadata))
        images = demo.ReferenceImages(self.song, self.reference, self.data_dir)
        self.assertEqual(images.source_dir, self.raw_dir)
        self.song["raw_source_dir"] = "raw/missing/test_F"
        images = demo.ReferenceImages(self.song, self.reference, self.data_dir)
        self.assertEqual(images.paths, {})
        self.assertEqual(images.missing_frame_indices, (60, 64, 70))

    def test_metadata_sequence_fallback_and_ambiguous_frame(self):
        metadata = dict(sequence="test_F", song="테스트", sequence_relative_path="test_F")
        self.metadata_path.write_text(json.dumps(metadata))
        parent = self.data_dir / "raw" / "[원천]테스트"
        parent.mkdir(exist_ok=True)
        self.raw_dir.rename(parent / "test_F")
        images = demo.ReferenceImages(self.song, self.reference, self.data_dir)
        self.assertEqual(len(images.paths), 3)
        (parent / "test_F/test_F_0000060.jpeg").write_bytes(b"duplicate fixture")
        with self.assertRaisesRegex(ValueError, "Ambiguous JPG"):
            demo.ReferenceImages(self.song, self.reference, self.data_dir)

    def test_all_ten_reference_jpg_counts_and_ringdingdong_exclusions(self):
        total = 0
        for song in demo.load_songs():
            with self.subTest(song_id=song["id"]):
                with np.load(demo.DATA_DIR / song["reference_npz"], allow_pickle=False) as data:
                    timestamps = data["timestamps_ms"].astype(float)
                    reference = SimpleNamespace(frame_indices=data["frame_indices"], times=(timestamps - timestamps[0]) / 1000)
                images = demo.ReferenceImages(song, reference)
                self.assertEqual(len(images.paths), len(reference.frame_indices))
                self.assertEqual(images.missing_frame_indices, ())
                self.assertEqual(set(images.paths), {int(frame) for frame in reference.frame_indices})
                if song["id"] == "008_A00_001_F":
                    self.assertEqual(images.excluded_frame_indices, (6372, 6374))
                    self.assertNotIn(6372, images.paths)
                    self.assertNotIn(6374, images.paths)
                total += len(images.paths)
        self.assertEqual(total, 31455)


class LandmarkAndControlTests(unittest.TestCase):
    def test_world_and_image_landmarks_are_independent_original_axes(self):
        world = np.arange(99).reshape(33, 3).astype(float) - 40
        image = image_coords()
        result = SimpleNamespace(pose_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in world]],
                                 pose_landmarks=[[SimpleNamespace(x=x, y=y) for x, y in image]])
        actual_world, actual_image = demo.extract_pose_landmarks(result)
        np.testing.assert_array_equal(actual_world, world)
        np.testing.assert_array_equal(actual_image, image)
        self.assertEqual(actual_world.dtype, np.float64)
        self.assertEqual(actual_image.shape, (33, 2))

    def test_pose_missing_and_image_only_never_becomes_world(self):
        self.assertEqual(demo.extract_pose_landmarks(None), (None, None))
        points = [SimpleNamespace(x=x, y=y) for x, y in image_coords()]
        world, image = demo.extract_pose_landmarks(SimpleNamespace(pose_world_landmarks=[], pose_landmarks=[points]))
        self.assertIsNone(world)
        self.assertEqual(image.shape, (33, 2))
        world, image = demo.extract_pose_landmarks(SimpleNamespace(pose_world_landmarks=[points[:32]], pose_landmarks=[]))
        self.assertIsNone(world)
        self.assertIsNone(image)

    def test_pixels_are_original_unmirrored_and_invalid_not_shifted(self):
        coords = image_coords()
        coords[0] = [-.1, 1.1]
        coords[1] = [np.nan, .5]
        pixels = demo.landmarks_to_pixels(coords, (100, 200, 3))
        self.assertEqual(pixels[11], (130, 20))
        self.assertEqual(pixels[12], (70, 20))
        self.assertEqual(pixels[0], (-20, 110))
        self.assertIsNone(pixels[1])
        with self.assertRaises(ValueError):
            demo.landmarks_to_pixels(np.zeros((33, 3)), (100, 200, 3))

    def test_timestamp_strictly_increases_independently_of_playback_reset(self):
        timestamps = demo.CaptureTimestamp(10)
        session = StubSession()
        controller = demo.PlaybackController(session, 10)
        first = timestamps.next(10)
        second = timestamps.next(10)
        controller.handle_key(32, 10)
        controller.handle_key(ord("r"), 10)
        third = timestamps.next(10)
        self.assertEqual((first, second, third), (0, 1, 2))
        self.assertGreater(timestamps.next(10.1), third)

    def test_playback_pause_resume_uses_elapsed_time_not_frame_count(self):
        session = StubSession()
        controller = demo.PlaybackController(session, 0)
        controller.advance(5)
        self.assertEqual(controller.playback_time, 0)
        prepare_countdown(controller, 5)
        controller.advance(10)
        controller.evaluate(None)
        self.assertEqual(controller.playback_time, 0)
        controller.advance(10.125)
        self.assertEqual(controller.playback_time, .125)
        controller.handle_key(32, 10.25)
        self.assertEqual(controller.playback_time, .25)
        controller.advance(20)
        self.assertEqual(controller.playback_time, .25)
        controller.evaluate(None)
        self.assertEqual(session.sample_count, 1)
        controller.handle_key(32, 20)
        controller.advance(20.125)
        self.assertEqual(controller.playback_time, .375)
        controller.evaluate(None)
        self.assertEqual(session.sample_count, 2)
        self.assertEqual(session.reset_count, 0)

    def test_reset_debug_and_end_controls(self):
        session = StubSession()
        controller = demo.PlaybackController(session, 0)
        prepare_countdown(controller, 0)
        controller.advance(5)
        controller.evaluate(None)
        controller.advance(5.5)
        controller.handle_key(ord("d"), 5.5)
        self.assertTrue(controller.debug)
        controller.handle_key(ord("r"), 5.5)
        self.assertEqual(controller.playback_time, 0)
        self.assertEqual(session.state, "paused")
        self.assertEqual(session.reset_count, 1)
        prepare_countdown(controller, 5.5)
        controller.advance(10.5)
        controller.evaluate(None)
        controller.advance(12)
        self.assertEqual(controller.playback_time, 1)
        result = controller.evaluate(None)
        self.assertTrue(controller.finished)
        self.assertEqual(session.end_count, 1)
        self.assertIs(controller.evaluate(None), result)
        controller.finish()
        self.assertEqual(session.end_count, 1)
        self.assertFalse(controller.handle_key(27, 12))

    def test_world_is_passed_to_session_without_transform(self):
        session = StubSession()
        controller = demo.PlaybackController(session, 0)
        prepare_countdown(controller, 0)
        controller.advance(5)
        world = np.arange(99).reshape(33, 3)
        result = controller.evaluate(world)
        self.assertIs(session.calls[-1][2], world)
        self.assertEqual(result["displayed_score"], 88)

    def test_demo_does_not_import_or_call_core_evaluation_algorithms(self):
        tree = ast.parse(Path(demo.__file__).read_text())
        forbidden = {"extract_bone_directions", "compare", "WindowedOnlineDTW", "LocalRecoveryDTW",
                     "pose_feedback", "final_feedback_metrics", "final_score_metrics", "ScoreEMAStabilizer",
                     "FeedbackStabilizer", "angle_deg", "angle_diff_deg", "torso_yaw_deg", "einsum", "dot"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertFalse(forbidden & {alias.name for alias in node.names})
            elif isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                self.assertNotIn(name, forbidden)

    def test_terminal_summary_uses_session_raw_mean(self):
        song = dict(title="테스트", artist="테스트")
        summary = dict(song_id="test", evaluation_sample_count=6, valid_sample_count=5,
                       mean_raw_final_score=.1234, displayed_score=88, recovery_trigger_count=2)
        with redirect_stdout(io.StringIO()) as output:
            demo.print_summary(song, summary)
        self.assertIn("0.1234", output.getvalue())
        self.assertNotIn("88", output.getvalue())
        summary["mean_raw_final_score"] = float("nan")
        with redirect_stdout(io.StringIO()) as output:
            demo.print_summary(song, summary)
        self.assertIn("평가 불가", output.getvalue())


class ReadinessControlTests(unittest.TestCase):
    def setUp(self):
        self.session = StubSession()
        self.session.reference.times = np.array([0., 20.])
        self.controller = demo.PlaybackController(self.session, 0)

    def test_initial_ready_and_no_session_update(self):
        self.assertEqual(self.controller.label, "READY")
        self.assertEqual(self.session.state, "paused")
        self.assertIsNone(self.controller.evaluate(np.zeros((33,3))))
        self.assertFalse(any(isinstance(call, tuple) for call in self.session.calls))

    def test_ready_space_without_t_pose_enters_waiting(self):
        self.assertFalse(self.controller.readiness["ready"])
        self.controller.handle_key(32, 1)
        self.assertEqual(self.controller.label, "TPOSE_WAITING")
        self.assertIsNone(self.controller.countdown_started_at)

    def test_waiting_without_readiness_stays_waiting(self):
        self.controller.handle_key(32, 0)
        for now in (1, 2, 50):
            self.controller.advance(now)
            self.controller.update_readiness(None, 640, 480)
            self.controller.evaluate(None)
            self.controller.handle_key(32, now)
            self.assertEqual(self.controller.label, "TPOSE_WAITING")
            self.assertEqual(self.controller.playback_time, 0)
        self.assertEqual(self.session.sample_count, 0)
        self.assertFalse(any(isinstance(call, tuple) for call in self.session.calls))

    def test_waiting_ready_starts_countdown_automatically(self):
        self.controller.handle_key(32, 0)
        self.controller.advance(10)
        self.controller.update_readiness(readiness_landmarks(), 640, 480)
        self.assertEqual(self.controller.label, "COUNTDOWN")
        self.assertEqual(self.controller.countdown_started_at, 10)
        self.assertEqual(self.controller.countdown_number, 5)
        self.assertEqual(self.session.state, "paused")

    def test_countdown_starts_at_detection_time_not_capture_time(self):
        self.controller.handle_key(32, 0)
        self.controller.advance(1)
        self.controller.update_readiness(readiness_landmarks(),640,480,now=3)
        self.assertEqual(self.controller.countdown_started_at,3)
        self.controller.advance(7.999)
        self.assertEqual(self.controller.label,"COUNTDOWN")
        self.controller.advance(8)
        self.assertEqual(self.controller.label,"RUNNING")
        self.assertEqual(self.controller.playback_time,0)

    def test_t_pose_before_space_does_not_start(self):
        self.controller.update_readiness(readiness_landmarks(), 640, 480)
        self.assertEqual(self.controller.label, "READY")
        self.controller.handle_key(32, 0)
        self.assertEqual(self.controller.label, "TPOSE_WAITING")
        self.assertFalse(self.controller.readiness["ready"])

    def test_countdown_repeated_space_does_not_restart(self):
        prepare_countdown(self.controller, 0)
        self.controller.handle_key(32, 2)
        self.assertEqual(self.controller.countdown_started_at, 0)
        self.assertEqual(self.controller.countdown_number, 3)
        self.assertEqual(self.controller.label, "COUNTDOWN")

    def test_countdown_numbers_and_no_playback_before_five_seconds(self):
        prepare_countdown(self.controller, 0)
        for now, number in [(0,5),(.99,5),(1,4),(2,3),(3,2),(4,1),(4.999,1)]:
            self.controller.advance(now)
            self.assertEqual(self.controller.countdown_number,number)
            self.assertEqual(self.controller.playback_time,0)
            self.assertEqual(self.controller.label,"COUNTDOWN")

    def test_countdown_never_calls_update_or_adds_samples(self):
        prepare_countdown(self.controller, 0)
        for now in [0,1,2,3,4.9]:
            self.controller.advance(now)
            self.controller.evaluate(np.zeros((33,3)))
        self.assertEqual(self.session.sample_count,0)
        self.assertFalse(any(isinstance(call,tuple) for call in self.session.calls))

    def test_exact_five_seconds_starts_first_evaluation_at_zero(self):
        prepare_countdown(self.controller, 100)
        self.controller.advance(105)
        self.assertEqual(self.controller.label,"RUNNING")
        self.assertEqual(self.controller.playback_time,0)
        self.controller.evaluate(None)
        self.assertEqual(self.session.calls[-1],("update",0.,None))
        self.controller.advance(105.25)
        self.assertEqual(self.controller.playback_time,.25)

    def test_delayed_frame_does_not_add_countdown_to_content_time(self):
        prepare_countdown(self.controller, 0)
        self.controller.advance(7)
        self.controller.evaluate(None)
        self.assertEqual(self.controller.playback_time,0)
        self.controller.advance(7.125)
        self.assertEqual(self.controller.playback_time,.125)

    def test_key_clock_crossing_starts_next_evaluated_frame_at_zero(self):
        prepare_countdown(self.controller, 0)
        self.controller.advance(4.99)
        self.controller.evaluate(None)
        self.controller.handle_key(255,5.01)
        self.controller.advance(5.2)
        self.controller.evaluate(None)
        self.assertEqual(self.session.calls[-1],("update",0.,None))

    def test_duplicate_space_at_countdown_boundary_does_not_pause(self):
        prepare_countdown(self.controller,0)
        self.controller.handle_key(32,5)
        self.assertEqual(self.controller.label,"RUNNING")
        self.assertEqual(self.session.state,"running")

    def test_loss_of_pose_does_not_cancel_committed_countdown(self):
        prepare_countdown(self.controller,0)
        self.controller.update_readiness(None,640,480)
        self.controller.advance(5)
        self.assertEqual(self.controller.label,"RUNNING")
        self.assertEqual(self.controller.playback_time,0)

    def test_pause_resume_preserves_previous_result_and_clock(self):
        prepare_countdown(self.controller,0)
        self.controller.advance(5)
        result=self.controller.evaluate(None)
        self.controller.handle_key(32,5.5)
        self.controller.advance(50)
        self.assertEqual(self.controller.playback_time,.5)
        self.assertEqual(self.controller.label,"PAUSED")
        self.assertIs(self.controller.last_result,result)
        self.controller.handle_key(32,50)
        self.controller.advance(50.25)
        self.assertEqual(self.controller.playback_time,.75)
        self.assertEqual(self.controller.label,"RUNNING")

    def test_reset_in_every_stage_returns_to_unready(self):
        for stage in ["READY","TPOSE_WAITING","COUNTDOWN","RUNNING","PAUSED","FINISHED"]:
            with self.subTest(stage=stage):
                session=StubSession()
                controller=demo.PlaybackController(session,0)
                if stage == "TPOSE_WAITING":controller.handle_key(32,0)
                elif stage != "READY":prepare_countdown(controller,0)
                if stage in ["RUNNING","PAUSED","FINISHED"]:
                    controller.advance(5);controller.evaluate(None)
                if stage=="PAUSED":controller.handle_key(32,5)
                if stage=="FINISHED":controller.finish()
                self.assertEqual(controller.label,stage)
                controller.handle_key(ord("r"),6)
                self.assertEqual(controller.label,"READY")
                self.assertEqual(controller.playback_time,0)
                self.assertFalse(controller.readiness["ready"])
                self.assertIsNone(controller.countdown_started_at)
                self.assertIsNone(controller.last_result)
                self.assertEqual(session.sample_count,0)
                self.assertEqual(session.reset_count,1)
                controller.handle_key(32,6)
                self.assertEqual(controller.label,"TPOSE_WAITING")

    def test_reset_cancels_countdown_and_later_time_cannot_start_it(self):
        prepare_countdown(self.controller,0)
        self.controller.handle_key(ord("r"),2)
        self.controller.advance(100)
        self.assertEqual(self.controller.label,"READY")
        self.assertEqual(self.controller.playback_time,0)

    def test_finished_no_samples_or_space_restart(self):
        prepare_countdown(self.controller,0)
        self.controller.advance(5);self.controller.evaluate(None)
        self.controller.advance(25);self.controller.evaluate(None)
        count=self.session.sample_count
        self.controller.handle_key(32,26)
        self.controller.evaluate(None)
        self.assertEqual(self.controller.label,"FINISHED")
        self.assertEqual(self.session.sample_count,count)
        self.assertEqual(self.session.end_count,1)

    def test_debug_and_quit_work_during_countdown(self):
        prepare_countdown(self.controller,0)
        self.controller.handle_key(ord("d"),1)
        self.assertTrue(self.controller.debug)
        self.assertFalse(self.controller.handle_key(ord("q"),2))
        self.assertFalse(self.controller.handle_key(27,2))

    def test_capture_timestamps_increase_across_all_stages(self):
        timestamps=demo.CaptureTimestamp(0)
        values=[timestamps.next(0)]
        prepare_countdown(self.controller,0)
        values.append(timestamps.next(0))
        self.controller.advance(5);self.controller.evaluate(None)
        values.append(timestamps.next(5))
        self.controller.handle_key(32,5)
        values.append(timestamps.next(5))
        self.controller.handle_key(ord("r"),5)
        values.append(timestamps.next(5))
        self.assertTrue(all(b>a for a,b in zip(values,values[1:])))


@unittest.skipUnless(HAS_CV2 and HAS_PIL, "OpenCV/Pillow가 설치된 로컬 환경에서 그리기 테스트를 실행하세요.")
class DrawingAndRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        cls.cv2 = cv2
        cls.runtime = SimpleNamespace(cv2=cv2, Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont)

    def setUp(self):
        self.image = np.zeros((380, 600, 3), dtype=np.uint8)
        self.image[:, :300] = [25, 40, 80]
        self.image[:, 300:] = [80, 40, 25]
        self.coords = image_coords()
        self.result = sample_result()

    def test_raw_bone_colors_and_priority_thickness(self):
        self.result["bone_status"]["left_upper_arm"] = "ERROR"
        self.result["bone_status"]["left_forearm"] = "INVALID"
        self.result["bone_status"]["right_upper_arm"] = "ERROR"
        self.result["priority_error_bone"] = "right_upper_arm"
        pixels = demo.landmarks_to_pixels(self.coords, self.image.shape)
        with patch.object(self.cv2, "line", wraps=self.cv2.line) as line:
            demo.draw_skeleton(self.image, self.coords, self.result, self.cv2)
        calls = line.call_args_list
        by_endpoints = {(call.args[1], call.args[2]): call.args for call in calls}
        self.assertEqual(by_endpoints[(pixels[11], pixels[13])][3], demo.BONE_COLORS["ERROR"])
        self.assertEqual(by_endpoints[(pixels[13], pixels[15])][3], demo.BONE_COLORS["INVALID"])
        self.assertEqual(by_endpoints[(pixels[23], pixels[25])][3], demo.BONE_COLORS["OK"])
        self.assertGreater(by_endpoints[(pixels[12], pixels[14])][4], by_endpoints[(pixels[11], pixels[13])][4])
        self.assertEqual(calls[-1].args[1:3], (pixels[12], pixels[14]))

    def test_white_body_joints_and_face_exclusion(self):
        pixels = demo.landmarks_to_pixels(self.coords, self.image.shape)
        with patch.object(self.cv2, "circle", wraps=self.cv2.circle) as circle:
            drawn = demo.draw_skeleton(self.image, self.coords, self.result, self.cv2)
        white_calls = [call for call in circle.call_args_list if call.args[3] == demo.WHITE]
        self.assertEqual(len(white_calls), 12)
        self.assertEqual({call.args[1] for call in white_calls}, {pixels[joint] for joint in demo.BODY_JOINTS})
        self.assertNotIn(pixels[0], {call.args[1] for call in circle.call_args_list})
        self.assertEqual(demo.BODY_JOINTS, (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28))
        np.testing.assert_array_equal(drawn[pixels[11][1], pixels[11][0]], [255, 255, 255])

    def test_torso_color_uses_returned_yaw_only(self):
        for error, expected in ((14.999, "OK"), (15, "ERROR"), (-15, "ERROR"), (np.nan, "INVALID")):
            self.result["torso_yaw_error_deg"] = error
            self.assertEqual(demo.torso_display_color(self.result), demo.BONE_COLORS[expected])

    def test_reference_and_webcam_mirror_only_display_copy(self):
        original = self.image.copy()
        expected_drawn = demo.draw_skeleton(self.image, self.coords, self.result, self.cv2)
        panel = demo.webcam_panel(self.image, self.coords, self.result, self.cv2)
        np.testing.assert_array_equal(panel, self.cv2.flip(expected_drawn, 1))
        np.testing.assert_array_equal(demo.reference_panel(self.image, self.cv2), self.cv2.flip(original, 1))
        np.testing.assert_array_equal(self.image, original)
        self.assertEqual(self.result["displayed_feedback"], "팔을 더 올리세요")

    def test_missing_image_landmarks_and_jpg_show_explicit_notice(self):
        with patch.object(demo, "draw_text", wraps=demo.draw_text) as text:
            demo.webcam_panel(self.image, None, self.result, self.cv2)
            demo.reference_panel(None, self.cv2)
        messages = [call.args[1] for call in text.call_args_list]
        self.assertIn("Skeleton Overlay: NOT DETECTED", messages)
        self.assertIn("No matching local JPG", messages)

    def test_renderer_uses_display_score_feedback_without_resmoothing(self):
        renderer = demo.DemoRenderer(self.runtime)
        controller = SimpleNamespace(label="RUNNING", playback_time=1.25, duration=10, finished=False, debug=False)
        song = dict(title="테스트 곡", artist="테스트")
        with patch.object(demo, "draw_text", wraps=demo.draw_text) as text:
            with patch.object(renderer, "korean", wraps=renderer.korean) as korean:
                canvas = renderer.render(song, controller, self.result, self.image, self.image, self.coords, 60)
        messages = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn("88", messages)
        self.assertNotIn("12", messages)
        self.assertIn("팔을 더 올리세요", [call.args[1] for call in korean.call_args_list])
        self.assertNotIn("동작이 조금 늦습니다.", [call.args[1] for call in korean.call_args_list])
        self.assertEqual(canvas.shape, (980, 1200, 3))

    def test_no_pose_marks_held_score_as_unavailable_current_evaluation(self):
        renderer = demo.DemoRenderer(self.runtime)
        controller = SimpleNamespace(label="RUNNING", playback_time=.5, duration=10, finished=False, debug=False)
        self.result.update(pose_detected=False, sample_recorded=False, raw_final_score=np.nan,
                           aligned_reference_index=-1, aligned_reference_time_sec=np.nan)
        with patch.object(demo, "draw_text", wraps=demo.draw_text) as text:
            renderer.render(dict(title="테스트", artist="테스트"), controller, self.result, None, self.image, None, 0)
        messages = [str(call.args[1]) for call in text.call_args_list]
        self.assertIn("Pose not detected - current evaluation unavailable", messages)
        self.assertIn("Score (held)", messages)

    def test_korean_jpg_path_and_decode_cache(self):
        with tempfile.TemporaryDirectory(prefix="danc3d_demo_decode_") as directory:
            path = Path(directory) / "한글 프레임.jpg"
            ok, encoded = self.cv2.imencode(".jpg", self.image)
            self.assertTrue(ok)
            path.write_bytes(encoded.tobytes())
            demo.read_jpg.cache_clear()
            with patch.object(self.cv2, "imdecode", wraps=self.cv2.imdecode) as decode:
                first = demo.read_jpg(path, self.cv2)
                second = demo.read_jpg(path, self.cv2)
            self.assertEqual(first.shape, self.image.shape)
            self.assertIs(first, second)
            self.assertEqual(decode.call_count, 1)

    def test_ready_screen_is_neutral_and_has_no_fake_score(self):
        renderer=demo.DemoRenderer(self.runtime)
        controller=demo.PlaybackController(StubSession(),0)
        controller.update_readiness(readiness_landmarks(),640,480)
        with patch.object(demo,"draw_text",wraps=demo.draw_text) as text, \
             patch.object(self.cv2,"line",wraps=self.cv2.line) as lines:
            canvas=renderer.render(dict(title="테스트",artist="테스트"),controller,None,self.image,self.image,self.coords,0)
        messages=[str(call.args[1]) for call in text.call_args_list]
        self.assertNotIn("T-POSE READY",messages)
        self.assertIn("--",messages)
        self.assertNotIn("88",messages)
        self.assertTrue(all(call.args[3]==demo.BONE_COLORS["INVALID"] for call in lines.call_args_list))
        self.assertEqual(canvas.shape,(980,1200,3))

    def test_countdown_screen_has_large_number_and_stationary_reference(self):
        renderer=demo.DemoRenderer(self.runtime)
        controller=demo.PlaybackController(StubSession(),0)
        prepare_countdown(controller,0)
        controller.advance(1)
        with patch.object(demo,"draw_text",wraps=demo.draw_text) as text, \
             patch.object(renderer,"korean",wraps=renderer.korean) as korean:
            renderer.render(dict(title="테스트",artist="테스트"),controller,None,self.image,self.image,self.coords,0)
        large=[call for call in text.call_args_list if call.kwargs.get("scale")==3.0]
        self.assertEqual(large[0].args[1],"4")
        self.assertEqual(len(korean.call_args_list),1)
        self.assertNotIn("Score",[call.args[1] for call in text.call_args_list])
        self.assertEqual(controller.playback_time,0)

    def test_ready_and_waiting_show_only_their_single_instruction(self):
        renderer=demo.DemoRenderer(self.runtime)
        controller=demo.PlaybackController(StubSession(),0)
        song=dict(title="테스트",artist="테스트")
        for stage,message in [("READY","SPACE를 눌러 시작하세요."),("TPOSE_WAITING","T자 자세를 취해주세요.")]:
            if stage=="TPOSE_WAITING":controller.handle_key(32,0)
            points=[SimpleNamespace(x=x,y=y) for x,y in self.coords]
            controller.update_readiness(points,640,480)
            with patch.object(renderer,"korean",wraps=renderer.korean) as text:
                renderer.render(song,controller,None,None,self.image,self.coords,0)
            self.assertEqual([call.args[1] for call in text.call_args_list][1:],[message])

    def test_mocked_webcam_loop_never_opens_real_camera_and_timestamps_survive_reset(self):
        with tempfile.TemporaryDirectory(prefix="danc3d_demo_loop_") as directory:
            data_dir = Path(directory)
            (data_dir / "references").mkdir()
            world = np.zeros((33, 3))
            for joint in demo.BODY_JOINTS:
                world[joint] = [joint, joint * .5, joint * .25]
            vectors, mask = extract_bone_directions(world)
            song = dict(id="test_F", title="테스트", artist="테스트",
                        reference_npz="references/test_F.npz", reference_metadata="references/test_F.json")
            (data_dir / "songs.json").write_text(json.dumps([song]))
            metadata = dict(sequence="test_F", bone_mapping={name: dict(index=index, parent=parent, child=child)
                                                             for index, (name, parent, child) in enumerate(BONES)})
            (data_dir / "references/test_F.json").write_text(json.dumps(metadata))
            np.savez(data_dir / "references/test_F.npz", frame_indices=np.array([0, 1]), timestamps_ms=np.array([0, 1000]),
                     raw_world=np.stack([world, world]), normalized_bone_vectors=np.stack([vectors, vectors]),
                     bone_valid_mask=np.stack([mask, mask]))
            session = DanceSession("test_F", data_dir=data_dir)
            camera = MagicMock()
            camera.isOpened.return_value = True
            camera.read.side_effect = [(True, self.image.copy()) for _ in range(7)]
            detector = MagicMock()
            pose = SimpleNamespace(pose_world_landmarks=[[SimpleNamespace(x=x, y=y, z=z) for x, y, z in world]],
                                   pose_landmarks=[[SimpleNamespace(x=x, y=y, visibility=v) for x, y, v in readiness_landmarks()]])
            detector.detect_for_video.return_value = pose
            landmarker = MagicMock()
            landmarker.create_from_options.return_value.__enter__.return_value = detector
            vision = SimpleNamespace(PoseLandmarker=landmarker, PoseLandmarkerOptions=MagicMock(),
                                     RunningMode=SimpleNamespace(VIDEO="VIDEO"))
            runtime = SimpleNamespace(cv2=self.cv2, vision=vision, BaseOptions=MagicMock(),
                                      mp=SimpleNamespace(Image=lambda **kwargs: SimpleNamespace(**kwargs),
                                                         ImageFormat=SimpleNamespace(SRGB="SRGB")))
            ticks = iter([0.,0., 0.,.01, 1.,1.005,1.01, 6.02,6.03, 6.12,6.13, 6.2,6.21, 6.3,6.31, 6.4,6.405,6.41])
            with patch.object(demo, "DanceSession", return_value=session), patch.object(demo, "DemoRenderer") as renderer, \
                 patch.object(demo, "ensure_pose_model") as ensure_model:
                renderer.return_value.render.return_value = np.zeros((980, 1200, 3), np.uint8)
                with patch.object(self.cv2, "VideoCapture", return_value=camera) as capture, \
                     patch.object(self.cv2, "namedWindow"), patch.object(self.cv2, "resizeWindow"), \
                     patch.object(self.cv2, "imshow"), patch.object(self.cv2, "destroyAllWindows") as destroy, \
                     patch.object(self.cv2, "getWindowProperty", return_value=1), \
                     patch.object(self.cv2, "waitKey", side_effect=[32, 32, 255, 32, ord("r"), 32, ord("q")]), \
                     patch.object(session, "update", wraps=session.update) as update, redirect_stdout(io.StringIO()):
                    demo.run_desktop(song, 2, runtime, data_dir=data_dir, clock=lambda: next(ticks))
                capture.assert_called_once_with(2)
                ensure_model.assert_called_once_with(demo.MODEL_PATH)
                camera.release.assert_called_once()
                destroy.assert_called_once()
            timestamps = [call.args[1] for call in detector.detect_for_video.call_args_list]
            self.assertTrue((np.diff(timestamps) > 0).all())
            self.assertEqual(len(timestamps), 7)
            self.assertEqual(update.call_count, 3)
            self.assertEqual([call.kwargs["playback_time"] for call in update.call_args_list if call.kwargs["playback_time"] == 0], [0.])
            for call in detector.detect_for_video.call_args_list:
                np.testing.assert_array_equal(call.args[0].data, self.cv2.cvtColor(self.image, self.cv2.COLOR_BGR2RGB))
            for call in update.call_args_list:
                np.testing.assert_array_equal(call.kwargs["world_landmarks"], world)
            options = vision.PoseLandmarkerOptions.call_args.kwargs
            self.assertEqual(options["running_mode"], "VIDEO")
            self.assertEqual(options["num_poses"], 1)
            self.assertFalse(options["output_segmentation_masks"])
            self.assertEqual(session.state, "ended")


if __name__ == "__main__":
    unittest.main()
