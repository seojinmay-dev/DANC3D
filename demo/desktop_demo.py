"""DanceSession을 사용하는 로컬 웹캠 Desktop Demo. 평가 계산은 서비스에 위임한다."""

import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
import unicodedata

import numpy as np

# python demo/desktop_demo.py로 실행해도 프로젝트 패키지를 찾도록 한다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.readiness import evaluate_readiness
from app.services.dance_session import DanceSession
from app.core.pose_features import BONES, PARTS, TORSO_EDGES
from app.core.feedback import TORSO_ROTATION_THRESHOLD_DEG
from app.services.pose_model import ensure_pose_model


DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_full.task"
PANEL_WIDTH, PANEL_HEIGHT = 600, 380
WINDOW_NAME = "DANC3D Desktop Demo"
BONE_COLORS = {"OK": (80, 215, 120), "ERROR": (35, 35, 245), "INVALID": (145, 145, 145)}
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
BODY_JOINTS = tuple(sorted({joint for _, parent, child in BONES for joint in (parent, child)}))
LOCAL_FONT_PATHS = (
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("C:/Windows/Fonts/malgun.ttf"),
)


def load_songs(data_dir=DATA_DIR):
    """로컬 songs.json 목록을 읽고 표시/선택에 필요한 id/title/artist를 검사한다."""
    path = Path(data_dir) / "songs.json"
    with path.open(encoding="utf-8") as handle:
        songs = json.load(handle)
    if not isinstance(songs, list) or not songs:
        raise ValueError("songs.json must contain a nonempty list")
    ids = []
    for song in songs:
        if not isinstance(song, dict) or not all(isinstance(song.get(key), str) and song[key]
                                               for key in ("id", "title", "artist")):
            raise ValueError("Each song must have a nonempty id, title and artist")
        ids.append(song["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate song_id in songs.json")
    return songs


def print_songs(songs):
    """선택 가능한 로컬 곡을 원래 catalog 순서와 song_id로 출력한다."""
    for number, song in enumerate(songs, 1):
        print("{}. {} / {} [{}]".format(number, song["title"], song["artist"], song["id"]))


def select_song(songs, song_id=None):
    """명시한 song_id 또는 터미널의 곡 번호/song_id로 catalog 항목을 선택한다."""
    if song_id is not None:
        match = next((song for song in songs if song["id"] == song_id), None)
        if match is None:
            raise ValueError("Unknown song_id: {}".format(song_id))
        return match
    while True:
        selection = input("Song number or song_id: ").strip()
        if selection.isdigit() and 1 <= int(selection) <= len(songs):
            return songs[int(selection) - 1]
        match = next((song for song in songs if song["id"] == selection), None)
        if match is not None:
            return match
        print("목록에 있는 곡 번호 또는 song_id를 입력하세요.")


def _nfc(value):
    """파일명을 변경하지 않고 한글 NFC/NFD 비교만 통일한다."""
    return unicodedata.normalize("NFC", str(value))


def _resolve_local_path(root, relative, metadata=False):
    """실제 디렉터리 이름을 NFC로 비교하여 data/ 상대 경로를 찾는다. 누락은 None이다."""
    if not isinstance(relative, str) or not relative:
        return None
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts:
        raise ValueError("Expected a local relative data path: {}".format(relative))
    if metadata and parts and parts[0] == "data":
        parts = parts[1:]
    current = Path(root)
    for part in parts:
        if not current.is_dir():
            return None
        matches = [child for child in current.iterdir() if _nfc(child.name) == _nfc(part)]
        if len(matches) > 1:
            raise ValueError("Ambiguous NFC/NFD path: {}".format(relative))
        if not matches:
            return None
        current = matches[0]
    return current


class ReferenceImages:
    """NPZ의 원본 frame_indices에 있는 JPG만 표시용으로 연결한다. 계산용 NPZ는 다시 읽지 않는다."""

    def __init__(self, song, reference, data_dir=DATA_DIR):
        self.reference = reference
        data_dir = Path(data_dir)
        metadata_path = _resolve_local_path(data_dir, song.get("reference_metadata"))
        if metadata_path is None:
            raise FileNotFoundError("Reference metadata unavailable for {}".format(song["id"]))
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("sequence") != song["id"]:
            raise ValueError("metadata sequence does not match song_id")
        sequence = metadata["sequence"]
        self.source_dir = _resolve_local_path(data_dir, metadata.get("raw_source_dir"), metadata=True)
        if self.source_dir is None or not self.source_dir.is_dir():
            self.source_dir = _resolve_local_path(data_dir, song.get("raw_source_dir"))
        if self.source_dir is None or not self.source_dir.is_dir():
            self.source_dir = None
            raw_dir = data_dir / "raw"
            parents = [path for path in raw_dir.iterdir()
                       if path.is_dir() and _nfc(path.name).startswith("[원천]")
                       and _nfc(path.name)[len("[원천]"):].strip() == _nfc(metadata.get("song", ""))] if raw_dir.is_dir() else []
            candidates = [_resolve_local_path(parent, metadata.get("sequence_relative_path", sequence))
                          for parent in parents]
            candidates = [path for path in candidates if path is not None and path.is_dir()]
            if len(candidates) > 1:
                raise ValueError("Ambiguous raw sequence directories")
            if candidates:
                self.source_dir = candidates[0]
        expected_frames = {int(frame) for frame in reference.frame_indices}
        self.paths = {}
        excluded = set()
        pattern = re.compile(re.escape(_nfc(sequence)) + r"_(\d+)$")
        if self.source_dir is not None:
            for path in self.source_dir.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in (".jpg", ".jpeg"):
                    continue
                match = pattern.fullmatch(_nfc(path.stem))
                if match is None:
                    continue
                frame = int(match.group(1))
                if frame not in expected_frames:
                    excluded.add(frame)
                    continue
                if frame in self.paths:
                    raise ValueError("Ambiguous JPG for frame_index {}".format(frame))
                self.paths[frame] = path
        self.excluded_frame_indices = tuple(sorted(excluded))
        self.missing_frame_indices = tuple(sorted(expected_frames - set(self.paths)))

    def at_playback(self, playback_time):
        """일반 playback에 가장 가까운 NPZ row/frame_index/JPG 경로를 반환한다.

        row는 DTW 정렬과 독립적인 표시 위치이며 동점은 이전 row다. JPG가 없으면
        path는 None이다. NPZ에 없는 인접 JPG로 대체하지 않는다.
        """
        times = self.reference.times
        right = min(int(np.searchsorted(times, playback_time)), len(times) - 1)
        left = max(0, right - 1)
        row = left if abs(times[left] - playback_time) <= abs(times[right] - playback_time) else right
        frame = int(self.reference.frame_indices[row])
        return row, frame, self.paths.get(frame)


def extract_pose_landmarks(result):
    """Tasks 결과에서 계산용 World (33, 3)와 표시용 image x/y (33, 2)를 분리한다.

    각 배열은 float64 원본 좌표이며 서로 대체하지 않는다. 미검출/잘못된 shape나
    landmark 형식은 해당 배열을 None으로 반환한다. World의 NaN은 서비스에 맡긴다.
    """
    arrays = []
    for field, axes in (("pose_world_landmarks", ("x", "y", "z")), ("pose_landmarks", ("x", "y"))):
        poses = getattr(result, field, None) if result is not None else None
        array = None
        if poses:
            try:
                points = np.asarray([tuple(getattr(point, axis) for axis in axes) for point in poses[0]], dtype=np.float64)
                if points.shape == (33, len(axes)):
                    array = points
            except (AttributeError, TypeError, ValueError):
                pass
        arrays.append(array)
    return tuple(arrays)


def landmarks_to_pixels(image_landmarks, image_shape):
    """원본 image x/y (33, 2)를 round(x*width), round(y*height) 픽셀 목록으로 변환한다.

    비유한 점은 None이고 이미지 밖 점은 임의로 이동시키지 않는다. 잘못된 shape는
    ValueError다. World 좌표나 표시용 mirror 좌표를 받지 않는다.
    """
    coords = np.asarray(image_landmarks, dtype=np.float64)
    if coords.shape != (33, 2):
        raise ValueError("image_landmarks must have shape (33, 2)")
    height, width = image_shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    return [(round(float(x) * width), round(float(y) * height))
            if math.isfinite(float(x)) and math.isfinite(float(y)) else None for x, y in coords]


def torso_display_color(result):
    """서비스의 yaw 오류와 core 임계값으로 몸통 표시 색상만 결정한다. 각도를 계산하지 않는다."""
    error = float(result.get("torso_yaw_error_deg", float("nan")))
    return BONE_COLORS["INVALID"] if not math.isfinite(error) else (
        BONE_COLORS["ERROR"] if abs(error) >= TORSO_ROTATION_THRESHOLD_DEG else BONE_COLORS["OK"])


def draw_skeleton(original, image_landmarks, result, cv2_module):
    """원본 BGR 영상 복사본에 RAW 상태의 Bone과 흰 Body Joint를 그린다. 반전은 하지 않는다."""
    image = original.copy()
    if image_landmarks is None:
        return image
    pixels = landmarks_to_pixels(image_landmarks, image.shape)
    height, width = image.shape[:2]
    display_scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    normal_width = max(1, round(4 / display_scale))
    priority_width = max(normal_width + 1, round(6 / display_scale))

    def line(parent, child, color, thickness):
        if pixels[parent] is not None and pixels[child] is not None:
            visible, start, end = cv2_module.clipLine((0, 0, width, height), pixels[parent], pixels[child])
            if visible:
                cv2_module.line(image, start, end, color, thickness, cv2_module.LINE_AA)

    for parent, child in TORSO_EDGES:
        line(parent, child, torso_display_color(result),
             priority_width if result.get("priority_feedback_type") == "torso_rotation" else normal_width)
    priority = result.get("priority_error_bone", "NONE")
    order = [bone for bone in BONES if bone[0] != priority] + [bone for bone in BONES if bone[0] == priority]
    for name, parent, child in order:
        color = BONE_COLORS[result["bone_status"].get(name, "INVALID")]
        line(parent, child, color, priority_width if name == priority else normal_width)
    joint_radius = max(1, round(3 / display_scale))
    joint_outline = max(1, round(1 / display_scale))
    for joint in BODY_JOINTS:
        point = pixels[joint]
        if point is not None and 0 <= point[0] < width and 0 <= point[1] < height:
            cv2_module.circle(image, point, joint_radius, WHITE, -1, cv2_module.LINE_AA)
            cv2_module.circle(image, point, joint_radius, BLACK, joint_outline, cv2_module.LINE_AA)
    return image


def draw_text(canvas, message, xy, cv2_module, color=WHITE, scale=.52, thickness=1):
    """OpenCV의 ASCII 텍스트를 그리며 평가값을 변경하지 않는다."""
    cv2_module.putText(canvas, str(message), xy, cv2_module.FONT_HERSHEY_SIMPLEX,
                       scale, color, thickness, cv2_module.LINE_AA)


def fit_panel(image, cv2_module, missing_message="No matching local JPG"):
    """종횡비와 Skeleton 위치를 유지해 600x380 패널에 영상을 맞춘다."""
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 24, dtype=np.uint8)
    if image is None:
        draw_text(panel, missing_message, (20, 50), cv2_module, BONE_COLORS["INVALID"])
        return panel
    height, width = image.shape[:2]
    scale = min(PANEL_WIDTH / width, PANEL_HEIGHT / height)
    resized = cv2_module.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))))
    height, width = resized.shape[:2]
    y, x = (PANEL_HEIGHT - height) // 2, (PANEL_WIDTH - width) // 2
    panel[y:y + height, x:x + width] = resized
    return panel


def reference_panel(image, cv2_module):
    """Reference 원본 JPG는 표시용 복사본만 mirror한다."""
    return fit_panel(cv2_module.flip(image, 1) if image is not None else None, cv2_module)


def webcam_panel(image, image_landmarks, result, cv2_module):
    """원본 image 좌표로 그리기를 끝낸 뒤 완성된 영상만 mirror한다."""
    drawn = draw_skeleton(image, image_landmarks, result, cv2_module)
    panel = fit_panel(cv2_module.flip(drawn, 1), cv2_module)
    if image_landmarks is None:
        draw_text(panel, "Skeleton Overlay: NOT DETECTED", (15, 360), cv2_module, BONE_COLORS["INVALID"])
    return panel


@lru_cache(maxsize=12)
def read_jpg(path, cv2_module):
    """한글 경로 JPG를 로컬에서 읽어 캐시한다. 누락/손상은 None이며 원본은 수정하지 않는다."""
    if path is None:
        return None
    try:
        return cv2_module.imdecode(np.fromfile(path, dtype=np.uint8), cv2_module.IMREAD_COLOR)
    except (OSError, ValueError, cv2_module.error):
        return None


class CaptureTimestamp:
    """MediaPipe VIDEO용 밀리초 시각. playback/Pause/Reset과 독립적으로 엄격히 증가한다."""

    def __init__(self, origin_time):
        self.origin_time = float(origin_time)
        self.last_timestamp_ms = -1

    def next(self, capture_time):
        """같은 시계값이 반복돼도 이전 값보다 최소 1ms 큰 int를 반환한다."""
        timestamp = max(self.last_timestamp_ms + 1, int((capture_time - self.origin_time) * 1000))
        self.last_timestamp_ms = timestamp
        return timestamp


COUNTDOWN_DURATION_SEC = 5.0


class PlaybackController:
    """준비/카운트다운과 재생 시계를 관리하고 평가는 기존 세션에 위임한다."""

    def __init__(self, session, now):
        self.session = session
        self.duration = float(session.reference.times[-1])
        self.playback_time = 0.0
        self.last_clock_time = float(now)
        self.debug = False
        self.stage = "READY"
        self.finished = False
        self.summary = self.last_result = None
        self.countdown_started_at = None
        self.running_start_pending = False
        self.readiness = evaluate_readiness(None, 1, 1)
        session.pause()

    @property
    def label(self):
        """화면용 명시적 상태를 반환한다."""
        return self.stage

    @property
    def countdown_number(self):
        """단조 증가 시계에 따른 5→1 표시 숫자를 반환한다."""
        if self.stage != "COUNTDOWN":
            return None
        remaining = COUNTDOWN_DURATION_SEC - (self.last_clock_time - self.countdown_started_at)
        return max(1, min(5, math.ceil(remaining)))

    def update_readiness(self, image_landmarks, width, height, now=None):
        """최신 준비 판정으로 자동 시작한다. now는 추론 완료 후 단조 시각이다."""
        self.readiness = evaluate_readiness(image_landmarks, width, height)
        if self.stage == "TPOSE_WAITING" and self.readiness["ready"]:
            if now is not None:
                self.last_clock_time = float(now)
            self.countdown_started_at = self.last_clock_time
            self.stage = "COUNTDOWN"
        return self.readiness

    def advance(self, now):
        """준비 중과 RUNNING 전환의 첫 프레임은 0초를 유지한다."""
        now = float(now)
        elapsed = max(0.0, now - self.last_clock_time)
        self.last_clock_time = now
        if self.stage == "COUNTDOWN":
            if now - self.countdown_started_at >= COUNTDOWN_DURATION_SEC:
                self.stage = "RUNNING"
                self.running_start_pending = True
                self.session.resume()
                self.playback_time = 0.0
        elif self.stage == "RUNNING" and not self.running_start_pending:
            self.playback_time = min(self.duration, self.playback_time + elapsed)

    def evaluate(self, world_landmarks):
        """READY/TPOSE_WAITING/COUNTDOWN/FINISHED에서는 평가를 호출하지 않는다."""
        if self.stage in ("READY", "TPOSE_WAITING", "COUNTDOWN", "FINISHED"):
            return self.last_result
        if self.stage == "RUNNING":
            self.running_start_pending = False
        self.last_result = self.session.update(playback_time=self.playback_time, world_landmarks=world_landmarks)
        if self.stage == "RUNNING" and self.playback_time >= self.duration:
            self.finish()
        return self.last_result

    def finish(self):
        """RAW 집계를 한 번만 보관하고 종료한다."""
        if not self.finished:
            self.summary = self.session.end()
            self.finished = True
            self.stage = "FINISHED"
        return self.summary

    def handle_key(self, key, now):
        """SPACE는 T-Pose 대기/정지/재개, R은 준비로 복귀한다."""
        previous_stage = self.stage
        self.advance(now)
        if key in (27, ord("q"), ord("Q")):
            return False
        if key in (ord("r"), ord("R")):
            self.session.reset()
            self.session.pause()
            self.playback_time = 0.0
            self.finished = False
            self.stage = "READY"
            self.summary = self.last_result = None
            self.countdown_started_at = None
            self.running_start_pending = False
            self.readiness = evaluate_readiness(None, 1, 1)
        elif key in (ord("d"), ord("D")):
            self.debug = not self.debug
        elif key == 32 and previous_stage != "COUNTDOWN":
            if self.stage == "READY":
                self.readiness = evaluate_readiness(None, 1, 1)
                self.stage = "TPOSE_WAITING"
            elif self.stage == "RUNNING":
                self.session.pause()
                self.stage = "PAUSED"
            elif self.stage == "PAUSED":
                self.session.resume()
                self.stage = "RUNNING"
        return True


def load_runtime():
    """이미 설치된 OpenCV/MediaPipe Tasks/Pillow만 읽는다. 없으면 설치 없이 오류를 알린다."""
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("Desktop dependencies unavailable in {}: {}. Use an existing environment with OpenCV, MediaPipe and Pillow; no installation/download attempted.".format(sys.executable, exc)) from exc
    return SimpleNamespace(cv2=cv2, mp=mp, vision=vision, BaseOptions=BaseOptions,
                           Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont)


class DemoRenderer:
    """두 영상 패널과 DanceSession의 반환값을 표시한다. 한국어는 로컬 Pillow 폰트를 사용한다."""

    def __init__(self, runtime, font_path=None):
        self.runtime = runtime
        self.cv2 = runtime.cv2
        font_path = Path(font_path) if font_path is not None else next((path for path in LOCAL_FONT_PATHS if path.is_file()), None)
        if font_path is None or not font_path.is_file():
            raise RuntimeError("No local Korean font found; no font download attempted")
        self.font = runtime.ImageFont.truetype(str(font_path), 28)

    def korean(self, canvas, message, xy, width=850, height=44):
        """한글 문자열을 표시 영역에만 렌더링하고 원본 메시지는 그대로 유지한다."""
        x, y = xy
        region = canvas[y:y + height, x:x + width]
        rgb = self.runtime.Image.fromarray(self.cv2.cvtColor(region, self.cv2.COLOR_BGR2RGB))
        self.runtime.ImageDraw.Draw(rgb).text((0, 0), str(message), font=self.font, fill=WHITE)
        region[:] = self.cv2.cvtColor(np.asarray(rgb), self.cv2.COLOR_RGB2BGR)

    def render(self, song, controller, result, reference_image, webcam_image, image_landmarks, reference_frame):
        """RAW Bone 상태와 서비스의 displayed_score/displayed_feedback를 1200x980 화면에 표시한다."""
        canvas = np.full((980, 1200, 3), 18, dtype=np.uint8)
        draw_text(canvas, "DANC3D  DESKTOP DEMO", (20, 30), self.cv2, scale=.75)
        self.korean(canvas, "{} / {}".format(song["title"], song["artist"]), (20, 43), width=1140)
        draw_text(canvas, "{}  Playback {:.3f}s / {:.3f}s | JPG frame {}".format(
            controller.label, controller.playback_time, controller.duration, reference_frame), (20, 99), self.cv2)
        draw_text(canvas, "REFERENCE / NORMAL PLAYBACK (mirror display)", (20, 119), self.cv2)
        draw_text(canvas, "WEBCAM USER / MIRROR DISPLAY", (620, 119), self.cv2)
        canvas[130:510, :600] = reference_panel(reference_image, self.cv2)
        canvas[130:510, 600:] = webcam_panel(webcam_image, image_landmarks, result if result is not None else {"bone_status": {}}, self.cv2)
        if controller.label in ("READY", "TPOSE_WAITING", "COUNTDOWN"):
            if controller.label == "COUNTDOWN":
                draw_text(canvas, str(controller.countdown_number), (560, 400), self.cv2, scale=3.0, thickness=5)
            else:
                message = "SPACE를 눌러 시작하세요." if controller.label == "READY" else "T자 자세를 취해주세요."
                self.korean(canvas, message, (20, 610), width=1140)
                draw_text(canvas, "Score", (950, 815), self.cv2, scale=.7)
                draw_text(canvas, "--", (980, 887), self.cv2, scale=1.9, thickness=2)
            return canvas
        if controller.finished:
            notice = "FINISHED: last evaluated frame frozen. R to restart."
        elif not result["pose_detected"]:
            notice = "Pose not detected - current evaluation unavailable"
        elif not result["evaluation_performed"]:
            notice = "Paused - current evaluation unavailable"
        elif result["aligned_reference_index"] < 0:
            notice = "Alignment unavailable - RAW score is NaN"
        else:
            notice = "Current RAW evaluation available"
        draw_text(canvas, notice, (20, 540), self.cv2, scale=.5)
        draw_text(canvas, "Overall Similarity: {:.3f}".format(result["overall_similarity"]), (20, 576), self.cv2, scale=.65)
        for index, name in enumerate(PARTS):
            draw_text(canvas, "{}: {:.3f}".format(name, result["part_similarities"][name]), (20, 615 + 28 * index), self.cv2)
        draw_text(canvas, "Priority Error: {}".format(result["priority_error_bone"]), (420, 576), self.cv2)
        draw_text(canvas, "Error bones: {} / 8 | Valid user bones: {}".format(result["error_bone_count"], result["valid_user_bone_count"]), (420, 610), self.cv2, scale=.45)
        draw_text(canvas, "Aligned row: {} | Aligned time: {:.3f}s".format(result["aligned_reference_index"], result["aligned_reference_time_sec"]), (420, 640), self.cv2, scale=.45)
        draw_text(canvas, "Timing offset: {:+.3f}s".format(result["timing_offset_sec"]), (420, 670), self.cv2)
        draw_text(canvas, "Feedback" + (" (held)" if not result["sample_recorded"] else ""), (20, 745), self.cv2, scale=.7)
        self.korean(canvas, result["displayed_feedback"], (20, 760))
        displayed_score = result["displayed_score"]
        score_text = str(displayed_score) if math.isfinite(float(displayed_score)) else "--"
        draw_text(canvas, "Score" + (" (held)" if not result["sample_recorded"] else ""), (950, 815), self.cv2, scale=.7)
        draw_text(canvas, score_text, (980, 887), self.cv2, scale=1.9, thickness=2)
        if controller.debug:
            for index, (name, _, _) in enumerate(BONES):
                draw_text(canvas, "{} {:.3f} {}".format(name, result["bone_similarities"][name], result["bone_status"][name]),
                          (900, 545 + 24 * index), self.cv2, BONE_COLORS[result["bone_status"][name]], scale=.32)
            draw_text(canvas, "RAW FinalScore (0-1): {:.4f} | Pose: {:.4f} | Timing factor: {:.4f}".format(
                result["raw_final_score"], result["pose_score"], result["score_timing_factor"]), (20, 835), self.cv2, scale=.42)
            self.korean(canvas, "RAW: " + result["raw_final_feedback"], (20, 848))
            draw_text(canvas, "Recovery: trigger={} count={} anchor={} | Torso yaw error: {:.1f} deg".format(
                result["recovery_triggered"], result["recovery_trigger_count"], result["recovery_anchor_index"],
                result["torso_yaw_error_deg"]), (20, 916), self.cv2, scale=.4)
        draw_text(canvas, "Green: OK | Red: ERROR | Gray: INVALID | Thick red: Priority Error | White: Body Joint", (20, 945), self.cv2, scale=.4)
        draw_text(canvas, "SPACE: Start/Pause/Resume | R: Reset | D: Debug | Q/ESC: Quit", (20, 970), self.cv2, scale=.45)
        return canvas


def print_summary(song, summary):
    """Session.end의 RAW 집계만 터미널에 출력한다. 평가 불가는 0점으로 바꾸지 않는다."""
    print("\n{} / {} [{}]".format(song["title"], song["artist"], summary["song_id"]))
    print("평가 전체 샘플: {} | 유효 RAW 샘플: {}".format(summary["evaluation_sample_count"], summary["valid_sample_count"]))
    mean = summary["mean_raw_final_score"]
    print("평균 RAW FinalScore (0~1): " + ("{:.4f}".format(mean) if math.isfinite(mean) else "NaN (평가 불가)"))
    print("Recovery 발생 횟수: {}".format(summary["recovery_trigger_count"]))


def run_desktop(song, camera_index, runtime, data_dir=DATA_DIR, clock=None):
    """사용자가 명시적으로 실행한 데모에서만 카메라를 열고 입력/그리기/세션 조작을 수행한다."""
    ensure_pose_model(MODEL_PATH)
    clock = time.perf_counter if clock is None else clock
    cv2 = runtime.cv2
    session = DanceSession(song["id"], data_dir=data_dir)
    references = ReferenceImages(song, session.reference, data_dir)
    renderer = DemoRenderer(runtime)
    print("Reference JPG: {}/{} | missing: {} | excluded non-NPZ: {}".format(
        len(references.paths), len(session.reference.frame_indices), len(references.missing_frame_indices), len(references.excluded_frame_indices)))
    controller = PlaybackController(session, clock())
    timestamps = CaptureTimestamp(clock())
    camera = None
    summary_printed = False
    frame = coords = result = None
    try:
        options = runtime.vision.PoseLandmarkerOptions(
            base_options=runtime.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=runtime.vision.RunningMode.VIDEO, num_poses=1, output_segmentation_masks=False)
        with runtime.vision.PoseLandmarker.create_from_options(options) as detector:
            camera = cv2.VideoCapture(camera_index)
            if not camera.isOpened():
                raise RuntimeError("Webcam {} could not open. Check camera connection/permission.".format(camera_index))
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, 1200, 980)
            while True:
                if not controller.finished:
                    ok, frame = camera.read()
                    if not ok:
                        raise RuntimeError("Webcam frame capture failed")
                    captured_at = clock()
                    controller.advance(captured_at)
                    # 추론에는 flip 전 원본 RGB를 전달하고 VIDEO timestamp는 reset하지 않는다.
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = runtime.mp.Image(image_format=runtime.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                    pose = detector.detect_for_video(image, timestamps.next(captured_at))
                    world, coords = extract_pose_landmarks(pose)
                    if controller.label == "TPOSE_WAITING":
                        poses = getattr(pose, "pose_landmarks", None)
                        controller.update_readiness(poses[0] if poses else None, frame.shape[1], frame.shape[0], now=clock())
                    result = controller.evaluate(world)
                _, frame_index, path = references.at_playback(controller.playback_time)
                canvas = renderer.render(song, controller, result, read_jpg(path, cv2), frame, coords, frame_index)
                cv2.imshow(WINDOW_NAME, canvas)
                if controller.finished and not summary_printed:
                    print_summary(song, controller.summary)
                    summary_printed = True
                key = cv2.waitKey(30 if controller.finished else 1) & 0xFF
                if not controller.handle_key(key, clock()) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if not controller.finished:
                    summary_printed = False
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
        summary = controller.finish()
        if not summary_printed:
            print_summary(song, summary)


def main(argv=None, data_dir=DATA_DIR):
    """로컬 곡 선택 CLI. 목록 출력은 카메라/MediaPipe/GUI 의존성을 불러오지 않는다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-songs", action="store_true", help="List local songs and exit")
    parser.add_argument("--song", metavar="SONG_ID", help="Song id from data/songs.json")
    parser.add_argument("--camera", type=int, default=0, metavar="CAMERA_INDEX", help="Webcam index (default: 0)")
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    try:
        songs = load_songs(data_dir)
        if args.list_songs or args.song is None:
            print_songs(songs)
        if args.list_songs:
            return 0
        song = select_song(songs, args.song)
        ensure_pose_model(MODEL_PATH)
        run_desktop(song, args.camera, load_runtime(), data_dir=data_dir)
        return 0
    except (OSError, ValueError, RuntimeError, EOFError) as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
