"""카메라·화면·네트워크와 독립적인 세션별 실시간 Pose 평가 파이프라인."""

from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from app.core.pose_features import BONES, PARTS, extract_bone_directions, validate_world_landmarks
from app.core.pose_similarity import compare, priority_error, status
from app.core.temporal_alignment import LocalRecoveryDTW, reference_times_from_timestamps
from app.core.feedback import FeedbackStabilizer, final_feedback_metrics, pose_feedback
from app.core.score import ScoreEMAStabilizer, final_score_metrics


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
# 전처리 검증과 같은 저장 float32 허용 오차이며 평가의 휴리스틱은 아니다.
REFERENCE_VECTOR_ATOL = 1e-6
REFERENCE_VECTOR_RTOL = 1e-5
REQUIRED_REFERENCE_ARRAYS = (
    "frame_indices", "timestamps_ms", "raw_world", "normalized_bone_vectors", "bone_valid_mask",
)


@dataclass(frozen=True)
class _ReferenceData:
    """한 번 로딩한 읽기 전용 Reference 배열. times만 첫 프레임 기준 초 단위이다."""

    frame_indices: np.ndarray
    timestamps_ms: np.ndarray
    raw_world: np.ndarray
    normalized_bone_vectors: np.ndarray
    bone_valid_mask: np.ndarray
    times: np.ndarray


def _read_json(path):
    """로컬 JSON을 읽고 손상된 JSON은 경로를 포함한 ValueError로 알린다."""
    with path.open(encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON: {}".format(path)) from exc


def _data_path(data_dir, relative_path, field):
    """catalog의 data/ 기준 상대 경로를 검사하여 로컬 파일 경로를 반환한다."""
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("{} must be a nonempty relative path".format(field))
    path = Path(relative_path)
    if path.is_absolute():
        raise ValueError("{} must be relative to data_dir".format(field))
    resolved = (data_dir / path).resolve()
    try:
        resolved.relative_to(data_dir)
    except ValueError as exc:
        raise ValueError("{} must stay inside data_dir".format(field)) from exc
    return resolved


def _load_reference(song_id, data_dir):
    """song_id의 catalog·metadata·NPZ를 읽고 shape/시간/DIR8 순서를 검증한다."""
    songs = _read_json(data_dir / "songs.json")
    if not isinstance(songs, list) or not all(isinstance(song, dict) for song in songs):
        raise ValueError("songs.json must contain a list of song objects")
    matches = [song for song in songs if song.get("id") == song_id]
    if not matches:
        raise ValueError("Unknown song_id: {}".format(song_id))
    if len(matches) != 1:
        raise ValueError("Duplicate song_id in songs.json: {}".format(song_id))
    song = matches[0]
    npz_path = _data_path(data_dir, song.get("reference_npz"), "reference_npz")
    metadata_path = _data_path(data_dir, song.get("reference_metadata"), "reference_metadata")
    metadata = _read_json(metadata_path)
    expected_mapping = {name: dict(index=index, parent=parent, child=child)
                        for index, (name, parent, child) in enumerate(BONES)}
    if not isinstance(metadata, dict) or metadata.get("sequence") != song_id:
        raise ValueError("Reference metadata sequence does not match song_id: {}".format(metadata_path))
    if metadata.get("bone_mapping") != expected_mapping:
        raise ValueError("Reference bone_mapping does not match DIR8 order: {}".format(metadata_path))
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = set(REQUIRED_REFERENCE_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError("Missing NPZ arrays {}: {}".format(sorted(missing), npz_path))
        arrays = {name: archive[name].copy() for name in REQUIRED_REFERENCE_ARRAYS}
    frames, timestamps = arrays["frame_indices"], arrays["timestamps_ms"]
    if (frames.ndim != 1 or len(frames) < 2 or frames.dtype.kind not in "iu"
            or (frames < 0).any() or not (frames[1:] > frames[:-1]).all()):
        raise ValueError("frame_indices must be strictly increasing nonnegative integers, N >= 2")
    count = len(frames)
    if timestamps.shape != (count,) or timestamps.dtype.kind not in "iuf":
        raise ValueError("timestamps_ms must be a real numeric (N,) array")
    times = reference_times_from_timestamps(timestamps)
    for name, shape in (("raw_world", (count, 33, 3)),
                        ("normalized_bone_vectors", (count, len(BONES), 3))):
        if arrays[name].shape != shape or arrays[name].dtype.kind not in "iuf":
            raise ValueError("{} must be a real numeric {} array".format(name, shape))
        arrays[name] = arrays[name].astype(np.float64)
    mask = arrays["bone_valid_mask"]
    if (mask.shape != (count, len(BONES)) or mask.dtype.kind not in "biuf"
            or not np.isin(mask, [0, 1]).all()):
        raise ValueError("bone_valid_mask must be bool/0/1 with shape (N, 8)")
    arrays["bone_valid_mask"] = mask.astype(bool)
    # 파일 로딩 시에만 기존 특징 추출을 재사용하여 잘못 저장된 Bone 순서도 확인한다.
    extracted = [extract_bone_directions(world) for world in arrays["raw_world"]]
    expected_vectors = np.stack([vectors for vectors, _ in extracted])
    expected_mask = np.stack([valid for _, valid in extracted])
    if not np.array_equal(arrays["bone_valid_mask"], expected_mask):
        raise ValueError("bone_valid_mask is inconsistent with raw_world")
    vectors = arrays["normalized_bone_vectors"]
    if (not np.isnan(vectors[~expected_mask]).all()
            or not np.allclose(vectors[expected_mask], expected_vectors[expected_mask],
                               rtol=REFERENCE_VECTOR_RTOL, atol=REFERENCE_VECTOR_ATOL)):
        raise ValueError("normalized_bone_vectors do not match raw_world DIR8 order/directions")
    arrays["times"] = times
    for array in arrays.values():
        array.setflags(write=False)
    return _ReferenceData(**arrays)


def _playback_seconds(playback_time):
    """초 단위 유한 비음수 실수 스칼라를 상태 변경 전에 검증한다."""
    try:
        if playback_time is None or np.iscomplexobj(playback_time):
            raise ValueError("expected a real scalar")
        value = np.asarray(playback_time, dtype=np.float64)
        if value.ndim != 0 or not np.isfinite(value) or value < 0:
            raise ValueError("expected finite nonnegative seconds")
        return float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("playback_time must be a finite nonnegative scalar in seconds") from exc


class DanceSession:
    """song_id로 생성되는 한 번의 연습. 생성/Reset 직후 state는 'running'이다.

    data_dir는 songs.json과 references/가 있는 로컬 디렉터리로 기본값은 프로젝트
    data/이다. 파일 누락은 FileNotFoundError, catalog/metadata/배열 계약 오류는
    ValueError이다. 검증된 Reference는 생성 시 한 번 읽고 readonly로 보관한다.

    각 세션의 alignment(LocalRecoveryDTW), score_stabilizer, feedback_stabilizer는
    독립적이며 update마다 재생성하지 않는다. state는 running/paused/ended이고
    외부에서 직접 변경하지 않는다. 시간 증가는 호출자가 관리하고 자동 종료나
    seek에 따른 자동 리셋은 없다. 새 연습은 명시적으로 reset()한다.
    """

    def __init__(self, song_id, data_dir=None):
        if not isinstance(song_id, str) or not song_id:
            raise ValueError("song_id must be a nonempty string")
        self.song_id = song_id
        self.data_dir = Path(data_dir if data_dir is not None else DEFAULT_DATA_DIR).resolve()
        self.reference = _load_reference(song_id, self.data_dir)
        self.reset()

    @property
    def last_result(self):
        """가장 최근 running update 결과의 독립적인 사본 또는 초기 None을 반환한다."""
        return deepcopy(self._last_result)

    def _empty_recovery_metrics(self):
        """새 관측이 없는 프레임에서 원본 run()의 무효 진단 규칙을 적용한다."""
        metrics = dict(self.alignment.metrics)
        for key in metrics:
            if key == "recovery_trigger_count":
                continue
            metrics[key] = (-1 if key.endswith("_index") else
                            0 if key in ("recovery_candidate", "recovery_triggered") else float("nan"))
        return metrics

    def _evaluate(self, playback, world, user_vectors, user_mask, aligned, recovery_metrics):
        """선택된 row만으로 순수 core 계산을 조합하며 세션의 누적 상태를 바꾸지 않는다."""
        if aligned is None:
            reference_world = np.full((33, 3), np.nan)
            reference_vectors = np.full((len(BONES), 3), np.nan)
            reference_mask = np.zeros(len(BONES), dtype=bool)
            aligned_time = None
        else:
            reference_world = self.reference.raw_world[aligned]
            reference_vectors = self.reference.normalized_bone_vectors[aligned]
            reference_mask = self.reference.bone_valid_mask[aligned]
            aligned_time = float(self.reference.times[aligned])
        values, overall, parts = compare(reference_vectors, user_vectors, reference_mask, user_mask)
        bone_states = {name: status(value) for (name, _, _), value in zip(BONES, values)}
        priority = priority_error(values)
        metrics, candidate = pose_feedback(reference_world, world, values)
        metrics.pop("priority_feedback")  # priority_pose_feedback를 유일한 RAW Pose 문장으로 사용한다.
        feedback_metrics = final_feedback_metrics(candidate, overall, aligned_time, playback, aligned)
        feedback_metrics.pop("aligned_overall_similarity")
        feedback_metrics["raw_final_feedback"] = feedback_metrics.pop("final_feedback")
        metrics.update(feedback_metrics)
        score_metrics = final_score_metrics(overall, parts, metrics["timing_offset_sec"])
        score_metrics.pop("score_old_overall")
        score_metrics["raw_final_score"] = score_metrics.pop("final_score")
        metrics.update(score_metrics)
        metrics.update(recovery_metrics)
        return dict(metrics, song_id=self.song_id, state=self.state, playback_time_sec=playback,
                    aligned_reference_frame_index=int(self.reference.frame_indices[aligned]) if aligned is not None else -1,
                    pose_detected=bool(user_mask.any()), valid_user_bone_count=int(user_mask.sum()),
                    user_dir8=user_vectors.copy(), user_bone_valid_mask=user_mask.copy(),
                    bone_similarities={name: float(value) for (name, _, _), value in zip(BONES, values)},
                    overall_similarity=overall,
                    part_similarities={name: value for name, value in zip(PARTS, parts)},
                    bone_status=bone_states, priority_error_bone=BONES[priority][0] if priority is not None else "NONE",
                    priority_error_similarity=float(values[priority]) if priority is not None else float("nan"),
                    error_bone_count=sum(value == "ERROR" for value in bone_states.values()))

    def update(self, playback_time, world_landmarks=None):
        """외부에서 추정한 World 한 프레임을 평가하여 RAW/표시/Recovery dict를 반환한다.

        playback_time은 초 단위 유한 비음수 스칼라, world_landmarks는 원본 좌표
        (33, 3) 또는 미검출 None이다. shape/실수 변환 오류는 상태 변경 전 ValueError다.
        유효 Bone이 없으면 DTW를 호출하지 않고 counter만 0으로 끊으며 과거 정렬을
        재사용하지 않는다. RAW Score는 NaN이지만 표시 EMA는 이전 값을 유지할 수 있다.

        paused에서는 DTW/안정화/통계/last_result를 유지하며 현재 평가를 수행하지 않는다.
        반환 정렬 index는 -1, RAW Score는 NaN, evaluation_performed/sample_recorded는
        False이다. running은 evaluation_performed=True이며 유한 RAW Score에만
        sample_recorded=True이다. ended에서는 RuntimeError, reset으로 새 연습을 시작한다.

        bone_similarities/bone_status는 BONES 이름의 dict, part_similarities는 PARTS
        이름의 dict이다. user_dir8은 float64 (8, 3), user_bone_valid_mask는 bool (8,)이다.
        aligned_reference_index는 NPZ row, aligned_reference_frame_index는 원본 프레임
        번호다. 정렬 불가는 -1/NaN이며 내부 NaN은 JSON용으로 변환하지 않는다.
        display_score는 RAW 반올림값, displayed_score는 EMA 표시값이며 RAW 통계와 분리한다.
        """
        if self.state == "ended":
            raise RuntimeError("Cannot update an ended session; call reset() for a new practice")
        playback = _playback_seconds(playback_time)
        world = (np.full((33, 3), np.nan) if world_landmarks is None
                 else validate_world_landmarks(world_landmarks))
        user_vectors, user_mask = extract_bone_directions(world)
        running = self.state == "running"
        aligned = None
        if running:
            if user_mask.any():
                aligned = self.alignment.update(user_vectors, user_mask, playback)
                recovery_metrics = dict(self.alignment.metrics)
            else:
                # 누적 DTW 경로는 유지하지만 Recovery의 연속 증거는 끊는다.
                self.alignment.counter = 0
                recovery_metrics = self._empty_recovery_metrics()
            self.last_aligned_index = aligned
        else:
            recovery_metrics = self._empty_recovery_metrics()
        result = self._evaluate(playback, world, user_vectors, user_mask, aligned, recovery_metrics)
        if running:
            self.score_stabilizer.update(result["raw_final_score"])
            self.feedback_stabilizer.update(result["raw_final_feedback"], playback)
            self.evaluation_sample_count += 1
        result.update(self.score_stabilizer.snapshot(result["raw_final_score"]))
        result.update(self.feedback_stabilizer.snapshot(result["raw_final_feedback"]))
        result["evaluation_performed"] = running
        result["sample_recorded"] = running and math.isfinite(result["raw_final_score"])
        if result["sample_recorded"]:
            self.valid_sample_count += 1
            self._raw_score_sum += result["raw_final_score"]
        if running:
            self._last_result = deepcopy(result)
        return result

    def pause(self):
        """running→paused로 전이하고 누적 상태를 보존한다. 재호출은 무해하며 ended는 RuntimeError다."""
        if self.state == "ended":
            raise RuntimeError("Cannot pause an ended session")
        self.state = "paused"

    def resume(self):
        """paused→running으로 전이하고 DTW/EMA/확인 상태를 이어간다. ended는 RuntimeError다."""
        if self.state == "ended":
            raise RuntimeError("Cannot resume an ended session; call reset()")
        self.state = "running"

    def reset(self):
        """Reference를 재사용하고 DTW/Recovery/표시/통계를 초기화하여 새 running 연습을 시작한다."""
        self.alignment = LocalRecoveryDTW(self.reference.normalized_bone_vectors,
                                          self.reference.bone_valid_mask, self.reference.times)
        self.score_stabilizer = ScoreEMAStabilizer()
        self.feedback_stabilizer = FeedbackStabilizer(0.0)
        self.last_aligned_index = None
        self._last_result = None
        self.evaluation_sample_count = 0
        self.valid_sample_count = 0
        self._raw_score_sum = 0.0
        self.state = "running"

    def end(self):
        """running/paused→ended로 전이하고 최종 결과 dict를 반환한다. 재호출은 같은 요약이다.

        평균은 초기 집계 정책인 유효 RAW FinalScore의 단순 표본 평균(0~1)이다.
        NaN 표본은 제외하고 유효 표본이 없으면 평균은 NaN, evaluation_available은
        False이다. displayed_score는 마지막 EMA의 100점 표시값이며 평균에 섞지 않는다.
        시간 가중 평균이 필요하면 표본 집계 책임만 확장할 수 있다. 종료 후 update,
        pause/resume은 RuntimeError이며 reset만 새 연습을 시작한다. 저장/DB 동작은 없다.
        """
        self.state = "ended"
        return dict(song_id=self.song_id, state=self.state, valid_sample_count=self.valid_sample_count,
                    evaluation_sample_count=self.evaluation_sample_count,
                    mean_raw_final_score=(self._raw_score_sum / self.valid_sample_count
                                          if self.valid_sample_count else float("nan")),
                    evaluation_available=self.valid_sample_count > 0,
                    displayed_score=self.score_stabilizer.snapshot(float("nan"))["displayed_score"],
                    recovery_trigger_count=self.alignment.trigger_count,
                    aggregation_policy="unweighted_raw_sample_mean")
