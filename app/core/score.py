"""부위 유사도와 Timing Offset으로 RAW 점수 및 독립적인 표시용 EMA를 계산한다."""

import math

import numpy as np

from .pose_features import PARTS
from .pose_similarity import valid_mean

POSE_MEAN_WEIGHT = 0.70
POSE_WORST_WEIGHT = 0.30
POSE_TIMING_GATE = 0.80
TIMING_FREE_SEC = 0.30
TIMING_MAX_PENALTY_SEC = 1.00
TIMING_MIN_FACTOR = 0.70
SCORE_EMA_ALPHA = 0.20


def _real_scalar(value, name):
    """실수 스칼라로 변환하며 비유한값은 유지하고 입력 오류는 ValueError로 알린다."""
    try:
        if value is None or np.iscomplexobj(value):
            raise ValueError("complex values are not supported")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 0:
            raise ValueError("expected a scalar")
        return float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a real scalar".format(name)) from exc


def final_score_metrics(overall, part_similarities, timing_offset_sec):
    """RAW MeanPart/WorstTwo/PoseScore/TimingFactor/FinalScore dict를 반환한다.

    overall은 기존 전체 DIR8 평균으로 score_old_overall 진단값에만 보관한다.
    part_similarities는 pose_similarity.compare()가 반환한 PARTS 순서의 실수 (4,)
    배열이다. NaN/Inf 부위는 제외하고 유효 부위를 오름차순 정렬하여 평균 및
    최저 두 부위 평균을 구한다. 하나만 유효하면 두 평균 모두 해당 값이다.
    유효 부위가 없으면 PoseScore/FinalScore/display_score는 NaN이다.

    timing_offset_sec는 aligned_reference_time_sec-playback_time_sec 실수 초다.
    비유한 offset이면 late_sec/TimingFactor는 NaN이고, PoseScore >=0.8인 경우
    FinalScore도 NaN이다. 그 미만에는 RAW PoseScore를 유지한다. 점수는 [0, 1]
    범위이며 display_score는 Python round(FinalScore*100) int 또는 NaN이다.
    shape/실수 변환 오류는 ValueError이며 입력과 기존 core 계산은 변경하지 않는다.
    """
    overall = _real_scalar(overall, "overall")
    timing_offset_sec = _real_scalar(timing_offset_sec, "timing_offset_sec")
    try:
        if np.iscomplexobj(part_similarities):
            raise ValueError("complex values are not supported")
        parts = np.asarray(part_similarities, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("part_similarities must contain real numbers") from exc
    if parts.shape != (len(PARTS),):
        raise ValueError("part_similarities must have shape (4,)")
    valid_parts = sorted(float(value) for value in parts if math.isfinite(float(value)))
    mean_part = valid_mean(valid_parts)
    worst_two = valid_mean(valid_parts[:2])
    weighted = POSE_MEAN_WEIGHT * mean_part + POSE_WORST_WEIGHT * worst_two
    pose_score = max(0.0, min(1.0, weighted)) if math.isfinite(weighted) else float("nan")
    late_sec = max(0.0, -timing_offset_sec) if math.isfinite(timing_offset_sec) else float("nan")
    if math.isfinite(late_sec):
        progress = (late_sec - TIMING_FREE_SEC) / (TIMING_MAX_PENALTY_SEC - TIMING_FREE_SEC)
        timing_factor = max(TIMING_MIN_FACTOR, min(1.0, 1.0 - progress * (1.0 - TIMING_MIN_FACTOR)))
    else:
        timing_factor = float("nan")
    final_score = pose_score * timing_factor if pose_score >= POSE_TIMING_GATE else pose_score
    display_score = round(final_score * 100) if math.isfinite(final_score) else float("nan")
    return dict(score_old_overall=overall, score_mean_part=mean_part, score_worst_two=worst_two,
                pose_score=pose_score, score_late_sec=late_sec, score_timing_factor=timing_factor,
                final_score=final_score, display_score=display_score)


class ScoreEMAStabilizer:
    """RAW FinalScore [0, 1]을 유지하고 100점 척도의 표시 점수만 EMA로 안정화한다.

    alpha=0.20이며 첫 유효값은 그대로 초기값으로 사용한다. NaN/Inf는 이전 값을
    보존하고 아직 유효값이 없으면 표시는 NaN이다. 새 세션은 새 인스턴스로
    시작한다. smoothed_score는 미초기화 None 또는 float의 100점 척도이다.
    """

    def __init__(self):
        self.smoothed_score = None

    def update(self, raw_final_score):
        """RAW 실수 스칼라로 표시 상태만 갱신한다. 반환은 None, 입력 오류는 ValueError이다.

        기존 UIStabilizer와 같이 RAW를 다시 clip하지 않고 유한값에만 EMA를 적용한다.
        """
        raw_final_score = _real_scalar(raw_final_score, "raw_final_score")
        if math.isfinite(raw_final_score):
            raw_score_100 = raw_final_score * 100.0
            self.smoothed_score = (raw_score_100 if self.smoothed_score is None else
                SCORE_EMA_ALPHA * raw_score_100 + (1.0 - SCORE_EMA_ALPHA) * self.smoothed_score)

    def snapshot(self, raw_final_score):
        """RAW값과 round한 표시 int(미초기화 NaN)를 새 dict로 반환하며 상태를 바꾸지 않는다."""
        return dict(raw_final_score=raw_final_score,
                    displayed_score=round(self.smoothed_score) if self.smoothed_score is not None else float("nan"))
