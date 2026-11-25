"""원본 World 좌표의 RAW 교정·Timing 피드백과 표시 메시지 안정화를 계산한다.

교정 각도는 방향 설명에만 사용한다. RAW 피드백 dict와 표시 상태는 별도로
반환하며, 이 모듈은 점수를 계산하거나 카메라·UI를 실행하지 않는다.
"""

import math

import numpy as np

from .pose_features import (
    BONES, BONE_LENGTH_EPSILON, angle_deg, angle_diff_deg, pelvis_center, validate_world_landmarks,
)
from .pose_similarity import BONE_ERROR_THRESHOLD

ARM_DIRECTION_THRESHOLD_DEG = 15.0
ELBOW_ANGLE_THRESHOLD_DEG = 15.0
KNEE_ANGLE_THRESHOLD_DEG = 15.0
LEG_OUTWARD_THRESHOLD_DEG = 15.0
TORSO_ROTATION_THRESHOLD_DEG = 15.0
POSE_GATE = 0.8
TIMING_THRESHOLD_SEC = 0.3
FEEDBACK_CONFIRM_FRAMES = 5
FEEDBACK_MIN_HOLD_SEC = 0.50

# 관절과 DIR8 index를 단일 BONES 정의에서 가져와 신체 좌우를 유지한다.
ARM_JOINTS = tuple(
    (side, BONES[bone][1], BONES[bone][2], BONES[bone + 1][2], bone)
    for side, bone in (("left", 0), ("right", 2))
)
LEG_JOINTS = tuple(
    (side, BONES[bone][1], BONES[bone][2], BONES[bone + 1][2], bone)
    for side, bone in (("left", 4), ("right", 6))
)


def _real_scalar(value, name):
    """실수 스칼라로 변환하며 shape/변환 오류는 ValueError로 알린다."""
    try:
        if value is None or np.iscomplexobj(value):
            raise ValueError("complex values are not supported")
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 0:
            raise ValueError("expected a scalar")
        return float(array)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must be a real scalar".format(name)) from exc


def _bone_similarities(values):
    """DIR8 순서의 실수 cosine (8,)를 읽으며 비유한값은 무효 gate로 남긴다."""
    try:
        if np.iscomplexobj(values):
            raise ValueError("complex values are not supported")
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bone_similarities must contain real numbers") from exc
    if values.shape != (len(BONES),):
        raise ValueError("bone_similarities must have shape (8,)")
    return values


def _bone_pair_has_error(values, bone):
    """부위의 두 Bone 중 유한한 RAW 오류가 있을 때만 각도 교정을 허용한다."""
    return any(math.isfinite(float(values[index])) and values[index] < BONE_ERROR_THRESHOLD
               for index in (bone, bone + 1))


def arm_measurements(world, shoulder, elbow, wrist):
    """World (33, 3)의 팔꿈치 각도와 팔 높이를 float 도 단위 tuple로 반환한다.

    관절 index는 BONES의 신체 좌우 index를 사용한다. 비유한/퇴화 벡터는
    해당 측정값 NaN이며 잘못된 World shape/실수 변환은 ValueError이다.
    팔 높이는 전역 Y축 대신 hip-to-shoulder 방향에 대한 90-angle이다.
    """
    world = validate_world_landmarks(world)
    elbow_angle = angle_deg(world[shoulder] - world[elbow], world[wrist] - world[elbow])
    torso_up = (world[11] + world[12] - world[23] - world[24]) / 2
    elevation = 90.0 - angle_deg(world[elbow] - world[shoulder], torso_up)
    return elbow_angle, elevation


def outward_measure_deg(world, hip, knee):
    """World (33, 3)의 대퇴 벌림 정도를 Python float 도 단위로 반환한다.

    pelvis-to-hip과 hip-to-knee 각도의 90-angle이며 원본 축을 유지한다.
    비유한/퇴화 측정은 NaN, World shape/실수 변환 오류는 ValueError이다.
    """
    world = validate_world_landmarks(world)
    hip_center = pelvis_center(world)
    return 90.0 - angle_deg(world[knee] - world[hip], world[hip] - hip_center)


def torso_rotation_deg(world):
    """World (33, 3) 어깨 x-z의 기존 unsigned 회전각을 float 도 단위로 반환한다.

    진단 필드 호환용이며 교정 방향/점수에 사용하지 않는다. 비유한 어깨나
    x-z 길이 <=1e-8은 NaN, World shape/실수 변환 오류는 ValueError이다.
    """
    world = validate_world_landmarks(world)
    shoulder = world[12] - world[11]
    if not np.isfinite(shoulder).all():
        return float("nan")
    x, z = float(shoulder[0]), float(shoulder[2])
    if math.hypot(x, z) <= BONE_LENGTH_EPSILON:
        return float("nan")
    return math.degrees(math.atan2(abs(z), abs(x)))


def torso_yaw_deg(world):
    """World (33, 3)의 신체 좌우 signed yaw를 float 도 단위로 반환한다.

    atan2(-z, -x)를 유지하며 양수는 자신의 왼쪽 회전이다. 표시 mirror는
    적용하지 않는다. 비유한/퇴화 측정은 NaN, World 입력 오류는 ValueError이다.
    """
    world = validate_world_landmarks(world)
    shoulder = world[12] - world[11]
    if not np.isfinite(shoulder).all():
        return float("nan")
    x, z = float(shoulder[0]), float(shoulder[2])
    if math.hypot(x, z) <= BONE_LENGTH_EPSILON:
        return float("nan")
    return math.degrees(math.atan2(-z, -x))


def feedback_action(error, threshold, gate, negative_action, positive_action):
    """유한한 도 단위 오류가 gate와 threshold 이상이면 부호별 action을 반환한다.

    gate가 False이거나 비유한/threshold 미만 오류는 'NONE'이다.
    error와 threshold는 실수 스칼라이며 변환 오류는 ValueError이다.
    """
    error = _real_scalar(error, "error")
    threshold = _real_scalar(threshold, "threshold")
    if not gate or not math.isfinite(error) or abs(error) < threshold:
        return "NONE"
    return negative_action if error < 0 else positive_action


def aggregate_feedback(kind, sides):
    """좌우 action dict를 기존 한국어 메시지의 후보 dict 또는 None으로 통합한다.

    kind는 arm_direction/elbow_angle/leg_direction/knee_angle이다.
    sides는 side/action/signed_error/reference/user 및 bone 또는 bones를 갖는
    dict 목록이다. 같은 action은 양측을 합치고, 다른 action은 절대 오류가
    큰 쪽을 선택한다. 동점은 입력 목록의 첫 후보이다. 후보가 없으면 None,
    잘못된 kind는 ValueError, 필수 dict 필드가 없으면 KeyError이다.
    """
    if kind not in ("arm_direction", "elbow_angle", "leg_direction", "knee_angle"):
        raise ValueError("unsupported feedback kind")
    active = [entry for entry in sides if entry["action"] != "NONE"]
    if not active:
        return None
    representative = max(active, key=lambda entry: abs(entry["signed_error"]))
    both = len(active) == 2 and active[0]["action"] == active[1]["action"]
    side = "both" if both else representative["side"]
    action = representative["action"]
    if kind == "arm_direction":
        subject = {"both": "팔을", "left": "왼팔을", "right": "오른팔을"}[side]
        verb = "올리세요" if action == "RAISE" else "내리세요"
    elif kind == "leg_direction":
        subject = {"both": "다리를", "left": "왼쪽 다리를", "right": "오른쪽 다리를"}[side]
        verb = "더 밖으로 보내세요" if action == "OUT" else "몸쪽으로 가져오세요"
    elif kind == "knee_angle":
        subject = {"both": "무릎을", "left": "왼쪽 무릎을", "right": "오른쪽 무릎을"}[side]
        verb = "펴세요" if action == "EXTEND" else "굽히세요"
    else:
        subject = {"both": "팔꿈치를", "left": "왼쪽 팔꿈치를", "right": "오른쪽 팔꿈치를"}[side]
        verb = "펴세요" if action == "EXTEND" else "굽히세요"
    chosen = active if both else [representative]
    bones = [bone for entry in chosen
             for bone in (entry["bones"] if "bones" in entry else [entry["bone"]])]
    message = "{} {}".format(subject, verb) if kind == "leg_direction" else "{} 더 {}".format(subject, verb)
    return dict(text=message, type=kind, side=side,
                error_deg=abs(representative["signed_error"]), signed_error=representative["signed_error"],
                reference=representative["reference"], user=representative["user"],
                representative_side=representative["side"], bones=bones)


def knee_feedback(reference_world, user_world, bone_similarities):
    """두 World (33, 3)와 RAW cosine (8,)에서 무릎 (metrics, candidate)을 반환한다.

    candidate는 dict 또는 None이며 error=user-reference이다. DIR8 오류 gate
    밖에서는 교정하지 않는다. 비유한 측정은 NaN/'NONE', 입력 오류는 ValueError이다.
    """
    reference = validate_world_landmarks(reference_world)
    current = validate_world_landmarks(user_world)
    values = _bone_similarities(bone_similarities)
    metrics, knees = {}, []
    for side, hip, knee, ankle, bone in LEG_JOINTS:
        ref_angle = angle_deg(reference[hip] - reference[knee], reference[ankle] - reference[knee])
        usr_angle = angle_deg(current[hip] - current[knee], current[ankle] - current[knee])
        error = usr_angle - ref_angle
        action = feedback_action(error, KNEE_ANGLE_THRESHOLD_DEG, _bone_pair_has_error(values, bone), "EXTEND", "BEND")
        metrics.update({side + "_knee_reference_deg": ref_angle,
                        side + "_knee_user_deg": usr_angle,
                        side + "_knee_error_deg": error, side + "_knee_action": action})
        knees.append(dict(side=side, action=action, signed_error=error,
                          reference=ref_angle, user=usr_angle, bones=[bone, bone + 1]))
    return metrics, aggregate_feedback("knee_angle", knees)


def leg_direction_feedback(reference_world, user_world, bone_similarities):
    """두 World (33, 3)와 RAW cosine (8,)에서 다리 방향 (metrics, candidate)을 반환한다.

    error=user-reference의 음수는 OUT, 양수는 IN이다. 후보는 dict 또는 None이다.
    비유한 측정은 NaN/'NONE', World 및 cosine 입력 오류는 ValueError이다.
    """
    reference_world = validate_world_landmarks(reference_world)
    user_world = validate_world_landmarks(user_world)
    values = _bone_similarities(bone_similarities)
    metrics, legs = {}, []
    for side, hip, knee, ankle, bone in LEG_JOINTS:
        reference = outward_measure_deg(reference_world, hip, knee)
        current = outward_measure_deg(user_world, hip, knee)
        error = current - reference
        action = feedback_action(error, LEG_OUTWARD_THRESHOLD_DEG, _bone_pair_has_error(values, bone), "OUT", "IN")
        metrics.update({side + "_leg_outward_error_deg": error, side + "_leg_action": action,
                        side + "_reference_outward_measure": reference,
                        side + "_user_outward_measure": current})
        legs.append(dict(side=side, action=action, signed_error=error,
                         reference=reference, user=current, bones=[bone]))
    return metrics, aggregate_feedback("leg_direction", legs)


def torso_feedback(reference_world, user_world):
    """두 World (33, 3)에서 몸통 (metrics, candidate)을 반환한다.

    error는 wrap된 reference yaw-user yaw이며 양수는 TURN_LEFT이다.
    DIR8 gate를 적용하지 않는다. 후보는 dict 또는 None이고 비유한 측정은
    NaN/'NONE'/오류 flag 0이다. World 입력 오류는 ValueError이다.
    """
    reference_world = validate_world_landmarks(reference_world)
    user_world = validate_world_landmarks(user_world)
    reference = torso_yaw_deg(reference_world)
    current = torso_yaw_deg(user_world)
    error = angle_diff_deg(reference, current)
    action = feedback_action(error, TORSO_ROTATION_THRESHOLD_DEG, True, "TURN_RIGHT", "TURN_LEFT")
    # 기존 unsigned 진단 필드는 user-reference이며 실제 교정은 signed yaw로 판단한다.
    legacy_reference = torso_rotation_deg(reference_world)
    legacy_current = torso_rotation_deg(user_world)
    metrics = dict(reference_torso_rotation_deg=legacy_reference,
                   user_torso_rotation_deg=legacy_current,
                   torso_rotation_error_deg=legacy_current - legacy_reference,
                   reference_torso_yaw_deg=reference, user_torso_yaw_deg=current,
                   torso_yaw_error_deg=error, torso_rotation_action=action,
                   torso_rotation_error_flag=int(action != "NONE"))
    candidate = None
    if action != "NONE":
        candidate = dict(text="몸을 더 왼쪽으로 돌리세요" if action == "TURN_LEFT" else "몸을 더 오른쪽으로 돌리세요",
                         type="torso_rotation", side="center", error_deg=abs(error), signed_error=error,
                         reference=reference, user=current, representative_side="center", bones=[])
    return metrics, candidate


def pose_feedback(reference_world, user_world, bone_similarities):
    """두 World (33, 3)와 RAW cosine (8,)에서 (metrics, priority candidate)을 반환한다.

    후보가 없으면 candidate는 None, metrics의 priority_feedback/type/side는
    'NONE', priority_feedback_error_deg는 NaN이다. 최대 절대 각도 오류를 선택하며
    동점은 팔 높이→팔꿈치→다리 방향→무릎→몸통 순서이다. 한국어 문장과
    Bone index는 기존 6-2 의미를 유지한다. 입력 오류는 ValueError이다.
    """
    reference_world = validate_world_landmarks(reference_world)
    user_world = validate_world_landmarks(user_world)
    values = _bone_similarities(bone_similarities)
    metrics = {}
    directions, elbows = [], []
    for side, shoulder, elbow, wrist, bone in ARM_JOINTS:
        ref_elbow, ref_elevation = arm_measurements(reference_world, shoulder, elbow, wrist)
        usr_elbow, usr_elevation = arm_measurements(user_world, shoulder, elbow, wrist)
        elbow_error = usr_elbow - ref_elbow
        direction_error = usr_elevation - ref_elevation
        gate = _bone_pair_has_error(values, bone)
        arm_action = feedback_action(direction_error, ARM_DIRECTION_THRESHOLD_DEG, gate, "RAISE", "LOWER")
        elbow_action = feedback_action(elbow_error, ELBOW_ANGLE_THRESHOLD_DEG, gate, "EXTEND", "BEND")
        metrics.update({side + "_elbow_reference_deg": ref_elbow, side + "_elbow_user_deg": usr_elbow,
                        side + "_elbow_error_deg": elbow_error, side + "_arm_direction_error_deg": direction_error,
                        side + "_arm_action": arm_action, side + "_elbow_action": elbow_action})
        directions.append(dict(side=side, action=arm_action, signed_error=direction_error,
                               reference=ref_elevation, user=usr_elevation, bone=bone))
        elbows.append(dict(side=side, action=elbow_action, signed_error=elbow_error,
                           reference=ref_elbow, user=usr_elbow, bone=bone + 1))
    knee_metrics, knee_candidate = knee_feedback(reference_world, user_world, values)
    metrics.update(knee_metrics)
    leg_metrics, leg_candidate = leg_direction_feedback(reference_world, user_world, values)
    metrics.update(leg_metrics)
    torso_metrics, torso_candidate = torso_feedback(reference_world, user_world)
    metrics.update(torso_metrics)
    candidates = [candidate for candidate in (
        aggregate_feedback("arm_direction", directions), aggregate_feedback("elbow_angle", elbows),
        leg_candidate, knee_candidate, torso_candidate,
    ) if candidate is not None]
    selected = max(candidates, key=lambda item: item["error_deg"]) if candidates else None
    metrics.update({"priority_feedback": selected["text"] if selected else "NONE",
                    "priority_feedback_type": selected["type"] if selected else "NONE",
                    "priority_feedback_side": selected["side"] if selected else "NONE",
                    "priority_feedback_error_deg": selected["error_deg"] if selected else float("nan")})
    return metrics, selected


def final_feedback_metrics(pose, aligned_overall_similarity, aligned_reference_time_sec,
                           playback_time_sec, aligned_reference_index=None, dtw_processing_ms=float("nan")):
    """Pose 후보와 정렬·playback 시간으로 RAW 최종 피드백 dict를 반환한다.

    pose는 pose_feedback의 후보 dict 또는 None이다. 시간은 초 단위 실수
    스칼라이며 aligned time의 None은 NaN으로 처리한다. offset=aligned-playback.
    Pose가 우선이며 없을 때만 유사도 >=0.8, offset <=-0.3이면 LATE 메시지다.
    비유한 유사도/offset에는 Timing을 표시하지 않는다. index는 0-based row int
    또는 None이며 반환 시 없으면 -1이다. 계산 가능 시간만 따로 전달할 수도 있다.
    dtw_processing_ms는 기존 진단용 밀리초이다. 잘못된 shape/변환/index는
    ValueError, pose의 필수 text 필드가 없으면 KeyError이다. 입력을 수정하지 않는다.
    """
    overall = _real_scalar(aligned_overall_similarity, "aligned_overall_similarity")
    aligned_time = (float("nan") if aligned_reference_time_sec is None
                    else _real_scalar(aligned_reference_time_sec, "aligned_reference_time_sec"))
    playback = _real_scalar(playback_time_sec, "playback_time_sec")
    processing_ms = _real_scalar(dtw_processing_ms, "dtw_processing_ms")
    if aligned_reference_index is not None:
        if (isinstance(aligned_reference_index, (bool, np.bool_))
                or not isinstance(aligned_reference_index, (int, np.integer)) or aligned_reference_index < 0):
            raise ValueError("aligned_reference_index must be a nonnegative row int or None")
        aligned_reference_index = int(aligned_reference_index)
    offset = aligned_time - playback
    pose_text = pose["text"] if pose else "NONE"
    late = (pose_text == "NONE" and math.isfinite(overall) and overall >= POSE_GATE
            and math.isfinite(offset) and offset <= -TIMING_THRESHOLD_SEC)
    final = pose_text if pose_text != "NONE" else "동작이 조금 늦습니다." if late else "NONE"
    return dict(aligned_reference_index=aligned_reference_index if aligned_reference_index is not None else -1,
                aligned_reference_time_sec=aligned_time, timing_offset_sec=offset,
                aligned_overall_similarity=overall, priority_pose_feedback=pose_text,
                timing_feedback="LATE" if late else "NONE", final_feedback=final,
                dtw_processing_ms=processing_ms)


class FeedbackStabilizer:
    """RAW 문장을 바꾸지 않고 5회 연속 확인과 playback 기준 0.5초 hold를 적용한다.

    최초 표시 'NONE'은 아직 확인된 문장이 아니므로 첫 교정에는 hold를 적용하지
    않는다. 'NONE'으로 돌아가는 것도 5회 확인 대상이다. 같은 객체를 세션 동안
    유지하며 새 인스턴스로 초기화한다. now는 유한한 비음수 초 단위 스칼라이다.
    """

    def __init__(self, now=0.0):
        now = self._playback_time(now)
        self.feedback_candidate = "NONE"
        self.feedback_candidate_count = 0
        self.displayed_feedback = "NONE"
        self.displayed_feedback_since_time = now
        self.has_displayed_confirmation = False

    @staticmethod
    def _playback_time(now):
        """표시 상태에 사용할 유한한 비음수 playback 초를 검증한다."""
        now = _real_scalar(now, "now")
        if not math.isfinite(now) or now < 0:
            raise ValueError("now must be finite nonnegative playback seconds")
        return now

    def update(self, raw_final_feedback, now):
        """RAW 문자열과 playback 초로 표시 상태만 갱신한다. 반환값은 None이다.

        문자열/시간 계약 오류는 ValueError이며 상태를 바꾸기 전에 검사한다.
        후보 변경 시 횟수는 1, 동일 후보는 1씩 증가한다. 시간 증가나 리셋은
        자동 수행하지 않는다. RAW 문자열을 수정하지 않는다.
        """
        if not isinstance(raw_final_feedback, str):
            raise ValueError("raw_final_feedback must be a string")
        now = self._playback_time(now)
        if raw_final_feedback == self.feedback_candidate:
            self.feedback_candidate_count += 1
        else:
            self.feedback_candidate = raw_final_feedback
            self.feedback_candidate_count = 1
        hold_elapsed = (not self.has_displayed_confirmation or
                        now - self.displayed_feedback_since_time >= FEEDBACK_MIN_HOLD_SEC)
        if (self.feedback_candidate_count >= FEEDBACK_CONFIRM_FRAMES and hold_elapsed
                and self.feedback_candidate != self.displayed_feedback):
            self.displayed_feedback = self.feedback_candidate
            self.displayed_feedback_since_time = now
            self.has_displayed_confirmation = True

    def snapshot(self, raw_final_feedback):
        """RAW/표시 문장과 후보 상태를 새 dict로 반환하며 내부 상태를 갱신하지 않는다."""
        return dict(raw_final_feedback=raw_final_feedback, displayed_feedback=self.displayed_feedback,
                    feedback_candidate=self.feedback_candidate,
                    feedback_candidate_count=self.feedback_candidate_count)
