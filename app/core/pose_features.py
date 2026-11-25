"""원본 MediaPipe World 좌표에서 DIR8 및 기본 3D 특징을 계산한다."""
import math
import numpy as np

BONES = (
    ("left_upper_arm", 11, 13), ("left_forearm", 13, 15),
    ("right_upper_arm", 12, 14), ("right_forearm", 14, 16),
    ("left_thigh", 23, 25), ("left_shin", 25, 27),
    ("right_thigh", 24, 26), ("right_shin", 26, 28),
)
PARTS = ("left_arm", "right_arm", "left_leg", "right_leg")
TORSO_EDGES = ((11, 12), (12, 24), (24, 23), (23, 11))
BONE_LENGTH_EPSILON = 1e-8
ANGLE_DENOMINATOR_EPSILON = 1e-12


def validate_world_landmarks(world_landmarks):
    """실수형 (33, 3) 입력을 복사하여 float64 배열로 반환한다.

    shape 오류 또는 실수로 변환할 수 없는 입력은 ValueError이다.
    NaN/Inf가 포함된 관절은 세 좌표 모두 NaN으로 처리한다.
    원본 입력과 좌표축, 실제 인체 LEFT/RIGHT는 변경하지 않는다.
    """
    if np.iscomplexobj(world_landmarks):
        raise ValueError("world_landmarks must be real-valued")
    try:
        world = np.array(world_landmarks, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("world_landmarks must be a numeric (33, 3) array") from exc
    if world.shape != (33, 3):
        raise ValueError("world_landmarks must have shape (33, 3)")
    world[~np.isfinite(world).all(axis=1)] = np.nan
    return world


def extract_bone_directions(world_landmarks):
    """World (33, 3)에서 (DIR8, bone_valid_mask)를 반환한다.

    DIR8은 float64 (8, 3), mask는 bool (8,)이다. 자식-부모 벡터의
    길이가 유한하고 1e-8 초과일 때만 단위벡터와 True를 반환한다.
    나머지는 NaN 벡터와 False이며 입력 오류는 검증 함수와 같다.
    """
    world = validate_world_landmarks(world_landmarks)
    directions = np.full((len(BONES), 3), np.nan, dtype=np.float64)
    bone_valid_mask = np.zeros(len(BONES), dtype=bool)
    for index, (_, parent, child) in enumerate(BONES):
        with np.errstate(over="ignore", invalid="ignore"):
            vector = world[child] - world[parent]
            length = float(np.linalg.norm(vector))
        if np.isfinite(vector).all() and math.isfinite(length) and length > BONE_LENGTH_EPSILON:
            directions[index] = vector / length
            bone_valid_mask[index] = True
    return directions, bone_valid_mask


def pelvis_center(world_landmarks):
    """World (33, 3)의 좌우 hip 평균을 float64 (3,)로 반환한다.

    어느 hip이든 무효이면 전체 NaN을 반환하며 입력 오류는 검증 함수와 같다.
    """
    world = validate_world_landmarks(world_landmarks)
    return (world[23] + world[24]) / 2.0


def center_world_landmarks(world_landmarks):
    """World (33, 3)에서 pelvis를 뺀 float64 (33, 3)를 반환한다.

    hip이 무효이면 전체 NaN이다. 별도 정렬 기능이며 DIR8 추출에는 적용하지
    않는다. 입력 오류는 검증 함수와 같고 원본을 수정하지 않는다.
    """
    world = validate_world_landmarks(world_landmarks)
    return world - pelvis_center(world)


def angle_deg(first, second):
    """실수형 (3,) 벡터 사이 각도를 Python float 도 단위로 반환한다.

    shape/실수 변환 오류는 ValueError이다. 비유한 좌표 또는 norm 곱이
    1e-12 이하이거나 비유한 경우 NaN이다. 기존 6-2의 계산식을 유지한다.
    """
    if np.iscomplexobj(first) or np.iscomplexobj(second):
        raise ValueError("angle vectors must be real-valued")
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != (3,) or second.shape != (3,):
        raise ValueError("angle vectors must have shape (3,)")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return float("nan")
    with np.errstate(over="ignore", invalid="ignore"):
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not math.isfinite(denominator) or denominator <= ANGLE_DENOMINATOR_EPSILON:
        return float("nan")
    return math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1, 1))))


def angle_diff_deg(target, current):
    """유한한 스칼라 각도의 target-current를 [-180, 180) 도로 반환한다.

    실수 변환 불가 입력은 ValueError/TypeError이며 비유한 입력은 NaN이다.
    """
    target, current = float(target), float(current)
    if not math.isfinite(target) or not math.isfinite(current):
        return float("nan")
    return ((target - current + 180.0) % 360.0) - 180.0
