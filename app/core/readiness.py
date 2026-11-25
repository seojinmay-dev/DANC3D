"""원본 2D Image Landmark의 전신 포함 및 정면 T-Pose 준비 판정."""
import math
from collections.abc import Mapping
import numpy as np

MIN_VISIBILITY = 0.5
FRAME_MARGIN = 0.03
MIN_ELBOW_ANGLE_DEG = 155.0
MAX_ARM_HEIGHT_ERROR_SHOULDER_RATIO = 0.25
MIN_WRIST_OUTWARD_SHOULDER_RATIO = 0.75
MIN_ELBOW_OUTWARD_SHOULDER_RATIO = 0.25
GEOMETRY_EPSILON = 1e-8
REQUIRED_LANDMARKS = (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 31, 32)


def _number(value):
    """누락·문자열·bool을 좌표로 보정하지 않고 유한 실수만 읽는다."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('Expected a real number')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Expected a finite number')
    return value


def _point(point):
    """객체/dict/세 열 배열에서 x,y,visibility를 명시적으로 읽는다."""
    if isinstance(point, Mapping):
        values = [point[key] for key in ('x', 'y', 'visibility')]
    elif hasattr(point, 'x'):
        values = [getattr(point, key) for key in ('x', 'y', 'visibility')]
    else:
        if len(point) != 3:
            raise ValueError('Expected x, y, visibility')
        values = point
    x, y, visibility = map(_number, values)
    if not 0 <= visibility <= 1:
        raise ValueError('Visibility must be in [0, 1]')
    return x, y, visibility


def evaluate_readiness(image_landmarks, image_width, image_height):
    """정면 T-Pose 준비 조건과 실패 코드를 반환하며 추론/그리기는 하지 않는다.

    입력은 None 또는 길이 33의 시퀀스다. 각 주요 점은 x/y/visibility 속성의
    MediaPipe 객체, 같은 키의 dict, 또는 [x,y,visibility] 세 열 배열이다.
    배열 세 번째 열은 z가 아니라 반드시 visibility여야 한다. World 좌표는 받지
    않는다. 주요 점만 유한 실수/visibility [0,1]을 검사하고 얼굴/손가락 세부 점은
    검사하지 않는다. 누락값을 보정하지 않는다. 너비/높이는 양의 유한 실수다.
    잘못된 입력은 예외 대신 ready=False와 INVALID_* 코드를 반환한다.
    pose_detected는 33개 입력 존재, full_body_visible은 주요 15개 점의 가시성과
    margin 포함 여부다. 각도/수평/펼침은 종횡비를 반영한 픽셀 기하로 검사한다.
    신체 좌우 인덱스/축을 바꾸지 않는다. 초기 휴리스틱이며 2D에서 깊이 방향
    자세나 모델 오검출을 완전히 구분할 수 없다.
    """
    result = dict(pose_detected=False, full_body_visible=False, arms_horizontal=False,
                  arms_extended=False, arms_spread=False, ready=False, reasons=[])
    reasons = result['reasons']
    if image_landmarks is None:
        reasons.append('NO_POSE')
        return result
    try:
        if len(image_landmarks) != 33:
            reasons.append('INVALID_LANDMARK_COUNT')
            return result
        result['pose_detected'] = True
        points = {index: _point(image_landmarks[index]) for index in REQUIRED_LANDMARKS}
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError):
        reasons.append('INVALID_LANDMARKS')
        return result
    try:
        width, height = _number(image_width), _number(image_height)
        if width <= 0 or height <= 0:
            raise ValueError('Positive image size required')
    except (ValueError, TypeError, OverflowError):
        reasons.append('INVALID_IMAGE_SIZE')
        return result
    if any(point[2] < MIN_VISIBILITY for point in points.values()):
        reasons.append('LOW_VISIBILITY')
    if any(not (FRAME_MARGIN <= x <= 1 - FRAME_MARGIN and FRAME_MARGIN <= y <= 1 - FRAME_MARGIN)
           for x, y, _ in points.values()):
        reasons.append('BODY_OUT_OF_FRAME')
    result['full_body_visible'] = not reasons
    pixels = {index: np.array([x * width, y * height]) for index, (x, y, _) in points.items()}
    shoulder_span = pixels[11][0] - pixels[12][0]
    shoulder_width = abs(shoulder_span)
    if not math.isfinite(shoulder_width) or shoulder_width <= GEOMETRY_EPSILON:
        reasons.append('DEGENERATE_GEOMETRY')
        return result
    outward_sign = 1 if shoulder_span > 0 else -1
    horizontal, extended, spread = [], [], []
    for side, shoulder, elbow, wrist, sign in (
            ('left', 11, 13, 15, outward_sign), ('right', 12, 14, 16, -outward_sign)):
        s, e, w = pixels[shoulder], pixels[elbow], pixels[wrist]
        error = max(abs(e[1] - s[1]), abs(w[1] - s[1])) / shoulder_width
        first, second = s - e, w - e
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        angle = (math.degrees(math.acos(float(np.clip(np.dot(first, second) / denominator, -1, 1))))
                 if math.isfinite(denominator) and denominator > GEOMETRY_EPSILON else None)
        horizontal.append(math.isfinite(error) and error <= MAX_ARM_HEIGHT_ERROR_SHOULDER_RATIO)
        extended.append(angle is not None and angle >= MIN_ELBOW_ANGLE_DEG)
        spread.append(sign * (w[0] - s[0]) >= MIN_WRIST_OUTWARD_SHOULDER_RATIO * shoulder_width
                      and sign * (e[0] - s[0]) >= MIN_ELBOW_OUTWARD_SHOULDER_RATIO * shoulder_width)
        result[side + '_elbow_angle_deg'] = angle
        result[side + '_height_error_ratio'] = float(error)
    result['arms_horizontal'] = all(horizontal)
    result['arms_extended'] = all(extended)
    result['arms_spread'] = all(spread)
    for condition, code in [('arms_horizontal', 'ARMS_NOT_HORIZONTAL'), ('arms_extended', 'ARMS_NOT_EXTENDED'),
                            ('arms_spread', 'ARMS_NOT_SPREAD')]:
        if not result[condition]:
            reasons.append(code)
    result['ready'] = result['full_body_visible'] and all((result['arms_horizontal'], result['arms_extended'], result['arms_spread']))
    return result
