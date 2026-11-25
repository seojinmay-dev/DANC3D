"""시간 정렬이 끝난 두 DIR8 프레임의 유사도와 오류 Bone을 반환한다."""
import math
import numpy as np
from .pose_features import BONES, PARTS

BONE_ERROR_THRESHOLD = 0.8


def valid_mean(values):
    """실수 스칼라 iterable의 유한값 평균을 Python float로 반환한다.

    유한값이 없으면 NaN이며 숫자로 변환할 수 없는 값은 ValueError/TypeError이다.
    """
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _validate_dir8(vectors, mask):
    """DIR8과 mask의 shape 및 값 계약을 확인한다."""
    if np.iscomplexobj(vectors):
        raise ValueError("DIR8 must be real-valued")
    vectors = np.asarray(vectors, dtype=np.float64)
    mask = np.asarray(mask)
    if vectors.shape != (len(BONES), 3) or mask.shape != (len(BONES),):
        raise ValueError("Expected DIR8 (8, 3) and bone_valid_mask (8,)")
    if not np.isin(mask, [False, True]).all():
        raise ValueError("bone_valid_mask must contain bool or 0/1")
    lengths = np.linalg.norm(vectors, axis=1)
    valid = mask.astype(bool) & np.isfinite(vectors).all(axis=1)
    if not np.allclose(lengths[valid], 1.0, atol=1e-4):
        raise ValueError("Valid DIR8 vectors must be unit vectors")
    return vectors, valid


def compare(reference_dir8, user_dir8, reference_mask, user_mask):
    """정렬된 두 float 변환 가능 DIR8 (8, 3)와 mask (8,)를 비교한다.

    mask는 bool/0/1만 허용한다. True인 유한 벡터는 단위벡터여야 하며
    shape/값 계약 오류는 ValueError이다. 양쪽 mask가 True이고 유한한
    Bone만 내적 후 [-1, 1]로 clip한다. 추가 정규화는 수행하지 않는다.
    반환값은 float64 Bone cosine (8,), Python float 전체 평균,
    PARTS 순서의 Python float 평균 list (길이 4)이다. 무효 Bone 및
    유효 Bone이 없는 전체/부위 평균은 NaN이다. 입력은 변경하지 않는다.
    """
    reference, reference_valid = _validate_dir8(reference_dir8, reference_mask)
    user, user_valid = _validate_dir8(user_dir8, user_mask)
    valid = reference_valid & user_valid
    values = np.full(len(BONES), np.nan, dtype=np.float64)
    values[valid] = np.clip(np.einsum("ij,ij->i", reference[valid], user[valid]), -1.0, 1.0)
    parts = [valid_mean(values[2 * index:2 * index + 2]) for index in range(len(PARTS))]
    return values, valid_mean(values), parts


def status(value):
    """실수 스칼라 cosine을 OK / ERROR / INVALID 문자열로 반환한다.

    비유한값은 INVALID, 0.8 미만은 ERROR, 이상은 OK이다.
    실수 변환 불가 입력은 ValueError/TypeError이다. UI 색상은 반환하지 않는다.
    """
    value = float(value)
    if not math.isfinite(value):
        return "INVALID"
    return "ERROR" if value < BONE_ERROR_THRESHOLD else "OK"


def priority_error(values):
    """실수 cosine (8,)에서 가장 낮은 오류 Bone의 int 인덱스를 반환한다.

    shape/실수 변환 오류는 ValueError이다. 유한한 0.8 미만 Bone만 고려하며
    동점은 BONES 순서의 첫 항목이다. 후보가 없으면 None이다.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(BONES),):
        raise ValueError("values must have shape (8,)")
    candidates = [index for index, value in enumerate(values)
                  if math.isfinite(float(value)) and value < BONE_ERROR_THRESHOLD]
    return min(candidates, key=lambda index: float(values[index])) if candidates else None
