"""DIR8 기반 Windowed Online DTW와 명시적 Local Recovery를 계산한다.

한 세션에서 같은 객체의 update()를 계속 호출한다. 입력 시간은 초 단위이며
반환 index는 Reference 배열의 0-based row index로, 원천 frame_index가 아니다.
NPZ timestamps_ms는 reference_times_from_timestamps()로 변환할 수 있다.
"""

import math

import numpy as np

from .pose_features import BONES

WARPING_WINDOW_SEC = 1.0
RECOVERY_LOCAL_SIM_THRESHOLD = 0.95
RECOVERY_SIM_GAIN_THRESHOLD = 0.05
RECOVERY_PLAYBACK_TOLERANCE_SEC = 0.20
RECOVERY_CONSECUTIVE_FRAMES = 3
RECOVERY_COOLDOWN_SEC = 0.5

# 기존 연구 코드의 load_reference()와 동일한 단위벡터 검증 허용 오차이다.
UNIT_VECTOR_ATOL = 1e-4
UNIT_VECTOR_RTOL = 1e-5


def _real_array(values, name):
    """실수 변환 가능한 입력을 float64로 읽고 변환 오류를 ValueError로 알린다."""
    try:
        if np.iscomplexobj(values):
            raise ValueError("complex values are not supported")
        return np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} must contain real numbers".format(name)) from exc


def _validate_dir8_arrays(vectors, mask, batched):
    """DIR8 shape와 mask를 검증하며 True인 Bone에만 유한 단위벡터를 요구한다."""
    vectors = _real_array(vectors, "vectors")
    expected_tail = (len(BONES), 3)
    if ((batched and (vectors.ndim != 3 or vectors.shape[1:] != expected_tail))
            or (not batched and vectors.shape != expected_tail)):
        raise ValueError("vectors must have shape (N, 8, 3) or (8, 3) as required")
    mask = np.asarray(mask)
    if mask.shape != vectors.shape[:-1]:
        raise ValueError("mask shape must match vectors without the coordinate axis")
    if mask.dtype.kind not in "biuf" or not np.isin(mask, [0, 1]).all():
        raise ValueError("mask must contain only bool or numeric 0/1 values")
    mask = mask.astype(bool)
    active = vectors[mask]
    if not np.isfinite(active).all():
        raise ValueError("mask=True requires finite DIR8 vectors")
    with np.errstate(over="ignore", invalid="ignore"):
        lengths = np.linalg.norm(active, axis=-1)
    if not np.allclose(lengths, 1.0, atol=UNIT_VECTOR_ATOL, rtol=UNIT_VECTOR_RTOL):
        raise ValueError("mask=True requires unit DIR8 vectors")
    return vectors, mask


def _validate_times(values, name):
    """유한한 비음수 1차원 시간축이 비어 있지 않고 엄격히 증가하는지 확인한다."""
    times = _real_array(values, name)
    if (times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all()
            or (times < 0).any() or not (times[1:] > times[:-1]).all()):
        raise ValueError("{} must be nonempty, finite, nonnegative and strictly increasing".format(name))
    return times


def _validate_playback_time(playback_time):
    """초 단위 playback을 유한한 비음수 Python float로 변환한다."""
    value = _real_array(playback_time, "playback_time")
    if value.ndim != 0 or not np.isfinite(value) or value < 0:
        raise ValueError("playback_time must be a finite nonnegative scalar in seconds")
    return float(value)


def reference_times_from_timestamps(timestamps_ms):
    """NPZ timestamps_ms (N,)를 첫 저장 프레임 기준 float64 초 (N,)로 반환한다.

    N >= 1이며 유한한 비음수 값이 엄격히 증가해야 한다. 잘못된 입력은
    ValueError이다. (timestamps_ms - timestamps_ms[0]) / 1000.0을 적용하며
    입력은 수정하지 않는다. 배열 row와 원천 frame_index를 연결하지 않는다.
    """
    timestamps = _validate_times(timestamps_ms, "timestamps_ms")
    return (timestamps - timestamps[0]) / 1000.0


class WindowedOnlineDTW:
    """playback ±1초에서 누적 비용과 경로 길이를 유지하는 인과적 Online DTW.

    reference_vectors는 (N, 8, 3), reference_mask는 (N, 8), reference_times는
    초 단위 (N,)이며 N >= 1이다. 시간축은 유한한 비음수 값으로 엄격히
    증가해야 한다. 첫 저장 프레임을 0초로 변환하여 전달하는 것이 사용 계약이다.
    벡터는 float64, mask는 bool로 복사하므로 호출자의 배열 변경은 전파되지 않는다.
    mask는 bool/0/1이며 True인 Bone은 유한 단위벡터여야 한다. False인 Bone의
    NaN은 허용한다. shape 및 값 계약 위반은 ValueError이며 재정규화하지 않는다.

    previous_cost(float64), previous_length(int64)는 길이 N+1의 누적 상태이다.
    위치 j는 Reference row j-1에 대응하고 위치 0은 최초 시작 경계이다.
    predicted_index는 아직 정렬하지 않았으면 -1이다. 새 세션은 새 객체로 시작한다.
    """

    def __init__(self, reference_vectors, reference_mask, reference_times):
        reference, valid = _validate_dir8_arrays(reference_vectors, reference_mask, batched=True)
        times = _validate_times(reference_times, "reference_times")
        if len(reference) != len(times):
            raise ValueError("reference vectors, mask and times must have the same N")
        self.reference = reference.copy()
        self.valid = valid.copy()
        self.times = times.copy()
        self.previous_cost = np.full(len(times) + 1, np.inf)
        self.previous_cost[0] = 0.0
        self.previous_length = np.zeros(len(times) + 1, dtype=np.int64)
        self.predicted_index = -1

    def update(self, user_vectors, user_mask, playback_time):
        """사용자 DIR8 한 프레임으로 상태를 갱신하고 Reference row int 또는 None을 반환한다.

        user_vectors는 (8, 3), user_mask는 (8,)이며 생성자와 같은 벡터 계약이다.
        playback_time은 유한한 비음수 초 단위 스칼라이다. 잘못된 입력은 상태를
        변경하기 전에 ValueError로 알린다. 유효 endpoint가 없으면 None이며
        이전 누적 상태를 보존한다. 일반 endpoint는 비감소하고 자동 리셋하지 않는다.
        """
        user, user_mask = _validate_dir8_arrays(user_vectors, user_mask, batched=False)
        playback_time = _validate_playback_time(playback_time)
        lo = int(np.searchsorted(self.times, playback_time - WARPING_WINDOW_SEC, side="left"))
        hi = int(np.searchsorted(self.times, playback_time + WARPING_WINDOW_SEC, side="right"))
        valid = self.valid[lo:hi] & user_mask
        with np.errstate(invalid="ignore", over="ignore"):
            cosines = np.clip(np.einsum("nbc,bc->nb", self.reference[lo:hi], user), -1.0, 1.0)
        counts = valid.sum(axis=1)
        means = np.zeros(hi - lo)
        np.divide(np.where(valid, cosines, 0.0).sum(axis=1), counts,
                  out=means, where=counts > 0)
        local = np.where(counts > 0, 1.0 - means, np.inf)
        cost = np.full(len(self.times) + 1, np.inf)
        length = np.zeros(len(self.times) + 1, dtype=np.int64)
        previous_cost, previous_length = self.previous_cost, self.previous_length
        for j in range(lo + 1, hi + 1):
            # 동점은 대각선, 수직, 수평 순서를 유지해야 원본 누적 경로와 같아진다.
            best_cost, best_length = previous_cost[j - 1], previous_length[j - 1]
            if previous_cost[j] < best_cost:
                best_cost, best_length = previous_cost[j], previous_length[j]
            if cost[j - 1] < best_cost:
                best_cost, best_length = cost[j - 1], length[j - 1]
            if np.isfinite(best_cost) and np.isfinite(local[j - 1 - lo]):
                cost[j] = local[j - 1 - lo] + best_cost
                length[j] = best_length + 1
        lower = max(lo, self.predicted_index, 0)
        if lower >= hi:
            return None
        normalized = np.full(len(self.times), np.inf)
        np.divide(cost[1:], length[1:], out=normalized, where=length[1:] > 0)
        predicted = lower + int(np.argmin(normalized[lower:hi]))
        if not np.isfinite(normalized[predicted]):
            return None
        self.previous_cost, self.previous_length = cost, length
        self.predicted_index = predicted
        return predicted


def _recovery_similarities(reference, reference_mask, user, user_mask):
    """검증된 배열에 기존 6-2의 유효 Bone 평균 연산 순서를 적용한다."""
    valid = reference_mask & user_mask
    with np.errstate(invalid="ignore", over="ignore"):
        cosines = np.clip(np.einsum("nbc,bc->nb", reference, user), -1.0, 1.0)
    counts = valid.sum(axis=1)
    result = np.full(len(reference), np.nan)
    np.divide(np.where(valid, cosines, 0.0).sum(axis=1), counts,
              out=result, where=counts > 0)
    return result


def recovery_similarities(reference_vectors, reference_mask, user_vectors, user_mask):
    """후보 DIR8 (M, 8, 3)와 사용자 (8, 3)의 평균 cosine float64 (M,)를 반환한다.

    mask는 각각 (M, 8), (8,)이며 양쪽 True인 Bone만 반영하고 cosine은
    [-1, 1]로 clip한다. 공통 유효 Bone이 없으면 NaN이다. M=0도 허용한다.
    벡터 및 mask의 값 계약은 WindowedOnlineDTW와 같고 위반은 ValueError이다.
    입력이나 누적 상태는 변경하지 않는다.
    """
    reference, reference_mask = _validate_dir8_arrays(reference_vectors, reference_mask, batched=True)
    user, user_mask = _validate_dir8_arrays(user_vectors, user_mask, batched=False)
    return _recovery_similarities(reference, reference_mask, user, user_mask)


class LocalRecoveryDTW:
    """Online DTW에 연속 증거와 cooldown을 적용해 명시적으로 anchor를 복구한다.

    생성자/update 입력 계약은 WindowedOnlineDTW와 같다. dtw 속성으로 누적
    상태를 확인할 수 있고 metrics는 매 update 후 최신 진단 dict로 교체된다.
    일반 endpoint는 비감소하지만 Recovery는 이전 Reference row로 되돌릴 수 있다.

    metrics의 모든 index는 0-based Reference row이며 없으면 -1이다.
    best_local_*는 현재 window의 최상위 후보이고, current_dtw_similarity와
    pre_recovery_aligned_*는 Recovery 전 DTW 결과이다. 시간은 초 단위이며
    측정 불가 시간/유사도/gain은 NaN이다. recovery_similarity_gain은
    best_local_similarity - current_dtw_similarity이다.
    recovery_candidate/triggered는 0/1, recovery_counter는 리셋 전 증거 횟수,
    recovery_trigger_count는 세션의 누적 발동 횟수이다. recovery_anchor_*는
    이번 update에서 발동했을 때만 채워지고 나머지는 -1/NaN이다.
    counter는 발동 후 0, last_trigger는 마지막 발동 playback(초)이며 초기 -inf다.
    """

    def __init__(self, reference_vectors, reference_mask, reference_times):
        self.dtw = WindowedOnlineDTW(reference_vectors, reference_mask, reference_times)
        self.counter = 0
        self.last_trigger = -math.inf
        self.trigger_count = 0
        self.metrics = dict(
            best_local_reference_index=-1,
            best_local_reference_time_sec=float("nan"), best_local_similarity=float("nan"),
            current_dtw_similarity=float("nan"), pre_recovery_aligned_index=-1,
            pre_recovery_aligned_time_sec=float("nan"), recovery_similarity_gain=float("nan"),
            recovery_candidate=0, recovery_counter=0, recovery_triggered=0,
            recovery_trigger_count=0, recovery_anchor_index=-1,
            recovery_anchor_time_sec=float("nan"))

    def update(self, user_vectors, user_mask, playback_time):
        """한 프레임을 누적 정렬하고 Recovery를 평가하여 row int 또는 None을 반환한다.

        잘못된 입력은 누적 상태와 metrics를 바꾸기 전에 ValueError로 알린다.
        유효 DTW 결과가 없어 current similarity가 NaN이면 Recovery도 발동하지
        않는다. 발동 시 현재 관측을 anchor의 길이 1 경로로 삼아 누적 상태 전체를
        재설정한다. cooldown은 wall-clock이 아닌 전달된 playback 초를 사용한다.
        """
        endpoint = self.dtw.update(user_vectors, user_mask, playback_time)
        # dtw.update에서 검증된 입력을 같은 dtype으로 읽어 Recovery에도 사용한다.
        user = np.asarray(user_vectors, dtype=np.float64)
        user_mask = np.asarray(user_mask, dtype=bool)
        playback = float(playback_time)
        times = self.dtw.times
        lo = int(np.searchsorted(times, playback - WARPING_WINDOW_SEC, side="left"))
        hi = int(np.searchsorted(times, playback + WARPING_WINDOW_SEC, side="right"))
        scores = _recovery_similarities(self.dtw.reference[lo:hi], self.dtw.valid[lo:hi], user, user_mask)
        local_index = None
        local_sim = float("nan")
        if np.isfinite(scores).any():
            offset = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
            local_index, local_sim = lo + offset, float(scores[offset])
        current_sim = float("nan")
        current_time = float("nan")
        if endpoint is not None:
            current_time = float(times[endpoint])
            current_sim = float(_recovery_similarities(
                self.dtw.reference[endpoint:endpoint + 1],
                self.dtw.valid[endpoint:endpoint + 1], user, user_mask)[0])
        gain = local_sim - current_sim
        local_time = float(times[local_index]) if local_index is not None else float("nan")
        enough_gain = math.isfinite(gain) and gain >= RECOVERY_SIM_GAIN_THRESHOLD
        displaced_or_inferior = (math.isfinite(current_time) and
            abs(current_time - playback) > RECOVERY_PLAYBACK_TOLERANCE_SEC) or enough_gain
        candidate = (math.isfinite(local_sim) and local_sim >= RECOVERY_LOCAL_SIM_THRESHOLD
                     and enough_gain and abs(local_time - playback) <= RECOVERY_PLAYBACK_TOLERANCE_SEC
                     and displaced_or_inferior)
        self.counter = self.counter + 1 if candidate else 0
        evidence_count = self.counter
        triggered = (candidate and self.counter >= RECOVERY_CONSECUTIVE_FRAMES
                     and playback - self.last_trigger >= RECOVERY_COOLDOWN_SEC)
        pre_endpoint = endpoint
        if triggered:
            # 반환 index만 바꾸지 않고 현재 관측을 새 anchor의 길이 1 경로로 만든다.
            self.dtw.previous_cost.fill(np.inf)
            self.dtw.previous_length.fill(0)
            self.dtw.previous_cost[local_index + 1] = 1.0 - local_sim
            self.dtw.previous_length[local_index + 1] = 1
            self.dtw.predicted_index = local_index
            endpoint = local_index
            self.counter = 0
            self.last_trigger = playback
            self.trigger_count += 1
        self.metrics = dict(
            best_local_reference_index=local_index if local_index is not None else -1,
            best_local_reference_time_sec=local_time, best_local_similarity=local_sim,
            current_dtw_similarity=current_sim,
            pre_recovery_aligned_index=pre_endpoint if pre_endpoint is not None else -1,
            pre_recovery_aligned_time_sec=current_time, recovery_similarity_gain=gain,
            recovery_candidate=int(candidate), recovery_counter=evidence_count,
            recovery_triggered=int(triggered), recovery_trigger_count=self.trigger_count,
            recovery_anchor_index=local_index if triggered else -1,
            recovery_anchor_time_sec=local_time if triggered else float("nan"))
        return endpoint
