"""내부 NumPy/비유한 값을 표준 JSON 값으로 변환한다."""
import math
import numpy as np


def json_safe(value):
    """중첩 배열과 숫자를 변환하며 평가 불가 값을 null로 보존한다."""
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
