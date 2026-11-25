"""Load frame-per-file AI Hub K-pop annotations in sequence order."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


AIHUB_TO_DANCE_SYNC = {
    "leftShoulder": 14,
    "rightShoulder": 19,
    "leftElbow": 15,
    "rightElbow": 20,
    "leftWrist": 16,
    "rightWrist": 21,
    "leftHip": 1,
    "rightHip": 6,
    "leftKnee": 2,
    "rightKnee": 7,
    "leftAnkle": 3,
    "rightAnkle": 8,
}


@dataclass(frozen=True)
class AIHubFrame:
    path: Path
    document: dict
    annotation: dict


def annotation_paths(source: Path) -> list[Path]:
    source = source.expanduser()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    paths = list(source.glob("*.json"))
    if not paths:
        paths = list(source.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no JSON annotations under {source}")
    return sorted(paths)


def load_frames(source: Path) -> list[AIHubFrame]:
    frames = []
    for path in annotation_paths(source):
        document = json.loads(path.read_text(encoding="utf-8"))
        annotations = document.get("annotations", [])
        if not annotations:
            continue
        for annotation in annotations:
            values = annotation.get("keypoints3d", [])
            if len(values) != 87:
                raise ValueError(
                    f"{path}: frame {annotation.get('frame_index')} has "
                    f"{len(values)} keypoints3d values, expected 87"
                )
            frames.append(AIHubFrame(path, document, annotation))
    frames.sort(key=lambda frame: frame.annotation["frame_index"])
    if not frames:
        raise ValueError(f"no annotations in {source}")
    return frames


def to_dance_sync_joints(keypoints3d: list[float], scale: float = 0.01) -> dict:
    """Preserve AI Hub X/Y/Z axes and convert centimetres to metres."""
    return {
        name: [value * scale for value in keypoints3d[index * 3:index * 3 + 3]]
        for name, index in AIHUB_TO_DANCE_SYNC.items()
    }
