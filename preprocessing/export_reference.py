"""Export semantic DanceSync reference JSON from AI Hub frame annotations."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
import unicodedata
from pathlib import Path

if __package__:
    from .load_aihub import AIHUB_TO_DANCE_SYNC, load_frames, to_dance_sync_joints
else:
    from load_aihub import AIHUB_TO_DANCE_SYNC, load_frames, to_dance_sync_joints

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "frontend/public/references"


def data_path(relative: str) -> Path:
    """Resolve catalog paths even when Korean filenames use different Unicode forms."""
    path = DATA_ROOT
    for part in Path(relative).parts:
        candidate = path / part
        if not candidate.exists() and path.is_dir():
            matches = [child for child in path.iterdir()
                       if unicodedata.normalize("NFC", child.name) == unicodedata.normalize("NFC", part)]
            if len(matches) == 1:
                candidate = matches[0]
        path = candidate
    return path


def validate_reference(payload: dict, sequence: str) -> None:
    """Validate the existing player's fixed-FPS, 12-joint input contract."""
    if payload.get("version") != 1 or payload.get("source") != "aihub-kpop" or payload.get("sequence") != sequence:
        raise ValueError("reference version/source/sequence mismatch")
    fps = payload.get("fps")
    if not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames or payload.get("numFrames") != len(frames):
        raise ValueError("missing frames or numFrames mismatch")
    if not math.isclose(payload.get("duration", 0), len(frames) / fps):
        raise ValueError("duration mismatch")
    if payload.get("coordinateSystem", {}).get("unit") != "meter":
        raise ValueError("joint coordinates must be in meters")
    previous = -1
    for index, frame in enumerate(frames):
        frame_index = frame["frameIndex"]
        if not isinstance(frame_index, int) or frame_index <= previous:
            raise ValueError("frame indices must be strictly increasing")
        previous = frame_index
        # DanceAvatar samples floor(currentTime * fps), so reject missing/irregular frames.
        if not math.isclose(frame["timestamp"], index / fps, abs_tol=1e-6):
            raise ValueError("irregular timestamps cannot be played by the fixed-FPS player")
        for name in AIHUB_TO_DANCE_SYNC:
            values = frame["joints"][name]
            if not isinstance(values, list) or len(values) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
            ):
                raise ValueError(f"invalid joint: frame {index}, {name}")


def valid_existing(output: Path, sequence: str) -> bool:
    try:
        validate_reference(json.loads(output.read_text(encoding="utf-8")), sequence)
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def export_reference(source: Path, output: Path, title=None, artist=None, image_directory=None) -> None:
    raw_frames = load_frames(source)
    first = raw_frames[0].document
    video = first.get("videos", [{}])[0]
    actor = first.get("actors", [{}])[0]
    indices = [frame.annotation["frame_index"] for frame in raw_frames]
    steps = [b - a for a, b in zip(indices, indices[1:]) if b > a]
    frame_step = statistics.median(steps) if steps else 1
    source_fps = float(video["frame_rate"])
    fps = source_fps / frame_step
    sequence = source.name if source.is_dir() else video["file_name"].rsplit(".", 1)[0]
    frames = []
    for raw in raw_frames:
        annotation = raw.annotation
        image = raw.document.get("images", [{}])[0]
        frames.append({
            "frameIndex": annotation["frame_index"],
            "timestamp": (annotation["frame_index"] - indices[0]) / source_fps,
            "rootScreenPosition": [0.5, 0.5],
            "joints": to_dance_sync_joints(annotation["keypoints3d"]),
            "actionClass": annotation.get("action_class", []),
            "danceMainpoint": annotation.get("dance_mainpoint", 0),
            "imageFileName": image.get("file_name"),
        })
    payload = {
        "version": 1,
        "source": "aihub-kpop",
        "sequence": sequence,
        "fps": fps,
        "numFrames": len(frames),
        "duration": len(frames) / fps,
        "coordinateSystem": {
            "axes": "X horizontal/camera-right, Y up, Z depth (world coordinates)",
            "handedness": "source world coordinates preserved",
            "unit": "meter",
            "sourceUnit": "centimeter",
            "transform": "[x, y, z] * 0.01; camera extrinsic not reapplied",
        },
        "jointMap": AIHUB_TO_DANCE_SYNC,
        "metadata": {
            "actorId": actor.get("id"),
            "actorHeight": actor.get("height"),
            "actorSex": actor.get("sex"),
            "actorAge": actor.get("age"),
            "title": title,
            "artist": artist,
            "videoFile": video.get("file_name"),
            "sourceFps": source_fps,
            "sourceFrameStep": frame_step,
            "imageDirectory": str(image_directory) if image_directory else None,
        },
        "frames": frames,
    }
    validate_reference(payload, sequence)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never leave a truncated JSON that a later run could mistake for a completed export.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"exported {len(frames)} frames at {fps:g} fps to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export AI Hub 3D labels for DanceAvatar; default: all catalog songs, skip valid outputs.")
    parser.add_argument("source", nargs="?", type=Path, help="label JSON or sequence directory")
    parser.add_argument("--sequence", help="export only this song ID from data/songs.json")
    parser.add_argument("--output", type=Path, help="output file (requires source or --sequence)")
    parser.add_argument("--force", action="store_true", help="regenerate even valid existing outputs")
    args = parser.parse_args()
    if args.source and args.sequence:
        parser.error("use either source or --sequence")
    if args.output and not (args.source or args.sequence):
        parser.error("--output requires source or --sequence")
    catalog = json.loads((DATA_ROOT / "songs.json").read_text(encoding="utf-8"))
    if args.source:
        source = args.source.expanduser().resolve()
        sequence = source.name if source.is_dir() else source.stem.rsplit("_", 1)[0]
        song = next((s for s in catalog if s["id"] == sequence), {"id": sequence})
        jobs = [(song, source)]
    else:
        selected = [s for s in catalog if not args.sequence or s["id"] == args.sequence]
        if not selected:
            parser.error(f"unknown catalog sequence: {args.sequence}")
        jobs = [(song, data_path(song["raw_label_dir"])) for song in selected]
    exported = skipped = failed = 0
    for song, source in jobs:
        sequence = song["id"]
        output = args.output or OUTPUT_DIR / f"{sequence}.json"
        if not args.force and valid_existing(output, sequence):
            print(f"[SKIP] {sequence}: valid JSON already exists", flush=True)
            skipped += 1
            continue
        try:
            images = data_path(song["raw_source_dir"]) if song.get("raw_source_dir") else None
            export_reference(source, output, song.get("title"), song.get("artist"), images)
            exported += 1
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"[ERROR] {sequence}: {exc}", flush=True)
            failed += 1
    print(f"완료: 생성 {exported}, 건너뜀 {skipped}, 실패 {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
