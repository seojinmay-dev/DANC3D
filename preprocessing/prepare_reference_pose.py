"""DANC3D 기준 안무의 F-view JPG를 MediaPipe 3D Pose NPZ로 전처리한다.

원천 JPG와 라벨 JSON의 파일명이 일치하는 프레임만 사용한다.
JSON에서는 frame_index와 원본 frame_rate만 읽으며, 3D Pose는 JPG에서
MediaPipe Tasks PoseLandmarker로 추정한다. 원본 데이터는 수정하지 않는다.
"""

import argparse
import importlib
import json
import math
import sys
import unicodedata
from pathlib import Path


# ============================================================
# 1. 경로와 비교 관절 정의
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REFERENCE_DIR = PROJECT_ROOT / "data" / "references"
MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_full.task"
EPSILON = 1e-8

LANDMARK_NAMES = (
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
    "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
)

# MediaPipe 관절 인덱스. LEFT/RIGHT는 실제 인체 기준이며 교환하지 않는다.
COMPARISON_JOINTS = {
    "pelvis": (23, 24),
    "left_shoulder": (11,), "right_shoulder": (12,),
    "left_elbow": (13,), "right_elbow": (14,),
    "left_wrist": (15,), "right_wrist": (16,),
    "left_hip": (23,), "right_hip": (24,),
    "left_knee": (25,), "right_knee": (26,),
    "left_ankle": (27,), "right_ankle": (28,),
}

# DIR8의 순서는 기존 비교·점수 코드와 동일하게 고정한다.
BONES = {
    "left_upper_arm": (11, 13), "left_forearm": (13, 15),
    "right_upper_arm": (12, 14), "right_forearm": (14, 16),
    "left_thigh": (23, 25), "left_shin": (25, 27),
    "right_thigh": (24, 26), "right_shin": (26, 28),
}
PART_GROUPS = {
    "LEFT_ARM": ("left_upper_arm", "left_forearm"),
    "RIGHT_ARM": ("right_upper_arm", "right_forearm"),
    "LEFT_LEG": ("left_thigh", "left_shin"),
    "RIGHT_LEG": ("right_thigh", "right_shin"),
}


def nfc(value: str) -> str:
    """macOS의 한글 폴더명 NFC/NFD 차이를 비교 단계에서만 통일한다."""
    return unicodedata.normalize("NFC", value)


# ============================================================
# 2. 원천·라벨 F-view 탐색 및 프레임 대응
# ============================================================

def collect_song_dirs():
    """[원천]/[라벨] 폴더를 정규화된 곡명으로 대응시킨다."""
    if not RAW_DIR.is_dir():
        raise FileNotFoundError(f"raw 폴더가 없습니다: {RAW_DIR}")
    groups = {"[원천]": {}, "[라벨]": {}}
    for path in sorted(RAW_DIR.iterdir()):
        if not path.is_dir():
            continue
        name = nfc(path.name)
        for prefix, songs in groups.items():
            if name.startswith(prefix):
                # 수정됨: 폴더명의 언더바(_)를 공백으로 치환하여 곡 이름 양식을 통일
                title = name[len(prefix):].strip().replace("_", " ")
                if title in songs:
                    raise ValueError(f"정규화한 곡 폴더명이 중복됩니다: {path}")
                songs[title] = path
                break
    source_titles = set(groups["[원천]"])
    label_titles = set(groups["[라벨]"])
    if source_titles != label_titles:
        raise ValueError(
            f"원천/라벨 곡 폴더가 일치하지 않습니다. "
            f"원천에만: {sorted(source_titles-label_titles)}, "
            f"라벨에만: {sorted(label_titles-source_titles)}"
        )
    if not source_titles:
        raise ValueError("[원천]/[라벨] 곡 폴더를 찾지 못했습니다.")
    return groups


def find_f_views(song_dir: Path):
    """동일 곡 내부의 F-view를 상대 경로 기준으로 수집한다."""
    views = {}
    for path in sorted(song_dir.rglob("*")):
        if not path.is_dir() or not nfc(path.name).upper().endswith("_F"):
            continue
        key = nfc(path.relative_to(song_dir).as_posix())
        if key in views:
            raise ValueError(f"F-view 경로 중복: {path}")
        views[key] = path
    return views


def file_map(folder: Path, suffixes: set[str]):
    """파일명(stem)을 키로 사용하며 동일 이름의 중복 파일은 허용하지 않는다."""
    files = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            stem = nfc(path.stem)
            if stem in files:
                raise ValueError(f"파일명 중복: {files[stem]} / {path}")
            files[stem] = path
    return files


def discover_sequences():
    """원천과 라벨에 모두 존재하는 F-view sequence를 찾는다."""
    groups = collect_song_dirs()
    sequences = []
    used_ids = set()
    for title in sorted(groups["[원천]"]):
        source_root = groups["[원천]"][title]
        label_root = groups["[라벨]"][title]
        source_views = find_f_views(source_root)
        label_views = find_f_views(label_root)
        if set(source_views) != set(label_views):
            raise ValueError(
                f"{title}: F-view 폴더 구성이 다릅니다. "
                f"원천에만 {sorted(set(source_views)-set(label_views))}, "
                f"라벨에만 {sorted(set(label_views)-set(source_views))}"
            )
        if not source_views:
            raise ValueError(f"{title}: F-view를 찾지 못했습니다.")
        for relative_path in sorted(source_views):
            sequence_id = nfc(source_views[relative_path].name)
            if sequence_id in used_ids:
                raise ValueError(f"출력 NPZ 이름이 충돌합니다: {sequence_id}")
            used_ids.add(sequence_id)
            images = file_map(source_views[relative_path], {".jpg", ".jpeg"})
            labels = file_map(label_views[relative_path], {".json"})
            if not (images.keys() & labels.keys()):
                raise ValueError(f"{title}/{sequence_id}: 대응되는 JPG/JSON이 없습니다.")
            sequences.append({
                "song": title,
                "id": sequence_id,
                "relative_path": relative_path,
                "images": images,
                "labels": labels,
                "source_dir": source_views[relative_path],
                "label_dir": label_views[relative_path],
            })
    return sequences


def load_frames(sequence):
    """동일 stem의 JPG/JSON만 선택하고 JSON에서 시간 정보만 읽는다."""
    images, labels = sequence["images"], sequence["labels"]
    matched_stems = sorted(images.keys() & labels.keys())
    unmatched_jpg = sorted(images.keys() - labels.keys())
    unmatched_json = sorted(labels.keys() - images.keys())

    print(f"  JPG {len(images)} / JSON {len(labels)} / 대응 {len(matched_stems)}")
    if unmatched_jpg:
        print(f"  [제외] 미대응 JPG {len(unmatched_jpg)}개: {', '.join(unmatched_jpg[:5])}")
    if unmatched_json:
        print(f"  [제외] 미대응 JSON {len(unmatched_json)}개: {', '.join(unmatched_json[:5])}")

    frames = []
    source_fps = None
    for stem in matched_stems:
        json_path = labels[stem]
        try:
            with json_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            raw_index = payload["annotations"][0]["frame_index"]
            raw_rate = payload["videos"][0]["frame_rate"]
            index = int(raw_index)
            rate = float(raw_rate)
            if isinstance(raw_index, bool) or index < 0 or float(raw_index) != index:
                raise ValueError("frame_index가 0 이상의 정수가 아닙니다.")
            if isinstance(raw_rate, bool) or not math.isfinite(rate) or rate <= 0:
                raise ValueError("frame_rate가 양의 유한한 수가 아닙니다.")
            if source_fps is not None and not math.isclose(source_fps, rate, rel_tol=1e-9):
                raise ValueError("sequence 내 frame_rate가 일치하지 않습니다.")
            source_fps = rate
        except (OSError, ValueError, TypeError, KeyError, IndexError, OverflowError) as exc:
            raise ValueError(f"라벨 JSON 오류: {json_path}: {exc}") from exc
        frames.append((index, images[stem]))

    frames.sort(key=lambda pair: pair[0])
    if not frames:
        raise ValueError("전처리 가능한 대응 프레임이 없습니다.")
    indices = [index for index, _ in frames]
    timestamps = [round(index * 1000 / source_fps) for index in indices]
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"{sequence['id']}: frame_index 중복/역순")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{sequence['id']}: timestamp_ms가 엄격히 증가하지 않습니다.")
    if max(timestamps) > 9223372036854775807 or max(indices) > 9223372036854775807:
        raise ValueError("frame_index/timestamp가 int64 범위를 초과합니다.")
    return frames, timestamps, source_fps, unmatched_jpg, unmatched_json


# ============================================================
# 3. MediaPipe Pose 및 DIR8 배열 구성
# ============================================================

def load_dependencies():
    """설치된 패키지만 불러온다. 자동 설치나 다운로드는 수행하지 않는다."""
    try:
        np = importlib.import_module("numpy")
        mp = importlib.import_module("mediapipe")
        # 수정됨: OpenCV 추가
        cv2 = importlib.import_module("cv2")
    except ImportError as exc:
        raise RuntimeError(f"NumPy/MediaPipe/OpenCV를 불러올 수 없습니다: {exc}") from exc
    return np, mp, cv2


def initialize_arrays(frames, timestamps, np):
    """추정 실패 시에도 원래 대응 프레임의 시간축을 유지하도록 NaN으로 초기화한다."""
    count = len(frames)
    arrays = {
        "frame_indices": np.array([frame[0] for frame in frames], dtype=np.int64),
        "timestamps_ms": np.array(timestamps, dtype=np.int64),
        "bone_valid_mask": np.zeros((count, 8), dtype=bool),
    }
    shapes = {
        "raw_world": (count, 33, 3),
        "pelvis": (count, 3),
        "centered_world": (count, 33, 3),
        "bone_vectors": (count, 8, 3),
        "normalized_bone_vectors": (count, 8, 3),
    }
    for name, shape in shapes.items():
        arrays[name] = np.full(shape, np.nan, dtype=np.float32)
    return arrays


def store_pose(arrays, row, world, np):
    """MediaPipe 원본 world 좌표를 보존하고 pelvis 및 DIR8을 계산한다."""
    if world.shape != (33, 3):
        return False
    # Reference와 사용자 모두 MediaPipe world를 사용하므로 sign flip을 하지 않는다.
    arrays["raw_world"][row] = world
    pelvis = (world[23].astype(np.float64) + world[24].astype(np.float64)) / 2
    if np.isfinite(pelvis).all():
        arrays["pelvis"][row] = pelvis
        arrays["centered_world"][row] = world - pelvis

    parents, children = zip(*BONES.values())
    vectors = world[list(children)].astype(np.float64) - world[list(parents)]
    lengths = np.linalg.norm(vectors, axis=1)
    finite = np.isfinite(vectors).all(axis=1) & (
        np.abs(vectors) <= np.finfo(np.float32).max
    ).all(axis=1)
    valid = finite & np.isfinite(lengths) & (lengths > EPSILON)
    arrays["bone_vectors"][row, finite] = vectors[finite]
    arrays["normalized_bone_vectors"][row, valid] = (
        vectors[valid] / lengths[valid, None]
    )
    arrays["bone_valid_mask"][row] = valid
    return bool(np.isfinite(world).all())


def validate_arrays(arrays, np):
    """저장 전 NPZ 배열의 형태, 시간 순서 및 유효성 마스크를 점검한다."""
    count = len(arrays["frame_indices"])
    expected_shapes = {
        "frame_indices": (count,), "timestamps_ms": (count,),
        "raw_world": (count, 33, 3), "pelvis": (count, 3),
        "centered_world": (count, 33, 3), "bone_vectors": (count, 8, 3),
        "normalized_bone_vectors": (count, 8, 3), "bone_valid_mask": (count, 8),
    }
    for name, shape in expected_shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"{name} shape 오류: {arrays[name].shape} != {shape}")
    for name in ("frame_indices", "timestamps_ms"):
        if not np.all(np.diff(arrays[name]) > 0):
            raise ValueError(f"{name}의 값이 엄격히 증가하지 않습니다.")
    raw = arrays["raw_world"].astype(np.float64)
    hip_valid = np.isfinite(raw[:, [23, 24]]).all(axis=(1, 2))
    midpoint = (raw[hip_valid, 23] + raw[hip_valid, 24]) / 2
    if not np.allclose(arrays["pelvis"][hip_valid], midpoint, atol=1e-6):
        raise ValueError("pelvis 계산 검증 실패")
    centered = arrays["centered_world"][hip_valid]
    if not np.allclose((centered[:, 23] + centered[:, 24]) / 2, 0, atol=1e-6):
        raise ValueError("centered_world 계산 검증 실패")
    vectors = arrays["bone_vectors"].astype(np.float64)
    lengths = np.linalg.norm(vectors, axis=2)
    expected_mask = (np.isfinite(vectors).all(axis=2) & np.isfinite(lengths)
                     & (lengths > EPSILON))
    mask = arrays["bone_valid_mask"]
    if not np.array_equal(mask, expected_mask):
        raise ValueError("bone_valid_mask 검증 실패")
    units = arrays["normalized_bone_vectors"]
    if not np.allclose(np.linalg.norm(units[mask], axis=1), 1, atol=1e-6):
        raise ValueError("DIR8 단위벡터 검증 실패")
    if not np.isnan(units[~mask]).all():
        raise ValueError("유효하지 않은 DIR8 단위벡터는 NaN이어야 합니다.")


# ============================================================
# 4. 한 곡 추론 및 결과 저장
# ============================================================

def process_sequence(sequence, np, mp, cv2, overwrite=False): # 수정됨: cv2 인자 추가
    """대응 프레임만 처리해 기존 4-1과 호환되는 NPZ 및 메타데이터를 저장한다."""
    frames, timestamps, source_fps, unmatched_jpg, unmatched_json = load_frames(sequence)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = REFERENCE_DIR / f"{sequence['id']}.npz"
    metadata_path = REFERENCE_DIR / f"{sequence['id']}.json"
    if not overwrite and (npz_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"이미 결과 파일이 있습니다: {npz_path.name} 또는 {metadata_path.name}. "
            "덮어쓰려면 --overwrite를 지정하세요."
        )

    arrays = initialize_arrays(frames, timestamps, np)
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        output_segmentation_masks=False,
    )
    valid_count = 0
    failures = {
        "image_read_error": 0, "no_world_pose": 0,
        "invalid_world_shape": 0, "nonfinite_world": 0,
    }
    print(f"  Source FPS: {source_fps:g}; input frames: {len(frames)}")
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for row, ((index, image_path), timestamp) in enumerate(zip(frames, timestamps)):
            try:
                # 수정됨: 한글 경로 및 유니코드(NFC/NFD) 대응을 위한 cv2 우회 로드
                img_array = np.fromfile(str(image_path), np.uint8)
                cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if cv_img is None:
                    raise ValueError(f"cv2.imdecode로 이미지를 읽을 수 없습니다: {image_path}")
                rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
            except Exception as exc:
                failures["image_read_error"] += 1
                print(f"  [경고] frame {index} JPG 읽기 실패: {exc}", file=sys.stderr)
            else:
                # 추론 엔진 오류는 단순 미검출과 구별하기 위해 즉시 알린다.
                try:
                    result = landmarker.detect_for_video(image, timestamp)
                except Exception as exc:
                    raise RuntimeError(f"frame {index} MediaPipe 추론 실패: {exc}") from exc
                if not result.pose_world_landmarks:
                    failures["no_world_pose"] += 1
                else:
                    world = np.asarray(
                        [(p.x, p.y, p.z) for p in result.pose_world_landmarks[0]],
                        dtype=np.float32,
                    )
                    if world.shape != (33, 3):
                        failures["invalid_world_shape"] += 1
                    elif store_pose(arrays, row, world, np):
                        valid_count += 1
                    else:
                        failures["nonfinite_world"] += 1
            if (row + 1) % 100 == 0 or row + 1 == len(frames):
                print(f"  처리: {row + 1}/{len(frames)}", flush=True)

    validate_arrays(arrays, np)
    metadata = {
        "song": sequence["song"],
        "sequence": sequence["id"],
        "sequence_relative_path": sequence["relative_path"],
        "view": "F",
        "raw_source_dir": sequence["source_dir"].relative_to(PROJECT_ROOT).as_posix(),
        "raw_label_dir": sequence["label_dir"].relative_to(PROJECT_ROOT).as_posix(),
        "source_fps": source_fps,
        "frame_count": len(frames),
        "valid_pose_frames": valid_count,
        "invalid_pose_frames": len(frames) - valid_count,
        "valid_pose_definition": "All 33 world landmarks have finite XYZ; individual bone validity is separate.",
        "unmatched_jpg_count": len(unmatched_jpg),
        "unmatched_json_count": len(unmatched_json),
        "unmatched_jpg_stems": unmatched_jpg,
        "unmatched_json_stems": unmatched_json,
        "failure_counts": failures,
        "mediapipe_version": mp.__version__,
        "model_path": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "running_mode": "VIDEO",
        "num_poses": 1,
        "coordinate_system": "Original MediaPipe Pose World Landmarks; no axis flip.",
        "pelvis_centering": "pelvis=(world[23]+world[24])/2; centered_world=world-pelvis",
        "timestamp_rule": "round(frame_index*1000/source_fps), strictly increasing",
        "input_policy": "Only matching JPG and JSON filename stems are processed; raw data is unchanged.",
        "missing_pose_policy": "Preserve matched frame and timestamp; pose arrays NaN, bone mask False.",
        "epsilon": EPSILON,
        "landmark_names": list(LANDMARK_NAMES),
        "comparison_joint_order": list(COMPARISON_JOINTS),
        "comparison_joint_mapping": COMPARISON_JOINTS,
        "bone_mapping": {
            name: {"index": i, "parent": pair[0], "child": pair[1]}
            for i, (name, pair) in enumerate(BONES.items())
        },
        "part_grouping": PART_GROUPS,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
    }

    # metadata 직렬화 오류를 먼저 점검한 뒤 NPZ를 저장한다.
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    np.savez_compressed(npz_path, **arrays)
    metadata_path.write_text(metadata_text, encoding="utf-8")
    print(
        f"  [완료] {sequence['id']} | 입력 {len(frames)} | 정상 Pose {valid_count} | "
        f"미검출/무효 {len(frames)-valid_count}\n"
        f"         NPZ: {npz_path}\n         메타데이터: {metadata_path}"
    )
    return {"id": sequence["id"], "frames": len(frames), "valid": valid_count}


# ============================================================
# 5. 명령행 인터페이스
# ============================================================

def select_sequences(sequences, args):
    """번호, 곡명, sequence ID 또는 전체 처리 요청을 해석한다."""
    if args.all:
        return sequences
    if args.song is None:
        for i, item in enumerate(sequences, 1):
            print(f"{i:2d}. {item['song']} / {item['id']}")
        try:
            choice = input("처리할 곡 번호 입력: ").strip()
        except EOFError as exc:
            raise ValueError("비대화형 실행에는 --song 또는 --all을 사용하세요.") from exc
    else:
        choice = args.song.strip()
    choice = nfc(choice)
    if choice.isdecimal() and 1 <= int(choice) <= len(sequences):
        return [sequences[int(choice)-1]]
    matched = [item for item in sequences if choice in (item["song"], item["id"])]
    if len(matched) != 1:
        raise ValueError("곡을 유일하게 선택할 수 없습니다. --list-songs로 확인하세요.")
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-songs", action="store_true", help="사용 가능한 F-view 곡 목록")
    parser.add_argument("--song", help="곡 번호, 정확한 곡명 또는 F-view sequence ID")
    parser.add_argument("--all", action="store_true", help="발견한 모든 F-view sequence 처리")
    parser.add_argument("--overwrite", action="store_true", help="기존 NPZ/메타데이터 덮어쓰기")
    args = parser.parse_args()
    if args.all and args.song is not None:
        parser.error("--song과 --all은 함께 사용할 수 없습니다.")
    try:
        sequences = discover_sequences()
        if args.list_songs:
            for i, item in enumerate(sequences, 1):
                print(f"{i:2d}. {item['song']} / {item['id']}")
            return 0
        selected = select_sequences(sequences, args)
        # python preprocessing/prepare_reference_pose.py로 실행해도 프로젝트 패키지를 찾도록 한다.
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from app.services.pose_model import ensure_pose_model
        ensure_pose_model(MODEL_PATH)
        # 수정됨: cv2 객체 추가로 전달 받음
        np, mp, cv2 = load_dependencies()
        successful, failed = [], []
        for sequence in selected:
            print("=" * 72)
            print(f"{sequence['song']} / {sequence['id']}")
            try:
                # 수정됨: cv2 전달
                successful.append(process_sequence(sequence, np, mp, cv2, args.overwrite))
            except (OSError, ValueError, RuntimeError, OverflowError) as exc:
                print(f"  [실패] {sequence['id']}: {exc}", file=sys.stderr)
                failed.append(sequence['id'])
                if not args.all:
                    break
        print("=" * 72)
        print(f"전처리 성공: {len(successful)}곡 / 실패: {len(failed)}곡")
        print(f"생성 입력 프레임 총합: {sum(item['frames'] for item in successful)}")
        print(f"유효 Pose 프레임 총합: {sum(item['valid'] for item in successful)}")
        if failed:
            print("실패한 sequence: " + ", ".join(failed))
        return 1 if failed else 0
    except (OSError, ValueError, RuntimeError, OverflowError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("사용자가 전처리를 중단했습니다.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())