"""MediaPipe Pose Landmarker 모델 파일을 준비한다.

로컬 파일이 있으면 네트워크 없이 그대로 사용하고, 없을 때만 Google 공식 저장소의
generation 고정 URL에서 받아 크기와 SHA-256을 모두 검증한 뒤 저장한다.
frontend/scripts/prepare-pose.mjs와 같은 상수와 규칙을 사용한다.
"""
from contextlib import suppress
import hashlib
import http.client
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "mediapipe" / "pose_landmarker_full.task"
# Pose Landmarker Full (float16). generation으로 고정해 latest가 바뀌어도 같은 바이트만 받는다.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/"
             "float16/latest/pose_landmarker_full.task?generation=1682642787774579")
MODEL_SHA256 = "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad"
MODEL_SIZE = 9398198
TIMEOUT_SECONDS = 60
CHUNK_SIZE = 1 << 16


class PoseModelError(RuntimeError):
    """모델을 받거나 검증하지 못했다. 호출부의 기존 RuntimeError 처리 경로를 그대로 탄다."""


def file_sha256(path):
    """파일의 SHA-256 hex 문자열."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _use_existing(path):
    """기존 파일은 그대로 사용한다. 크기/해시가 다르면 경고만 하고 삭제·교체하지 않는다."""
    size = path.stat().st_size
    if size != MODEL_SIZE:
        print("Warning: {} is {} bytes, expected {}; using the existing file as-is.".format(
            path, size, MODEL_SIZE), file=sys.stderr)
    elif file_sha256(path) != MODEL_SHA256:
        print("Warning: {} SHA-256 differs from {}; using the existing file as-is.".format(
            path, MODEL_SHA256), file=sys.stderr)
    return path


def _download(file, opener, timeout):
    """고정 URL을 스트리밍으로 받아 file에 쓰고, 크기와 SHA-256을 모두 검증한다."""
    digest, received = hashlib.sha256(), 0
    with opener(MODEL_URL, timeout=timeout) as response:
        status = getattr(response, "status", 200)
        if status != 200:
            raise PoseModelError("HTTP {}".format(status))
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            received += len(chunk)
            if received > MODEL_SIZE:
                raise PoseModelError("download exceeds {} bytes".format(MODEL_SIZE))
            digest.update(chunk)
            file.write(chunk)
    if received != MODEL_SIZE:
        raise PoseModelError("downloaded {} bytes, expected {}".format(received, MODEL_SIZE))
    if digest.hexdigest() != MODEL_SHA256:
        raise PoseModelError("SHA-256 mismatch: got {}, expected {}".format(digest.hexdigest(), MODEL_SHA256))
    file.flush()
    os.fsync(file.fileno())


def _failure_message(path, exc):
    message = ("Could not prepare MediaPipe model {}: {}\n"
               "Download it manually from {}\n"
               "and save it to that path ({} bytes, SHA-256 {}).").format(
                   path, exc, MODEL_URL, MODEL_SIZE, MODEL_SHA256)
    if "CERTIFICATE_VERIFY_FAILED" in str(exc):
        message += "\nOn macOS python.org builds, run 'Install Certificates.command' first."
    return message


def ensure_pose_model(path=DEFAULT_MODEL_PATH, *, opener=urllib.request.urlopen, timeout=TIMEOUT_SECONDS):
    """모델 경로를 반환한다. 파일이 없을 때만 다운로드한다.

    검증을 통과한 고유 임시 파일을 os.link로 최종 경로에 연결한다. link는 대상이 이미
    있으면 실패하므로 동시에 다른 프로세스가 만든 최종 파일을 덮어쓰지 않는다.
    """
    path = Path(path)
    if path.is_file():
        return _use_existing(path)
    temp_name = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        print("MediaPipe model not found: {}\nDownloading {}".format(path, MODEL_URL), file=sys.stderr)
        fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".part")
        with os.fdopen(fd, "wb") as file:
            _download(file, opener, timeout)
        os.chmod(temp_name, 0o644)  # mkstemp의 0600 대신 일반 파일 권한
        try:
            os.link(temp_name, path)
        except FileExistsError:
            print("{} was created by another process; keeping it.".format(path), file=sys.stderr)
            return _use_existing(path)
    except (OSError, ValueError, http.client.HTTPException, PoseModelError) as exc:
        raise PoseModelError(_failure_message(path, exc)) from exc
    finally:
        if temp_name is not None:
            with suppress(FileNotFoundError):
                os.unlink(temp_name)
    print("Saved MediaPipe model: {}".format(path), file=sys.stderr)
    return path


def main():
    """python -m app.services.pose_model: 모델을 수동으로 준비한다."""
    try:
        print(ensure_pose_model())
        return 0
    except PoseModelError as exc:
        print("Error: {}".format(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
