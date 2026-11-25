"""MediaPipe 모델 준비 헬퍼 테스트. 네트워크와 실제 모델 파일 없이 임시 폴더에서만 실행한다."""
from contextlib import redirect_stderr
import hashlib
import http.client
import io
from pathlib import Path
import re
import socket
import stat
import tempfile
import threading
import unittest
import urllib.error
from unittest.mock import patch

from app.services import pose_model


ROOT = Path(__file__).resolve().parents[1]
DATA = bytes(range(256)) * 1000  # 256,000 bytes: 여러 청크로 나뉘어 스트리밍된다.
DATA_SHA256 = hashlib.sha256(DATA).hexdigest()


class FakeResponse:
    """urlopen 응답 대역. on_read는 첫 read 직후 호출된다."""

    def __init__(self, data, status=200, on_read=None, fail_after=None):
        self.stream, self.status = io.BytesIO(data), status
        self.on_read, self.fail_after, self.reads = on_read, fail_after, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, size):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise self.fail_after_exc
        chunk = self.stream.read(size)
        if self.reads == 1 and self.on_read:
            self.on_read()
        return chunk


class FakeOpener:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def __call__(self, url, timeout):
        self.calls.append((url, timeout))
        if self.error is not None:
            raise self.error
        return self.response


class PoseModelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="danc3d_pose_model_")
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name)
        self.path = self.dir / "models" / "mediapipe" / "pose_landmarker_full.task"
        for name, value in (("MODEL_SIZE", len(DATA)), ("MODEL_SHA256", DATA_SHA256)):
            patcher = patch.object(pose_model, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def ensure(self, opener, **kwargs):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = pose_model.ensure_pose_model(self.path, opener=opener, **kwargs)
        return result, stderr.getvalue()

    def assert_failure(self, opener, text):
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(pose_model.PoseModelError) as caught:
            pose_model.ensure_pose_model(self.path, opener=opener)
        self.assertIn(text, str(caught.exception))
        self.assertIn(pose_model.MODEL_URL, str(caught.exception))  # 수동 다운로드 안내
        self.assertFalse(self.path.exists())
        self.assert_no_temp_files()
        return caught.exception

    def assert_no_temp_files(self):
        self.assertEqual(list(self.path.parent.glob("*.part")) if self.path.parent.exists() else [], [])

    def write_existing(self, data):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(data)

    # ---------- 기존 파일 ----------
    def test_existing_model_is_used_without_network(self):
        self.write_existing(DATA)
        opener = FakeOpener(error=AssertionError("network must not be used"))
        result, stderr = self.ensure(opener)
        self.assertEqual(result, self.path)
        self.assertEqual(opener.calls, [])
        self.assertEqual(stderr, "")
        self.assertEqual(self.path.read_bytes(), DATA)

    def test_existing_model_with_wrong_size_warns_and_is_kept(self):
        self.write_existing(b"user model")
        opener = FakeOpener(error=AssertionError("network must not be used"))
        result, stderr = self.ensure(opener)
        self.assertEqual(result, self.path)
        self.assertEqual(opener.calls, [])
        self.assertIn("Warning", stderr)
        self.assertIn("expected {}".format(len(DATA)), stderr)
        self.assertEqual(self.path.read_bytes(), b"user model")

    def test_existing_model_with_wrong_hash_warns_and_is_kept(self):
        other = bytes(len(DATA))
        self.write_existing(other)
        opener = FakeOpener(error=AssertionError("network must not be used"))
        result, stderr = self.ensure(opener)
        self.assertEqual(opener.calls, [])
        self.assertIn("SHA-256 differs", stderr)
        self.assertEqual(self.path.read_bytes(), other)

    # ---------- 다운로드 ----------
    def test_missing_model_is_downloaded_and_verified(self):
        opener = FakeOpener(FakeResponse(DATA))
        result, stderr = self.ensure(opener)
        self.assertEqual(result, self.path)
        self.assertEqual(opener.calls, [(pose_model.MODEL_URL, pose_model.TIMEOUT_SECONDS)])
        self.assertEqual(self.path.read_bytes(), DATA)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o644)
        self.assertIn("Saved MediaPipe model", stderr)
        self.assert_no_temp_files()

    def test_hash_mismatch_saves_nothing(self):
        self.assert_failure(FakeOpener(FakeResponse(bytes(len(DATA)))), "SHA-256 mismatch")

    def test_oversized_download_is_aborted(self):
        self.assert_failure(FakeOpener(FakeResponse(DATA + b"x")), "exceeds")

    def test_truncated_download_saves_nothing(self):
        self.assert_failure(FakeOpener(FakeResponse(DATA[:-1])), "expected {}".format(len(DATA)))

    def test_network_error_saves_nothing(self):
        self.assert_failure(FakeOpener(error=urllib.error.URLError("no route")), "no route")

    def test_connect_timeout_saves_nothing(self):
        self.assert_failure(FakeOpener(error=socket.timeout("timed out")), "timed out")

    def test_timeout_while_reading_cleans_partial_file(self):
        response = FakeResponse(DATA, fail_after=2)
        response.fail_after_exc = TimeoutError("read timed out")
        self.assert_failure(FakeOpener(response), "read timed out")

    def test_incomplete_read_cleans_partial_file(self):
        response = FakeResponse(DATA, fail_after=1)
        response.fail_after_exc = http.client.IncompleteRead(b"", 10)
        self.assert_failure(FakeOpener(response), "IncompleteRead")

    def test_http_error_saves_nothing(self):
        error = urllib.error.HTTPError(pose_model.MODEL_URL, 404, "Not Found", {}, None)
        self.assert_failure(FakeOpener(error=error), "404")

    def test_non_200_status_saves_nothing(self):
        self.assert_failure(FakeOpener(FakeResponse(DATA, status=204)), "HTTP 204")

    def test_certificate_error_adds_macos_hint(self):
        error = urllib.error.URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        exception = self.assert_failure(FakeOpener(error=error), "CERTIFICATE_VERIFY_FAILED")
        self.assertIn("Install Certificates.command", str(exception))

    def test_interrupt_cleans_partial_file_and_propagates(self):
        response = FakeResponse(DATA, fail_after=1)
        response.fail_after_exc = KeyboardInterrupt()
        with redirect_stderr(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            pose_model.ensure_pose_model(self.path, opener=FakeOpener(response))
        self.assertFalse(self.path.exists())
        self.assert_no_temp_files()

    # ---------- 동시성 ----------
    def test_final_file_created_by_another_process_during_download_is_kept(self):
        other = b"created by another process"

        def create_final():
            self.path.write_bytes(other)

        result, stderr = self.ensure(FakeOpener(FakeResponse(DATA, on_read=create_final)))
        self.assertEqual(result, self.path)
        self.assertEqual(self.path.read_bytes(), other)  # 덮어쓰지 않는다.
        self.assertIn("created by another process", stderr)
        self.assertIn("Warning", stderr)  # 기존 파일 규칙대로 검증 경고
        self.assert_no_temp_files()

    def test_concurrent_downloads_keep_one_verified_file(self):
        barrier = threading.Barrier(2, timeout=5)
        results, errors = [], []

        def worker():
            response = FakeResponse(DATA, on_read=barrier.wait)  # 두 다운로드가 모두 진행 중일 때 만난다.
            try:
                results.append(pose_model.ensure_pose_model(self.path, opener=FakeOpener(response)))
            except Exception as exc:  # pragma: no cover - 실패 시 보고용
                errors.append(exc)

        with redirect_stderr(io.StringIO()):
            threads = [threading.Thread(target=worker) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(results, [self.path, self.path])
        self.assertEqual(self.path.read_bytes(), DATA)
        self.assert_no_temp_files()

    # ---------- CLI ----------
    def test_cli_reports_failure_with_exit_code(self):
        with patch.object(pose_model, "ensure_pose_model", side_effect=pose_model.PoseModelError("boom")), \
             redirect_stderr(io.StringIO()) as stderr:
            self.assertEqual(pose_model.main(), 1)
        self.assertIn("boom", stderr.getvalue())


class PinnedConstantTests(unittest.TestCase):
    def test_pinned_model_constants(self):
        self.assertEqual(pose_model.MODEL_URL,
                         "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/"
                         "float16/latest/pose_landmarker_full.task?generation=1682642787774579")
        self.assertEqual(pose_model.MODEL_SHA256, "4eaa5eb7a98365221087693fcc286334cf0858e2eb6e15b506aa4a7ecdcec4ad")
        self.assertEqual(pose_model.MODEL_SIZE, 9398198)
        self.assertEqual(pose_model.DEFAULT_MODEL_PATH, ROOT / "models" / "mediapipe" / "pose_landmarker_full.task")

    def test_python_and_js_constants_match(self):
        source = (ROOT / "frontend" / "scripts" / "prepare-pose.mjs").read_text(encoding="utf-8")

        def js_const(name):
            match = re.search(r"export const {}=('([^']*)'|(\d+));".format(name), source)
            self.assertIsNotNone(match, name)
            return match.group(2) if match.group(2) is not None else int(match.group(3))

        self.assertEqual(js_const("MODEL_URL"), pose_model.MODEL_URL)
        self.assertEqual(js_const("MODEL_SHA256"), pose_model.MODEL_SHA256)
        self.assertEqual(js_const("MODEL_SIZE"), pose_model.MODEL_SIZE)
        self.assertEqual(js_const("TIMEOUT_MS"), pose_model.TIMEOUT_SECONDS * 1000)
        self.assertIn("models/mediapipe/pose_landmarker_full.task", source)


if __name__ == "__main__":
    unittest.main()
