"""Local Vite origins may connect; unrelated and explicitly excluded origins may not."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from app.main import create_app


class WebSocketOriginTests(unittest.TestCase):
    def test_default_local_origins_and_foreign_origin(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
            os.environ.pop('DANC3D_CORS_ORIGINS', None)
            with TestClient(create_app(db_path=Path(directory) / 'test.db')) as client:
                for origin in ('http://localhost:5173', 'http://127.0.0.1:5173'):
                    # An application error proves the handshake succeeded; no record is created.
                    with client.websocket_connect('/ws/sessions/missing', headers={'origin':origin}) as socket:
                        self.assertEqual(socket.receive_json()['code'], 'session_not_found_or_expired')
                with self.assertRaises(WebSocketDisconnect) as error:
                    with client.websocket_connect('/ws/sessions/missing', headers={'origin':'https://example.com'}):
                        pass
                self.assertEqual(error.exception.code, 1008)

    def test_explicit_configuration_overrides_defaults(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'DANC3D_CORS_ORIGINS':'http://localhost:5173'}):
            with TestClient(create_app(db_path=Path(directory) / 'test.db')) as client:
                with self.assertRaises(WebSocketDisconnect):
                    with client.websocket_connect('/ws/sessions/missing', headers={'origin':'http://127.0.0.1:5173'}):
                        pass
