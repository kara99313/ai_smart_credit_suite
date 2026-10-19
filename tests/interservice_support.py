"""Explicit authenticated transport for existing business regression tests."""
import os
from unittest.mock import patch
from fastapi.testclient import TestClient as RawTestClient

TEST_SECRET = "test-interservice-credential"

class TestClient(RawTestClient):
    __test__ = False
    def request(self, *args, **kwargs):
        supplied = dict(kwargs.pop("headers", {}) or {})
        supplied.setdefault("X-Interservice-Secret", TEST_SECRET)
        with patch.dict(os.environ, {"INTERSERVICE_SECRET": TEST_SECRET}):
            return super().request(*args, headers=supplied, **kwargs)
