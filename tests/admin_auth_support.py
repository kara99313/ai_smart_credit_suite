"""Explicit authenticated context for existing authorized-operation regression tests."""
import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from utils import admin_auth as auth

PASSWORD = 'test-only-password'
SALT = b'test-only-salt-16'
CONFIG = {'ADMIN_USERNAME': 'test-admin', 'ADMIN_PASSWORD_HASH':
          'pbkdf2_sha256$600000$' + SALT.hex() + '$' +
          hashlib.pbkdf2_hmac('sha256', PASSWORD.encode(), SALT, 600000).hex()}


def authenticated_admin_test(cls):
    original = cls.setUp
    def setup(self):
        original(self)
        state = {}
        for item in (patch.dict(os.environ, CONFIG), patch.object(auth, '_session', return_value=state)):
            item.start()
            self.addCleanup(item.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        item = patch('utils.audit.AUDIT_PATH', Path(temporary.name) / 'audit.csv')
        # Preserve audit paths already isolated by the test itself.
        from utils import audit
        if audit.AUDIT_PATH == Path(__file__).resolve().parents[1] / 'data/audit_log.csv':
            item.start()
            self.addCleanup(item.stop)
        with auth._LOCK:
            auth._FAILURES.clear()
        assert auth.login(CONFIG['ADMIN_USERNAME'], PASSWORD)
        self.addCleanup(auth.logout)
    cls.setUp = setup
    return cls
