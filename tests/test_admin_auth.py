"""Lot 2A: real credentials, session isolation and authorization before effects."""
import os
from pathlib import Path
from unittest.mock import patch, Mock
import pytest
from streamlit.testing.v1 import AppTest
from utils import admin_auth as auth, integration_services as services, health_diagnostics as health, audit
from admin_auth_support import CONFIG, PASSWORD

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def context(monkeypatch, tmp_path):
    for key, value in CONFIG.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(audit, 'AUDIT_PATH', tmp_path / 'audit.csv')
    state = {}
    monkeypatch.setattr(auth, '_session', lambda: state)
    auth._FAILURES.clear()
    auth._GRANTS.clear()
    yield state
    auth._FAILURES.clear()
    auth._GRANTS.clear()


def test_denies_direct_operations_before_effects(context, tmp_path):
    with patch.object(Path, 'read_text') as read, patch.object(health, 'build_client') as build:
        with pytest.raises(PermissionError):
            services.save_llm_configuration('openai', 'model', path=tmp_path / 'env')
        with pytest.raises(PermissionError):
            health.provider_check('openai')
        read.assert_not_called()
        build.assert_not_called()
    context.update(admin=True, actor='test-admin', _admin_grant='forged')
    assert not auth.is_admin()
    assert not audit.AUDIT_PATH.exists()


@pytest.mark.parametrize('value', ['', 'bad', 'pbkdf2_sha256$1$00$00'])
def test_invalid_config_fails_closed(context, monkeypatch, value):
    monkeypatch.setenv('ADMIN_PASSWORD_HASH', value)
    assert not auth.login('test-admin', PASSWORD)
    assert not auth.is_admin()


def test_missing_config(context, monkeypatch):
    monkeypatch.delenv('ADMIN_USERNAME')
    assert not auth.login('test-admin', PASSWORD)
    assert not auth.is_admin()


def test_password_and_username_required(context):
    assert not auth.login('wrong', PASSWORD)
    assert not auth.login('test-admin', 'wrong')
    assert auth.login('test-admin', PASSWORD)
    assert auth.require_admin() == 'test-admin'
    assert PASSWORD not in repr(context)


def test_throttle_shared_across_sessions(context, monkeypatch):
    for _ in range(5):
        assert not auth.login('test-admin', 'bad')
    monkeypatch.setattr(auth, '_session', lambda: {})
    assert not auth.login('test-admin', PASSWORD)
    with patch.object(auth.time, 'monotonic', return_value=auth.time.monotonic() + 301):
        # Use a stable new browser session.
        new = {}
        monkeypatch.setattr(auth, '_session', lambda: new)
        assert auth.login('test-admin', PASSWORD)


def test_expiration_logout_rotation_and_new_session(context, monkeypatch):
    assert auth.login('test-admin', PASSWORD)
    token = context[auth._KEY]
    other = {auth._KEY: token}
    with patch.object(auth, '_session', return_value=other):
        assert not auth.is_admin()
    assert auth.login('test-admin', PASSWORD)
    with patch.object(auth.time, 'monotonic', return_value=auth.time.monotonic() + 1801):
        assert not auth.is_admin()
    assert auth.login('test-admin', PASSWORD)
    monkeypatch.setenv('ADMIN_USERNAME', 'changed')
    assert not auth.is_admin()
    monkeypatch.setenv('ADMIN_USERNAME', 'test-admin')
    assert auth.login('test-admin', PASSWORD)
    auth.logout()
    assert not auth.is_admin()


def test_authorized_save_audit_without_credentials(context, tmp_path):
    assert auth.login('test-admin', PASSWORD)
    with patch.dict(os.environ):
        services.save_llm_configuration('openai', 'example-model', 'test-provider-secret', path=tmp_path / 'env')
    raw = audit.AUDIT_PATH.read_text()
    assert 'test-admin' in raw
    for secret in (PASSWORD, CONFIG['ADMIN_PASSWORD_HASH'], 'test-provider-secret'):
        assert secret not in raw


def test_direct_admin_page_requires_login(monkeypatch, tmp_path):
    for key, value in CONFIG.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(audit, 'AUDIT_PATH', tmp_path / 'audit.csv')
    auth._FAILURES.clear()
    with patch('utils.llm_providers.simple_chat') as chat, patch.object(services, 'save_llm_configuration') as save:
        app = AppTest.from_file(str(ROOT / 'app/admin.py')).run()
        assert not app.exception
        assert not any(b.label == 'Tester le LLM' for b in app.button)
        app.text_input[0].set_value('test-admin')
        app.text_input[1].set_value(PASSWORD)
        next(b for b in app.button if b.label == 'Se connecter').click().run()
        assert not app.exception
        assert any(b.label == 'Tester le LLM' for b in app.button)
        app.run()
        assert any(b.label == 'Se deconnecter' for b in app.button)
        fresh = AppTest.from_file(str(ROOT / 'app/admin.py')).run()
        assert not any(b.label == 'Tester le LLM' for b in fresh.button)
        next(b for b in app.button if b.label == 'Se deconnecter').click().run()
        assert not any(b.label == 'Tester le LLM' for b in app.button)
        chat.assert_not_called()
        save.assert_not_called()


@pytest.mark.parametrize('filename,label', [
    ('integrations.py', 'Tester le fournisseur (appel reel)'),
    ('health_check.py', 'Tester le fournisseur selectionne (appel reel)')])
def test_other_pages_disable_provider_tests(monkeypatch, filename, label):
    monkeypatch.delenv('ADMIN_USERNAME', raising=False)
    with patch.object(health, 'build_client') as build:
        app = AppTest.from_file(str(ROOT / 'app' / filename), default_timeout=30).run()
        assert not app.exception
        assert next(b for b in app.button if b.label == label).disabled
        build.assert_not_called()


def test_direct_admin_test_function_denied(context):
    import runpy
    with patch('utils.llm_providers.simple_chat') as chat:
        namespace = runpy.run_path(str(ROOT / 'app/admin.py'), run_name='admin_auth_test')
        with pytest.raises(PermissionError):
            namespace['test_admin_llm']('test')
        chat.assert_not_called()


def test_expired_session_denies_provider_before_client_creation(context):
    assert auth.login('test-admin', PASSWORD)
    with patch.object(auth.time, 'monotonic', return_value=auth.time.monotonic() + 1801):
        with patch.object(health, 'build_client') as build:
            with pytest.raises(PermissionError):
                health.provider_check('openai')
            build.assert_not_called()


def test_provider_audit_uses_verified_identity_without_output(context):
    from types import SimpleNamespace
    assert auth.login('test-admin', PASSWORD)
    with patch.object(health, 'get_setting', return_value='test-provider-secret'), patch.object(health, 'build_client'), patch.object(health, 'invoke', return_value=SimpleNamespace(content='private-response')):
        assert health.provider_check('openai')['state'] == 'Fonctionnel'
    raw = audit.AUDIT_PATH.read_text()
    assert 'test-admin' in raw and 'provider_check' in raw
    assert 'private-response' not in raw and 'test-provider-secret' not in raw


def test_no_streamlit_context_denies_by_default():
    with pytest.raises(PermissionError):
        auth.require_admin()


def test_observability_requires_valid_admin_before_reading(monkeypatch):
    for key, value in CONFIG.items():
        monkeypatch.setenv(key, value)
    auth._FAILURES.clear()
    with patch('utils.observability_data.prediction_snapshot') as predictions, patch('utils.observability_data.csv_snapshot') as journals:
        app = AppTest.from_file(str(ROOT / 'app/observability.py')).run()
        assert not app.exception
        assert not app.metric
        assert not any(b.label == 'Actualiser les journaux' for b in app.button)
        predictions.assert_not_called()
        journals.assert_not_called()
        app.text_input[0].set_value('test-admin')
        app.text_input[1].set_value('invalid')
        next(b for b in app.button if b.label == 'Se connecter').click().run()
        predictions.assert_not_called()
        journals.assert_not_called()
    # Real grants and the real page gate, with temporary empty sources.
    with patch('utils.observability_data.prediction_snapshot', return_value={
            'source': {'state': 'missing', 'size': None}, 'warnings': []}) as predictions, patch(
            'utils.observability_data.csv_snapshot', return_value={
                'source': {'state': 'missing', 'size': None}, 'warnings': []}) as journals:
        app.text_input[0].set_value('test-admin')
        app.text_input[1].set_value(PASSWORD)
        next(b for b in app.button if b.label == 'Se connecter').click().run()
        assert not app.exception
        assert any(b.label == 'Actualiser les journaux' for b in app.button)
        assert predictions.called and journals.called
        predictions.reset_mock(); journals.reset_mock()
        next(b for b in app.button if b.label == 'Se deconnecter').click().run()
        predictions.assert_not_called(); journals.assert_not_called()
