"""Lot 2B-1: authorization, disabled source downloads and public API surface."""
from pathlib import Path
from unittest.mock import patch
import pytest
from streamlit.testing.v1 import AppTest
from utils import admin_auth as auth, settings, web_search as web, health_diagnostics as checks

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('state', [{}, {'admin': True, '_admin_grant': 'forged'}])
def test_preferences_denied_before_filesystem(state):
    with patch.object(auth, '_session', return_value=state), patch.object(settings, '_ensure_data_dir') as mkdir, patch.object(Path, 'write_text') as write:
        with pytest.raises(PermissionError):
            settings.save_settings(settings.AppSettings(enable_web_search=True))
        with pytest.raises(PermissionError):
            settings.update_settings({'enable_web_search': True})
        mkdir.assert_not_called()
        write.assert_not_called()

@pytest.mark.parametrize('url', ['https://example.org', 'http://127.0.0.1/private', 'http://169.254.169.254/', 'file:///etc/passwd'])
def test_source_download_never_uses_network(url):
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')) as network, patch.object(web, 'web_configuration') as config:
        assert web.fetch_and_clean(url) == ''
        network.assert_not_called()
        config.assert_not_called()

@pytest.mark.parametrize('fetch', [False, True])
def test_fetch_full_never_fetches_sources(fetch):
    hit = dict(title='Source', href='https://example.org/report', body='Extrait fournisseur fiable.')
    with patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden')) as network, patch.object(web, 'fetch_and_clean', side_effect=AssertionError('Source download forbidden')) as download:
        result = web.build_web_context([hit], fetch_full=fetch)
        assert hit['body'] in result
        assert hit['href'] in result
        network.assert_not_called()
        download.assert_not_called()

@pytest.mark.parametrize('configured', [False, True])
def test_health_page_denied_before_any_technical_read(configured):
    with patch.object(auth, '_session', return_value={'admin': True, '_admin_grant': 'forged'}), patch.object(auth, 'configured', return_value=configured), patch.object(checks, 'environment_inventory') as environment, patch.object(checks, 'provider_inventory') as providers, patch.object(checks, 'corpus_inventory') as corpus, patch.object(checks, 'backend_check') as backend, patch('requests.sessions.Session.request') as network:
        app = AppTest.from_file(str(ROOT / 'app/health_check.py')).run()
        assert not app.exception
        for probe in (environment, providers, corpus, backend, network):
            probe.assert_not_called()
        assert not app.json
        assert 'health_checks' not in app.session_state


def test_public_health_and_removed_debug_routes():
    from interservice_support import TestClient
    import server
    client = TestClient(server.app)
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status': 'ok', 'model_loaded': True}
    schema = server.app.openapi()
    for path in ('/debug/imports', '/debug/dtypes'):
        assert client.get(path).status_code == 404
        assert path not in schema['paths']
        assert all(getattr(route, 'path', None) != path for route in server.app.routes)
