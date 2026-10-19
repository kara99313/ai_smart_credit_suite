"""Lot 4A: untrusted API errors remain private."""
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import requests
from interservice_support import TestClient
import server
from test_prediction import PAYLOAD
from utils import api_client, health_diagnostics

PRIVATE = 'Traceback RuntimeError https://user:SIMULATED_PASSWORD@host/?token=SIMULATED_TOKEN SIMULATED_KEY'

def safe(value):
    for fragment in PRIVATE.split():
        assert fragment not in str(value)

@pytest.mark.parametrize('body', [{'detail': PRIVATE}, [PRIVATE], PRIVATE])
def test_client_http_body(body):
    response = SimpleNamespace(ok=False, status_code=500, text=PRIVATE, json=lambda: body)
    with patch.object(api_client.requests, 'post', return_value=response):
        with pytest.raises(api_client.ApiError) as caught:
            api_client.predict_credit_api({}, base='http://localhost')
    safe(caught.value)

@pytest.mark.parametrize('error', [requests.ConnectionError(PRIVATE), ValueError(PRIVATE)])
def test_client_transport_and_json_errors(error):
    with patch.object(api_client.requests, 'post', side_effect=error):
        with pytest.raises(api_client.ApiError) as caught:
            api_client.predict_credit_api({}, base='http://localhost')
    safe(caught.value)
    assert caught.value.__suppress_context__
    with patch.object(api_client.requests, 'get', side_effect=error):
        ok, message = api_client.ping_backend(base='http://localhost')
    assert not ok
    safe(message)

def test_health_body_is_not_echoed():
    for status in (200, 500):
        response = SimpleNamespace(status_code=status, json=lambda: {'status': 'ok', 'model_loaded': True, 'detail': PRIVATE})
        with patch.object(health_diagnostics.requests, 'get', return_value=response):
            safe(health_diagnostics.backend_check('http://localhost'))

@pytest.mark.parametrize('route,payload', [
    ('/tools/predict_credit', {'data': {}}),
    ('/tools/rag_search', {'query': 'test'}),
    ('/tools/web_search', {'query': 'test'}),
    ('/tools/generate_report_html', {'client_id': 'test', 'pd_value': .2, 'threshold': .1, 'inputs_json': '{}'}),
    ('/tools/now', None),
])
def test_tool_errors(route, payload):
    with patch.object(server, '_import_agent_tools', side_effect=RuntimeError(PRIVATE)):
        client = TestClient(server.app)
        response = client.get(route) if payload is None else client.post(route, json=payload)
    assert response.status_code == 500
    safe(response.text)

def test_prediction_and_explanation_errors():
    with patch.object(server, '_compute_pd', side_effect=RuntimeError(PRIVATE)):
        for route in ('/api/predict', '/tools/explain'):
            response = TestClient(server.app).post(route, json=PAYLOAD)
            assert response.status_code == 500
            safe(response.text)
    with patch('utils.scoring_explanations.global_coefficient_importance', side_effect=ValueError(PRIVATE)):
        response = TestClient(server.app).post('/tools/explain', json=PAYLOAD)
        assert response.status_code == 200
        assert response.json()['ok'] is False
        safe(response.text)

def test_validation_hides_input():
    response = TestClient(server.app).post('/api/predict', json=dict(PAYLOAD, Income=PRIVATE))
    assert response.status_code == 422
    safe(response.text)

def test_assistant_creation_error_is_safe_and_remains_safe_in_history():
    from streamlit.testing.v1 import AppTest
    with patch('utils.llm_status.llm_status_badge'), patch(
            'utils.agents.create_credit_agent', side_effect=RuntimeError(PRIVATE)) as create:
        app = AppTest.from_file('app/chatbot_assistant.py', default_timeout=30).run()
        app.chat_input[0].set_value('question').run()
        assert not app.exception
        create.assert_called_once()
        safe(' '.join(item.value for item in app.markdown))
        assert any('Erreur agent' in item.value for item in app.markdown)
        app.run()
        safe(' '.join(item.value for item in app.markdown))
