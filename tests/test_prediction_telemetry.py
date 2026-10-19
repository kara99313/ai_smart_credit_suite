import json
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import patch
from uuid import UUID
import pytest
from fastapi.testclient import TestClient
import server
from utils import api_client, health_diagnostics
from utils import prediction_telemetry as telemetry
from test_prediction import PAYLOAD

@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / 'technical.sqlite3'
    monkeypatch.setattr(telemetry, 'STORE_PATH', path)
    return path

def records(path):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM prediction_events')]

def test_terminals_privacy_metadata(store):
    client = TestClient(server.app)
    secret = 'SECRET_client_password_https://user:password@host/token'
    assert client.post('/api/predict', json=dict(PAYLOAD, client_id=secret, prompt=secret)).status_code == 200
    assert client.post('/api/predict', json=dict(PAYLOAD, Income=secret)).status_code == 422
    assert client.post('/api/predict', content=secret).status_code == 422
    with patch.object(server, '_compute_pd', side_effect=RuntimeError(secret)):
        assert client.post('/api/predict', json=PAYLOAD).status_code == 500
    rows = records(store)
    assert [r['status'] for r in rows] == ['success', 'validation_error', 'validation_error', 'internal_error']
    assert [r['error_category'] for r in rows] == [None, 'request_validation', 'request_validation', 'prediction_failure']
    assert len({UUID(r['request_id']) for r in rows}) == 4
    for row in rows:
        assert datetime.fromisoformat(row['timestamp_utc']).utcoffset() == timedelta(0)
        assert row['duration_ms'] >= 0
        assert row['model_id'] == server.MODEL_ID
        assert row['origin'] == 'business'
        assert set(row) == {'request_id', 'schema_version', 'timestamp_utc', 'duration_ms', 'status', 'error_category', 'model_id', 'origin', 'call_type'}
    for forbidden in [secret, *PAYLOAD, 'client_id', 'prompt', 'traceback', 'input_id', 'score_1000', 'decision', 'rating']:
        assert forbidden not in json.dumps(rows)

def test_synthetic_diagnostic_and_history(store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data = tmp_path / 'data'
    data.mkdir()
    for name in ('predictions_log.csv', 'predictions_log.results.jsonl'):
        (data / name).write_bytes(b'PRIVATE HISTORY\n')
    before = {p.name: p.read_bytes() for p in data.iterdir()}
    client = TestClient(server.app)
    def post(url, **kwargs):
        from types import SimpleNamespace
        response = client.post('/api/predict', json=kwargs['json'], headers=kwargs.get('headers', {}))
        return SimpleNamespace(ok=response.is_success, status_code=response.status_code, json=response.json)
    with patch.object(api_client.requests, 'post', side_effect=post) as transport:
        assert health_diagnostics.inference_check('http://localhost')['state'] == 'Fonctionnel'
        assert transport.call_args.kwargs['headers'] == {'X-Prediction-Origin': 'synthetic'}
    assert {p.name: p.read_bytes() for p in data.iterdir()} == before
    assert len(records(store)) == 1
    assert records(store)[0]['origin'] == 'synthetic'
    client.get('/health')
    client.get('/api/predict_ping')
    assert len(records(store)) == 1
    client.post('/api/predict', json=PAYLOAD, headers={'X-Prediction-Origin': 'SECRET'})
    assert records(store)[-1]['origin'] == 'business'

@pytest.mark.parametrize('code', [200, 422, 500])
def test_unavailable_storage(store, code):
    store.mkdir()
    with patch.object(server, '_compute_pd', **({'side_effect': RuntimeError('private')} if code == 500 else {'return_value': .2})):
        response = TestClient(server.app).post('/api/predict', json={} if code == 422 else PAYLOAD)
    assert response.status_code == code

def test_idempotence_retention_warmup(store):
    from datetime import timezone
    attempt = telemetry.Attempt('secret_model_path')
    attempt.event['timestamp_utc'] = (datetime.now(timezone.utc) - timedelta(days=31)).isoformat()
    attempt.finish(200)
    assert records(store) == []
    server._warmup_model()
    with patch.object(server, '_compute_pd', side_effect=RuntimeError('PRIVATE')):
        server._warmup_model()
    assert [r['status'] for r in records(store)] == ['success', 'internal_error']
    assert all(r['origin'] == 'synthetic' and r['call_type'] == 'warmup' for r in records(store))
    attempt = telemetry.Attempt('secret_model_path')
    attempt.finish(200)
    attempt.finish(500)
    assert len(records(store)) == 3
    assert records(store)[-1]['model_id'] is None

def test_unhandled_exception_and_writer_failure(store):
    from fastapi import FastAPI
    app = FastAPI()
    app.add_middleware(telemetry.PredictionTelemetryMiddleware, model_id=server.MODEL_ID)
    @app.post('/api/predict')
    def fail():
        raise RuntimeError('PRIVATE')
    assert TestClient(app, raise_server_exceptions=False).post('/api/predict').status_code == 500
    assert len(records(store)) == 1
    assert records(store)[0]['status'] == 'internal_error'
    with patch.object(telemetry, 'persist', side_effect=RuntimeError('PRIVATE')):
        assert TestClient(server.app).post('/api/predict', json=PAYLOAD).status_code == 200

