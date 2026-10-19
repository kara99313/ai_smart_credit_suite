"""Allowlisted prediction telemetry; no payloads, exception text or client IDs."""
from datetime import datetime, timedelta, timezone
from contextlib import closing
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

STORE_PATH = Path(__file__).resolve().parents[1] / 'logs' / 'prediction_telemetry.sqlite3'
RETENTION_DAYS = 30


def persist(event):
    """Atomic terminal records, bounded lock wait, best effort storage and retention."""
    try:
        STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(STORE_PATH, timeout=0.05)) as db, db:
            db.execute('''CREATE TABLE IF NOT EXISTS prediction_events (
                request_id TEXT PRIMARY KEY, schema_version INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL, duration_ms REAL NOT NULL,
                status TEXT NOT NULL, error_category TEXT, model_id TEXT,
                origin TEXT NOT NULL, call_type TEXT NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS prediction_events_timestamp ON prediction_events(timestamp_utc)')
            db.execute('INSERT INTO prediction_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', (
                event['request_id'], 1, event['timestamp_utc'], event['duration_ms'],
                event['status'], event['error_category'], event['model_id'],
                event['origin'], event['call_type']))
            cutoff = (datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)).isoformat()
            db.execute('DELETE FROM prediction_events WHERE timestamp_utc < ?', (cutoff,))
    except Exception:
        # Never print storage exceptions: paths/messages may contain secrets.
        pass


class Attempt:
    def __init__(self, model_id, *, synthetic=False, warmup=False):
        self.started = time.perf_counter()
        self.event = dict(request_id=str(uuid4()), schema_version=1,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_id=model_id if isinstance(model_id, str) and re.fullmatch(r'[0-9a-f]{64}', model_id) else None,
            origin='synthetic' if synthetic or warmup else 'business',
            call_type='warmup' if warmup else 'api_predict')
        self.finished = False

    def finish(self, status_code):
        if self.finished:
            return
        self.finished = True
        status, category = ('success', None) if 200 <= status_code < 300 else (
            ('validation_error', 'request_validation') if 400 <= status_code < 500
            else ('internal_error', 'prediction_failure'))
        try:
            persist(dict(self.event, duration_ms=max(0., (time.perf_counter() - self.started) * 1000),
                         status=status, error_category=category))
        except Exception:
            pass


class PredictionTelemetryMiddleware:
    """One event per HTTP attempt, including pre-handler validation failures."""
    def __init__(self, app, model_id):
        self.app, self.model_id = app, model_id

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('method') != 'POST' or scope.get('path') != '/api/predict':
            return await self.app(scope, receive, send)
        synthetic = (b'x-prediction-origin', b'synthetic') in scope.get('headers', [])
        attempt = Attempt(self.model_id, synthetic=synthetic)
        status = 500

        async def capture(message):
            nonlocal status
            if message['type'] == 'http.response.start':
                status = message['status']
            await send(message)

        try:
            await self.app(scope, receive, capture)
        except BaseException:
            status = 500
            raise
        finally:
            attempt.finish(status)
