"""Read-only, content-free aggregates of the version 1 technical stores."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import re
import sqlite3
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
PRED_LOG = ROOT / 'logs/prediction_telemetry.sqlite3'
AGENT_LOG = ROOT / 'logs/agent_telemetry.sqlite3'
TOOL_NAMES = {'predict_credit', 'rag_retrieve', 'search_web', 'build_report', 'now', 'client_context', 'unknown'}
TOOL_STATUSES = ('success', 'error', 'no_result', 'unknown', 'interrupted', 'running')


def _read(path, queries, hours, now):
    if hours not in (24, 168, 720):
        raise ValueError('Unsupported period')
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    bounds = ((now - timedelta(hours=hours)).isoformat(), now.isoformat())
    try:
        path = Path(path)
        if not path.exists():
            return 'missing', []
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=.05)) as db:
            db.execute('PRAGMA query_only=ON')
            db.execute('BEGIN')
            rows = [db.execute(sql, bounds + params).fetchall() for sql, params in queries]
        return ('available' if any(rows) else 'empty'), rows
    except (OSError, sqlite3.Error, ValueError):
        return 'unavailable', []


def _latencies(rows):
    values = [r['duration_ms'] for r in rows if isinstance(r['duration_ms'], (int, float))
              and math.isfinite(r['duration_ms']) and r['duration_ms'] >= 0]
    return dict(mean_ms=mean(values) if values else None,
                median_ms=median(values) if values else None, latency_count=len(values))


def _stats(rows, statuses, terminal):
    result = dict(total=len(rows), **{s: sum(r['status'] == s for r in rows) for s in statuses})
    denominator = sum(r['status'] in terminal for r in rows)
    result.update(terminal=denominator, success_rate=result.get('success', 0) / denominator if denominator else None)
    result.update(_latencies([r for r in rows if r['status'] in terminal]))
    return result


def _daily(rows):
    counts = {}
    for row in rows:
        key = (row['timestamp_utc'][:10], row['status'])
        counts[key] = counts.get(key, 0) + 1
    return [dict(date=d, status=s, calls=n) for (d, s), n in sorted(counts.items())]


def _groups(rows, keys, statuses, terminal):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    return [dict(zip(keys, key), **_stats(items, statuses, terminal))
            for key, items in sorted(groups.items())]


def _label(value, pattern):
    return value if isinstance(value, str) and re.fullmatch(pattern, value) else 'Non renseigné'


def prediction_snapshot(path=PRED_LOG, *, hours=720, origin=None, now=None):
    if origin not in (None, 'business', 'synthetic'):
        raise ValueError('Unsupported origin')
    sql = '''SELECT strftime('%Y-%m-%d', timestamp_utc), duration_ms, status, model_id, call_type, schema_version, origin FROM prediction_events
             WHERE julianday(timestamp_utc) >= julianday(?) AND julianday(timestamp_utc) <= julianday(?)'''
    params = ()
    if origin:
        sql += ' AND origin = ?'
        params = (origin,)
    state, tables = _read(path, [(sql, params)], hours, now)
    result = dict(source=dict(state=state), kpis=None, daily=[], models=[], warmups=None)
    if not tables:
        return result
    statuses = ('success', 'validation_error', 'internal_error')
    rows = [dict(zip(('timestamp_utc', 'duration_ms', 'status', 'model_id', 'call_type', 'schema_version', 'origin'), r)) for r in tables[0]]
    for row in rows:
        if row['origin'] not in ('business', 'synthetic') or row['schema_version'] != 1 or row['status'] not in statuses or row['call_type'] not in ('api_predict', 'warmup'):
            result['source']['state'] = 'unavailable'
            return result
        row['model_id'] = _label(row['model_id'], r'[0-9a-f]{64}')
    api = [r for r in rows if r['call_type'] == 'api_predict']
    result.update(kpis=_stats(api, statuses, statuses), daily=_daily(api),
                  models=_groups(api, ('model_id',), statuses, statuses),
                  warmups=_stats([r for r in rows if r['call_type'] == 'warmup'], statuses, statuses))
    return result


def agent_snapshot(path=AGENT_LOG, *, hours=720, now=None):
    specs = [('agent_runs', 'timestamp_utc, duration_ms, status, provider, model, schema_version'),
             ('agent_attempts', 'timestamp_utc, duration_ms, status, attempt'),
             ('agent_tools', 'timestamp_utc, duration_ms, status, tool_name')]
    queries = []
    for table, cols in specs:
        selected = cols.replace('timestamp_utc', "strftime('%Y-%m-%d', timestamp_utc)", 1)
        queries.append((f'SELECT {selected} FROM {table} '
                        'WHERE julianday(timestamp_utc) >= julianday(?) '
                        'AND julianday(timestamp_utc) <= julianday(?)', ()))
    state, tables = _read(path, queries, hours, now)
    result = dict(source=dict(state=state), kpis=None, daily=[], models=[], tools=None, tool_groups=[])
    if not tables:
        return result
    runs, attempts, tools = [[dict(zip(cols.split(', '), row)) for row in rows]
                             for (_, cols), rows in zip(specs, tables)]
    if (any(r['schema_version'] != 1 or r['status'] not in ('success', 'error', 'running') for r in runs)
            or any(r['status'] not in ('success', 'error', 'retry', 'running') or not isinstance(r['attempt'], int) or r['attempt'] < 1 for r in attempts)
            or any(r['status'] not in TOOL_STATUSES for r in tools)):
        result['source']['state'] = 'unavailable'
        return result
    for row in runs:
        row['provider'] = _label(row['provider'], r'(?:groq|openai|ollama|anthropic)')
        row['model'] = _label(row['model'], r'[A-Za-z0-9_./:-]{1,100}')
    for row in tools:
        row['tool_name'] = row['tool_name'] if row['tool_name'] in TOOL_NAMES else 'unknown'
    k = _stats(runs, ('success', 'error', 'running'), ('success', 'error'))
    k.update(attempts=len(attempts), retry_decided=sum(r['status'] == 'retry' for r in attempts),
             retries_started=sum(r['attempt'] > 1 for r in attempts))
    terminal = ('success', 'error', 'no_result', 'unknown')
    result.update(kpis=k, daily=_daily(runs), models=_groups(runs, ('provider', 'model'), ('success', 'error', 'running'), ('success', 'error')),
                  tools=_stats(tools, TOOL_STATUSES, terminal),
                  tool_groups=_groups(tools, ('tool_name',), TOOL_STATUSES, terminal))
    return result
