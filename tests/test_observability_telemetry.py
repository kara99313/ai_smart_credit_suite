from datetime import datetime, timedelta, timezone
import sqlite3
from unittest.mock import patch
import pytest
from utils import observability_telemetry as obs
from utils import prediction_telemetry as pred, agent_telemetry as agent

NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)
PRIVATE = 'FAKE_SECRET prompt client_id RAG Web <html>private</html>'


@pytest.fixture
def stores(tmp_path):
    p, a = tmp_path/'prediction.db', tmp_path/'agent.db'
    with patch.object(pred, 'STORE_PATH', p), patch.object(agent, 'STORE_PATH', a):
        for i, (status, origin, kind, days, duration) in enumerate([
            ('success', 'business', 'api_predict', 0, 10),
            ('validation_error', 'synthetic', 'api_predict', 2, 30),
            ('internal_error', 'business', 'api_predict', 8, 80),
            ('success', 'synthetic', 'warmup', 0, 1000)]):
            pred.persist(dict(request_id=PRIVATE+str(i), timestamp_utc=(NOW-timedelta(days=days)).isoformat(),
                             duration_ms=duration, status=status, error_category=PRIVATE,
                             model_id='a'*64, origin=origin, call_type=kind))
        for i, status in enumerate(('success', 'error', 'running')):
            agent.persist('agent_runs', (PRIVATE+str(i), 1, NOW.isoformat(), [10,30,999][i], status, 2, None, None, 0,0,0))
        for i, (attempt, status) in enumerate(((1,'retry'), (2,'success'), (1,'retry'))):
            agent.persist('agent_attempts', (PRIVATE+str(i), attempt, NOW.isoformat(), 10, status))
        for i, status in enumerate(obs.TOOL_STATUSES):
            agent.persist('agent_tools', (PRIVATE, 1, i, NOW.isoformat(), 'rag_retrieve', (i+1)*10, status))
    return p,a


def test_prediction_aggregates(stores):
    p,_ = stores
    before = p.read_bytes()
    s = obs.prediction_snapshot(p, now=NOW)
    assert s['kpis']['total'] == 3
    assert s['kpis']['success_rate'] == 1/3
    assert s['kpis']['median_ms'] == 30
    assert s['kpis']['mean_ms'] == 40
    assert s['warmups']['total'] == 1
    assert len(s['daily']) == 3
    assert s['models'][0]['total'] == 3
    assert p.read_bytes() == before
    assert PRIVATE not in str(s) and 'request_id' not in str(s)


@pytest.mark.parametrize('hours,total', [(24,1),(168,2),(720,3)])
def test_periods(stores, hours, total):
    assert obs.prediction_snapshot(stores[0], hours=hours, now=NOW)['kpis']['total'] == total
    assert obs.agent_snapshot(stores[1], hours=hours, now=NOW)['kpis']['total'] == 3


@pytest.mark.parametrize('origin,total', [('business',2),('synthetic',1)])
def test_origin(stores,origin,total):
    assert obs.prediction_snapshot(stores[0], origin=origin, now=NOW)['kpis']['total'] == total


def test_agent_aggregates(stores):
    s = obs.agent_snapshot(stores[1], now=NOW)
    k = s['kpis']
    assert (k['total'], k['success'], k['error'], k['running']) == (3,1,1,1)
    assert k['success_rate'] == .5 and k['median_ms'] == 20 and k['mean_ms'] == 20
    assert (k['attempts'], k['retry_decided'], k['retries_started']) == (3,2,1)
    assert s['models'][0]['provider'] == s['models'][0]['model'] == 'Non renseigné'
    assert s['tools']['terminal'] == 4
    assert s['tools']['total'] == 6
    assert s['tools']['median_ms'] == 25
    assert all(s['tool_groups'][0][status] == 1 for status in obs.TOOL_STATUSES)
    assert PRIVATE not in str(s) and 'request_id' not in str(s)


@pytest.mark.parametrize('reader', [obs.prediction_snapshot, obs.agent_snapshot])
@pytest.mark.parametrize('kind', ['missing','empty','corrupt','incompatible','locked'])
def test_unavailable(tmp_path, reader, kind):
    path = tmp_path/'source.db'
    db = None
    if kind == 'empty':
        path.touch()
    elif kind == 'corrupt':
        path.write_text(PRIVATE)
    elif kind in ('incompatible', 'locked'):
        db = sqlite3.connect(path)
        db.execute('CREATE TABLE unrelated (secret TEXT)')
        db.commit()
        if kind == 'locked':
            db.execute('BEGIN EXCLUSIVE')
    try:
        result = reader(path, now=NOW)
        assert result['source']['state'] == ('missing' if kind == 'missing' else 'unavailable')
        assert result['kpis'] is None and PRIVATE not in str(result)
        if kind == 'missing':
            assert not path.exists()
    finally:
        if db:
            db.close()


def test_valid_empty(stores):
    for path,reader in zip(stores,(obs.prediction_snapshot,obs.agent_snapshot)):
        result=reader(path, now=NOW+timedelta(days=60))
        assert result['source']['state']=='empty'
        assert result['kpis']['total']==0
        assert result['kpis']['success_rate'] is None


def test_read_only_connection(stores):
    real = sqlite3.connect
    connections=[]
    def connect(*args, **kwargs):
        assert args[0].endswith('?mode=ro') and kwargs['uri'] and kwargs['timeout'] <= .1
        db=real(*args, **kwargs)
        connections.append(db)
        return db
    with patch.object(obs.sqlite3,'connect',side_effect=connect):
        obs.agent_snapshot(stores[1],now=NOW)
    with pytest.raises(sqlite3.ProgrammingError):
        connections[0].execute('SELECT 1')

@pytest.mark.parametrize('failed', ['prediction', 'agent'])
def test_page_source_failure_is_independent(stores, failed):
    import runpy
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    p,a = stores
    (p if failed == 'prediction' else a).write_text(PRIVATE)
    script = ('import runpy\n'
              + f"ns=runpy.run_path({str(Path('app/observability.py').resolve())!r})\n"
              + "ns['main'].__globals__['render_gate']=lambda: True\n"
              + f"ns['main'].__globals__['PRED_LOG']={str(p)!r}\n"
              + f"ns['main'].__globals__['AGENT_LOG']={str(a)!r}\n"
              + "ns['main']()\n")
    with patch.object(obs, 'datetime') as clock:
        clock.now.return_value = NOW
        app=AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception and app.error
    labels={m.label for m in app.metric}
    assert ('Exécutions' if failed=='prediction' else 'Tentatives API') in labels
    assert PRIVATE not in str(app) and 'request_id' not in str(app)


def test_gate_precedes_any_sqlite_read():
    import runpy
    ns=runpy.run_path('app/observability.py')
    ns['main'].__globals__['render_gate']=lambda: False
    with patch.object(obs.sqlite3,'connect') as connect:
        ns['main']()
    connect.assert_not_called()


@pytest.mark.parametrize('source', ['prediction','agent'])
def test_incompatible_version_and_extra_sensitive_columns(stores, source):
    path=stores[0 if source=='prediction' else 1]
    table='prediction_events' if source=='prediction' else 'agent_runs'
    reader=obs.prediction_snapshot if source=='prediction' else obs.agent_snapshot
    with sqlite3.connect(path) as db:
        db.execute(f'ALTER TABLE {table} ADD COLUMN prompt TEXT')
        db.execute(f'UPDATE {table} SET prompt = ?', (PRIVATE,))
    assert PRIVATE not in str(reader(path, now=NOW))
    with sqlite3.connect(path) as db:
        db.execute(f'UPDATE {table} SET schema_version = 999')
    result=reader(path, now=NOW)
    assert result['source']['state']=='unavailable' and result['kpis'] is None

@pytest.mark.parametrize('hours,expected', [(24,3),(168,4),(720,5)])
def test_agent_window_applies_to_each_table(stores,hours,expected):
    path=stores[1]
    with sqlite3.connect(path) as db:
        for i,days in enumerate((2,8,31)):
            stamp=(NOW-timedelta(days=days)).isoformat()
            db.execute('INSERT INTO agent_runs VALUES (?,1,?,10,?,1,NULL,NULL,0,0,0)',
                       ('old'+str(i),stamp,'success'))
            db.execute('INSERT INTO agent_attempts VALUES (?,1,?,10,?)',
                       ('old'+str(i),stamp,'success'))
            db.execute('INSERT INTO agent_tools VALUES (?,1,1,?,?,10,?)',
                       ('old'+str(i),stamp,'search_web','success'))
    result=obs.agent_snapshot(path,hours=hours,now=NOW)
    assert result['kpis']['total']==expected
    assert result['kpis']['attempts']==expected
    assert result['tools']['total']==expected+3


def test_utc_dates_and_numeric_sqlite_timestamps(stores):
    with sqlite3.connect(stores[0]) as db:
        db.execute('UPDATE prediction_events SET timestamp_utc=?', ('2026-10-18T23:30:00-00:30',))
    result=obs.prediction_snapshot(stores[0],now=NOW)
    assert all(row['date']=='2026-10-19' for row in result['daily'])
    with sqlite3.connect(stores[0]) as db:
        db.execute('UPDATE prediction_events SET timestamp_utc=julianday(?)',(NOW.isoformat(),))
    result=obs.prediction_snapshot(stores[0],now=NOW)
    assert all(row['date']=='2026-10-19' for row in result['daily'])
