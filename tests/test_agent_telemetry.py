"""Persistent Agent lifecycle and privacy tests."""
import sqlite3
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4
import httpx
import pytest
from utils import agent_telemetry as t
from utils.agents import run_agent_once
from utils.agent_diagnostics import AgentExecutionError

@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / 'agent.sqlite3'
    monkeypatch.setattr(t, 'STORE_PATH', path)
    return path

def rows(path, table):
    with sqlite3.connect(path) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute('SELECT * FROM ' + table)]

@pytest.mark.parametrize('failure', [False, True])
def test_retry(store, failure):
    agent = Mock(invoke=Mock(side_effect=[httpx.ReadTimeout('PRIVATE'),
        httpx.ReadTimeout('PRIVATE') if failure else {'output': 'ok'}]))
    with patch('time.sleep'):
        if failure:
            with pytest.raises(AgentExecutionError):
                run_agent_once(agent, 'PRIVATE_PROMPT', [])
        else:
            run_agent_once(agent, 'PRIVATE_PROMPT', [])
    runs = rows(store, 'agent_runs')
    assert len(runs) == 1 and runs[0]['attempts'] == 2
    assert runs[0]['status'] == ('error' if failure else 'success')
    attempts = rows(store, 'agent_attempts')
    assert [a['status'] for a in attempts] == ['retry', 'error' if failure else 'success']
    assert {a['request_id'] for a in attempts} == {runs[0]['request_id']}
    assert rows(store, 'agent_tools') == []

@pytest.mark.parametrize('ending', ['success', 'exception', 'text_error', 'pending'])
def test_no_retry_after_tool(store, ending):
    def invoke(payload, config):
        cb, rid = config['callbacks'][0], uuid4()
        cb.on_tool_start({'name': 'search_web'}, 'PRIVATE_ARGUMENT', run_id=rid)
        if ending == 'exception':
            cb.on_tool_error(ValueError('PRIVATE_EXCEPTION'), run_id=rid)
        elif ending != 'pending':
            cb.on_tool_end('ERROR: PRIVATE' if ending == 'text_error' else '[W1] title\nhttps://example.org\nPRIVATE', run_id=rid)
        raise httpx.ReadTimeout('PRIVATE')
    agent = Mock(invoke=Mock(side_effect=invoke))
    run_agent_once(agent, 'PRIVATE', [])
    assert agent.invoke.call_count == 1
    run, tool = rows(store, 'agent_runs')[0], rows(store, 'agent_tools')[0]
    assert run['status'] == 'error' and run['tools_started'] == 1
    assert run['tools_completed'] == (ending != 'pending')
    assert run['tools_errors'] == (ending in {'exception', 'text_error'})
    assert tool['status'] == dict(success='success', exception='error', text_error='error', pending='interrupted')[ending]
    assert tool['duration_ms'] >= 0 and tool['request_id'] == run['request_id']
    assert tool['attempt'] == 1 and tool['tool_name'] == 'search_web'

def test_privacy_metadata_and_deduplication(store):
    def invoke(payload, config):
        cb = config['callbacks'][0]
        for name, output in [('rag_retrieve', '[D1] PRIVATE_DOCUMENT\nPRIVATE_PASSAGE erreur'),
                ('search_web', '[W1] PRIVATE_QUERY\nhttps://example.org\nPRIVATE_RESULT error'),
                ('build_report', '<html>PRIVATE_HTML error erreur</html>'),
                ('predict_credit', 'PRIVATE_CLIENT PD=0.123 rating=A decision=ACCEPT')]:
            rid = uuid4()
            cb.on_tool_start({'name': name}, 'PRIVATE_ARGUMENT sk-fake-secret', run_id=rid)
            cb.on_tool_end(output, run_id=rid)
            cb.on_tool_end(output, run_id=rid)
        return {'output': 'PRIVATE_RESPONSE'}
    with patch.object(t, 'model_metadata', return_value=('groq', 'llama-3.3')):
        run_agent_once(Mock(invoke=Mock(side_effect=invoke)), 'PRIVATE_PROMPT', ['PRIVATE_HISTORY'])
    run, tools = rows(store, 'agent_runs')[0], rows(store, 'agent_tools')
    assert run['status'] == 'success' and run['schema_version'] == 1
    assert (run['provider'], run['model']) == ('groq', 'llama-3.3')
    assert run['duration_ms'] >= 0 and run['timestamp_utc'].endswith('+00:00')
    assert datetime.fromisoformat(run['timestamp_utc']).utcoffset() == timedelta(0)
    assert run['tools_started'] == run['tools_completed'] == 4
    assert run['tools_errors'] == 0
    assert {x['tool_name'] for x in tools} == {'rag_retrieve', 'search_web', 'build_report', 'predict_credit'}
    assert {x['request_id'] for x in tools} == {run['request_id']}
    with sqlite3.connect(store) as db:
        dump = '\n'.join(db.iterdump())
    for secret in ['PRIVATE', 'sk-fake-secret', '<html>', '0.123', 'ACCEPT', 'https://']:
        assert secret not in dump

@pytest.mark.parametrize('output,status', [('ERROR: JSON invalide', 'error'),
    ('RAG indisponible : secret', 'error'), ('ERROR: appel', 'error'),
    ('Aucun passage pertinent', 'no_result'), ('[D1] source\nerror in document', 'success')])
def test_text_classification(output, status):
    assert t.tool_status('rag_retrieve', output) == status

def test_unavailable_storage(store, monkeypatch):
    monkeypatch.setattr(t, 'STORE_PATH', store.parent)
    def invoke(payload, config):
        cb, rid = config['callbacks'][0], uuid4()
        cb.on_tool_start({'name': 'now'}, 'PRIVATE', run_id=rid)
        cb.on_tool_end('2026-01-01', run_id=rid)
        return {'output': 'ok'}
    assert 'ok' in run_agent_once(Mock(invoke=Mock(side_effect=invoke)), 'private', [])['output']
    with patch.object(t, 'persist', side_effect=RuntimeError('PRIVATE')):
        assert 'ok' in run_agent_once(Mock(invoke=Mock(side_effect=invoke)), 'private', [])['output']

def test_retention_unknown_name(store):
    session = t.AgentTelemetry('test', None)
    session.timestamp = '2000-01-01T00:00:00+00:00'
    session.run('success')
    assert rows(store, 'agent_runs') == []
    session.begin_attempt(1)
    session.start_tool(uuid4(), 'PRIVATE_TOOL_NAME')
    assert rows(store, 'agent_tools')[0]['tool_name'] == 'unknown'

def test_actual_model_configuration():
    from types import SimpleNamespace
    from langchain_groq import ChatGroq
    llm = ChatGroq(model='llama-3.3-70b-versatile', api_key='FAKE_TEST_KEY')
    agent = SimpleNamespace(agent=SimpleNamespace(runnable=SimpleNamespace(steps=[llm.bind_tools([])])))
    assert t.model_metadata(agent) == ('groq', 'llama-3.3-70b-versatile')


def test_tool_correlated_to_second_attempt(store):
    count = 0
    def invoke(payload, config):
        nonlocal count
        count += 1
        if count == 1:
            raise httpx.ConnectError('PRIVATE')
        cb, rid = config['callbacks'][0], uuid4()
        cb.on_tool_start({'name': 'now'}, '', run_id=rid)
        cb.on_tool_end('2026-01-01', run_id=rid)
        return {'output': 'ok'}
    with patch('time.sleep'):
        run_agent_once(Mock(invoke=Mock(side_effect=invoke)), 'private', [])
    run = rows(store, 'agent_runs')[0]
    tool = rows(store, 'agent_tools')[0]
    assert run['attempts'] == 2 and run['tools_started'] == run['tools_completed'] == 1
    assert tool['attempt'] == 2 and tool['request_id'] == run['request_id']


@pytest.mark.parametrize('name,output', [
    ('rag_retrieve', 'Aucune erreur d\u00e9tect\u00e9e'),
    ('search_web', 'No error found'),
    ('rag_retrieve', '[D1] source\nUne erreur m\u00e9tier'),
    ('search_web', '[W1] source\nAn error occurred'),
    ('rag_retrieve', ' \n\t[D1] source\nUne erreur m\u00e9tier'),
    ('search_web', ' \n\t[W1] source\nAn error occurred'),
    ('build_report', '\n<!doctype html><html>error erreur</html>'),
    ('predict_credit', 'R\u00e9sultat m\u00e9tier : \u00e9chec ou erreur historique'),
])
def test_valid_content_is_not_error(name, output):
    assert t.tool_status(name, output) == 'success'


@pytest.mark.parametrize('name,output', [
    ('predict_credit', "\u274c \u00c9chec d'appel /api/predict : PRIVATE"),
    ('predict_credit', 'Format JSON invalide pour predict_credit'),
    ('rag_retrieve', '\u26a0\ufe0f RAG indisponible : PRIVATE'),
    ('search_web', '\u26a0\ufe0f Recherche Web indisponible : PRIVATE'),
    ('search_web', 'ERROR: PRIVATE'),
])
def test_tool_failure_contracts(name, output):
    assert t.tool_status(name, output) == 'error'


def test_structured_error_overrides_content():
    from types import SimpleNamespace
    assert t.tool_status('rag_retrieve', SimpleNamespace(
        status='error', content='[D1] source valide')) == 'error'


@pytest.mark.parametrize('first', ['success', 'error', 'retry'])
def test_attempt_terminal_is_idempotent(store, first):
    session = t.AgentTelemetry('test', None)
    with patch.object(t.time, 'perf_counter', return_value=10.):
        session.begin_attempt(1)
    with patch.object(t.time, 'perf_counter', return_value=11.):
        session.end_attempt(first)
    original = rows(store, 'agent_attempts')
    assert original[0]['status'] == first
    assert original[0]['duration_ms'] == 1000.
    with patch.object(t.time, 'perf_counter', return_value=20.):
        session.end_attempt('error' if first != 'error' else 'success')
        session.end_attempt('running')
    assert rows(store, 'agent_attempts') == original
    session.begin_attempt(2)
    session.end_attempt('success')
    assert [a['status'] for a in rows(store, 'agent_attempts')] == [first, 'success']
