"""Production log volume, correlation, isolation and privacy guarantees."""
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest
from utils import agents, credit_agent_contract as contract
from utils import agent_preflight_diagnostics as diagnostics
from utils import agent_telemetry


def events(caplog, prefix, level):
    return [json.loads(r.getMessage()[len(prefix) + 1:]) for r in caplog.records
            if r.levelno == level and r.getMessage().startswith(prefix + ' ')]


@pytest.mark.parametrize('query,reason,count,missing,allowed', [
    ('Quelles variables sont requises ?', 'schema_question', 0, 15, False),
    ('Prediction Income=918273', 'missing_features', 1, 14, False),
    ('Prediction ' + json.dumps(dict.fromkeys(contract.FEATURES, 918273)),
     'complete_features', 15, 0, True),
    ('Bonjour PRIVATE_USER', 'unrecognized_request', 0, 15, True),
    ('Prediction Age=null threshold=0.876543', 'missing_features', 1, 15, False),
])
def test_branches_levels_correlation_and_privacy(query, reason, count, missing, allowed, caplog, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'PRIVATE_API_KEY')
    history = [('assistant', 'PRIVATE_HISTORY client_id=PRIVATE_CLIENT score=PRIVATE_SCORE')]
    agent = Mock(invoke=Mock(return_value={'output': 'PRIVATE_RESULT'}))
    expected = contract.preflight(query, contract.user_values(query, history))
    with caplog.at_level(logging.DEBUG):
        result = agents.run_agent_once(agent, query + ' PRIVATE_PROMPT secret=PRIVATE_SECRET', history)
    detail, = events(caplog, 'Agent preflight', logging.DEBUG)
    assert detail == dict(request_id=detail['request_id'], preflight_executed=True,
                          reason=reason, recognized_features_count=count,
                          missing_features_count=missing, invoke_allowed=allowed,
                          history_messages_count=1 if allowed else 0)
    info = events(caplog, 'Agent preflight', logging.INFO)
    assert info == ([dict(request_id=detail['request_id'], reason=reason,
                          invoke_allowed=False, missing_features_count=missing)]
                    if reason == 'missing_features' else [])
    assert bool(agent.invoke.call_count) == allowed
    if expected:
        assert result == dict(output=expected, intermediate_steps=[], artifacts=[])
    else:
        assert agent.invoke.call_args.args[0]['chat_history'] is history
        execution = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Agent execution ')]
        assert len(execution) == 1
        assert 'request_id=' + detail['request_id'] in execution[0]
    for forbidden in ('PRIVATE_', '918273', '0.876543', 'client_id', 'threshold'):
        assert forbidden not in caplog.text
    assert all(re.search(r'\b' + name + r'\b', caplog.text) is None for name in contract.FEATURES)
    assert all(r.exc_info is None for r in caplog.records)


def test_info_only_has_compact_blocking_event(caplog):
    with caplog.at_level(logging.INFO):
        agents.run_agent_once(Mock(), 'Prediction Income=918273', [])
    event, = events(caplog, 'Agent preflight', logging.INFO)
    assert set(event) == {'request_id', 'reason', 'invoke_allowed', 'missing_features_count'}
    assert not events(caplog, 'Agent preflight', logging.DEBUG)


@pytest.mark.parametrize('query', ['Bonjour', 'Quelles variables sont requises ?',
                                  json.dumps(dict.fromkeys(contract.FEATURES, 918273))])
def test_normal_preflight_has_no_info_event(query, caplog):
    with caplog.at_level(logging.INFO):
        agents.run_agent_once(Mock(invoke=Mock(return_value={'output': 'ok'})), query, [])
    assert events(caplog, 'Agent preflight', logging.INFO) == []
    assert events(caplog, 'Agent preflight', logging.DEBUG) == []


def test_runtime_once_with_concurrent_requests(caplog, monkeypatch):
    monkeypatch.setattr(diagnostics, '_runtime_pid', None)
    with caplog.at_level(logging.INFO), ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: diagnostics.log_runtime(), range(32)))
        diagnostics.log_runtime()
    runtime, = events(caplog, 'Agent runtime', logging.INFO)
    assert set(runtime) == {'pid', 'python_version', 'implementation'}
    assert runtime['pid'] > 0 and runtime['python_version'] and runtime['implementation']
    assert not any(marker in caplog.text for marker in
                   ('origin', 'executable', 'sha256', 'modules', 'loaded_code', 'request_id'))


def test_runtime_lock_never_waits(monkeypatch):
    monkeypatch.setattr(diagnostics, '_runtime_pid', None)
    lock = Mock(acquire=Mock(return_value=False))
    monkeypatch.setattr(diagnostics, '_runtime_lock', lock)
    with patch.object(diagnostics, '_emit') as emit:
        diagnostics.log_runtime()
    lock.acquire.assert_called_once_with(blocking=False)
    emit.assert_not_called()


def test_runtime_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(diagnostics, '_runtime_pid', None)
    with patch.object(diagnostics._logger, 'log', side_effect=RuntimeError('PRIVATE')) as log:
        diagnostics.log_runtime()
        diagnostics.log_runtime()
    assert log.call_count == 1


def test_bypassed_preflight_only_in_debug(caplog):
    with patch.object(contract, 'preflight', return_value=None), caplog.at_level(logging.DEBUG):
        agents.run_agent_once(Mock(invoke=Mock(return_value={'output': 'ok'})), 'PRIVATE', [])
    event, = events(caplog, 'Agent preflight', logging.DEBUG)
    assert event['preflight_executed'] is False
    assert event['reason'] is None
    assert event['invoke_allowed'] is True
    assert not events(caplog, 'Agent preflight', logging.INFO)


@pytest.mark.parametrize('query,allowed', [('Prediction Income=12', False), ('Bonjour', True)])
def test_logging_failure_does_not_change_behavior(query, allowed, caplog):
    with caplog.at_level(logging.DEBUG), patch.object(
            diagnostics._logger, 'log', side_effect=RuntimeError('PRIVATE')):
        agent = Mock(invoke=Mock(return_value={'output': 'ok'}))
        result = agents.run_agent_once(agent, query, [])
    assert bool(agent.invoke.call_count) == allowed
    if not allowed:
        assert result['output'] == contract.preflight(query, {'Income': 12})
    assert diagnostics._state.get() is None


def test_context_reset_on_exception(caplog):
    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError):
        with diagnostics.preflight_diagnostics('test', {}, []):
            diagnostics.mark_preflight()
            raise ValueError('PRIVATE_EXCEPTION')
    assert diagnostics._state.get() is None
    assert 'PRIVATE_EXCEPTION' not in caplog.text


def test_concurrent_preflights_keep_separate_contexts(caplog):
    from threading import Barrier
    barrier = Barrier(2)
    def run(reason):
        with diagnostics.preflight_diagnostics(reason, {}, []) as state:
            diagnostics.mark_preflight(reason)
            barrier.wait(timeout=10)
            assert state['reason'] == reason
        assert diagnostics._state.get() is None
    with caplog.at_level(logging.DEBUG), ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, ('schema_question', 'missing_features')))
    assert {e['request_id']: e['reason'] for e in events(caplog, 'Agent preflight', logging.DEBUG)} == {
        'schema_question': 'schema_question', 'missing_features': 'missing_features'}


def test_blocked_preflight_does_not_create_lot4c_telemetry():
    with patch.object(agent_telemetry, 'AgentTelemetry') as telemetry:
        agents.run_agent_once(Mock(), 'Prediction Income=12', [])
        agents.run_agent_once(Mock(), 'Quelles variables sont requises ?', [])
    telemetry.assert_not_called()


@pytest.mark.parametrize('level', [logging.INFO, logging.DEBUG])
@pytest.mark.parametrize('blocked', [False, True])
def test_sensitive_content_never_enters_production_or_debug_logs(level, blocked, caplog, monkeypatch):
    monkeypatch.setattr(diagnostics, '_runtime_pid', None)
    monkeypatch.setenv('GROQ_API_KEY', 'PRIVATE_KEY_SENTINEL')
    sensitive = ('PRIVATE_USER_SENTINEL PRIVATE_CLIENT_SENTINEL PRIVATE_SECRET_SENTINEL '
                 'PRIVATE_RAG_SENTINEL PRIVATE_WEB_SENTINEL PRIVATE_HISTORY_SENTINEL '
                 'PRIVATE_SCORING_SENTINEL PRIVATE_KEY_SENTINEL '
                 'client_id=PRIVATE_CLIENT_SENTINEL https://private.example/sensitive')
    history = [('user', 'Income=918273 ' + sensitive), ('assistant', sensitive)]
    query = 'Prediction ' + ('' if blocked else json.dumps(dict.fromkeys(contract.FEATURES, 918273)))
    agent = Mock(invoke=Mock(return_value={'output': sensitive}))
    with caplog.at_level(level):
        agents.run_agent_once(agent, query + ' ' + sensitive, history)
    assert bool(agent.invoke.call_count) is not blocked
    for forbidden in (*sensitive.split(), '918273', 'client_id', 'https://private.example', *contract.FEATURES):
        assert re.search(re.escape(forbidden) + (r'\b' if forbidden in contract.FEATURES else ''),
                         caplog.text) is None
    runtime, = events(caplog, 'Agent runtime', logging.INFO)
    assert set(runtime) == {'pid', 'python_version', 'implementation'}
    for record in caplog.records:
        if record.name != diagnostics.__name__:
            continue
        if record.getMessage().startswith('Agent preflight '):
            fields = json.loads(record.getMessage().partition('Agent preflight ')[2])
            assert set(fields) == ({'request_id', 'reason', 'invoke_allowed', 'missing_features_count'}
                                   if record.levelno == logging.INFO else
                                   {'request_id', 'reason', 'invoke_allowed', 'missing_features_count',
                                    'preflight_executed', 'recognized_features_count', 'history_messages_count'})
        assert record.exc_info is None
