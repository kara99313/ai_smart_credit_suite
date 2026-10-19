import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from utils import agents, credit_agent_contract as contract
from utils import agent_preflight_diagnostics as diagnostics


def events(caplog, prefix):
    return [json.loads(r.getMessage()[len(prefix) + 1:]) for r in caplog.records
            if r.getMessage().startswith(prefix + ' ')]


@pytest.mark.parametrize('query,reason,count,missing,allowed', [
    ('Quelles variables sont requises ?', 'schema_question', 0, 15, False),
    ('Prediction Income=918273', 'missing_features', 1, 14, False),
    ('Prediction ' + json.dumps(dict.fromkeys(contract.FEATURES, 918273)),
     'complete_features', 15, 0, True),
    ('Bonjour PRIVATE_USER', 'unrecognized_request', 0, 15, True),
    ('Prediction Age=null threshold=0.876543', 'missing_features', 1, 15, False),
])
def test_branches_and_privacy(query, reason, count, missing, allowed, caplog, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'PRIVATE_API_KEY')
    history = [('assistant', 'PRIVATE_HISTORY client_id=PRIVATE_CLIENT score=PRIVATE_SCORE')]
    agent = Mock(invoke=Mock(return_value={'output': 'PRIVATE_RESULT'}))
    expected = contract.preflight(query, contract.user_values(query, history))
    with caplog.at_level(logging.INFO, logger='utils.agent_diagnostics'):
        result = agents.run_agent_once(agent, query + ' PRIVATE_PROMPT secret=PRIVATE_SECRET', history)
    event, = events(caplog, 'Agent preflight')
    assert event == dict(request_id=event['request_id'], preflight_executed=True,
                         reason=reason, recognized_features_count=count,
                         missing_features_count=missing, invoke_allowed=allowed,
                         history_messages_count=1 if allowed else 0)
    assert bool(agent.invoke.call_count) == allowed
    if expected:
        assert result == dict(output=expected, intermediate_steps=[], artifacts=[])
    else:
        assert agent.invoke.call_args.args[0]['chat_history'] is history
        execution = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Agent execution ')]
        assert len(execution) == 1
        assert 'request_id=' + event['request_id'] in execution[0]
    runtime, = events(caplog, 'Agent runtime')
    assert runtime['request_id'] == event['request_id']
    assert runtime['pid'] > 0 and runtime['python_version'] and runtime['executable']
    for module in (agents, contract):
        metadata = runtime['modules'][module.__name__]
        assert metadata['origin'] == module.__file__
        assert metadata['file_sha256'] == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        assert len(metadata['loaded_code_sha256']) == 64
    for forbidden in ('PRIVATE_', '918273', '0.876543', 'client_id', 'threshold', 'Income', 'Age='):
        assert forbidden not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


def test_bypassed_preflight(caplog):
    with patch.object(contract, 'preflight', return_value=None), caplog.at_level(logging.INFO):
        agents.run_agent_once(Mock(invoke=Mock(return_value={'output': 'ok'})), 'PRIVATE', [])
    event, = events(caplog, 'Agent preflight')
    assert event['preflight_executed'] is False
    assert event['reason'] is None
    assert event['invoke_allowed'] is True


def test_logging_failure_does_not_change_behavior():
    with patch.object(diagnostics._logger, 'info', side_effect=RuntimeError('PRIVATE')):
        agent = Mock()
        result = agents.run_agent_once(agent, 'Prediction Income=12', [])
        assert result['output'] == contract.preflight('Prediction Income=12', {'Income': 12})
        agent.invoke.assert_not_called()


def test_context_reset_on_exception(caplog):
    with caplog.at_level(logging.INFO), pytest.raises(ValueError):
        with diagnostics.preflight_diagnostics('test', {}, []):
            diagnostics.mark_preflight()
            raise ValueError('PRIVATE_EXCEPTION')
    assert diagnostics._state.get() is None
    assert 'PRIVATE_EXCEPTION' not in caplog.text
