"""Targeted exception boundary and Streamlit diagnostic tests."""
import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from streamlit.testing.v1 import AppTest
from utils.agents import run_agent_once
from utils.agent_diagnostics import AgentExecutionError, public_message


@pytest.mark.parametrize('error, expected', [
    (TimeoutError('private detail'), 'Delai'),
    (ConnectionError('connection refused'), 'Connexion'),
    (TypeError('unexpected local argument'), 'Connexion'),
])
def test_original_cause_and_safe_public_message(error, expected, caplog):
    with caplog.at_level(logging.ERROR), pytest.raises(AgentExecutionError) as caught:
        run_agent_once(Mock(invoke=Mock(side_effect=error)), 'question', [])
    assert caught.value.__cause__ is error
    assert isinstance(caught.value.__cause__, type(error))
    assert expected in public_message(caught.value)
    assert str(error) not in public_message(caught.value)
    assert type(error).__name__ in caplog.text
    assert str(error) not in caplog.text
    assert 'AgentExecutor.invoke' in caplog.text
    assert all(record.exc_info is None for record in caplog.records)


@pytest.mark.parametrize('response_status', [False, True])
def test_http_status_and_secrets(response_status, caplog, monkeypatch):
    monkeypatch.setenv('GROQ_API_KEY', 'configured-credential')
    error = RuntimeError('tool_use_failed configured-credential https://u:pass@host/?q=private ' 
                         'gsk_unconfigured Bearer hidden-token\nAuthorization: hidden-header\n'
                         'api_key=hidden-key\npassword=hidden-password\nsecret=hidden-secret')
    if response_status:
        error.response = SimpleNamespace(status_code=400)
    else:
        error.status_code = 429
    with caplog.at_level(logging.ERROR), pytest.raises(AgentExecutionError) as caught:
        run_agent_once(Mock(invoke=Mock(side_effect=error)), 'question', [])
    assert caught.value.__cause__ is error
    assert ('http_status=400' if response_status else 'http_status=429') in caplog.text
    assert 'tool_use_failed' not in caplog.text
    for secret in ['configured-credential', 'u:pass', 'private', 'gsk_unconfigured',
                   'hidden-token', 'hidden-header', 'hidden-key', 'hidden-password', 'hidden-secret']:
        assert secret not in caplog.text
        assert secret not in str(caught.value)


def test_result_processing_stage(caplog):
    with caplog.at_level(logging.ERROR), pytest.raises(AgentExecutionError) as caught:
        run_agent_once(Mock(invoke=Mock(return_value={})), 'question', [])
    assert isinstance(caught.value.__cause__, ValueError)
    assert 'run_agent_once.result' in caplog.text


def test_streamlit_keeps_safe_timeout_message(caplog):
    executor = Mock(invoke=Mock(side_effect=TimeoutError('private diagnostic detail')))
    with patch('utils.llm_status.llm_status_badge'), patch(
            'utils.multitask_agent.create_multitask_agent', return_value=executor), caplog.at_level(logging.ERROR):
        app = AppTest.from_file('app/agent.py', default_timeout=30).run()
        app.chat_input[0].set_value('Recherche BCEAO').run()
    assert not app.exception
    text = ' '.join(item.value for item in app.markdown)
    assert 'Delai' in text
    assert 'private diagnostic detail' not in text
    assert 'Traceback' not in text
    assert sum('Agent execution' in record.message for record in caplog.records) == 1


def test_streamlit_secrets_are_redacted(caplog):
    with patch('streamlit.secrets', {'provider': {'api_key': 'streamlit-credential'}}), caplog.at_level(logging.ERROR):
        with pytest.raises(AgentExecutionError):
            run_agent_once(Mock(invoke=Mock(side_effect=ValueError('invalid streamlit-credential'))), 'x', [])
    assert 'streamlit-credential' not in caplog.text
    assert 'invalid' not in caplog.text
