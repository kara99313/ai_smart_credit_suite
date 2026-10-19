"""Deterministic bounded retries and request-local tool evidence."""
from unittest.mock import Mock, patch
from uuid import uuid4
import ssl
import httpx
import groq
import pytest
from utils.agents import run_agent_once
from utils.agent_diagnostics import AgentExecutionError
from utils.agent_robustness import retryable_network_error


@pytest.mark.parametrize('failures, calls', [(1, 2), (2, 2)])
def test_bounded_retry(failures, calls):
    error = httpx.ReadTimeout('secret')
    agent = Mock(invoke=Mock(side_effect=[error] * failures + [{'output': 'ok'}]))
    with patch('time.sleep') as sleep, patch('random.uniform', return_value=.125):
        if failures == 2:
            with pytest.raises(AgentExecutionError):
                run_agent_once(agent, 'private prompt', [])
        else:
            assert 'ok' in run_agent_once(agent, 'private prompt', [])['output']
    assert agent.invoke.call_count == calls
    sleep.assert_called_once_with(1.125)


@pytest.mark.parametrize('error', [RuntimeError('ReadTimeout'), ValueError('network'),
    httpx.PoolTimeout('x'), httpx.LocalProtocolError('x'), httpx.UnsupportedProtocol('x'),
    PermissionError('x'), ssl.SSLError('x'), httpx.HTTPStatusError('x', request=Mock(), response=Mock())])
def test_non_retryable(error):
    agent = Mock(invoke=Mock(side_effect=error))
    with pytest.raises(AgentExecutionError):
        run_agent_once(agent, 'x', [])
    assert agent.invoke.call_count == 1


@pytest.mark.parametrize('name', ['predict_credit', 'build_report', 'unknown', 'search_web', 'rag_retrieve', 'client_context', 'now'])
@pytest.mark.parametrize('completion', ['pending', 'confirmed', 'error'])
def test_no_replay_and_partial_evidence(name, completion):
    html = '<html><body>PRIVATE_REPORT</body></html>'
    def invoke(payload, config):
        callback = config['callbacks'][0]
        run_id = uuid4()
        callback.on_tool_start({'name': name}, 'PRIVATE_INPUT', run_id=run_id)
        assert callback.tool_started
        if completion == 'confirmed':
            callback.on_tool_end(html if name == 'build_report' else 'confirmed observation', run_id=run_id)
        elif completion == 'error':
            callback.on_tool_error(ValueError('PRIVATE_ERROR'), run_id=run_id)
        raise httpx.ReadError('PRIVATE_ERROR')
    agent = Mock(invoke=Mock(side_effect=invoke))
    result = run_agent_once(agent, 'x', [])
    assert agent.invoke.call_count == 1
    assert 'PRIVATE_REPORT' not in result['output']
    assert 'PRIVATE_INPUT' not in str(result['intermediate_steps'])
    if completion == 'pending':
        assert 'incertain' in result['output']
    if completion == 'confirmed':
        if name == 'build_report':
            assert result['artifacts'][0]['data'] == html
        else:
            assert 'confirmed observation' in str(result['intermediate_steps'])
            assert 'confirmed observation' not in result['output']


def test_causes_and_tls_exclusion():
    error = groq.APIConnectionError(request=httpx.Request('GET', 'https://example.org'))
    assert not retryable_network_error(error)
    error.__cause__ = httpx.ConnectError('x')
    assert retryable_network_error(error)
    error.__cause__.__cause__ = ssl.SSLError('x')
    assert not retryable_network_error(error)
    generic = RuntimeError('x')
    generic.__cause__ = httpx.ReadTimeout('x')
    assert not retryable_network_error(generic)


def test_request_isolation_and_safe_diagnostics(caplog):
    callbacks = []
    def invoke(payload, config):
        callbacks.append(config['callbacks'][0])
        raise ValueError('PRIVATE_PROMPT PRIVATE_HTML PRIVATE_KEY')
    with caplog.at_level('ERROR'):
        for _ in range(2):
            with pytest.raises(AgentExecutionError):
                run_agent_once(Mock(invoke=Mock(side_effect=invoke)), 'PRIVATE_PROMPT', [])
    assert callbacks[0] is not callbacks[1]
    assert 'PRIVATE' not in caplog.text
    assert 'request_id=' in caplog.text


def test_streaming_disabled():
    from utils import multitask_agent as module
    with patch.object(module, 'load_llm'), patch.object(module, 'create_tool_calling_agent'), patch.object(module, 'AgentExecutor') as factory:
        module.create_multitask_agent()
    assert factory.call_args.kwargs['stream_runnable'] is False

@pytest.mark.parametrize('file, namespace', [('app/agent.py', 'agent'), ('app/chatbot_assistant.py', 'assistant')])
def test_streamlit_partial_report(file, namespace):
    from streamlit.testing.v1 import AppTest
    html = '<html><body>PRIVATE_REPORT</body></html>'
    def invoke(payload, config):
        callback = config['callbacks'][0]
        run_id = uuid4()
        callback.on_tool_start({'name': 'build_report'}, '', run_id=run_id)
        callback.on_tool_end(html, run_id=run_id)
        callback.on_tool_start({'name': 'unknown'}, '', run_id=uuid4())
        raise httpx.ReadTimeout('PRIVATE_FAILURE')
    executor = Mock(invoke=Mock(side_effect=invoke))
    with patch('utils.llm_status.llm_status_badge'), patch('utils.multitask_agent.create_multitask_agent', return_value=executor):
        app = AppTest.from_file(file, default_timeout=30).run()
        app.chat_input[0].set_value('rapport').run()
        assert not app.exception
        assert app.session_state[namespace + '_results'][0]['artifacts'][0]['data'] == html
        assert 'PRIVATE_REPORT' not in ' '.join(m.value for m in app.markdown)
        assert 'incertain' in ' '.join(m.value for m in app.markdown)
        assert len(app.get('download_button')) == 1
        assert executor.invoke.call_count == 1


def test_real_executor_callbacks_prevent_replay():
    from langchain.agents import AgentExecutor
    from langchain_core.agents import AgentAction
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import StructuredTool
    calls = []
    def tool():
        calls.append('executed')
        return 'confirmed'
    def plan(payload):
        if payload['intermediate_steps']:
            raise httpx.ReadTimeout('failure after tool')
        return AgentAction(tool='future_tool', tool_input={}, log='')
    executor = AgentExecutor(agent=RunnableLambda(plan), stream_runnable=False,
        tools=[StructuredTool.from_function(tool, name='future_tool', description='test')],
        return_intermediate_steps=True)
    result = run_agent_once(executor, 'x', [])
    assert calls == ['executed']
    assert 'confirmed' in str(result['intermediate_steps'])
    assert result['intermediate_steps'][0]['status'] == 'uncertain'

@pytest.mark.parametrize('kind', [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
    httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError])
def test_allowed_transport_classes(kind):
    assert retryable_network_error(kind('ignored message'))


def test_groq_timeout_and_status():
    request = httpx.Request('GET', 'https://example.org')
    assert retryable_network_error(groq.APITimeoutError(request=request))
    response = httpx.Response(503, request=request)
    assert not retryable_network_error(groq.APIStatusError('x', response=response, body=None))
