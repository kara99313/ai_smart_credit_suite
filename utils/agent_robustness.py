"""Request-local execution evidence and conservative transport retry policy."""
import logging
import ssl
from types import SimpleNamespace
import httpx
import groq
from langchain_core.callbacks import BaseCallbackHandler


def exception_chain(error):
    chain, seen = [], set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        chain.append(error)
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return chain


def retryable_network_error(error):
    chain = exception_chain(error)
    allowed = (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.WriteTimeout,
               httpx.ConnectError, httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)
    excluded = (ssl.SSLError, PermissionError, ValueError, TypeError, FileNotFoundError, httpx.HTTPStatusError,
                groq.APIStatusError, httpx.LocalProtocolError,
                httpx.UnsupportedProtocol, httpx.PoolTimeout)
    if any(isinstance(item, excluded) for item in chain):
        return False
    # Never unwrap a generic, parsing or business exception into a retry decision.
    if isinstance(error, groq.APITimeoutError):
        return True
    if isinstance(error, groq.APIConnectionError):
        return len(chain) > 1 and isinstance(chain[1], allowed)
    return isinstance(error, allowed)


class ToolProgress(BaseCallbackHandler):
    run_inline = True
    raise_error = True

    def __init__(self):
        self.tool_started = False
        self.started = {}
        self.observations = []
        self.errors = []

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        self.tool_started = True
        self.started[run_id] = dict(name=(serialized or {}).get('name', 'outil'), state='pending')

    def on_tool_end(self, output, *, run_id, **kwargs):
        entry = self.started[run_id]
        entry['state'] = 'confirmed'
        self.observations.append((entry['name'], getattr(output, 'content', output)))

    def on_tool_error(self, error, *, run_id, **kwargs):
        self.started[run_id]['state'] = 'error'
        self.errors.append(run_id)

    def steps(self):
        return [(SimpleNamespace(tool=name, tool_input=''), observation)
                for name, observation in self.observations]


def log_attempt(agent, request_id, attempt, stage, progress, duration, decision, error=None, delay=0):
    # Read only known LangChain model fields; never serialize model/config objects.
    from langchain_core.language_models.chat_models import BaseChatModel
    model = getattr(getattr(agent, 'agent', None), 'runnable', None)
    steps = getattr(model, 'steps', [])
    steps = steps if isinstance(steps, (list, tuple)) else []
    llm = next((getattr(step, 'bound', step) for step in steps
                if isinstance(getattr(step, 'bound', step), BaseChatModel)), None)
    provider = type(llm).__name__ if llm is not None else 'unknown'
    model_name = getattr(llm, 'model_name', None)
    # Model identifiers are configuration, restricted to plain identifier characters.
    import re
    model_name = model_name if isinstance(model_name, str) and re.fullmatch(r'[A-Za-z0-9_./:-]{1,100}', model_name) else 'unknown'
    chain = exception_chain(error) if error else []
    status = next((getattr(item, 'status_code', None) or getattr(getattr(item, 'response', None), 'status_code', None)
                   for item in chain if isinstance(getattr(item, 'status_code', None) or getattr(getattr(item, 'response', None), 'status_code', None), int)), None)
    logging.getLogger('utils.agent_diagnostics').log(logging.ERROR if error else logging.INFO,
        'Agent execution request_id=%s attempt=%s provider=%s model=%s stage=%s exception=%s http_status=%s tool_started=%s tools_started=%s tools_completed=%s tools_errors=%s decision=%s delay=%.3f duration=%.3f',
        request_id, attempt, provider, model_name, stage,
        ','.join(type(item).__name__ for item in chain) or None, status,
        progress.tool_started, len(progress.started), len(progress.observations) + len(progress.errors),
        len(progress.errors), decision, delay, duration)
