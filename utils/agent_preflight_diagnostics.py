"""Content-free production diagnostics, independent of Agent lifecycle telemetry."""
import json
import logging
import os
import platform
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Lock

_state = ContextVar('preflight_diagnostic_state', default=None)
_logger = logging.getLogger(__name__)
_runtime_pid = None
_runtime_lock = Lock()


def mark_preflight(reason=None):
    state = _state.get()
    if state is not None:
        state['preflight_executed'] = True
        state['reason'] = reason


def _emit(event, fields, level=logging.INFO):
    # Diagnostics must never interrupt execution, including logging/IO failures.
    try:
        _logger.log(level, '%s %s', event, json.dumps(fields, sort_keys=True))
    except Exception:
        pass


def log_runtime():
    """Attempt one event per process without waiting on another request's lock."""
    global _runtime_pid
    try:
        pid = os.getpid()
        if _runtime_pid == pid or not _runtime_lock.acquire(blocking=False):
            return
        try:
            if _runtime_pid == pid:
                return
            _runtime_pid = pid
        finally:
            _runtime_lock.release()
        _emit('Agent runtime', dict(pid=pid, python_version=platform.python_version(),
                                  implementation=platform.python_implementation()))
    except Exception:
        pass


@contextmanager
def preflight_diagnostics(request_id, values, history):
    from utils.credit_agent_contract import FEATURES, missing_values
    state = dict(preflight_executed=False, reason=None, invoke_allowed=False)
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)
        try:
            if state['reason'] == 'missing_features':
                _emit('Agent preflight', dict(request_id=request_id,
                      reason=state['reason'], invoke_allowed=state['invoke_allowed'],
                      missing_features_count=len(missing_values(values))))
            if _logger.isEnabledFor(logging.DEBUG):
                _emit('Agent preflight', dict(request_id=request_id, **state,
                      recognized_features_count=sum(name in values for name in FEATURES),
                      missing_features_count=len(missing_values(values)),
                      history_messages_count=len(history or []) if state['invoke_allowed'] else 0),
                      level=logging.DEBUG)
        except Exception:
            pass
