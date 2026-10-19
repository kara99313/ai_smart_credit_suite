"""Temporary, metadata-only diagnostics for local/Render comparisons."""
import hashlib
import json
import logging
import marshal
import os
import platform
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import FunctionType

_state = ContextVar('preflight_diagnostic_state', default=None)
_logger = logging.getLogger(__name__)
# Keep temporary diagnostics visible even when the root defaults to WARNING.
_logger.setLevel(logging.INFO)
if not _logger.hasHandlers():
    _logger.addHandler(logging.StreamHandler())


def mark_preflight(reason=None):
    state = _state.get()
    if state is not None:
        state['preflight_executed'] = True
        state['reason'] = reason


def _emit(event, fields):
    # Diagnostics must never interrupt execution, including logging/IO failures.
    try:
        _logger.info('%s %s', event, json.dumps(fields, sort_keys=True))
    except Exception:
        pass


def log_runtime(request_id):
    try:
        modules = {}
        for name in ('utils.agents', 'utils.credit_agent_contract'):
            module = sys.modules[name]
            origin = module.__file__
            try:
                file_hash = hashlib.sha256(Path(origin).read_bytes()).hexdigest()
            except (OSError, TypeError):
                file_hash = 'unavailable'
            code = b''.join(marshal.dumps(value.__code__)
                            for key, value in sorted(vars(module).items())
                            if isinstance(value, FunctionType) and value.__module__ == name)
            modules[name] = dict(origin=origin, file_sha256=file_hash,
                                 loaded_code_sha256=hashlib.sha256(code).hexdigest())
        _emit('Agent runtime', dict(request_id=request_id, pid=os.getpid(),
              python_version=platform.python_version(), executable=sys.executable,
              implementation=platform.python_implementation(), modules=modules))
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
            _emit('Agent preflight', dict(request_id=request_id, **state,
                  recognized_features_count=sum(name in values for name in FEATURES),
                  missing_features_count=len(missing_values(values)),
                  history_messages_count=len(history or []) if state['invoke_allowed'] else 0))
        except Exception:
            pass
