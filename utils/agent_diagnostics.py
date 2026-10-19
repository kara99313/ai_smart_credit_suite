"""Safe server diagnostics for agent failures; no raw exception logging."""
import logging
import os
import re
from utils.llm_runtime import ProviderError, explain

logger = logging.getLogger(__name__)


class AgentExecutionError(RuntimeError):
    """Safe public message with the original exception retained as cause."""


def sanitized_message(error):
    text = str(error)
    for key, value in os.environ.items():
        if value and re.search(r'key|token|password|secret|authorization', key, re.I):
            text = text.replace(value, '[masque]')
    # Streamlit credentials may be supplied independently of environment variables.
    try:
        import streamlit as st
        def mask_settings(settings):
            nonlocal text
            for key, value in settings.items():
                if hasattr(value, 'items'):
                    mask_settings(value)
                elif value and re.search(r'key|token|password|secret|authorization', str(key), re.I):
                    text = text.replace(str(value), '[masque]')
        mask_settings(st.secrets)
    except Exception:
        pass  # Missing secrets.toml is normal; do not log configuration failures.
    text = re.sub(r'https?://[^\s]+', '[URL masquee]', text)
    text = re.sub(r'(?i)\b(?:bearer|basic)\s+[^\s,;]+', '[identifiant masque]', text)
    text = re.sub(r'(?i)[\w-]*(?:api[_-]?key|token|password|passwd|secret|authorization)[\w-]*[^\r\n]*', '[champ sensible masque]', text)
    text = re.sub(r'\b(?:sk-|gsk_)[A-Za-z0-9_-]+', '[cle masquee]', text)
    return re.sub(r'[\x00-\x1f\x7f]', ' ', text)[:1000]


def public_message(error):
    if isinstance(error, AgentExecutionError):
        return str(error)
    if isinstance(error, ProviderError):
        return 'Connexion ou appel fournisseur en echec.'
    return explain(error)


def log_agent_failure(error, stage):
    from utils.agent_robustness import exception_chain
    chain = exception_chain(error)
    logger.error('Agent failure stage=%s exception=%s http_status=%s',
                 stage, ','.join(type(item).__name__ for item in chain),
                 next((item.status_code for item in chain
                       if isinstance(getattr(item, 'status_code', None), int)), None))
