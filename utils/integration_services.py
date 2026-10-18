"""Configuration writes for the active LLM integration, without touching app data."""
from __future__ import annotations
import os
import re
from pathlib import Path
from urllib.parse import urlsplit
from utils.runtime_config import ENV_PATH
from utils.llm_runtime import PROVIDERS, get_setting


def validate_base_url(value):
    try:
        parsed = urlsplit(value)
        valid = (not any(c.isspace() for c in value) and parsed.scheme in {'http', 'https'} and parsed.hostname and not parsed.username
                 and not parsed.password and not parsed.query and not parsed.fragment)
        port = parsed.port  # reject malformed port syntax too
        if not valid or (port is not None and not 1 <= port <= 65535):
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError('URL HTTP(S) absolue sans identifiants, query ni fragment requise.') from None
    return value.rstrip('/')


def _set_env_var(content, key, value):
    if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key) or any(c in value for c in '\r\n\x00'):
        raise ValueError('Valeur de configuration invalide (retour a la ligne interdit).')
    # python-dotenv single quoted syntax preserves #, $, quotes and backslashes.
    quoted = "'" + value.replace('\\', '\\\\').replace("'", "\\'") + "'"
    pattern = re.compile(r'^\s*(?:export\s+)?' + re.escape(key) + r'\s*=')
    lines = []
    found = False
    for line in content.splitlines():
        if pattern.match(line):
            if not found:
                lines.append(f'{key}={quoted}')
                found = True
        else:
            lines.append(line)
    if not found:
        lines.append(f'{key}={quoted}')
    return '\n'.join(lines) + '\n'


from utils.admin_auth import sensitive


@sensitive
def save_llm_configuration(provider, model, api_key='', base_url='', path=None):
    if provider not in PROVIDERS:
        raise ValueError('Fournisseur non supporte.')
    if not isinstance(model, str) or not model.strip() or any(c in model for c in '\r\n\x00'):
        raise ValueError('Modele vide ou invalide.')
    key_name, model_name, _, _, _ = PROVIDERS[provider]
    values = {'LLM_PROVIDER': provider, model_name: model.strip()}
    # An empty form field retains the configured secret; never send it back to UI.
    if key_name and api_key.strip():
        values[key_name] = api_key.strip()
    if provider == 'ollama':
        values['OLLAMA_BASE_URL'] = validate_base_url(base_url.strip())
    target = Path(path) if path is not None else ENV_PATH
    content = target.read_text(encoding='utf-8') if target.exists() else ''
    for key, value in values.items():
        content = _set_env_var(content, key, value)
    tmp = target.with_name(target.name + '.tmp')
    try:
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(target)
    except OSError:
        raise OSError('Enregistrement de la configuration impossible ; environnement inchange.') from None
    # Update only after the complete write succeeded. Secrets in Streamlit retain priority.
    os.environ.update(values)
    return values.keys()


def effective_llm_identity():
    """Hash the effective configuration so old test results are never reused."""
    import hashlib
    names = ['LLM_PROVIDER', 'LLM_TIMEOUT', 'OLLAMA_TIMEOUT', 'OLLAMA_BASE_URL', 'OPENAI_BASE_URL', 'GROQ_BASE_URL']
    for key_name, model_name, *_ in PROVIDERS.values():
        names.append(model_name)
        if key_name:
            names.append(key_name)
    return hashlib.sha256(repr([(name, get_setting(name)) for name in names]).encode()).hexdigest()
