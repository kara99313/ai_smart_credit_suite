"""Single server-configured administrator. Credentials and grants never leave the server."""
import hashlib
import hmac
import os
import secrets
import time
from threading import RLock

_LOCK = RLock()
_GRANTS = {}
_FAILURES = []
_KEY = '_admin_grant'


def _session():
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    context = get_script_run_ctx(suppress_warning=True)
    if context is None:
        raise PermissionError('Authentification administrateur requise.')
    import streamlit as st
    return st.session_state


def _binding(state):
    if isinstance(state, dict):
        return id(state)
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    context = get_script_run_ctx(suppress_warning=True)
    return context.session_id if context is not None else id(state)


def _config():
    try:
        username = os.environ['ADMIN_USERNAME']
        encoded = os.environ['ADMIN_PASSWORD_HASH']
        algorithm, rounds, salt, digest = encoded.split('$')
        rounds = int(rounds)
        salt, digest = bytes.fromhex(salt), bytes.fromhex(digest)
        ttl = int(os.environ.get('ADMIN_SESSION_TTL_SECONDS', '1800'))
        if (not username.strip() or len(username) > 128 or not username.isprintable()
                or algorithm != 'pbkdf2_sha256' or not 600000 <= rounds <= 2000000
                or not 16 <= len(salt) <= 64 or len(digest) != 32 or not 60 <= ttl <= 28800):
            return None
        return username, rounds, salt, digest, ttl
    except (KeyError, ValueError, TypeError):
        return None


def configured():
    return _config() is not None


def logout():
    state = _session()
    with _LOCK:
        token = state.pop(_KEY, None)
        if isinstance(token, str):
            _GRANTS.pop(token, None)
    for key in list(state):
        if key.startswith(('admin_', 'admin_new_', '_admin_')) or key in ('integration_llm_test', 'health_checks'):
            state.pop(key, None)


def login(username, password):
    state = _session()
    logout()
    cfg = _config()
    if cfg is None or not isinstance(username, str) or not isinstance(password, str):
        return False
    with _LOCK:
        now = time.monotonic()
        _FAILURES[:] = [t for t in _FAILURES if now - t < 300]
        if len(_FAILURES) >= 5:
            return False
        expected, rounds, salt, digest, ttl = cfg
        # Bound input before the expensive derivation; never retain credentials.
        if len(password) > 4096 or len(username) > 128:
            _FAILURES.append(now)
            return False
        derived = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, rounds)
        valid = hmac.compare_digest(username.encode(), expected.encode()) & hmac.compare_digest(derived, digest)
        if not valid:
            _FAILURES.append(now)
            return False
        for token, grant in list(_GRANTS.items()):
            if grant[1] <= now:
                del _GRANTS[token]
        token = secrets.token_urlsafe(32)
        _GRANTS[token] = (expected, now + ttl, cfg, _binding(state))
        state[_KEY] = token
        return True


def require_admin():
    state = _session()
    with _LOCK:
        token = state.get(_KEY)
        grant = _GRANTS.get(token) if isinstance(token, str) else None
        if grant and grant[3] == _binding(state) and grant[1] > time.monotonic() and grant[2] == _config():
            return grant[0]
        if isinstance(token, str):
            _GRANTS.pop(token, None)
        state.pop(_KEY, None)
    raise PermissionError('Authentification administrateur requise.')


def is_admin():
    try:
        require_admin()
        return True
    except PermissionError:
        return False


def sensitive(function):
    """Check before any side effect and audit only verified identity and safe metadata."""
    from functools import wraps
    @wraps(function)
    def guarded(*args, **kwargs):
        actor = require_admin()
        outcome = False
        try:
            result = function(*args, **kwargs)
            outcome = True
            return result
        finally:
            from utils import audit
            try:
                audit.append_audit(function.__name__, actor=actor, details={'completed': outcome})
            except Exception:
                import warnings
                warnings.warn('Journal audit indisponible.', RuntimeWarning)
    return guarded


def render_gate():
    import streamlit as st
    if is_admin():
        st.caption('Administrateur : ' + require_admin())
        if st.button('Se deconnecter'):
            logout()
            st.rerun()
        return True
    if not configured():
        st.error('Authentification administrateur non configuree ou invalide ; acces refuse.')
        return False
    with st.form('admin_login', clear_on_submit=True):
        username = st.text_input('Identifiant administrateur')
        password = st.text_input('Mot de passe administrateur', type='password')
        submitted = st.form_submit_button('Se connecter')
    if submitted:
        if login(username, password):
            st.rerun()
        st.error('Connexion refusee ou trop de tentatives. Reessayer plus tard.')
    return False
