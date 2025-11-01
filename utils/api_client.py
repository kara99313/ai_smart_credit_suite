# utils/api_client.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Union
import os
import requests

# Par défaut, on colle à ton backend réel (18000)
_DEFAULT_BACKEND = "http://127.0.0.1:18000"

def _resolve_base(base: Optional[str]) -> str:
    """
    Résout l'URL du backend en respectant (par ordre) :
    - argument 'base'
    - BACKEND_URL (env)
    - API_BASE (env)
    - fallback local 127.0.0.1:18000
    """
    env = base or os.getenv("BACKEND_URL") or os.getenv("API_BASE") or _DEFAULT_BACKEND
    return str(env).rstrip("/")


class ApiError(RuntimeError):
    pass


def ping_backend(base: Optional[str] = None, timeout: float = 3.0) -> Tuple[bool, Union[str, Dict[str, Any]]]:
    url_base = _resolve_base(base)
    try:
        r = requests.get(f"{url_base}/health", timeout=timeout)
        if r.ok:
            try:
                return True, r.json()
            except Exception:
                return True, "OK"
        return False, f"HTTP {r.status_code} sur {url_base}/health"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _err_text(resp: requests.Response) -> str:
    try:
        data = resp.json()
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
    except Exception:
        pass
    return resp.text.strip()[:2000]


def predict_credit_api(
    payload: Dict[str, Any],
    timeout: float = 30.0,
    base: Optional[str] = None
) -> Dict[str, Any]:
    url_base = _resolve_base(base)
    try:
        r = requests.post(f"{url_base}/api/predict", json=payload, timeout=timeout)
        if not r.ok:
            raise ApiError(f"HTTP {r.status_code}: {_err_text(r)}")
        return r.json()
    except ApiError:
        raise
    except requests.RequestException as e:
        raise ApiError(f"API /api/predict error ({url_base}): {e}") from e
    except Exception as e:
        raise ApiError(f"API /api/predict unexpected error ({url_base}): {e}") from e


def explain_credit_api(
    payload: Dict[str, Any],
    top_k: int = 10,
    timeout: float = 10.0,
    base: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Optionnel : si un endpoint /tools/explain existe côté backend, on l'utilise.
    Sinon on renvoie None proprement.
    """
    url_base = _resolve_base(base)
    try:
        r = requests.post(f"{url_base}/tools/explain?top_k={int(top_k)}", json=payload, timeout=timeout)
        if r.status_code == 404:
            return None
        if not r.ok:
            return None
        return r.json()
    except Exception:
        return None


# Aliases compat (si du code legacy les appelle)
def predict(payload: Dict[str, Any], timeout: float = 30.0, base: Optional[str] = None) -> Dict[str, Any]:
    return predict_credit_api(payload, timeout=timeout, base=base)

def explain(payload: Dict[str, Any], top_k: int = 10, timeout: float = 10.0, base: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return explain_credit_api(payload, top_k=top_k, timeout=timeout, base=base)
