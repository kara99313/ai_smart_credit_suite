# utils/api_client.py
from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Union
import math
from numbers import Real
import requests
from utils.runtime_config import backend_url

# Par défaut, on colle à ton backend réel (18000)
def _resolve_base(base: Optional[str]) -> str:
    return backend_url(base)


class ApiError(RuntimeError):
    pass


def _validate_prediction_response(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise ApiError("Reponse de prediction invalide : objet JSON attendu.")
    pd_value = result.get("pd")
    if isinstance(pd_value, bool) or not isinstance(pd_value, Real) or not math.isfinite(pd_value) or not 0 <= pd_value <= 1:
        raise ApiError("Reponse invalide : PD numerique finie entre 0 et 1 requise.")
    score = result.get("score_1000")
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 1000:
        raise ApiError("Reponse invalide : score_1000 requis.")
    if not isinstance(result.get("rating"), str) or not result["rating"]:
        raise ApiError("Reponse invalide : rating requis.")
    if result.get("decision") not in {"ACCEPT", "REVIEW/REJECT"}:
        raise ApiError("Reponse invalide : decision requise.")
    threshold = result.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, Real) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ApiError("Reponse invalide : threshold requis.")
    features = result.get("used_features")
    if not isinstance(features, list) or not features or not all(isinstance(f, str) for f in features):
        raise ApiError("Reponse invalide : used_features requis.")
    return result


def ping_backend(base: Optional[str] = None, timeout: float = 3.0) -> Tuple[bool, Union[str, Dict[str, Any]]]:
    try:
        url_base = _resolve_base(base)
        r = requests.get(f"{url_base}/health", timeout=timeout)
        if r.ok:
            try:
                return True, r.json()
            except Exception:
                return True, "OK"
        return False, f"HTTP {r.status_code} : service indisponible."
    except Exception:
        return False, "Connexion API impossible ou configuration invalide."


def _err_text(resp: requests.Response) -> str:
    """Never expose an untrusted HTTP body, including JSON detail fields."""
    return "La requete API a echoue."


def predict_credit_api(
    payload: Dict[str, Any],
    timeout: float = 30.0,
    base: Optional[str] = None,
    *, synthetic: bool = False
) -> Dict[str, Any]:
    url_base = "configuration API invalide"
    try:
        url_base = _resolve_base(base)
        options = {"headers": {"X-Prediction-Origin": "synthetic"}} if synthetic else {}
        r = requests.post(f"{url_base}/api/predict", json=payload, timeout=timeout, **options)
        if not r.ok:
            raise ApiError(f"HTTP {r.status_code}: {_err_text(r)}")
        return _validate_prediction_response(r.json())
    except ApiError:
        raise
    except requests.RequestException:
        raise ApiError("Connexion API impossible ou delai depasse.") from None
    except Exception:
        raise ApiError("Configuration API ou reponse de prediction invalide.") from None


def explain_credit_api(
    payload: Dict[str, Any],
    top_k: int = 10,
    timeout: float = 10.0,
    base: Optional[str] = None,
    *,
    allow_global: bool = False,
    prediction: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return validated global data only to callers explicitly supporting that scope.

    Legacy report consumers expect individual contributions, so they receive None
    rather than misinterpreting coefficient magnitudes as signed contributions.
    """
    try:
        body = dict(payload)
        if prediction is not None:
            if not prediction.get("model_id") or not prediction.get("input_id"):
                return None
            body.update(expected_model_id=prediction["model_id"],
                        expected_input_id=prediction["input_id"], expected_pd=prediction["pd"])
        url_base = _resolve_base(base)
        response = requests.post(f"{url_base}/tools/explain?top_k={int(top_k)}", json=body, timeout=timeout)
        if not response.ok:
            return None
        result = response.json()
        if not allow_global or not isinstance(result, dict) or result.get("ok") is not True:
            return None
        if result.get("method") != "global_importance" or result.get("scope") != "global":
            return None
        if result.get("space") != "transformed_features" or result.get("coefficient_space") != "log_odds" or result.get("class_label") != 1:
            return None
        if prediction is not None:
            if any(result.get(key) != prediction[key] for key in ("model_id", "input_id")):
                return None
            probability = result.get("predicted_pd")
            if not isinstance(probability, Real) or not math.isfinite(probability) or not math.isclose(probability, prediction["pd"], rel_tol=1e-12, abs_tol=1e-12):
                return None
        items = result.get("items")
        feature_count = result.get("feature_count")
        if isinstance(feature_count, bool) or not isinstance(feature_count, int) or feature_count <= 0:
            return None
        if not isinstance(items, list) or len(items) != min(int(top_k), feature_count):
            return None
        names = set()
        for item in items:
            if not isinstance(item, dict) or "contribution" in item:
                return None
            name = item.get("feature")
            if not isinstance(name, str) or not name or name in names:
                return None
            names.add(name)
            coefficient, importance = item.get("coefficient"), item.get("importance")
            if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in (coefficient, importance)):
                return None
            if importance < 0 or not math.isclose(importance, abs(coefficient), rel_tol=1e-12, abs_tol=1e-12):
                return None
        return result
    except Exception:
        return None


# Aliases compat (si du code legacy les appelle)
def predict(payload: Dict[str, Any], timeout: float = 30.0, base: Optional[str] = None) -> Dict[str, Any]:
    return predict_credit_api(payload, timeout=timeout, base=base)

def explain(payload: Dict[str, Any], top_k: int = 10, timeout: float = 10.0, base: Optional[str] = None) -> Optional[Dict[str, Any]]:
    return explain_credit_api(payload, top_k=top_k, timeout=timeout, base=base)
