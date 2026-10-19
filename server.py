# server.py
from __future__ import annotations
import json, os, sys, time
import hashlib, io
from typing import Literal, Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path
from functools import lru_cache

# ===== UTF-8 partout =====
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for k in ("LC_ALL","LANG","LANGUAGE"):
    os.environ.setdefault(k, "C.UTF-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from utils.interservice_auth import InterserviceAuthMiddleware, backend_loopback
from utils.runtime_config import ROOT
from utils.prediction_telemetry import Attempt, PredictionTelemetryMiddleware

# ===== Chemins / modèle =====
PROJECT_ROOT = ROOT
MODEL_PATH = Path(os.getenv("MODEL_PATH") or "model/logistic_pipeline_best.pkl")
if not MODEL_PATH.is_absolute():
    MODEL_PATH = PROJECT_ROOT / MODEL_PATH

# ===== Features (15) =====
FEATURES = [
    "DTIRatio","TrustScorePsychometric","HouseholdSize","NumCreditLines","Income",
    "CommunityGroupMember","HasMortgage","MonthsEmployed","HasSocialAid",
    "MobileMoneyTransactions","Age","InterestRate","LoanTerm","LoanAmount","InformalIncome",
]
CATEGORICAL: set[str] = {"CommunityGroupMember", "HasMortgage", "HasSocialAid"}
NUMERIC: set[str] = set(FEATURES) - CATEGORICAL

# ===== Chargement pipeline =====
try:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    _model_bytes = MODEL_PATH.read_bytes()
    MODEL_ID = hashlib.sha256(_model_bytes).hexdigest()
    _pipe = joblib.load(io.BytesIO(_model_bytes))
    del _model_bytes
except Exception as e:
    raise RuntimeError(
        f"Impossible de charger le modèle: {MODEL_PATH}\n"
        f"Cause: {e}\n"
        "Vérifie scikit-learn et le chemin MODEL_PATH."
    )

def _normalize_missing(x: Any) -> Any:
    if isinstance(x, str) and x.strip().lower() in {"", "na", "n/a", "nan", "none", "null"}:
        return np.nan
    return x

# ===== Alignement robuste =====
def _align_df(d: Dict[str, Any]) -> pd.DataFrame:
    x = dict(d or {})
    for k in FEATURES:
        if k not in x or x[k] is None:
            x[k] = (False if k in CATEGORICAL else np.nan)
    for k in list(x.keys()):
        x[k] = _normalize_missing(x[k])

    for k in FEATURES:
        v = x[k]
        if k in NUMERIC:
            x[k] = pd.to_numeric(v, errors="coerce")
        else:
            if isinstance(v, str):
                v2 = v.strip().lower()
                if v2 in {"1","true","vrai","oui","yes","y","t"}:
                    x[k] = "True"
                elif v2 in {"0","false","faux","non","no","n"}:
                    x[k] = "False"
                else:
                    x[k] = "False"
            else:
                x[k] = "True" if bool(v) else "False"

    df = pd.DataFrame([x]).replace([np.inf, -np.inf], np.nan)

    feats = FEATURES
    try:
        if hasattr(_pipe, "feature_names_in_"):
            feats = [str(c) for c in _pipe.feature_names_in_]
    except Exception:
        pass

    for col in feats:
        if col not in df.columns:
            df[col] = ("False" if col in CATEGORICAL else np.nan)

    for n in NUMERIC:
        if n in df.columns:
            df[n] = pd.to_numeric(df[n], errors="coerce").astype("float64")
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("string")

    bad_obj = [c for c in df.columns if df[c].dtype == "object"]
    for c in bad_obj:
        if c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        else:
            df[c] = df[c].astype("string")

    try:
        return df[feats]
    except Exception:
        return df.reindex(columns=feats, fill_value=np.nan)

def _align_categoricals(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    df2 = df.copy()
    for c in CATEGORICAL:
        if c not in df2.columns:
            continue
        if mode == "str":
            df2[c] = df2[c].astype("string").map(lambda v: "True" if str(v).strip().lower() in {"true","1"} else "False")
        elif mode == "bool":
            df2[c] = df2[c].astype("string").map(lambda v: str(v).strip().lower() in {"true","1"})
        elif mode == "int":
            df2[c] = df2[c].astype("string").map(lambda v: 1 if str(v).strip().lower() in {"true","1"} else 0).astype("int64")
    return df2

def _sigmoid(z: float) -> float:
    z = max(min(z, 50), -50)
    return 1.0 / (1.0 + np.exp(-z))

def _compute_pd(df: pd.DataFrame) -> float:
    if not callable(getattr(_pipe, "predict_proba", None)):
        raise RuntimeError("Le modele ne fournit pas predict_proba ; aucune PD disponible.")
    classes = np.asarray(getattr(_pipe, "classes_", []))
    if classes.ndim != 1 or classes.size != 2:
        raise RuntimeError("Un modele binaire avec classes_ est requis.")
    default_indices = np.flatnonzero(classes == 1)
    if default_indices.size != 1:
        raise RuntimeError("Classe defaut numerique 1 absente ou ambigue dans classes_.")
    proba = np.asarray(_pipe.predict_proba(df), dtype=float)
    if proba.shape != (len(df), len(classes)) or len(df) != 1:
        raise RuntimeError("Dimensions invalides de predict_proba.")
    if not np.isfinite(proba).all() or ((proba < 0) | (proba > 1)).any():
        raise RuntimeError("predict_proba a renvoye une probabilite invalide.")
    if not np.allclose(proba.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise RuntimeError("La somme des probabilites doit etre egale a 1.")
    return float(proba[0, default_indices[0]])


def _normalize_prediction_category(value: Any) -> str:
    """Map supported boolean representations to the learned No/Yes labels."""
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if value else "No"
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return "Yes"
        if value == 0:
            return "No"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"yes", "oui", "true", "vrai", "1", "y", "t"}:
            return "Yes"
        if text in {"no", "non", "false", "faux", "0", "n", "f"}:
            return "No"
    raise ValueError("Categorie inconnue : utiliser Oui/Non, Yes/No ou un booleen.")


def _prepare_prediction_df(raw: Dict[str, Any]) -> pd.DataFrame:
    """Prepare inference inputs without changing the legacy explanation path."""
    features = [str(name) for name in getattr(_pipe, "feature_names_in_", [])]
    if len(features) != len(FEATURES) or set(features) != set(FEATURES):
        raise RuntimeError("Features du modele incompatibles avec les 15 variables attendues.")
    missing = [name for name in features if name not in raw]
    if missing:
        raise ValueError(f"Variables manquantes : {', '.join(missing)}")
    row = {}
    for name in features:
        if name in CATEGORICAL:
            row[name] = _normalize_prediction_category(raw[name])
        else:
            value = float(raw[name])
            if not np.isfinite(value):
                raise ValueError(f"Valeur numerique non finie pour {name}.")
            row[name] = value
    # Validate before handle_unknown='ignore' can hide an encoding mismatch.
    try:
        preprocessor = _pipe.named_steps["preprocessing"]
        categorical_names = next(list(columns) for name, _, columns in preprocessor.transformers_ if name == "cat")
        categories = preprocessor.named_transformers_["cat"].categories_
    except (AttributeError, KeyError, StopIteration, TypeError) as error:
        raise RuntimeError("Impossible de verifier l'encodeur categoriel ajuste.") from error
    if set(categorical_names) != CATEGORICAL or len(categories) != len(categorical_names):
        raise RuntimeError("Schema categoriel du modele incompatible.")
    for name, learned in zip(categorical_names, categories):
        if row[name] not in learned:
            raise RuntimeError(f"Categorie normalisee de {name} absente de l'encodeur.")
    return pd.DataFrame([row], columns=features)


def _rating_from_pd(v: float) -> str:
    v = float(v)
    if v < 0.01: return "AAA"
    if v < 0.02: return "AA"
    if v < 0.03: return "A"
    if v < 0.05: return "BBB"
    if v < 0.08: return "BB"
    if v < 0.12: return "B"
    if v < 0.20: return "CCC"
    if v < 0.35: return "CC"
    if v < 0.50: return "C"
    return "D"

def _decision_from_pd(pd_val: float, threshold: float) -> str:
    return "ACCEPT" if pd_val < threshold else "REVIEW/REJECT"

# ===== Pydantic =====
class ScoringInputs(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    DTIRatio: float
    TrustScorePsychometric: float
    HouseholdSize: int
    NumCreditLines: int
    Income: float
    CommunityGroupMember: bool
    HasMortgage: bool
    MonthsEmployed: int
    HasSocialAid: bool
    MobileMoneyTransactions: int
    Age: int
    InterestRate: float
    LoanTerm: int
    LoanAmount: float
    InformalIncome: float

    @field_validator("CommunityGroupMember", "HasMortgage", "HasSocialAid", mode="before")
    @classmethod
    def normalize_category(cls, value):
        return _normalize_prediction_category(value)


class PredictRequest(ScoringInputs):
    threshold: float = Field(0.10, ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    pd: float
    score_1000: int
    rating: str
    decision: Literal["ACCEPT", "REVIEW/REJECT"]
    threshold: float
    used_features: List[str]
    model_id: str
    input_id: str

class PredictToolBody(BaseModel):
    data: Dict[str, Any]

class RAGPayload(BaseModel):
    query: str
    mode: Optional[str] = "docs"
    k: Optional[int] = 4

class WebSearchPayload(BaseModel):
    query: str
    k: Optional[int] = 3
    fetch: Optional[bool] = False

class ReportPayload(BaseModel):
    client_id: str
    timestamp: Optional[str] = None
    pd_value: float
    threshold: float
    inputs_json: str
    model_tag: Optional[str] = "pipeline_v1"
    logo_path: Optional[str] = None

class ExplainPayload(ScoringInputs):
    expected_model_id: Optional[str] = None
    expected_input_id: Optional[str] = None
    expected_pd: Optional[float] = Field(None, ge=0.0, le=1.0)


def _prediction_input_id(df: pd.DataFrame) -> str:
    normalized = json.dumps(df.iloc[0].to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

# ===== FastAPI =====
app = FastAPI(title="AI Smart Credit Suite — Backend", version="1.3.0")
app.add_middleware(PredictionTelemetryMiddleware, model_id=MODEL_ID)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.add_middleware(InterserviceAuthMiddleware)

@app.exception_handler(RequestValidationError)
async def invalid_request(_request, _error):
    # Validation errors can echo submitted credentials or arbitrary input values.
    return JSONResponse(status_code=422, content={"detail": "Requete invalide ; verifier les parametres."})


# ===== Warm-up =====
def _warmup_model() -> None:
    attempt = Attempt(MODEL_ID, warmup=True)
    status = 500
    try:
        dummy = {
            "DTIRatio": 0.3, "TrustScorePsychometric": 0.6, "HouseholdSize": 3, "NumCreditLines": 2,
            "Income": 100000, "CommunityGroupMember": False, "HasMortgage": False, "MonthsEmployed": 12,
            "HasSocialAid": False, "MobileMoneyTransactions": 10, "Age": 30, "InterestRate": 10.0,
            "LoanTerm": 12, "LoanAmount": 100000, "InformalIncome": 0.0
        }
        _compute_pd(_prepare_prediction_df(dummy))
        status = 200
    except Exception as e:
        print("[WARMUP] modele non prechauffe.")
    finally:
        attempt.finish(status)

@app.on_event("startup")
def _on_startup():
    t0 = time.perf_counter()
    _warmup_model()
    print(f"[WARMUP] model preheated in {time.perf_counter() - t0:.2f}s")

# ===== Routes =====
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.get("/api/predict_ping")
def predict_ping():
    return {"ok": True, "msg": "predict service alive"}

@app.post("/api/predict_echo")
def predict_echo(req: PredictRequest):
    df = _prepare_prediction_df(req.model_dump())
    return {
        "ok": True,
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "row0": df.iloc[0].to_dict()
    }

@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    try:
        raw = req.model_dump()
        df = _prepare_prediction_df(raw)
        pd_val = _compute_pd(df)
        score = int(round((1 - pd_val) * 1000))
        rating = _rating_from_pd(pd_val)
        decision = _decision_from_pd(pd_val, req.threshold)
        return PredictResponse(
            pd=pd_val, score_1000=score, rating=rating, decision=decision,
            threshold=req.threshold, used_features=list(df.columns),
            model_id=MODEL_ID, input_id=_prediction_input_id(df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail="Operation indisponible ; reessayer plus tard.") from None

# ===== Outils Agent IA =====
@lru_cache(maxsize=1)
def _import_agent_tools() -> Tuple[
    Callable[[str], str],
    Callable[[str, str, int], str],
    Callable[[str, int, bool], str],
    Callable[[str], str],
    Callable[[str], str],
]:
    from utils import agent_tools as at  # type: ignore
    predict_fn = getattr(at, "predict_credit_tool", lambda _:"predict_credit_tool indisponible")
    rag_fn = getattr(at, "rag_retrieve_tool", getattr(at, "rag_search_tool", lambda q,mode="docs",k=4:"RAG tool indisponible"))
    web_fn = getattr(at, "tool_web_search", getattr(at, "search_web_tool", lambda q,k=3,fetch=False:"Recherche Web indisponible"))
    rep_fn = getattr(at, "generate_report_tool", getattr(at, "build_report_tool", lambda ctx:"Génération de rapport indisponible"))
    now_fn = getattr(at, "now_tool", lambda fmt="%Y-%m-%d %H:%M:%S": __import__("datetime").datetime.now().isoformat())
    return predict_fn, rag_fn, web_fn, rep_fn, now_fn

# Optional tool dependencies are loaded only when their routes are called.

@app.post("/tools/predict_credit")
def tools_predict_credit(body: PredictToolBody):
    try:
        token = backend_loopback.set(True)
        try:
            txt = _import_agent_tools()[0](json.dumps(body.data, ensure_ascii=False))
        finally:
            backend_loopback.reset(token)
        return {"ok": True, "result": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Operation indisponible ; reessayer plus tard.") from None

@app.post("/tools/rag_search")
def tools_rag_search(body: RAGPayload):
    try:
        txt = _import_agent_tools()[1](body.query, mode=body.mode or "docs", k=body.k or 4)
        return {"ok": True, "result": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Operation indisponible ; reessayer plus tard.") from None

@app.post("/tools/web_search")
def tools_web_search(body: WebSearchPayload):
    try:
        txt = _import_agent_tools()[2](body.query, k=body.k or 3, fetch=bool(body.fetch))
        return {"ok": True, "result": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Operation indisponible ; reessayer plus tard.") from None

@app.post("/tools/generate_report_html")
def tools_generate_report(body: ReportPayload):
    try:
        ctx = {
            "client_id": body.client_id,
            "timestamp": body.timestamp,
            "pd_value": body.pd_value,
            "threshold": body.threshold,
            "inputs_json": body.inputs_json,
            "model_tag": body.model_tag,
            "logo_path": body.logo_path,
        }
        html = _import_agent_tools()[3](json.dumps(ctx, ensure_ascii=False))
        return {"ok": True, "html": html}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Operation indisponible ; reessayer plus tard.") from None

@app.get("/tools/now")
def tools_now(fmt: str = "%Y-%m-%d %H:%M:%S"):
    try:
        return {"ok": True, "now": _import_agent_tools()[4](fmt)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Operation indisponible ; reessayer plus tard.") from None

# ===== Explain: validated global fallback, separate from individual contributions =====
@app.post("/tools/explain")
def tools_explain(body: ExplainPayload, top_k: int = Query(10, ge=1, le=30)):
    from utils.scoring_explanations import global_coefficient_importance
    try:
        df = _prepare_prediction_df(body.model_dump())
        input_id = _prediction_input_id(df)
        pd_value = _compute_pd(df)
        if body.expected_model_id is not None and body.expected_model_id != MODEL_ID:
            raise HTTPException(status_code=409, detail="Explanation model differs from prediction model.")
        if body.expected_input_id is not None and body.expected_input_id != input_id:
            raise HTTPException(status_code=409, detail="Explanation inputs differ from prediction inputs.")
        if body.expected_pd is not None and not np.isclose(body.expected_pd, pd_value, rtol=1e-12, atol=1e-12):
            raise HTTPException(status_code=409, detail="Explanation probability differs from prediction probability.")
        provenance = {"model_id": MODEL_ID, "input_id": input_id, "predicted_pd": pd_value}
        try:
            result = global_coefficient_importance(_pipe, df)
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as error:
            return {"ok": False, "method": "unavailable", "scope": "global",
                    "reason": "invalid_global_explanation", "detail": "Explication globale indisponible.",
                    "items": [], **provenance}
        result["items"] = sorted(result["items"], key=lambda item: item["importance"], reverse=True)[:int(top_k)]
        return {**result, **provenance}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail="Operation indisponible ; reessayer plus tard.") from None
