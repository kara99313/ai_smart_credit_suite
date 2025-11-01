# server.py
from __future__ import annotations
import json, os, sys, time
from typing import Literal, Optional, Dict, Any, List, Tuple, Callable
from pathlib import Path

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
from pydantic import BaseModel, Field

# ===== Chemins / modèle =====
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model" / "logistic_pipeline_best.pkl")))

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
    _pipe = joblib.load(MODEL_PATH)
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
    if hasattr(_pipe, "predict_proba"):
        proba = _pipe.predict_proba(df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        if proba.ndim == 1:
            return float(proba[0])
    if hasattr(_pipe, "decision_function"):
        s = float(np.array(_pipe.decision_function(df)).ravel()[0])
        return float(_sigmoid(s))
    if hasattr(_pipe, "predict"):
        y = int(np.array(_pipe.predict(df)).ravel()[0])
        return 0.8 if y == 1 else 0.2
    return 0.5

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
class PredictRequest(BaseModel):
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
    threshold: float = Field(0.10, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    pd: float
    score_1000: int
    rating: str
    decision: Literal["ACCEPT", "REVIEW/REJECT"]
    threshold: float
    used_features: List[str]

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

class ExplainPayload(BaseModel):
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

# ===== FastAPI =====
app = FastAPI(title="AI Smart Credit Suite — Backend", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ===== Warm-up =====
def _warmup_model() -> None:
    try:
        dummy = {
            "DTIRatio": 0.3, "TrustScorePsychometric": 0.6, "HouseholdSize": 3, "NumCreditLines": 2,
            "Income": 100000, "CommunityGroupMember": False, "HasMortgage": False, "MonthsEmployed": 12,
            "HasSocialAid": False, "MobileMoneyTransactions": 10, "Age": 30, "InterestRate": 10.0,
            "LoanTerm": 12, "LoanAmount": 100000, "InformalIncome": 0.0
        }
        base = _align_df(dummy)
        for mode in ("str", "bool", "int"):
            df = _align_categoricals(base, mode)
            _ = _compute_pd(df)
            break
    except Exception as e:
        print(f"[WARMUP] warning: {e}")

@app.on_event("startup")
def _on_startup():
    t0 = time.perf_counter()
    _warmup_model()
    print(f"[WARMUP] model preheated in {time.perf_counter() - t0:.2f}s")

# ===== Routes =====
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_path": str(MODEL_PATH), "features": FEATURES}

@app.get("/api/predict_ping")
def predict_ping():
    return {"ok": True, "msg": "predict service alive"}

@app.post("/api/predict_echo")
def predict_echo(req: PredictRequest):
    df = _align_df(req.model_dump())
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
        base_df = _align_df(raw)
        last_err: Optional[Exception] = None
        for mode in ("str", "bool", "int"):
            try:
                df = _align_categoricals(base_df, mode)
                pd_val = _compute_pd(df)
                score = int(round((1 - pd_val) * 1000))
                rating = _rating_from_pd(pd_val)
                decision = _decision_from_pd(pd_val, req.threshold)
                return PredictResponse(
                    pd=pd_val, score_1000=score, rating=rating, decision=decision,
                    threshold=req.threshold, used_features=list(df.columns),
                )
            except (ValueError, TypeError) as e:
                last_err = e
                continue
        raise RuntimeError(f"All category coercion modes failed: {last_err}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict failed: {type(e).__name__}: {e}")

# ===== Outils Agent IA =====
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

_predict_tool, _rag_tool, _web_tool, _report_tool, _now_tool = _import_agent_tools()

@app.post("/tools/predict_credit")
def tools_predict_credit(body: PredictToolBody):
    try:
        txt = _predict_tool(json.dumps(body.data, ensure_ascii=False))
        return {"ok": True, "result": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"predict_credit error: {e}")

@app.post("/tools/rag_search")
def tools_rag_search(body: RAGPayload):
    try:
        txt = _rag_tool(body.query, mode=body.mode or "docs", k=body.k or 4)
        return {"ok": True, "result": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"rag_search error: {e}")

@app.post("/tools/web_search")
def tools_web_search(body: WebSearchPayload):
    try:
        txt = _web_tool(body.query, k=body.k or 3, fetch=bool(body.fetch))
        return {"ok": True, "result": txt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"web_search error: {e}")

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
        html = _report_tool(json.dumps(ctx, ensure_ascii=False))
        return {"ok": True, "html": html}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"generate_report error: {e}")

@app.get("/tools/now")
def tools_now(fmt: str = "%Y-%m-%d %H:%M:%S"):
    try:
        return {"ok": True, "now": _now_tool(fmt)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"now error: {e}")

# ===== Explain (SHAP + fallback) =====
def _extract_estimator_and_features():
    est = _pipe
    feats: List[str] = []
    if hasattr(_pipe, "feature_names_in_"):
        feats = [str(c) for c in getattr(_pipe, "feature_names_in_")]
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(_pipe, Pipeline):
            est = _pipe.steps[-1][1]
    except Exception:
        pass
    return est, feats or FEATURES

def _try_shap_values(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    try:
        import shap  # type: ignore
        explainer = shap.Explainer(_pipe)   # auto: Tree/Linear/Kernel
        sv = explainer(df.iloc[[0]])
        vals = None
        if hasattr(sv, "values"):
            arr = sv.values
            if isinstance(arr, list):   # multi-classes -> dernière classe
                arr = arr[-1]
            arr = getattr(arr, "data", arr)
            try:
                arr = arr.reshape(-1)
            except Exception:
                pass
            vals = [float(x) for x in list(arr)]
        if vals is None:
            return None
        _, feats = _extract_estimator_and_features()
        out = {}
        for i, name in enumerate(feats[: len(vals)]):
            out[name] = float(vals[i])
        return out
    except Exception:
        return None

def _simple_importances(df: pd.DataFrame) -> Dict[str, float]:
    est, feats = _extract_estimator_and_features()
    imp: Dict[str, float] = {}
    coef = getattr(est, "coef_", None)
    if coef is not None:
        arr = np.array(coef).ravel()
        for i, name in enumerate(feats[: len(arr)]):
            imp[name] = float(abs(arr[i]))
        if imp:
            return imp
    fimp = getattr(est, "feature_importances_", None)
    if fimp is not None:
        arr = np.array(fimp).ravel()
        for i, name in enumerate(feats[: len(arr)]):
            imp[name] = float(abs(arr[i]))
        if imp:
            return imp
    try:
        var = df.var(numeric_only=True).to_dict()
        imp = {k: float(abs(v) if v is not None else 0.0) for k, v in var.items()}
    except Exception:
        imp = {k: 1.0 for k in feats}
    return imp

@app.post("/tools/explain")
def tools_explain(body: ExplainPayload, top_k: int = Query(10, ge=1, le=30)):
    try:
        base_df = _align_df(body.model_dump())
        df = _align_categoricals(base_df, "bool")

        shap_imp = _try_shap_values(df)
        if shap_imp:
            items = sorted(shap_imp.items(), key=lambda kv: abs(kv[1]), reverse=True)[: int(top_k)]
            return {"ok": True, "method": "shap", "items": [
                {"feature": k, "contribution": float(v)} for k, v in items
            ]}

        imp = _simple_importances(df)
        if not imp:
            return {"ok": False, "reason": "no_importance_available", "items": []}
        items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[: int(top_k)]
        return {"ok": True, "method": "fallback_importance", "items": [
            {"feature": k, "importance": float(v)} for k, v in items
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"explain error: {e}")

# ===== Debug utiles =====
@app.get("/debug/imports")
def debug_imports():
    out = {}
    for m in ["numpy","pandas","sklearn","torch","transformers","tokenizers","sentence_transformers","faiss","langchain","shap"]:
        try:
            mod = __import__(m)
            ver = getattr(mod, "__version__", "n/a")
            out[m] = {"ok": True, "version": ver}
        except Exception as e:
            out[m] = {"ok": False, "error": str(e)}
    out["sys_executable"] = sys.executable
    return out

@app.get("/debug/dtypes")
def debug_dtypes(
    DTIRatio: float = 0.3,
    TrustScorePsychometric: float = 0.6,
    HouseholdSize: int = 3,
    NumCreditLines: int = 2,
    Income: float = 100000,
    CommunityGroupMember: bool = False,
    HasMortgage: bool = False,
    MonthsEmployed: int = 12,
    HasSocialAid: bool = False,
    MobileMoneyTransactions: int = 10,
    Age: int = 30,
    InterestRate: float = 10.0,
    LoanTerm: int = 12,
    LoanAmount: float = 100000,
    InformalIncome: float = 0.0,
):
    df = _align_df(locals())
    return {"dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()}, "row0": df.iloc[0].to_dict()}
