# utils/predictor.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path("model")
DATA_DIR  = Path("data")
MODEL_PATH = MODEL_DIR / "logistic_pipeline_best.pkl"
LOG_PATH   = DATA_DIR / "predictions_log.csv"

ALL_FEATURES_ORDER = [
    "DTIRatio","TrustScorePsychometric","HouseholdSize","NumCreditLines","Income",
    "CommunityGroupMember","HasMortgage","MonthsEmployed","HasSocialAid","MobileMoneyTransactions",
    "Age","InterestRate","LoanTerm","LoanAmount","InformalIncome",
]
CATEGORICAL_VARS = ["CommunityGroupMember","HasMortgage","HasSocialAid"]

def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def pd_to_rating(pd_val: float) -> str:
    v = float(pd_val)
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

def rating_to_risk_level(r: str) -> str:
    lut = {
        "AAA":"Très faible","AA":"Très faible","A":"Faible","BBB":"Modéré",
        "BB":"Élevé","B":"Très élevé","CCC":"Critique","CC":"Critique","C":"Critique","D":"Défaut imminent",
    }
    return lut.get(r, "Inconnu")

def decision_policy(pd_val: float) -> tuple[str,str]:
    if pd_val < 0.08:  return ("✅ Accord","Accord sous réserve KYC/AML et contrôles internes.")
    if pd_val < 0.15:  return ("🟠 À analyser","Garanties / ajuster montant / pièces complémentaires.")
    return ("❌ Refus","Risque trop élevé. Alternatives: montant plus faible, co-emprunteur...")

# ----------------------------------------------------------------

def load_pipeline():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def _features_from_pipeline(pipe) -> list[str] | None:
    try:
        if hasattr(pipe, "feature_names_in_"):
            arr = list(pipe.feature_names_in_)
            if arr: return [str(x) for x in arr]
    except Exception:
        pass
    # fouille d’un ColumnTransformer
    try:
        from sklearn.compose import ColumnTransformer
        steps = [est for name, est in getattr(pipe, "steps", [])] or [pipe]
        for est in steps:
            if isinstance(est, ColumnTransformer):
                cols = []
                for _, _, c in est.transformers_:
                    if isinstance(c,(list,tuple)): cols += list(c)
                if cols:
                    seen, ordered = set(), []
                    for c in cols:
                        c = str(c)
                        if c not in seen:
                            seen.add(c); ordered.append(c)
                    return ordered
    except Exception:
        pass
    return None

def get_expected_features(pipe) -> list[str]:
    feats = _features_from_pipeline(pipe)
    return feats if feats else ALL_FEATURES_ORDER.copy()

def _coerce_bool_to_category(val, categories: list) -> object:
    """
    Map True/False/1/0 vers la catégorie apprise par l’encoder.
    On couvre les cas: ['Oui','Non'], ['Yes','No'], ['True','False'], ['1','0'], [True, False]
    """
    v = val
    if isinstance(v, (np.bool_, bool, int)):
        b = bool(int(v))
        cats = [str(c) for c in categories]
        if "Oui" in cats or "Non" in cats:
            return "Oui" if b else "Non"
        if "Yes" in cats or "No" in cats:
            return "Yes" if b else "No"
        if "True" in cats or "False" in cats:
            return "True" if b else "False"
        if "1" in cats or "0" in cats:
            return "1" if b else "0"
        # sinon si l’encoder a vraiment appris des bool:
        if True in categories or False in categories:
            return b
        # fallback: chaîne
        return "True" if b else "False"
    # si string déjà fournie
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"oui","yes","true","1"}:  return _coerce_bool_to_category(True, categories)
        if s in {"non","no","false","0"}:   return _coerce_bool_to_category(False, categories)
    return v

def prepare_input_df(pipe, raw: dict, features: list[str]) -> pd.DataFrame:
    """
    Aligne les features et convertit les 3 colonnes catégorielles
    vers ce que le pipeline attend (en lisant les categories_ de l’OneHotEncoder).
    """
    x = dict(raw or {})
    # valeurs manquantes -> 0/False
    for f in features:
        if f not in x:
            x[f] = False if f in CATEGORICAL_VARS else 0

    df = pd.DataFrame([x], dtype=object)[features]

    # essai de coercition en s’appuyant sur le ColumnTransformer
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder
        steps = [est for name, est in getattr(pipe, "steps", [])] or [pipe]
        for est in steps:
            if isinstance(est, ColumnTransformer):
                for name, trf, cols in est.transformers_:
                    if isinstance(trf, OneHotEncoder):
                        cats = trf.categories_
                        for i, col in enumerate(cols):
                            if col in df.columns:
                                df[col] = [_coerce_bool_to_category(df[col].iloc[0], list(cats[i]))]
    except Exception:
        # silencieux: si rien trouvé on laisse tel quel
        pass

    # conversions numériques de base
    for f in df.columns:
        if f not in CATEGORICAL_VARS:
            try:
                df[f] = pd.to_numeric(df[f])
            except Exception:
                df[f] = 0
    return df

def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -50, 50))
    return 1.0 / (1.0 + np.exp(-z))

def compute_pd(pipe, X: pd.DataFrame) -> float:
    # priorité à predict_proba
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)
        if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        if np.ndim(proba) == 1:
            return float(np.ravel(proba)[0])
    # sinon decision_function
    if hasattr(pipe, "decision_function"):
        d = float(np.ravel(pipe.decision_function(X))[0])
        return float(_sigmoid(d))
    # fallback pauvre
    if hasattr(pipe, "predict"):
        y = int(np.ravel(pipe.predict(X))[0])
        return 0.8 if y == 1 else 0.2
    return 0.5

def predict_from_dict(inputs: dict, threshold: float = 0.10) -> dict:
    """
    Entrée: dict des 15 variables (clés = noms features)
    Sortie: dict complet (pd, score, rating, risk, décision, X_aligned)
    """
    pipe = load_pipeline()
    feats = get_expected_features(pipe)
    X = prepare_input_df(pipe, inputs, feats)
    pd_val = compute_pd(pipe, X)
    score_1000 = int(round((1 - pd_val) * 1000))
    rating = pd_to_rating(pd_val)
    risk = rating_to_risk_level(rating)
    decision, note = decision_policy(pd_val)

    # log minimal
    try:
        _ensure_data_dir()
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "client_id": inputs.get("client_id","N/A"),
            "prob_default": pd_val,
            "pred_label": int(pd_val >= threshold),
            "threshold": threshold,
            "inputs_json": json.dumps(inputs, ensure_ascii=False),
            "model_tag": "pipeline_v1"
        }
        pd.DataFrame([row]).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False, encoding="utf-8")
    except Exception:
        pass

    return {
        "pd": pd_val,
        "score_1000": score_1000,
        "rating": rating,
        "risk_level": risk,
        "decision": decision,
        "decision_note": note,
        "features": feats,
        "X_aligned": X.to_dict(orient="records")[0],
    }
