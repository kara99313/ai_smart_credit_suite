# utils/scoring_runtime.py
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("model/logistic_pipeline_best.pkl")

FEATURES = [
    "DTIRatio",
    "TrustScorePsychometric",
    "HouseholdSize",
    "NumCreditLines",
    "Income",
    "CommunityGroupMember",
    "HasMortgage",
    "MonthsEmployed",
    "HasSocialAid",
    "MobileMoneyTransactions",
    "Age",
    "InterestRate",
    "LoanTerm",
    "LoanAmount",
    "InformalIncome",
]

CATEG = ["CommunityGroupMember", "HasMortgage", "HasSocialAid"]

def _to_bool(v):
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1","true","vrai","yes","oui"}: return True
        if s in {"0","false","faux","no","non"}: return False
    return False

def _align(df: pd.DataFrame) -> pd.DataFrame:
    for col in FEATURES:
        if col not in df.columns:
            df[col] = False if col in CATEG else 0
    df = df[FEATURES]
    for c in CATEG:
        df[c] = df[c].map(_to_bool)
    return df

def _sigmoid(z: float) -> float:
    z = float(np.clip(z, -50, 50))
    return 1.0 / (1.0 + np.exp(-z))

def _pd_to_rating(pd_val: float) -> str:
    if pd_val < 0.01: return "AAA"
    if pd_val < 0.02: return "AA"
    if pd_val < 0.03: return "A"
    if pd_val < 0.05: return "BBB"
    if pd_val < 0.08: return "BB"
    if pd_val < 0.12: return "B"
    if pd_val < 0.20: return "CCC"
    if pd_val < 0.35: return "CC"
    if pd_val < 0.50: return "C"
    return "D"

def _rating_to_risk(r: str) -> str:
    m = {
        "AAA":"Très faible","AA":"Très faible","A":"Faible",
        "BBB":"Modéré","BB":"Élevé","B":"Très élevé",
        "CCC":"Critique","CC":"Critique","C":"Critique","D":"Défaut imminent"
    }
    return m.get(r, "Inconnu")

def _decision(pd_val: float):
    if pd_val < 0.08: return "✅ Accord"
    if pd_val < 0.15: return "🟠 À analyser"
    return "❌ Refus"

def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def score_dict(payload: dict) -> dict:
    pipe = _load()
    df = pd.DataFrame([payload])
    df = _align(df)

    # priorité proba -> decision_function -> predict
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(df)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            pd_val = float(proba[0,1])
        else:
            pd_val = float(np.ravel(proba)[0])
    elif hasattr(pipe, "decision_function"):
        z = float(np.ravel(pipe.decision_function(df))[0])
        pd_val = _sigmoid(z)
    else:
        y = int(np.ravel(pipe.predict(df))[0])
        pd_val = 0.8 if y == 1 else 0.2

    score_1000 = int(round((1.0 - pd_val) * 1000))
    rating = _pd_to_rating(pd_val)
    risk = _rating_to_risk(rating)
    dec = _decision(pd_val)
    return {
        "pd": pd_val,
        "score_1000": score_1000,
        "rating": rating,
        "risk_level": risk,
        "decision": dec,
    }
