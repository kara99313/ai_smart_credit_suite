# utils/metrics.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

LOG_PATH = Path("data/predictions_log.csv")

def load_prediction_logs() -> pd.DataFrame | None:
    if not LOG_PATH.exists():
        return None
    try:
        df = pd.read_csv(LOG_PATH)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
        else:
            # fallback: create now column if missing
            df["timestamp"] = pd.Timestamp.utcnow()
        # Ensure basic columns exist
        for c in ["client_id","prob_default","pred_label","threshold","model_tag"]:
            if c not in df.columns:
                df[c] = np.nan
        df["prob_default"] = pd.to_numeric(df["prob_default"], errors="coerce").fillna(0.0)
        df["pred_label"]  = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(int)
        df["threshold"]   = pd.to_numeric(df["threshold"], errors="coerce").fillna(0.10)
        return df
    except Exception:
        return None

def rating_from_pd(pd_val: float) -> str:
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

def compute_core_kpis(df: pd.DataFrame) -> dict:
    n = len(df)
    appr = (df["pred_label"] == 0).mean() * 100 if n else 0.0
    pd_median = df["prob_default"].median() if n else 0.0
    score_mean = (1 - df["prob_default"]).mean() * 1000 if n else 0.0
    return {
        "n_predictions": int(n),
        "approval_rate": float(appr),
        "pd_median": float(pd_median),
        "score_mean": float(score_mean),
    }
