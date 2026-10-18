# utils/csv_utils.py
from __future__ import annotations
import os, csv, json
from typing import List
import pandas as pd

LOG_PATH_DEFAULT = os.path.join("data", "predictions_log.csv")

REQUIRED_COLS: List[str] = [
    "timestamp","client_id","prob_default","pred_label",
    "threshold","inputs_json","model_tag","rating","decision"
]

def read_logs_robust(path: str = LOG_PATH_DEFAULT) -> pd.DataFrame:
    """Read every supported historical record without rewriting or skipping rows."""
    from utils.prediction_logs import read_prediction_records
    records = read_prediction_records(path)
    df = pd.DataFrame(records)
    for name in REQUIRED_COLS:
        if name not in df:
            df[name] = None
    for name in ("prob_default", "threshold", "pred_label"):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.sort_values("timestamp", ascending=False, kind="stable")


def sanitize_predictions_log(path: str = LOG_PATH_DEFAULT) -> str:
    """Compatibility validation only: historical records are never rewritten here."""
    read_logs_robust(path)
    return path


def read_any_csv_robust(path: str) -> pd.DataFrame:
    """
    Helper générique pour d'autres CSV (ex: agent_usage.csv).
    """
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip")
