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
    """
    Lecture tolérante:
    - supporte anciens (7 colonnes) / nouveaux (9 colonnes)
    - engine='python' + on_bad_lines='skip' pour ignorer les lignes cassées
    - garde inputs_json avec virgules/quotes
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=REQUIRED_COLS)

    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception:
        # filet de sécurité si le header est lui-même altéré
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            rows = list(csv.reader(f))
        if not rows:
            return pd.DataFrame(columns=REQUIRED_COLS)
        header, data = rows[0], rows[1:]
        width = max((len(r) for r in data), default=len(header))
        header = (header + [f"col_{i}" for i in range(width-len(header))])[:width]
        df = pd.DataFrame(data, columns=header)

    # Colonnes manquantes → compléter
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("rating","decision") else None

    # Types
    if "prob_default" in df.columns:
        df["prob_default"] = pd.to_numeric(df["prob_default"], errors="coerce")
    if "threshold" in df.columns:
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    if "pred_label" in df.columns:
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce")

    # Tri
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", ascending=False)

    # Ordre cohérent
    cols = [c for c in REQUIRED_COLS if c in df.columns]
    return df[cols]

def sanitize_predictions_log(path: str = LOG_PATH_DEFAULT) -> str:
    """
    Réécrit le CSV de log de manière propre (quoting minimal).
    """
    df = read_logs_robust(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
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
