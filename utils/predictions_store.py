# utils/predictions_store.py
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

PRED_PATH = Path("data/predictions_log.csv")

class PredictionsStore:
    def __init__(self, path: str | Path = PRED_PATH):
        self.path = Path(path)
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.path, low_memory=False)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        for c in ["prob_default","threshold","score","pred_label"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def last_by_client(self, client_id: str) -> dict:
        if self.df.empty or "client_id" not in self.df.columns:
            return {}
        sub = self.df[self.df["client_id"].astype(str) == str(client_id)].sort_values("timestamp")
        if sub.empty:
            return {}
        row = sub.iloc[-1].to_dict()
        try:
            row["inputs_dict"] = json.loads(row.get("inputs_json") or "{}")
        except Exception:
            row["inputs_dict"] = {}
        return row

    def list_clients(self, limit=50) -> list[str]:
        if self.df.empty or "client_id" not in self.df.columns:
            return []
        ids = self.df["client_id"].astype(str).unique().tolist()
        return ids[:limit]
