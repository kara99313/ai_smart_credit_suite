# utils/audit.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

AUDIT_PATH = Path("data/audit_log.csv")
AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)

def append_audit(event: str, actor: str = "admin", details: dict | None = None) -> None:
    row = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "actor": actor,
        "event": event,
        "details": json.dumps(details or {}, ensure_ascii=False),
    }
    exists = AUDIT_PATH.exists()
    with open(AUDIT_PATH, "a", encoding="utf-8", newline="") as f:
        pd.DataFrame([row]).to_csv(f, header=not exists, index=False)
