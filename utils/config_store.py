# utils/config_store.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path("data")
CFG_PATH = DATA_DIR / "app_config.json"

_DEFAULTS: Dict[str, Any] = {
    "risk_threshold": 0.10,
    "web_search_enabled": True,
    "telemetry_opt_in": False,
    "api_enabled": False,
    "api_port": 8010,
    "allow_public_web_sources": True,
    "allowed_domains": [],
}

def load_config() -> Dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CFG_PATH.exists():
        try:
            return {**_DEFAULTS, **json.loads(CFG_PATH.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return _DEFAULTS.copy()

def save_config(cfg: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with CFG_PATH.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
