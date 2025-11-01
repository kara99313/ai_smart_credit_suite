# utils/settings.py 
from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml  # PyYAML est déjà dans requirements

# ============================
# Dossiers et fichiers
# ============================
ROOT_DIR = Path(os.getcwd())
DATA_DIR = ROOT_DIR / "data"
SETTINGS_PATH = DATA_DIR / "app_settings.json"
CONFIG_PATH = ROOT_DIR / "config.yaml"

# ============================
# Helper secrets/ENV (Cloud ou local)
# ============================
def _get_secret(key: str, default: str = "") -> str:
    """
    Lecture sécurisée : st.secrets (Streamlit Cloud) ou os.getenv (local).
    """
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return os.getenv(key, default)

# ============================
# Défauts (côté code)
# ============================
@dataclass
class AppSettings:
    # --- App ---
    app_name: str = "AI Smart Credit Suite"
    default_language: str = "fr"             # fr | en | bi
    theme: str = "auto"                      # auto | light | dark
    logo_path: str = "assets/logo.png"

    # --- Frontend / Backend ---
    ui_port: int = 8501
    api_base_url: str = "http://127.0.0.1:18000"

    # --- Modèle / décision ---
    decision_threshold: float = 0.10
    rating_scale: str = "standard"
    model_path: str = "model/logistic_pipeline_best.pkl"

    # --- LLM / Chatbots ---
    llm_provider: str = _get_secret("LLM_PROVIDER", "groq")  # groq | ollama | openai
    llm_model: str = (
        _get_secret("GROQ_MODEL", "")
        or _get_secret("OPENAI_MODEL", "")
        or _get_secret("OLLAMA_MODEL", "")
        or "llama-3.1-8b-instant"
    )
    llm_temperature: float = 0.2
    llm_max_tokens: int = 768
    llm_timeout: int = 60

    # --- Clés API (jamais persistées dans app_settings.json) ---
    groq_api_key: str = _get_secret("GROQ_API_KEY", "")
    openai_api_key: str = _get_secret("OPENAI_API_KEY", "")
    serpapi_key: str = _get_secret("SERPAPI_KEY", "")  # ← pour app/integrations.py

    # --- RAG ---
    rag_vector_store: str = "faiss"
    rag_docs_dir: str = "docs_rag"
    rag_hybrid_enabled: bool = True
    rag_top_k: int = 3
    rag_embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Web Search ---
    enable_web_search: bool = False
    web_search_max: int = 6
    allowed_domains: list[str] = field(default_factory=list)

    # --- Rapports ---
    reports_dir: str = "reports"
    report_primary_color: str = "#0F766E"
    report_footer_text: str = "Sous réserve des politiques internes de risque & conformité."
    report_author: str = "Credit Risk Analytics"

    # --- Observabilité ---
    kpi_window_days: int = 30
    enable_usage_metrics: bool = True

# ============================
# Utilitaires
# ============================
def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def _deep_update(base: dict, patch: Mapping[str, Any]) -> dict:
    """Fusion profonde (dict imbriqués)."""
    for k, v in (patch or {}).items():
        if isinstance(v, Mapping) and isinstance(base.get(k), Mapping):
            base[k] = _deep_update(dict(base[k]), v)
        else:
            base[k] = v
    return base

def _load_config_yaml() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # Aplatir les clés utiles vers le format AppSettings
        flat: dict[str, Any] = {}

        # app
        app = data.get("app", {})
        flat["app_name"] = app.get("name") or flat.get("app_name")
        flat["default_language"] = app.get("language_default") or flat.get("default_language")
        flat["theme"] = app.get("theme") or flat.get("theme")
        flat["logo_path"] = app.get("logo_path") or flat.get("logo_path")

        # frontend
        fe = data.get("frontend", {})
        if "port" in fe: flat["ui_port"] = fe["port"]

        # backend
        be = data.get("backend", {})
        if be.get("base_url"):
            flat["api_base_url"] = be["base_url"]
        else:
            host = be.get("host", "127.0.0.1")
            port = be.get("port", 18000)
            flat["api_base_url"] = f"http://{host}:{port}"

        # llm
        llm = data.get("llm", {})
        flat["llm_provider"] = llm.get("provider") or flat.get("llm_provider")
        flat["llm_model"] = llm.get("model") or flat.get("llm_model")
        if "temperature" in llm: flat["llm_temperature"] = float(llm["temperature"])
        if "max_tokens" in llm: flat["llm_max_tokens"] = int(llm["max_tokens"])
        if "timeout" in llm: flat["llm_timeout"] = int(llm["timeout"])

        # rag
        rag = data.get("rag", {})
        flat["rag_vector_store"] = rag.get("vector_store", "faiss")
        flat["rag_docs_dir"] = rag.get("docs_dir", "docs_rag")
        flat["rag_hybrid_enabled"] = bool(rag.get("hybrid_enabled", True))
        flat["rag_top_k"] = int(rag.get("top_k", 3))
        flat["rag_embeddings_model"] = rag.get("embeddings_model", "sentence-transformers/all-MiniLM-L6-v2")

        # reports
        rep = data.get("reports", {})
        flat["reports_dir"] = rep.get("output_dir", "reports")
        flat["report_primary_color"] = rep.get("primary_color", "#0F766E")
        flat["report_footer_text"] = rep.get("footer_text", "Sous réserve des politiques internes de risque & conformité.")

        # model
        mdl = data.get("model", {})
        flat["model_path"] = mdl.get("path", "model/logistic_pipeline_best.pkl")
        if "decision_threshold" in mdl:
            flat["decision_threshold"] = float(mdl["decision_threshold"])
        flat["rating_scale"] = mdl.get("rating_scale", "standard")

        return flat
    except Exception:
        return {}

def _load_app_settings_json() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

# ============================
# API publique
# ============================
def get_settings() -> AppSettings:
    """
    Priorité de fusion (du plus faible au plus fort) :
      1) Valeurs par défaut (AppSettings)
      2) config.yaml (réglages globaux)
      3) data/app_settings.json (préférences sauvegardées par l’utilisateur)
      4) Secrets .env/Streamlit (déjà injectés via _get_secret)
    """
    _ensure_data_dir()

    base = asdict(AppSettings())           # 1
    cfg_yaml = _load_config_yaml()         # 2
    base = _deep_update(base, cfg_yaml)

    file_obj = _load_app_settings_json()   # 3
    if isinstance(file_obj, dict):
        base = _deep_update(base, file_obj)

    return AppSettings(**base)

def save_settings(cfg: AppSettings) -> None:
    """
    Sauvegarde les paramètres (hors secrets API).
    """
    _ensure_data_dir()
    payload: Dict[str, Any] = asdict(cfg)
    # On ne sauvegarde pas les clés API dans le fichier
    for secret_key in ["groq_api_key", "openai_api_key", "serpapi_key"]:
        payload.pop(secret_key, None)
    tmp = SETTINGS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(SETTINGS_PATH)

def update_settings(patch: Dict[str, Any]) -> AppSettings:
    """Met à jour certains paramètres et les sauvegarde."""
    cur = get_settings()
    base = asdict(cur)
    base = _deep_update(base, patch or {})
    new_cfg = AppSettings(**base)
    save_settings(new_cfg)
    return new_cfg

def safe_get_settings() -> Dict[str, Any]:
    """Retourne les paramètres d'application sous forme de dict — jamais None, jamais exception."""
    try:
        return asdict(get_settings())
    except Exception:
        return asdict(AppSettings())
