# utils/llm_status.py
from __future__ import annotations
import os, streamlit as st

def _get_secret(k: str, default: str = "") -> str:
    try:
        return st.secrets.get(k, default)
    except Exception:
        return os.getenv(k, default)

def llm_status_badge():
    """Badge d'état LLM (Groq/OpenAI/Ollama) basé sur utils.llm_providers.load_llm()."""
    try:
        from utils.llm_providers import load_llm
        llm = load_llm()
        provider = (_get_secret("LLM_PROVIDER", os.getenv("LLM_PROVIDER","")) or "auto").lower()
        # Compat LangChain: certains connecteurs exposent model, d'autres model_name
        model = getattr(llm, "model", None) or getattr(llm, "model_name", None) \
                or _get_secret("GROQ_MODEL", os.getenv("GROQ_MODEL","")) or "—"
        st.caption(f"✅ LLM prêt ({provider}) — modèle : {model}")
    except Exception as e:
        msg = str(e).split("\n")[0][:180]
        st.caption(f"❌ LLM KO — {msg}")
