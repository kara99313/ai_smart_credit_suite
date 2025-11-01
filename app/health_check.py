# app/health_check.py
from __future__ import annotations
import os
import requests
import streamlit as st

st.title("🩺 Health Check")

def get_secret(k: str, default: str = "") -> str:
    """Lit d'abord st.secrets (Cloud), sinon ENV (local)."""
    try:
        return st.secrets.get(k, default)
    except Exception:
        return os.getenv(k, default)

LLM_PROVIDER = (get_secret("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "")) or "").lower()
GROQ_MODEL   = get_secret("GROQ_MODEL",   os.getenv("GROQ_MODEL", ""))
GROQ_API_KEY = get_secret("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
OPENAI_KEY   = get_secret("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
BACKEND_URL  = get_secret("BACKEND_URL",  os.getenv("BACKEND_URL", ""))
OLLAMA_URL   = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

c1, c2, c3, c4 = st.columns(4)
c1.metric("LLM_PROVIDER", LLM_PROVIDER or "—")
c2.metric("LLM_MODEL", GROQ_MODEL or "—")
c3.metric("BACKEND_URL", BACKEND_URL or "—")
c4.metric("OLLAMA", OLLAMA_URL or "—")
st.write(
    "**Groq key ?**", "✅" if bool(GROQ_API_KEY) else "❌",
    " • **OpenAI key ?**", "✅" if bool(OPENAI_KEY) else "❌"
)

# 1) LLM loader (via utilitaire)
try:
    from utils.llm_providers import load_llm
    _ = load_llm()
    st.success("LLM OK (load_llm)")
except Exception as e:
    st.error(f"LLM KO: {e}")

# 2) Backend /health (optionnel)
if BACKEND_URL:
    try:
        r = requests.get(BACKEND_URL.rstrip("/") + "/health", timeout=5)
        st.info(f"/health → {r.status_code} {r.text[:160]}")
    except Exception as e:
        st.error(f"Backend KO: {e}")
else:
    st.warning("BACKEND_URL non défini (OK si tu n'utilises pas /api/predict).")

# 3) RAG (index documents)
try:
    from utils.rag_utils import build_or_load_faiss_docs
    vs, path = build_or_load_faiss_docs(rebuild=False)
    if vs:
        st.success(f"RAG OK : {path}")
    else:
        st.warning("RAG: index absent. Ajoute des fichiers dans docs_rag/ puis recharge.")
except Exception as e:
    st.error(f"RAG KO: {e}")
