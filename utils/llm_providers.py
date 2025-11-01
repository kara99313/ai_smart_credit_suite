# utils/llm_providers.py
from __future__ import annotations
import os
from typing import Optional

import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

# ------------ Helper secrets/ENV (Cloud puis local) ------------
def _get_secret(k: str, default: str = "") -> str:
    try:
        import streamlit as st  # dispo en local et sur Cloud
        return st.secrets.get(k, default)
    except Exception:
        return os.getenv(k, default)

# ------------ Détection des connecteurs disponibles ------------
_HAS_GROQ = True
try:
    from langchain_groq import ChatGroq  # compatible avec langchain-core 0.3.x
except Exception:
    _HAS_GROQ = False

_HAS_OPENAI = True
try:
    from langchain_openai import ChatOpenAI
except Exception:
    _HAS_OPENAI = False

_HAS_OLLAMA = True
try:
    from langchain_ollama import ChatOllama
except Exception:
    _HAS_OLLAMA = False


# ------------ Pings légers pour éviter des blocages ------------
def _ping_ollama(base_url: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=timeout)
        return r.ok
    except Exception:
        return False


# ------------ Constructeurs sûrs par provider ------------
def _groq_or_raise() -> BaseChatModel:
    if not _HAS_GROQ:
        raise RuntimeError(
            "Connecteur 'langchain-groq' manquant. Installe : pip install -U 'langchain-groq<1.0.0'"
        )
    key = (_get_secret("GROQ_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY manquante (LLM_PROVIDER=groq). Ajoute ta clé 'gsk_...'."
        )
    model = _get_secret("GROQ_MODEL", "llama-3.1-8b-instant")

    # ✅ Bornes pour palier gratuit : éviter 413 / TPM exceeded
    try:
        return ChatGroq(
            model=model,
            api_key=key,
            temperature=0.2,
            max_tokens=768,   # borne la longueur des réponses
            timeout=60,
        )
    except Exception as e:
        raise RuntimeError(
            f"Erreur Groq : {e}\n"
            "Réduis la taille envoyée (historique + contexte) si besoin."
        )


def _openai_or_raise() -> BaseChatModel:
    if not _HAS_OPENAI:
        raise RuntimeError(
            "Connecteur 'langchain-openai' manquant. Installe : pip install -U langchain-openai"
        )
    key = (_get_secret("OPENAI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY manquante (LLM_PROVIDER=openai).")
    model = _get_secret("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, api_key=key, temperature=0.2)


def _ollama_or_raise() -> BaseChatModel:
    if not _HAS_OLLAMA:
        raise RuntimeError(
            "Connecteur 'langchain-ollama' manquant. Installe : pip install -U langchain-ollama"
        )
    base = _get_secret("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = _get_secret("OLLAMA_MODEL", "llama3.2:3b")

    if not _ping_ollama(base):
        raise RuntimeError(
            f"Ollama injoignable sur {base}. Lance 'ollama serve' (et 'ollama pull {model}' la 1ère fois), "
            "ou utilise LLM_PROVIDER=groq/openai avec la clé correspondante."
        )

    # num_predict petit pour CPU / latence locale
    return ChatOllama(model=model, base_url=base, temperature=0.2, num_predict=128, keep_alive="5m")


# ------------ Sélection/fallback du LLM ------------
def load_llm() -> BaseChatModel:
    """
    Stratégie robuste :
    1) Respecter LLM_PROVIDER si défini : 'groq' | 'openai' | 'ollama'
    2) Sinon : tenter groq (si GROQ_API_KEY) -> openai (si OPENAI_API_KEY) -> ollama (si joignable)
    """
    provider = (_get_secret("LLM_PROVIDER", os.getenv("LLM_PROVIDER", "")) or "").strip().lower()

    if provider == "groq":
        return _groq_or_raise()
    if provider == "openai":
        return _openai_or_raise()
    if provider == "ollama":
        return _ollama_or_raise()

    # Mode auto (si non fixé)
    groq_key = (_get_secret("GROQ_API_KEY", "")).strip()
    if groq_key and _HAS_GROQ:
        try:
            return _groq_or_raise()
        except Exception:
            pass

    oai_key = (_get_secret("OPENAI_API_KEY", "")).strip()
    if oai_key and _HAS_OPENAI:
        try:
            return _openai_or_raise()
        except Exception:
            pass

    if _HAS_OLLAMA:
        try:
            return _ollama_or_raise()
        except Exception as e_ollama:
            raise RuntimeError(
                "LLM indisponible en mode auto.\n\n"
                f"Détail Ollama: {e_ollama}\n"
                "Solutions: fournis GROQ_API_KEY (LLM_PROVIDER=groq) ou OPENAI_API_KEY (LLM_PROVIDER=openai), "
                "ou lance 'ollama serve', puis relance l'application."
            )

    raise RuntimeError(
        "LLM indisponible. Installe un connecteur (groq/openai/ollama) et configure la clé/API."
    )


# ------------ Utilitaire simple de chat ------------
def simple_chat(prompt: str, system: Optional[str] = None) -> str:
    llm = load_llm()
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    out = llm.invoke(msgs)
    return out.content if hasattr(out, "content") else str(out)
