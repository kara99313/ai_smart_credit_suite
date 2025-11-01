# utils/i18n.py
from __future__ import annotations
import os
import yaml
import streamlit as st
from pathlib import Path

# Emplacements des fichiers de traduction
BASE_DIR = Path(__file__).resolve().parent.parent
TRANS_DIR = BASE_DIR / "assets" / "translations"

# Session keys
_LANG_KEY = "APP_LANG"   # 'fr', 'en', 'bi'
_TMODE    = "APP_MODE"   # alias interne si besoin (optionnel)

def _load_yaml(lang: str) -> dict:
    path = TRANS_DIR / f"{lang}.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

@st.cache_resource(show_spinner=False)
def _load_all():
    # charge FR/EN une fois
    return {
        "fr": _load_yaml("fr"),
        "en": _load_yaml("en"),
    }

def get_mode() -> str:
    return st.session_state.get(_LANG_KEY, os.getenv("APP_LANG", "fr"))

def set_mode(lang: str):
    st.session_state[_LANG_KEY] = lang

def init_i18n():
    # priorité URL ?lang=en|fr|bi
    try:
        q = st.query_params.get("lang", None)
    except Exception:
        q = None
    if q in ("fr", "en", "bi"):
        set_mode(q)
    else:
        # défaut (si jamais)
        st.session_state.setdefault(_LANG_KEY, os.getenv("APP_LANG", "fr"))

def t(key: str, **kwargs) -> str:
    """
    Récupère la traduction pour 'key'.
    - 'bi' => FR + (EN) pour le même texte
    - Si clé manquante => retourne la clé elle-même
    """
    lang = get_mode()
    bundle = _load_all()
    def _lookup(langcode: str) -> str:
        cur = bundle.get(langcode, {})
        # descente par 'a.b.c'
        node = cur
        for part in key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return key  # clé manquante
        if isinstance(node, str):
            return node.format(**kwargs) if kwargs else node
        return key

    if lang == "bi":
        fr = _lookup("fr")
        en = _lookup("en")
        # format bilingue compact
        if fr == en:   # même texte
            return fr
        return f"{fr} / ({en})"
    else:
        return _lookup(lang)
