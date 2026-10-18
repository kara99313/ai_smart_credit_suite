"""Configuration badge only; service availability requires an explicit real call."""
import streamlit as st
from utils.llm_runtime import select_provider,get_setting,PROVIDERS


def llm_status_badge():
    try:
        provider=select_provider()
        key_name,model_name,default,_,_=PROVIDERS[provider]
        configured=not key_name or bool(get_setting(key_name).strip())
        st.caption(f"LLM {provider} : {'Configure seulement' if configured else 'Indisponible (cle absente)'} ; reponse fonctionnelle non testee.")
    except Exception:
        st.caption('LLM : configuration invalide ; fonctionnement non teste.')
