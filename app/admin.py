# app/admin.py
from __future__ import annotations
import os
from pathlib import Path
import re
import streamlit as st

# UTF-8 (sécurité)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# === Utils existants ===
try:
    from utils.settings import safe_get_settings
except Exception:
    class _Cfg:
        llm_provider = os.getenv("LLM_PROVIDER", "groq")
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        groq_model   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        openai_model   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        ollama_model    = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    def safe_get_settings():  # fallback simple
        return _Cfg()

from utils.llm_providers import simple_chat

ROOT = Path.cwd()
ENV_PATH = ROOT / ".env"

def _read_env() -> str:
    return ENV_PATH.read_text(encoding="utf-8", errors="ignore") if ENV_PATH.exists() else ""

def _write_env(content: str) -> None:
    ENV_PATH.write_text(content, encoding="utf-8")

def _set_env_var(content: str, key: str, value: str) -> str:
    """
    Remplace ou ajoute key=value (sans guillemets) de manière idempotente.
    Conserve les autres lignes. Supprime les doublons de la même clé.
    """
    lines = content.splitlines()
    pat = re.compile(rf"^\s*{re.escape(key)}\s*=")
    new_lines = []
    replaced = False
    for ln in lines:
        if pat.match(ln):
            if not replaced:
                new_lines.append(f"{key}={value}")
                replaced = True
            # si doublon: skip
        else:
            new_lines.append(ln)
    if not replaced:
        # ajoute à la fin
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append(f"{key}={value}")
        else:
            # évite double ligne vide
            new_lines[-1:] = [f"{key}={value}"] if new_lines else [f"{key}={value}"]
    # nettoie doublons éventuels restants
    seen = set()
    final_lines = []
    for ln in new_lines:
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*=", ln)
        if m:
            k = m.group(1)
            if k in seen:
                continue
            seen.add(k)
        final_lines.append(ln)
    return "\n".join(final_lines) + ("\n" if not final_lines or final_lines[-1] != "" else "")

def _mask(s: str, show_last: int = 4) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= show_last:
        return "*" * len(s)
    return "*" * (len(s) - show_last) + s[-show_last:]

def main():
    st.set_page_config(page_title="Administration", page_icon="⚙️", layout="wide")
    st.markdown("## ⚙️ Administration — Configuration LLM & Système")

    cfg = safe_get_settings()

    providers = ["groq", "openai", "ollama"]
    def safe_index(val: str, arr: list[str]) -> int:
        return arr.index(val) if val in arr else 0

    colA, colB = st.columns([2, 1])
    with colA:
        st.subheader("Fournisseur LLM")
        idx = safe_index(getattr(cfg, "llm_provider", "groq"), providers)
        provider = st.selectbox("Fournisseur LLM", providers, index=idx)

        # GROQ
        if provider == "groq":
            st.markdown("#### 🔑 GROQ")
            cur_key = os.getenv("GROQ_API_KEY", "")
            key_input = st.text_input(
                "GROQ_API_KEY (gsk_…)",
                value=cur_key if not cur_key else cur_key,  # on montre la vraie valeur localement
                type="password",
                help="Colle ta clé Groq (https://console.groq.com/keys)."
            )
            model = st.text_input("GROQ_MODEL", value=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))

        # OPENAI
        if provider == "openai":
            st.markdown("#### 🔑 OpenAI")
            cur_key = os.getenv("OPENAI_API_KEY", "")
            key_input = st.text_input(
                "OPENAI_API_KEY (sk-…)",
                value=cur_key if not cur_key else cur_key,
                type="password",
                help="Colle ta clé OpenAI."
            )
            model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

        # OLLAMA
        if provider == "ollama":
            st.markdown("#### 🖥️ Ollama (local)")
            base = st.text_input("OLLAMA_BASE_URL", value=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
            model = st.text_input("OLLAMA_MODEL", value=os.getenv("OLLAMA_MODEL", "llama3.2:3b"))
            key_input = ""  # pas de clé

        st.divider()
        if st.button("💾 Enregistrer dans .env", type="primary"):
            env_txt = _read_env()
            env_txt = _set_env_var(env_txt, "LLM_PROVIDER", provider)
            # set variables selon provider
            if provider == "groq":
                env_txt = _set_env_var(env_txt, "GROQ_API_KEY", (key_input or "").strip())
                env_txt = _set_env_var(env_txt, "GROQ_MODEL", (model or "llama-3.1-8b-instant").strip())
            elif provider == "openai":
                env_txt = _set_env_var(env_txt, "OPENAI_API_KEY", (key_input or "").strip())
                env_txt = _set_env_var(env_txt, "OPENAI_MODEL", (model or "gpt-4o-mini").strip())
            elif provider == "ollama":
                env_txt = _set_env_var(env_txt, "OLLAMA_BASE_URL", (base or "http://127.0.0.1:11434").strip())
                env_txt = _set_env_var(env_txt, "OLLAMA_MODEL", (model or "llama3.2:3b").strip())

            _write_env(env_txt)
            st.success("✅ Paramètres sauvegardés dans .env. Relance en cours…")
            st.rerun()

        st.caption(f"Fichier .env : `{ENV_PATH}`")

    with colB:
        st.subheader("Test rapide du LLM")
        prompt = st.text_area("Prompt", "Dis simplement: Bonjour, je suis prêt.")
        if st.button("▶️ Tester"):
            try:
                out = simple_chat(prompt)
                st.success("Réponse du LLM :")
                st.write(out)
            except Exception as e:
                st.error(f"Impossible d'appeler le LLM : {e}")

        st.divider()
        st.subheader("Infos")
        st.write(f"LLM_PROVIDER courant : **{os.getenv('LLM_PROVIDER', 'non défini')}**")
        st.write(f"GROQ_API_KEY : **{_mask(os.getenv('GROQ_API_KEY',''))}**")
        st.write(f"OPENAI_API_KEY : **{_mask(os.getenv('OPENAI_API_KEY',''))}**")
        st.write(f"OLLAMA_BASE_URL : **{os.getenv('OLLAMA_BASE_URL','')}**")
        st.write(f"MODEL (GROQ) : **{os.getenv('GROQ_MODEL','')}**")
        st.write(f"MODEL (OPENAI) : **{os.getenv('OPENAI_MODEL','')}**")
        st.write(f"MODEL (OLLAMA) : **{os.getenv('OLLAMA_MODEL','')}**")

if __name__ == "__main__":
    main()
