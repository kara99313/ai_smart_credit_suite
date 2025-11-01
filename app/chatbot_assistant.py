# app/chatbot_assistant.py
from __future__ import annotations
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from utils.llm_status import llm_status_badge

# i18n léger (fallback)
try:
    from utils.lang_utils import detect_lang, system_prompt_for
except Exception:
    def detect_lang(text: str) -> str:
        return "FR" if any(c in text for c in "éèàùçôîï") else "EN"
    def system_prompt_for(lang: str) -> str:
        if lang == "FR":
            return ("Tu es un assistant expert en scoring de crédit et en finance. "
                    "Réponds en français, clair et concis.")
        return ("You are a finance & credit scoring assistant. "
                "Answer in English, clearly and concisely.")

from utils.web_search import search_web, build_web_context

def _sidebar():
    st.markdown("### Options")
    st.session_state.assistant_use_web = st.checkbox(
        "🔎 Activer la recherche Web (outil de l’agent)",
        value=st.session_state.get("assistant_use_web", True)
    )
    st.session_state.assistant_agent = st.checkbox(
        "🤖 Mode Agent (LangChain + Tools)",
        value=st.session_state.get("assistant_agent", True)
    )
    st.caption("• “/reset” efface le chat.  • “/fr” / “/en” force la langue.")
    st.divider()

def _init_state():
    st.session_state.setdefault("assistant_msgs", [])
    st.session_state.setdefault("force_lang", None)
    st.session_state.setdefault("agent", None)

def _render_history():
    for role, msg in st.session_state.assistant_msgs:
        st.chat_message("user" if role == "user" else "assistant").markdown(msg)

def _handle_commands(text: str) -> bool:
    t = text.strip().lower()
    if t == "/reset":
        st.session_state.assistant_msgs = []
        st.session_state.agent = None
        st.toast("Historique effacé.", icon="🗑️")
        st.rerun()
        return True
    if t == "/fr":
        st.session_state.force_lang = "FR"
        st.toast("Langue forcée : français.", icon="🇫🇷")
        return True
    if t == "/en":
        st.session_state.force_lang = "EN"
        st.toast("Language forced: English.", icon="🇬🇧")
        return True
    return False

def _ensure_agent():
    if st.session_state.agent is None:
        try:
            # Ton create_credit_agent / run_agent_once sont déjà dans utils.agents
            from utils.agents import create_credit_agent
            st.session_state.agent = create_credit_agent()
        except Exception as e:
            raise RuntimeError(
                "Impossible de créer l’agent. Vérifie LLM (LLM_PROVIDER=groq + GROQ_API_KEY).\n"
                f"Détail: {e}"
            )

def main():
    st.markdown("## 🤖 Chatbot Assistant")
    st.caption("Agent IA multi-outils (Crédit API + Web) — explications traçables.")
    llm_status_badge()  # ← badge LLM

    with st.sidebar:
        _sidebar()

    _init_state()
    _render_history()

    user_msg = st.chat_input("Pose ta question… (ex : ‘Calcule la PD et la note pour DTI=0.4…’) ")
    if not user_msg:
        return
    if _handle_commands(user_msg):
        return

    st.session_state.assistant_msgs.append(("user", user_msg))
    st.chat_message("user").markdown(user_msg)

    lang = st.session_state.force_lang or detect_lang(user_msg)
    system_prompt = system_prompt_for(lang)

    if st.session_state.assistant_agent:
        try:
            _ensure_agent()
        except Exception as e:
            msg = f"Erreur agent : {e}"
            st.session_state.assistant_msgs.append(("assistant", msg))
            st.chat_message("assistant").markdown(msg)
            return

        chat_history = []
        for role, msg in st.session_state.assistant_msgs[:-1]:
            chat_history.append(HumanMessage(msg) if role == "user" else AIMessage(msg))

        # ✅ Limiter l'historique pour Groq free (évite TPM exceeded)
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

        query = user_msg if st.session_state.assistant_use_web else f"[NO_WEB] {user_msg}"

        try:
            from utils.agents import run_agent_once
            out = run_agent_once(st.session_state.agent, query, chat_history)
            answer = out.get("output", "")
            st.session_state.assistant_msgs.append(("assistant", answer))
            st.chat_message("assistant").markdown(answer)

            steps = out.get("intermediate_steps", []) or []
            if steps:
                with st.expander("🧩 Étapes & outils utilisés"):
                    for i, (action, observation) in enumerate(steps, 1):
                        st.markdown(f"**Étape {i}: tool → `{getattr(action, 'tool', 'n/a')}`**")
                        st.code(str(getattr(action, 'tool_input', '')), language="json")
                        st.markdown("**Observation**")
                        st.write(observation if isinstance(observation, str) else str(observation)[:2000])
        except Exception as e:
            msg = f"Erreur agent : {e}"
            st.session_state.assistant_msgs.append(("assistant", msg))
            st.chat_message("assistant").markdown(msg)
    else:
        from utils.llm_providers import load_llm
        web_ctx = ""
        if st.session_state.assistant_use_web:
            hits = search_web(user_msg, max_results=2)  # ✅ plus léger
            web_ctx = build_web_context(hits, max_chars_total=1200, fetch_full=False)

        human = user_msg if not web_ctx else (
            f"Tu peux t'appuyer sur ces résultats Web :\n\n{web_ctx}\n\n"
            f"Question utilisateur : {user_msg}"
        )
        try:
            llm = load_llm()
            resp = llm.invoke([
                {"type": "system", "content": system_prompt},
                {"type": "human",  "content": human},
            ])
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            answer = f"Erreur LLM : {e}"

        st.session_state.assistant_msgs.append(("assistant", answer))
        st.chat_message("assistant").markdown(answer)

        if web_ctx:
            with st.expander("🔗 Sources Web"):
                st.markdown(web_ctx)

if __name__ == "__main__":
    main()
