# app/agent.py
from __future__ import annotations
import streamlit as st
from utils.multitask_agent import create_multitask_agent, as_messages
from utils.llm_status import llm_status_badge

@st.cache_resource(show_spinner=False)
def _get_agent():
    # Création agent protégée : s'il échoue, on remonte l'erreur proprement
    return create_multitask_agent()

def main():
    st.markdown("## 🧠 Agent IA multitâche (LangChain)")
    st.caption("Outils: prédiction crédit, RAG (docs), recherche Web, génération de rapport, date/heure…")
    llm_status_badge()  # ← badge LLM

    st.divider()
    col_chat, col_help = st.columns([3, 2], gap="large")

    with col_chat:
        st.markdown("#### Chat")
        st.session_state.setdefault("agent_history", [])
        for role, msg in st.session_state.agent_history:
            st.chat_message("user" if role == "user" else "assistant").markdown(msg)

        user_msg = st.chat_input("Ex : « Prédit le risque pour ce JSON et génère un rapport. »")
        if user_msg:
            st.session_state.agent_history.append(("user", user_msg))
            st.chat_message("user").markdown(user_msg)

            try:
                agent = _get_agent()
                # Historique tronqué par as_messages() pour limiter les tokens
                msgs = as_messages(st.session_state.agent_history[:-1])
                out = agent.invoke({"input": user_msg, "chat_history": msgs})
                answer = out["output"] if isinstance(out, dict) and "output" in out else str(out)
            except Exception as e:
                answer = f"Erreur agent : {e}"

            st.session_state.agent_history.append(("assistant", answer))
            st.chat_message("assistant").markdown(answer)

        with st.expander("🧹 Effacer l'historique"):
            if st.button("Reset", use_container_width=True):
                st.session_state.agent_history = []
                st.rerun()

    with col_help:
        st.markdown("#### Outils (appelés automatiquement)")
        st.write("- **predict_credit** : PD, score (0–1000), rating, décision")
        st.write("- **rag_retrieve** : passages des documents internes (k=3, tronqués)")
        st.write("- **search_web** : résultats web (résumé court)")
        st.write("- **build_report** : génère un **HTML** de rapport (tronqué)")
        st.write("- **now** : heure actuelle")

        st.markdown("#### Exemple JSON pour `predict_credit`")
        st.code('''{
  "client_id": "CLI-0001",
  "DTIRatio": 0.35, "TrustScorePsychometric": 0.62, "HouseholdSize": 4, "NumCreditLines": 2,
  "Income": 300000, "CommunityGroupMember": true, "HasMortgage": false, "MonthsEmployed": 36,
  "HasSocialAid": false, "MobileMoneyTransactions": 120, "Age": 32, "InterestRate": 12.0,
  "LoanTerm": 24, "LoanAmount": 800000, "InformalIncome": 50000, "threshold": 0.10
}''', language="json")

if __name__ == "__main__":
    main()
