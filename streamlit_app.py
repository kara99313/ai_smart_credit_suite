# streamlit_app.py
from __future__ import annotations
import os, sys, csv
import streamlit as st

# --- UTF-8 partout (console / Streamlit logs)
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# 1) Config générale
st.set_page_config(page_title="AI Smart Credit Suite", page_icon="💠", layout="wide")

# 2) i18n (robuste : fallback si utils.i18n absent)
try:
    from utils.i18n import t, set_mode, get_mode, init_i18n
    init_i18n()
except Exception:
    def t(key: str, **kw): return kw.get("default", key)
    def set_mode(x: str):
        st.session_state.setdefault("APP_LANG", "fr")
        st.session_state["APP_LANG"] = x
    def get_mode() -> str:
        return st.session_state.get("APP_LANG", "fr")

# 3) S’assure que le log existe (pour pages qui lisent data/predictions_log.csv)
LOG_DIR = "data"
LOG_PATH = os.path.join(LOG_DIR, "predictions_log.csv")
LOG_HEADER = ["timestamp","client_id","prob_default","pred_label","threshold","inputs_json","model_tag"]
os.makedirs(LOG_DIR, exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(LOG_HEADER)

# 4) Barre latérale “globale”
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3039/3039394.png", width=60)
    st.markdown("### " + t("app.language", default="Langue"))
    current = get_mode()
    choix = st.radio(
        label=t("app.language", default="Langue"),
        options=[("fr", "Français"), ("en", "English"), ("bi", "Bilingue")],
        index={"fr": 0, "en": 1, "bi": 2}.get(current, 0),
        format_func=lambda x: x[1],
        horizontal=True,
        key="lang_mode",
        label_visibility="collapsed",
    )
    if choix[0] != current:
        set_mode(choix[0])
        # Essaye de propager la langue dans l’URL (compatibilité 1.50)
        try:
            qp = st.query_params
            qp["lang"] = choix[0]
            st.query_params = qp
        except Exception:
            try:
                st.experimental_set_query_params(lang=choix[0])
            except Exception:
                pass
        st.rerun()

# 5) Déclaration des pages
home              = st.Page("app/home.py",              title="Accueil",             icon="🏠")
prediction        = st.Page("app/prediction.py",        title="Prédiction Scoring",  icon="📊")
client_dashboard  = st.Page("app/client_dashboard.py",  title="Dashboard Client",    icon="👤")
global_dashboard  = st.Page("app/global_dashboard.py",  title="Dashboard Global",    icon="📈")
report            = st.Page("app/report.py",            title="Rapports",            icon="📝")

agent             = st.Page("app/agent.py",             title="Agent IA",            icon="🧠")
chatbot_assistant = st.Page("app/chatbot_assistant.py", title="Chatbot Assistant",   icon="🤖")
rag_chatbot       = st.Page("app/rag_chatbot.py",       title="Chatbot RAG",         icon="📚")

health_check      = st.Page("app/health_check.py",      title="Health Check",        icon="🩺")
observability     = st.Page("app/observability.py",     title="Observabilité",       icon="📡")
integrations      = st.Page("app/integrations.py",      title="Intégrations & API",  icon="🔗")
admin             = st.Page("app/admin.py",             title="Administration",      icon="⚙️")

# ⚠️ Si une page n’existe pas physiquement, enlève-la de la liste.

# 6) Navigation (groupes + ordre)
nav = st.navigation({
    "🏠 Application": [home, prediction, report, client_dashboard, global_dashboard],
    "🤖 Assistants":  [agent, chatbot_assistant, rag_chatbot],
    "🛠️ Opérations":  [health_check, observability, integrations, admin],
})

# 7) Lance la page choisie
nav.run()
