# app/home.py
from __future__ import annotations
from pathlib import Path
import os

import pandas as pd
import plotly.express as px
import streamlit as st

LOG_PATH = Path("data/predictions_log.csv")

# ========= Utils =========
@st.cache_data(show_spinner=False)
def load_logs(path: Path = LOG_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[
            "timestamp", "client_id", "prob_default", "pred_label", "threshold", "inputs_json", "model_tag"
        ])
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["prob_default", "threshold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pred_label" in df.columns:
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    for c in ["client_id", "prob_default", "threshold", "model_tag"]:
        if c not in df.columns:
            df[c] = None
    return df


def kpis_from_df(df: pd.DataFrame, threshold: float = 0.50):
    if df.empty:
        empty_daily = pd.DataFrame({"d": [], "count": []})
        fig_daily = px.line(empty_daily, x="d", y="count", title="Volume de prédictions (par jour)", height=260)
        fig_daily.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        return {
            "n_scored": 0, "accept_rate": None, "auc_proxy": 0.90, "new_clients": 0,
            "hist": None, "daily": fig_daily
        }

    n_scored = len(df)
    accept_rate = float((df["prob_default"] < threshold).mean()) if "prob_default" in df else None
    auc_proxy = 0.91  # proxy d’affichage

    df = df.copy()
    df["d"] = df["timestamp"].dt.date
    if df["timestamp"].notna().any():
        this_month = df["timestamp"].dt.to_period("M").max()
        new_clients = df[df["timestamp"].dt.to_period("M") == this_month]["client_id"].nunique()
    else:
        new_clients = 0

    hist = None
    if "prob_default" in df:
        hist = px.histogram(
            df, x="prob_default", nbins=30, title="Distribution des probabilités", height=260
        )
        hist.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    daily = (df.groupby("d").size().reset_index(name="count")
             if "d" in df.columns else pd.DataFrame({"d": [], "count": []}))
    fig_daily = px.line(
        daily, x="d", y="count", markers=True, title="Volume de prédictions (par jour)", height=260
    )
    fig_daily.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    return {
        "n_scored": int(n_scored),
        "accept_rate": accept_rate,
        "auc_proxy": auc_proxy,
        "new_clients": int(new_clients),
        "hist": hist,
        "daily": fig_daily
    }


# ========= Page =========
def main():
    st.set_page_config(layout="wide", page_title="Smart Credit Scoring", page_icon=":bar_chart:")

    # ===== Sidebar (AIDE + ÉTAT ENV) — PAS DE page_link ICI =====
    with st.sidebar:
        st.header("Aide")
        st.info(
            "Utilisez le menu principal (dans la page `streamlit_app.py`) pour naviguer : "
            "Prédiction, Dashboards, Agent IA, Chatbots, Rapports, etc.",
            icon="ℹ️"
        )
        st.divider()
        st.subheader("État de l’environnement")
        st.code(
            f"LLM_PROVIDER = {os.environ.get('LLM_PROVIDER', 'ollama')}\n"
            f"OLLAMA_MODEL = {os.environ.get('OLLAMA_MODEL', 'llama3.2:3b')}\n"
            f"OLLAMA_BASE_URL = {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}\n"
            f"BACKEND_URL = {os.environ.get('BACKEND_URL', 'http://127.0.0.1:18000')}"
        )

    # --- En-tête compact : bannière + titre + slogan ---
    st.markdown(
        """
        <div style='display:flex;align-items:center;justify-content:space-between;background:#0B1F33;
                    border-radius:14px;padding:10px 26px;box-shadow:0 4px 14px #00142855;'>
            <div style='display:flex;align-items:center;gap:14px;'>
                <img src='https://images.unsplash.com/photo-1461749280684-dccba630e2f6?auto=format&fit=crop&w=720&q=80'
                     style='width:110px;height:48px;object-fit:cover;border-radius:8px;box-shadow:0 2px 8px #001;'/>
                <span style='color:#F9FAFB;font-size:1.8rem;font-weight:700;letter-spacing:0.5px;'>
                    Smart Credit Scoring
                </span>
            </div>
            <div style='text-align:right;'>
                <div style='color:#83C5FF;font-size:0.95rem;'>La nouvelle génération d’évaluation du risque</div>
                <div style='color:#FFD700;font-size:0.95rem;'>Powered by AI & Data Science</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # --- Corps : 3 colonnes équilibrées ---
    col1, col2, col3 = st.columns([1.25, 1.6, 1.25], gap="large")

    # ===== Colonne 1 : À-PROPOS =====
    with col1:
        tabs = st.tabs(["À propos", "Pourquoi", "Navigation"])
        with tabs[0]:
            st.markdown(
                """
                <div style='background:#F6FAFF;border-radius:14px;padding:12px 12px 6px;'>
                  <ul style='margin:0 0 0.2rem 1.1rem; line-height:1.55rem; font-size:0.98rem;'>
                    <li><b>Inclusif</b> : données bancaires & socio-éco</li>
                    <li><b>Explicable</b> : décisions justifiées (SHAP/LIME)</li>
                    <li><b>Ultra-analytique</b> : vues portefeuille & client</li>
                    <li><b>Ouvert</b> : APIs, chatbots, RAG documentaire</li>
                    <li><b>Personnalisable</b> : thèmes, langues, préférences</li>
                  </ul>
                </div>
                """, unsafe_allow_html=True
            )
        with tabs[1]:
            st.markdown(
                """
                <div style='background:#F6FFFB;border-radius:14px;padding:12px 12px 6px;'>
                  <ul style='margin:0 0 0.2rem 1.1rem; line-height:1.55rem; font-size:0.98rem;'>
                    <li>Modélisation <b>hybride</b> IA / métier</li>
                    <li>Interface <b>premium</b>, claire et rapide</li>
                    <li>Traçabilité, <b>audit</b> et conformité intégrés</li>
                    <li>Architecture <b>modulaire</b>, évolutive</li>
                  </ul>
                </div>
                """, unsafe_allow_html=True
            )
        with tabs[2]:
            st.markdown(
                """
                <div style='background:#FFF9F0;border-radius:14px;padding:10px 12px 2px;'>
                  <p style='margin:0; line-height:1.55rem; font-size:0.98rem;'>
                  Utilisez le menu principal (géré par <code>streamlit_app.py</code>) pour accéder aux
                  pages : Prédiction, Dashboards, Agent IA, Chatbots, Rapports, etc.
                  </p>
                </div>
                """, unsafe_allow_html=True
            )

    # ===== Colonne 2 : KPIs + Graphiques =====
    with col2:
        df = load_logs()
        k = kpis_from_df(df, threshold=0.50)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Clients scorés", f"{k['n_scored']:,}")
        k2.metric("Taux d’acceptation", "--" if k["accept_rate"] is None else f"{k['accept_rate']:.0%}")
        k3.metric("AUC (proxy)", f"{k['auc_proxy']:.2f}")
        k4.metric("Nouveaux clients (mois)", f"{k['new_clients']:,}")

        g1, g2 = st.columns(2)
        with g1:
            if k["hist"] is not None:
                st.plotly_chart(k["hist"], use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Histogramme indisponible (pas de données).")
        with g2:
            if k["daily"] is not None:
                st.plotly_chart(k["daily"], use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Série quotidienne indisponible.")

        st.markdown(
            "<div style='text-align:center;color:#0B5ED7;margin-top:4px;'>"
            "« Un crédit plus juste, une analyse plus fine, un avenir plus sûr. »</div>",
            unsafe_allow_html=True
        )

    # ===== Colonne 3 : Branding =====
    with col3:
        st.markdown(
            """
            <div style='background:#F6FAFF;border-radius:14px;padding:14px;text-align:center;'>
              <img src='https://cdn-icons-png.flaticon.com/512/3039/3039394.png' width='60' style='margin-bottom:10px;' />
              <div style='color:#053B74;font-weight:600;font-size:1.04rem;'>Plateforme de scoring IA</div>
              <div style='color:#053B74;font-size:0.96rem;'>Explicabilité • Sécurité • Excellence</div>
              <hr style='margin:0.8rem 0 0.6rem 0;'>
              <div style='font-size:0.94rem;color:#0B5ED7;'>Afrique francophone — Référence</div>
            </div>
            """, unsafe_allow_html=True
        )

        st.markdown(
            """
            <div style='background:#EEF6FF;border-radius:14px;padding:10px;margin-top:10px;'>
              <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:0.95rem;'>
                <div>🔮 Prédiction</div>
                <div>📈 Portefeuille</div>
                <div>🧑‍💼 Client</div>
                <div>🤖 Chatbot</div>
              </div>
            </div>
            """, unsafe_allow_html=True
        )

    # Footer
    st.markdown(
        """
        <div style='width:100%;margin-top:12px;'>
          <div style='background:linear-gradient(90deg,#001f3f 60%,#0B5ED7 100%);
                      border-radius:12px;padding:8px 12px;text-align:center;'>
            <span style='color:#FFD700;font-size:0.98rem;font-weight:600;'>
              Contact : <a href="mailto:contact@credit-scoring.ai" style="color:#FFF;">contact@credit-scoring.ai</a>
            </span>
            <span style='color:#EAF2FF;font-size:0.95rem;margin-left:1.2rem;'>
              Mentions légales | © 2025 Idriss & OpenAI — Smart Credit Scoring
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
