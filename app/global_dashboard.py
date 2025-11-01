# app/global_dashboard.py
import os, json
import pandas as pd
import plotly.express as px
import streamlit as st
from utils.csv_utils import read_logs_robust, sanitize_predictions_log

LOG_PATH = "data/predictions_log.csv"

st.set_page_config(layout="wide", page_title="Dashboard Global", page_icon="📊")

@st.cache_data(show_spinner=False)
def load_logs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp","client_id","prob_default","pred_label","threshold","inputs_json","model_tag","rating","decision"
        ])
    sanitize_predictions_log(path)
    df = read_logs_robust(path)
    # Timestamps → datetime naive
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    # Types numériques
    for c in ["prob_default","threshold","pred_label"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def kpi_block(df: pd.DataFrame, threshold: float):
    c1, c2, c3, c4 = st.columns(4)
    total = len(df)
    def_rate = float(df["pred_label"].mean()) if total and "pred_label" in df else 0.0
    avg_proba = float(df["prob_default"].mean()) if total and "prob_default" in df else 0.0

    now = pd.Timestamp.now()
    last_7d = df[df["timestamp"] >= (now - pd.Timedelta(days=7))] if "timestamp" in df else pd.DataFrame()
    new_last_7d = len(last_7d)

    c1.metric("Prédictions totales", f"{total:,}")
    c2.metric("Taux défaut prédit", f"{def_rate:.1%}")
    c3.metric("Proba défaut moyenne", f"{avg_proba:.2f}")
    c4.metric("Nouvelles (7j)", f"{new_last_7d}")

    st.caption(f"Seuil courant pour acceptation/rejet : **{threshold:.2f}**")

def chart_time_series(df: pd.DataFrame):
    if df.empty or "timestamp" not in df:
        st.info("Aucune donnée dans l’intervalle sélectionné.")
        return
    tmp = df.copy()
    tmp["date"] = tmp["timestamp"].dt.date
    g = tmp.groupby("date", as_index=False).agg(
        avg_proba=("prob_default", "mean"),
        volume=("client_id", "count"),
        def_rate=("pred_label", "mean"),
    )
    fig1 = px.line(g, x="date", y="avg_proba", markers=True,
                   title="Évolution de la probabilité moyenne de défaut (jour)")
    fig1.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(g, x="date", y="volume",
                  title="Volume de prédictions par jour")
    fig2.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig2, use_container_width=True)

def chart_distros(df: pd.DataFrame, threshold: float):
    if df.empty:
        return
    if "prob_default" in df:
        fig_h = px.histogram(df, x="prob_default", nbins=30,
                             title="Distribution des probabilités de défaut")
        fig_h.add_vline(x=threshold, line_width=3, line_dash="dash", line_color="red",
                        annotation_text=f"Seuil={threshold:.2f}", annotation_position="top right")
        fig_h.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    else:
        fig_h = None

    if "pred_label" in df:
        fig_p = px.pie(df, names="pred_label", hole=0.45,
                       title="Répartition des classes prédites (0=Bon payeur / 1=Défaillant)")
        fig_p.update_layout(height=350, margin=dict(l=10, r=10, t=50, b=10))
    else:
        fig_p = None

    c1, c2 = st.columns(2)
    if fig_h: c1.plotly_chart(fig_h, use_container_width=True)
    if fig_p: c2.plotly_chart(fig_p, use_container_width=True)

def table_preview(df: pd.DataFrame):
    st.markdown("### Aperçu des dernières prédictions")
    if df.empty:
        st.info("Aucune ligne à afficher.")
        return
    show_cols = [c for c in ["timestamp","client_id","prob_default","pred_label","threshold","model_tag","rating","decision"] if c in df.columns]
    st.dataframe(df.sort_values("timestamp", ascending=False)[show_cols].head(50), use_container_width=True)

def main():
    st.title("📊 Dashboard Global — Portefeuille & Tendances")
    df = load_logs(LOG_PATH)

    # Filtres
    st.markdown("#### 🔎 Filtres")
    c1, c2, c3, c4 = st.columns([1,1,1,1])

    # Intervalle de dates
    if not df.empty and "timestamp" in df:
        min_d = df["timestamp"].min().date()
        max_d = df["timestamp"].max().date()
    else:
        today = pd.Timestamp.now().date()
        min_d = max_d = today

    with c1:
        date_range = st.date_input("Intervalle de dates", (min_d, max_d), min_value=min_d, max_value=max_d)
        start_date, end_date = (date_range if isinstance(date_range, tuple) else (min_d, max_d))

    with c2:
        threshold = st.slider("Seuil (visuel)", min_value=0.0, max_value=1.0, value=0.50, step=0.01)

    with c3:
        models = ["(Tous)"] + sorted(df["model_tag"].dropna().unique().tolist() if "model_tag" in df else [])
        model_tag = st.selectbox("Modèle", options=models)

    with c4:
        only_last_30 = st.checkbox("Derniers 30 jours", value=False)

    # Application des filtres
    dff = df.copy()
    if not dff.empty and "timestamp" in dff:
        start_ts = pd.to_datetime(pd.Timestamp(start_date), utc=False)
        end_ts = pd.to_datetime(pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), utc=False)
        dff = dff[(dff["timestamp"] >= start_ts) & (dff["timestamp"] <= end_ts)]

    if only_last_30 and "timestamp" in dff:
        dff = dff[dff["timestamp"] >= (pd.Timestamp.now() - pd.Timedelta(days=30))]

    if model_tag != "(Tous)" and "model_tag" in dff.columns:
        dff = dff[dff["model_tag"] == model_tag]

    st.markdown("---")
    kpi_block(dff, threshold)

    st.markdown("---")
    chart_time_series(dff)

    st.markdown("---")
    chart_distros(dff, threshold)

    st.markdown("---")
    table_preview(dff)

    with st.expander("📦 Exemple d’une ligne (inputs_json)"):
        if not dff.empty and "inputs_json" in dff.columns:
            try:
                sample = json.loads(dff.iloc[-1]["inputs_json"])
                st.json(sample)
            except Exception:
                st.write("`inputs_json` non exploitable pour cet exemple.")

if __name__ == "__main__":
    main()
