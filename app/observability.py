# app/observability.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import streamlit as st
from utils.csv_utils import read_logs_robust, sanitize_predictions_log, read_any_csv_robust

DATA_DIR = Path("data")
PRED_LOG = DATA_DIR / "predictions_log.csv"
AGENT_LOG = DATA_DIR / "agent_usage.csv"

def _kpi_card(title: str, value: str):
    st.markdown(
        f"""
        <div style="border:1px solid #e5e7eb;border-radius:12px;padding:14px;background:#fafafa">
          <div style="font-size:.75rem;color:#6b7280;text-transform:uppercase;letter-spacing:.08em">{title}</div>
          <div style="font-size:1.5rem;font-weight:800;margin-top:2px">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def main():
    st.set_page_config(page_title="Observabilité", page_icon="📡", layout="wide")
    st.title("📡 Observabilité")
    st.caption("KPIs d’usage, santé du scoring et suivi de l’agent IA (si activé).")

    # ---------- Section Scoring / Prédictions ----------
    st.subheader("Scoring — Journal des prédictions")
    if PRED_LOG.exists():
        sanitize_predictions_log(str(PRED_LOG))
    dfp = read_logs_robust(str(PRED_LOG))

    if dfp.empty:
        st.info("Aucun log de prédiction pour le moment.")
    else:
        dfp["timestamp"] = pd.to_datetime(dfp["timestamp"], errors="coerce")
        dfp["prob_default"] = pd.to_numeric(dfp["prob_default"], errors="coerce")
        dfp["threshold"] = pd.to_numeric(dfp["threshold"], errors="coerce")
        dfp["approved"] = (dfp["prob_default"] < dfp["threshold"]).astype(int)

        total_preds = len(dfp)
        avg_pd = dfp["prob_default"].mean() if total_preds else 0.0
        appr_rate = dfp["approved"].mean() if total_preds else 0.0

        c1, c2, c3 = st.columns(3)
        _kpi_card("Volume prédictions (total)", f"{total_preds:,}")
        _kpi_card("PD moyenne", f"{avg_pd*100:,.2f}%")
        _kpi_card("Taux d’accord", f"{appr_rate*100:,.1f}%")

        df_day = (
            dfp.assign(day=dfp["timestamp"].dt.date)
               .groupby("day", as_index=False)
               .agg(preds=("client_id","count"), avg_pd=("prob_default","mean"), acc=("approved","mean"))
        )
        st.markdown("##### Évolution quotidienne")
        st.line_chart(df_day.set_index("day")[["preds","avg_pd","acc"]])

        with st.expander("Données brutes (dernières lignes)"):
            st.dataframe(dfp.sort_values("timestamp", ascending=False).head(200), use_container_width=True)

    st.divider()

    # ---------- Section Agent IA ----------
    st.subheader("Agent IA — Usage & Outils")
    dfa = read_any_csv_robust(str(AGENT_LOG))

    if dfa.empty:
        st.info("Aucun log d’agent (agent_usage.csv).")
    else:
        if "timestamp" in dfa:
            dfa["timestamp"] = pd.to_datetime(dfa["timestamp"], errors="coerce")
            dfa["day"] = dfa["timestamp"].dt.date
        if "success" in dfa:
            dfa["success"] = dfa["success"].astype(str)

        total_calls = len(dfa)
        tools_counts = dfa["tool"].value_counts().to_dict() if "tool" in dfa else {}
        ok_rate = (dfa["success"] == "True").mean() if "success" in dfa and total_calls else 0.0

        c1, c2, c3 = st.columns(3)
        _kpi_card("Appels agent (total)", f"{total_calls:,}")
        _kpi_card("Taux succès outils", f"{ok_rate*100:,.1f}%")
        most_used = max(tools_counts, key=tools_counts.get) if tools_counts else "-"
        _kpi_card("Outil le plus utilisé", most_used)

        if "tool" in dfa and "day" in dfa:
            st.markdown("##### Volume par outil")
            tool_pivot = dfa.pivot_table(index="day", columns="tool", values="user", aggfunc="count").fillna(0)
            st.area_chart(tool_pivot)

        with st.expander("Données d’usage (dernières lignes)"):
            show = dfa.sort_values("timestamp", ascending=False).head(300) if "timestamp" in dfa else dfa.head(300)
            def _fmt_extra(x):
                try:
                    obj = json.loads(x) if isinstance(x, str) else x
                    return json.dumps(obj, ensure_ascii=False)
                except Exception:
                    return str(x)
            if "extra" in show.columns:
                show["extra"] = show["extra"].map(_fmt_extra)
            st.dataframe(show, use_container_width=True)

if __name__ == "__main__":
    main()
