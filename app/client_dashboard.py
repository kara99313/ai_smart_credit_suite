# app/client_dashboard.py
import streamlit as st
import pandas as pd
import json
import os
from io import StringIO, BytesIO
from datetime import datetime, timedelta

# Plotly pour des graphes interactifs
import plotly.express as px
import plotly.graph_objects as go

LOG_PATH = "data/predictions_log.csv"

# =====================================================================
# Helpers: chargement, parsing, enrichissement
# =====================================================================
def ensure_log_file():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("timestamp,client_id,prob_default,pred_label,threshold,inputs_json,model_tag\n")

def load_logs() -> pd.DataFrame:
    ensure_log_file()
    try:
        df = pd.read_csv(LOG_PATH)
    except Exception:
        df = pd.DataFrame(columns=[
            "timestamp","client_id","prob_default","pred_label","threshold","inputs_json","model_tag"
        ])
    # horodatage -> UTC naive (sans timezone) pour comparaisons robustes
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    # types
    for c in ["prob_default", "threshold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pred_label" in df.columns:
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").astype("Int64")
    # inputs_json -> dict à la volée
    if "inputs_json" in df.columns:
        def _safe_json(x):
            try:
                return json.loads(x) if isinstance(x, str) and x.strip() else {}
            except Exception:
                return {}
        df["inputs_dict"] = df["inputs_json"].apply(_safe_json)
    return df

def extract_inputs_to_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des colonnes dérivées des inputs (sans explosion)."""
    if "inputs_dict" not in df.columns or df.empty:
        return df
    all_keys = []
    for d in df["inputs_dict"]:
        if isinstance(d, dict):
            all_keys.extend(list(d.keys()))
    if not all_keys:
        return df
    key_counts = pd.Series(all_keys).value_counts()
    top_keys = key_counts.index[:20]
    for k in top_keys:
        df[f"inp__{k}"] = df["inputs_dict"].apply(lambda d: d.get(k, None) if isinstance(d, dict) else None)
    return df

# =====================================================================
# KPI Block
# =====================================================================
def kpi_block(df: pd.DataFrame, threshold: float):
    total = len(df)
    if total == 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prédictions", "0")
        c2.metric("Taux défaut (>= seuil)", "--")
        c3.metric("Seuil courant", f"{threshold:.2f}")
        c4.metric("Clients uniques", "0")
        return

    # Calculs généraux
    bad = (df["prob_default"] >= threshold).sum()
    rate = bad / total if total > 0 else 0.0
    unique_clients = df["client_id"].nunique()

    # 7 derniers jours (comparaison de timestamps naïfs)
    now = pd.Timestamp.utcnow().tz_localize(None)
    last_7d = df[df["timestamp"] >= (now - timedelta(days=7))]
    last_7d_cnt = len(last_7d)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prédictions", f"{total:,}")
    c2.metric("Taux défaut (>= seuil)", f"{rate:.1%}")
    c3.metric("Seuil courant", f"{threshold:.2f}")
    c4.metric("Préd. (7j)", f"{last_7d_cnt:,}")

    c5, c6 = st.columns(2)
    c5.metric("Clients uniques", f"{unique_clients:,}")
    if "pred_label" in df.columns and df["pred_label"].notna().any():
        acc = (df["pred_label"] == (df["prob_default"] >= df["threshold"]).astype("Int64")).mean()
        c6.metric("Cohérence préd./seuil", f"{acc:.1%}")
    else:
        c6.metric("Cohérence préd./seuil", "—")

# =====================================================================
# Graphiques
# =====================================================================
def timeline_block(df: pd.DataFrame, title: str = "Évolution des probabilités"):
    if df.empty:
        st.info("Aucune donnée pour la période / les filtres sélectionnés.")
        return
    df = df.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["prob_default"],
        mode="lines+markers",
        name="Probabilité de défaut"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Temps",
        yaxis_title="Probabilité",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
    return fig  # on retourne la figure pour le rapport

def dist_block(df: pd.DataFrame, threshold: float):
    if df.empty:
        return None
    fig = px.histogram(df, x="prob_default", nbins=40, title="Distribution des probabilités")
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Seuil {threshold:.2f}", annotation_position="top")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    return fig  # pour le rapport

# =====================================================================
# Génération de rapport (PDF + fallback HTML)
# =====================================================================
def _fig_to_png_bytes(fig) -> bytes:
    """
    Convertit une figure Plotly en PNG (en mémoire).
    Nécessite kaleido: pip install kaleido
    """
    return fig.to_image(format="png")  # peut lever une exception si kaleido absent

def generate_client_report_pdf(df: pd.DataFrame, client_id: str, threshold: float) -> bytes:
    """
    Génère un PDF (bytes) avec ReportLab + images des figures Plotly.
    Requiert: reportlab et kaleido installés.
    """
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Titre et contexte
    story.append(Paragraph(f"Rapport Client – {client_id}", styles['Title']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Seuil de décision : <b>{threshold:.2f}</b>", styles['Normal']))
    story.append(Spacer(1, 8))

    total = len(df)
    avg_prob = df["prob_default"].mean() if total > 0 else 0
    default_rate = (df["prob_default"] >= threshold).mean() if total > 0 else 0

    story.append(Paragraph(f"Total prédictions : <b>{total}</b>", styles['Normal']))
    story.append(Paragraph(f"Probabilité moyenne : <b>{avg_prob:.2%}</b>", styles['Normal']))
    story.append(Paragraph(f"Taux défaut (>= seuil) : <b>{default_rate:.2%}</b>", styles['Normal']))
    story.append(Spacer(1, 12))

    # Figures: timeline + distribution si existantes
    # On re-génère des figures propres (tri)
    if not df.empty:
        df_sorted = df.sort_values("timestamp")
        fig_t = go.Figure()
        fig_t.add_trace(go.Scatter(x=df_sorted["timestamp"], y=df_sorted["prob_default"],
                                   mode="lines+markers", name="Prob. défaut"))
        fig_t.update_layout(title="Évolution des probabilités", xaxis_title="Temps", yaxis_title="Probabilité")
        img_t = _fig_to_png_bytes(fig_t)
        story.append(Image(BytesIO(img_t), width=400, height=220))
        story.append(Spacer(1, 10))

        fig_d = px.histogram(df, x="prob_default", nbins=30, title="Distribution des probabilités")
        fig_d.add_vline(x=threshold, line_dash="dash", line_color="red",
                        annotation_text=f"Seuil {threshold:.2f}", annotation_position="top")
        img_d = _fig_to_png_bytes(fig_d)
        story.append(Image(BytesIO(img_d), width=400, height=220))
        story.append(Spacer(1, 12))

    # Conclusion simple
    story.append(Paragraph(
        "Interprétation rapide : plus la probabilité est élevée, plus le risque de défaut est important. "
        "La ligne verticale sur l’histogramme représente le seuil de décision utilisé.",
        styles['Italic']
    ))

    doc.build(story)
    return buffer.getvalue()

def generate_client_report_html(df: pd.DataFrame, client_id: str, threshold: float) -> bytes:
    """
    Fallback: Génère un rapport HTML autonome (aucune dépendance supplémentaire).
    Les graphiques restent interactifs (embed Plotly).
    """
    # Figures en HTML
    df_sorted = df.sort_values("timestamp")
    fig_t = go.Figure()
    if not df_sorted.empty:
        fig_t.add_trace(go.Scatter(x=df_sorted["timestamp"], y=df_sorted["prob_default"],
                                   mode="lines+markers", name="Prob. défaut"))
    fig_t.update_layout(title="Évolution des probabilités (interactif)", xaxis_title="Temps", yaxis_title="Probabilité")
    html_t = fig_t.to_html(full_html=False, include_plotlyjs='cdn')

    fig_d = px.histogram(df, x="prob_default", nbins=30, title="Distribution des probabilités (interactive)")
    fig_d.add_vline(x=threshold, line_dash="dash", line_color="red",
                    annotation_text=f"Seuil {threshold:.2f}", annotation_position="top")
    html_d = fig_d.to_html(full_html=False, include_plotlyjs='cdn')

    total = len(df)
    avg_prob = df["prob_default"].mean() if total > 0 else 0
    default_rate = (df["prob_default"] >= threshold).mean() if total > 0 else 0

    html = f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Rapport Client – {client_id}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<body style="font-family: Arial; margin: 20px;">
<h1>Rapport Client – {client_id}</h1>
<p><b>Seuil de décision :</b> {threshold:.2f}</p>
<ul>
  <li><b>Total prédictions :</b> {total}</li>
  <li><b>Probabilité moyenne :</b> {avg_prob:.2%}</li>
  <li><b>Taux défaut (≥ seuil) :</b> {default_rate:.2%}</li>
</ul>
<hr>
<h3>Timeline</h3>
{html_t}
<hr>
<h3>Distribution</h3>
{html_d}
<p style="font-style:italic; color:#444; margin-top:16px;">
  Interprétation rapide : plus la probabilité est élevée, plus le risque de défaut est important.
  La ligne verticale sur l’histogramme représente le seuil de décision utilisé.
</p>
</body>
</html>"""
    return html.encode("utf-8")

# =====================================================================
# UI principale
# =====================================================================
def main():
    st.title("Dashboard Client / Historique")
    st.caption("Filtre par client, date et seuil • KPIs • Timeline • Export • Rapport")

    df = load_logs()
    df = df.dropna(subset=["timestamp", "client_id"]).copy()
    df = extract_inputs_to_cols(df)

    # ------------------ Barre de filtres ------------------
    with st.container():
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 1])
        clients = ["(Tous)"] + sorted(df["client_id"].dropna().astype(str).unique().tolist())
        client_pick = c1.selectbox("Client", clients, index=0)

        if not df.empty:
            min_date = df["timestamp"].min().date()
            max_date = df["timestamp"].max().date()
        else:
            today = datetime.utcnow().date()
            min_date = max_date = today

        date_from = c2.date_input("Du", value=min_date, min_value=min_date, max_value=max_date)
        date_to = c3.date_input("Au", value=max_date, min_value=min_date, max_value=max_date)

        threshold = float(c4.slider("Seuil", 0.05, 0.95, 0.50, 0.01))

    # ------------------ Application des filtres ------------------
    dff = df.copy()
    start_ts = pd.to_datetime(str(date_from))  # naive
    end_ts = pd.to_datetime(str(date_to)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    dff = dff[(dff["timestamp"] >= start_ts) & (dff["timestamp"] <= end_ts)]
    if client_pick != "(Tous)":
        dff = dff[dff["client_id"].astype(str) == client_pick]

    st.markdown("---")
    kpi_block(dff, threshold)

    # ------------------ Graphes ------------------
    g1, g2 = st.columns(2)
    with g1:
        fig_t = timeline_block(dff, title="Timeline des probabilités")
    with g2:
        fig_d = dist_block(dff, threshold)

    st.markdown("---")

    # ------------------ Table + Export + Rapports ------------------
    st.subheader("Événements filtrés")
    if not dff.empty:
        show_cols = ["timestamp", "client_id", "prob_default", "pred_label", "threshold", "model_tag"]
        extra_cols = [c for c in dff.columns if c.startswith("inp__")]
        show_cols += extra_cols
        st.dataframe(dff[show_cols].sort_values("timestamp", ascending=False), use_container_width=True, height=280)

        # Export CSV
        buff = StringIO()
        dff.to_csv(buff, index=False)
        st.download_button(
            "Télécharger CSV filtré",
            data=buff.getvalue().encode("utf-8"),
            file_name="client_dashboard_export.csv",
            mime="text/csv"
        )

        # Boutons de rapport
        st.markdown("#### 📄 Rapports")
        cpdf, chtml = st.columns(2)
        # Nom du “client” pour le rapport (si Tous → 'Portefeuille')
        report_id = client_pick if client_pick != "(Tous)" else "Portefeuille"

        with cpdf:
            if st.button("Générer PDF"):
                try:
                    pdf_bytes = generate_client_report_pdf(dff, report_id, threshold)
                    st.download_button(
                        label="⬇️ Télécharger le PDF",
                        data=pdf_bytes,
                        file_name=f"rapport_{report_id}.pdf",
                        mime="application/pdf"
                    )
                except Exception as e:
                    st.error(f"Échec PDF (installez 'reportlab' et 'kaleido') : {e}")

        with chtml:
            if st.button("Générer HTML"):
                html_bytes = generate_client_report_html(dff, report_id, threshold)
                st.download_button(
                    label="⬇️ Télécharger le HTML",
                    data=html_bytes,
                    file_name=f"rapport_{report_id}.html",
                    mime="text/html"
                )

    else:
        st.info("Aucun enregistrement avec ces filtres.")

    st.caption("Source : data/predictions_log.csv (alimenté automatiquement par la page Prédiction).")

if __name__ == "__main__":
    main()
