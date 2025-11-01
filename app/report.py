# app/report.py
from __future__ import annotations
import os, sys, json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

# Explicabilité pour l’aperçu écran (facultatif si l’API est dispo)
try:
    from utils.api_client import explain_credit_api  # type: ignore
except Exception:  # pas bloquant si absent
    explain_credit_api = None  # type: ignore

from utils.report_generator import build_report_context, render_html_report
from utils.reporting_pdf import build_pdf_bytes
from utils.settings import get_settings

DATA_DIR    = os.path.join(os.getcwd(), "data")
REPORTS_DIR = os.path.join(os.getcwd(), "reports")
LOG_PATH    = os.path.join(DATA_DIR, "predictions_log.csv")

st.set_page_config(page_title="Rapports (HTML & PDF)", page_icon="🧾", layout="wide")


# ---------------- Helpers ----------------
def _safe_ts_to_str(ts_val: Any, fmt: str = "%Y-%m-%dT%H:%M:%S") -> str:
    """Transforme un timestamp potentiellement NaT/None/str en texte sûr."""
    if ts_val is None:
        return datetime.now().strftime(fmt)
    try:
        ts = pd.to_datetime(ts_val, errors="coerce")
        if pd.isna(ts):
            return datetime.now().strftime(fmt)
        if hasattr(ts, "to_pydatetime"):
            ts = ts.to_pydatetime()
        return ts.strftime(fmt)
    except Exception:
        return datetime.now().strftime(fmt)


def _read_logs(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "timestamp","client_id","prob_default","pred_label","threshold",
            "inputs_json","model_tag","rating","decision"
        ])
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python", on_bad_lines="skip")
    except Exception:
        df = pd.read_csv(path, encoding="latin-1", engine="python", on_bad_lines="skip")

    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    for c in ("prob_default", "threshold"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pred_label" in df:
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").fillna(0).astype(int)
    return df


def _rating_to_risk_level(rating: str) -> str:
    table = {
        "AAA":"Très faible","AA":"Très faible","A":"Faible","BBB":"Modéré",
        "BB":"Élevé","B":"Très élevé","CCC":"Critique","CC":"Critique",
        "C":"Critique","D":"Défaut imminent"
    }
    return table.get(str(rating), "Inconnu")


def _plot_contrib_bar(df: pd.DataFrame):
    """Graphique Plotly – importance absolue (Top 10)."""
    try:
        import plotly.express as px
    except Exception:
        return None
    d = df.copy()
    if "abs_contribution" not in d.columns:
        d["abs_contribution"] = d["contribution"].abs()
    d = d.sort_values("abs_contribution", ascending=True).tail(10)
    fig = px.bar(
        d, x="abs_contribution", y="feature", orientation="h",
        title="Principaux facteurs (importance absolue)"
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ---------------- UI ----------------
def main():
    st.title("🧾 Génération de rapports — HTML & PDF")
    st.caption("Sélectionnez une prédiction, générez le rapport HTML/PDF (structure CA), puis sauvegardez-le.")

    df = _read_logs(LOG_PATH)
    if df.empty:
        st.info("Aucune donnée dans le journal `data/predictions_log.csv`. Lancez au moins une prédiction.")
        return

    # Aperçu des dernières prédictions
    st.markdown("### Aperçu des dernières prédictions ↪")
    cols_to_show = ["timestamp","client_id","prob_default","pred_label","threshold","model_tag","rating","decision"]
    preview = df.sort_values("timestamp", ascending=False).head(10).copy()
    st.dataframe(preview[cols_to_show], use_container_width=True, hide_index=True)
    st.markdown("---")

    # Sélection
    st.markdown("#### 🔎 Sélection")
    c1, c2, c3 = st.columns([2,2,1])
    df_sorted = df.sort_values("timestamp", ascending=False)

    with c1:
        clients = df_sorted["client_id"].dropna().astype(str).unique().tolist()
        client_id = st.selectbox("Client", options=clients, index=0 if clients else None)

    with c2:
        same_client = df_sorted[df_sorted["client_id"].astype(str) == str(client_id)]
        dates = same_client["timestamp"].astype(str).tolist()
        date_choice = st.selectbox("Occurrence (horodatage)", options=dates, index=0 if dates else None)

    with c3:
        show_raw = st.checkbox("Voir la ligne brute", value=False)

    if date_choice:
        row = same_client[same_client["timestamp"].astype(str) == date_choice].iloc[0].to_dict()
    else:
        row = same_client.iloc[0].to_dict()

    if show_raw:
        with st.expander("Ligne brute sélectionnée"):
            st.json(row)

    # Champs sûrs
    safe_timestamp = _safe_ts_to_str(row.get("timestamp"))
    pd_value       = float(row.get("prob_default", 0.0) or 0.0)
    threshold      = float(row.get("threshold", 0.1) or 0.1)
    model_tag      = str(row.get("model_tag", "pipeline_v1"))
    rating         = str(row.get("rating", "N/A"))
    decision       = str(row.get("decision", "N/A"))
    risk_level     = _rating_to_risk_level(rating)
    try:
        inputs_obj = json.loads(row.get("inputs_json") or "{}")
    except Exception:
        inputs_obj = {}
    score_1000 = int(round((1.0 - max(0.0, min(1.0, pd_value))) * 1000))

    # Tableau récapitulatif (metrics)
    st.markdown("#### Récapitulatif")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("PD", f"{pd_value*100:.2f}%")
    r2.metric("Score (0–1000)", f"{score_1000}")
    r3.metric("Notation", rating)
    r4.metric("Niveau de risque", risk_level)
    r5.metric("Décision", decision)

    # Résultats & Explicabilité (graph + tableau)
    st.markdown("#### Résultats & Explicabilité")
    df_exp: pd.DataFrame | None = None
    if explain_credit_api is not None:
        try:
            exp = explain_credit_api(inputs_obj, top_k=30)
            if exp and exp.get("ok"):
                items = exp.get("items") or []
                contrib = {}
                for it in items:
                    f = str(it.get("feature"))
                    if "contribution" in it:
                        contrib[f] = float(it.get("contribution", 0.0))
                    elif "importance" in it:
                        contrib[f] = float(it.get("importance", 0.0))
                if contrib:
                    df_exp = pd.DataFrame({
                        "feature": list(contrib.keys()),
                        "contribution": list(contrib.values())
                    })
                    df_exp["value"] = df_exp["feature"].map(lambda f: inputs_obj.get(f))
                    df_exp["abs_contribution"] = df_exp["contribution"].abs()
        except Exception:
            df_exp = None

    if df_exp is not None:
        fig = _plot_contrib_bar(df_exp)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            df_exp.sort_values("abs_contribution", ascending=False)
                 .loc[:, ["feature","value","contribution","abs_contribution"]]
                 .head(10),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Explications détaillées non disponibles pour cette occurrence.")

    # Préparation HTML
    cfg = get_settings()
    logo_path = None
    for p in ["assets/logo.png", os.path.join("assets", "logo.png")]:
        if os.path.exists(p):
            logo_path = p
            break

    ctx_html = build_report_context(
        client_id=str(client_id),
        timestamp=safe_timestamp,
        pd_value=pd_value,
        threshold=threshold,
        inputs_json=json.dumps(inputs_obj, ensure_ascii=False),
        model_tag=model_tag,
        logo_path=logo_path,
        primary_color=cfg.report_primary_color,
        footer_text=cfg.report_footer_text,
    )
    html_doc = render_html_report(ctx_html)

    # Téléchargements
    st.markdown("### ⬇️ Téléchargements")
    colH, colP = st.columns(2)
    with colH:
        st.download_button(
            "📥 Télécharger le rapport (HTML)",
            data=html_doc.encode("utf-8"),
            file_name=f"report_{client_id}.html",
            mime="text/html",
            use_container_width=True,
        )
    with colP:
        pdf_bytes = build_pdf_bytes(
            ctx={
                "client_id": str(client_id),
                "timestamp": safe_timestamp,
                "pd_value": pd_value,
                "threshold": threshold,
                "model_tag": model_tag,
                "inputs": inputs_obj,
                "score_1000": score_1000,
                "rating": rating,
                "risk_level": risk_level,
                "decision": decision,
                "footer_text": cfg.report_footer_text,
            },
            df_exp=df_exp,
            logo_path=logo_path,
        )
        st.download_button(
            "🧾 Télécharger le rapport (PDF corporate)",
            data=pdf_bytes,
            file_name=f"report_{client_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Enregistrement local + ouverture dossier
    st.markdown("### 💾 Enregistrer dans /reports")
    try:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        b1, b2 = st.columns([2,1])
        with b1:
            if st.button("📂 Enregistrer HTML + PDF (horodaté)", use_container_width=True):
                ts = datetime.now().isoformat(timespec="seconds").replace(":", "").replace(" ", "_")
                html_path = os.path.join(REPORTS_DIR, f"report_{client_id}_{ts}.html")
                pdf_path  = os.path.join(REPORTS_DIR, f"report_{client_id}_{ts}.pdf")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(html_doc)
                with open(pdf_path, "wb") as f:
                    f.write(pdf_bytes)
                st.success(f"Rapports enregistrés : `{html_path}` et `{pdf_path}`")
        with b2:
            if st.button("📁 Ouvrir le dossier /reports", use_container_width=True):
                try:
                    if os.name == "nt":
                        os.startfile(REPORTS_DIR)  # type: ignore[attr-defined]
                    elif sys.platform == "darwin":
                        os.system(f'open "{REPORTS_DIR}"')
                    else:
                        os.system(f'xdg-open "{REPORTS_DIR}"')
                except Exception as e:
                    st.warning(f"Impossible d’ouvrir le dossier automatiquement : {e}")
    except Exception as e:
        st.warning(f"Enregistrement impossible : {e}")


if __name__ == "__main__":
    main()
