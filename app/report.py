from __future__ import annotations
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import streamlit as st
from utils.csv_utils import read_logs_robust
from utils.report_generator import build_report_context, render_html_report, report_display_values
from utils.reporting_pdf import build_pdf_bytes
from utils.settings import get_settings

DATA_DIR = os.path.join(os.getcwd(), "data")
REPORTS_DIR = os.path.join(os.getcwd(), "reports")
LOG_PATH = os.path.join(DATA_DIR, "predictions_log.csv")


def _read_logs(path):
    return read_logs_robust(path)


def _report_context_from_row(row):
    ctx = build_report_context(
        client_id=row.get("client_id"), timestamp=row.get("timestamp"),
        pd_value=row.get("prob_default"), threshold=row.get("threshold"),
        inputs_json=row.get("inputs_json"), model_tag=row.get("model_tag"),
        rating=row.get("rating"), decision=row.get("decision"),
        score_1000=row.get("score_1000"), risk_level=row.get("risk_level"),
        model_id=row.get("model_id"), input_id=row.get("input_id"),
    )
    if row.get("_history_warning"):
        ctx["data_notes"].append(str(row["_history_warning"]))
    return ctx


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value or "dossier")).strip("._") or "dossier"


def main():
    st.set_page_config(page_title="Rapports HTML et PDF", page_icon="R", layout="wide")
    st.title("Rapports HTML et PDF")
    st.caption("Selectionnez un dossier. Les valeurs disponibles de la prediction sont conservees.")
    try:
        df = _read_logs(LOG_PATH)
    except (ValueError, OSError) as error:
        st.error(f"Lecture du journal impossible : {error}")
        return
    if df.empty:
        st.info("Aucune prediction enregistree.")
        return
    df = df.copy()
    df["_sort_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.sort_values("_sort_timestamp", ascending=False, kind="stable", na_position="last")
    st.caption(f"Historique : {len(df)} dossiers")
    preview = ["timestamp", "client_id", "prob_default", "threshold", "rating", "decision"]
    st.dataframe(df[preview].head(10), hide_index=True, width="stretch")
    client_keys = df["client_id"].fillna("").astype(str)
    clients = client_keys.drop_duplicates().tolist()
    client_id = st.selectbox("Client", clients, format_func=lambda value: value or "Client non renseigne")
    same_client = df[client_keys == client_id]
    if same_client.empty:
        st.info("Aucun dossier pour cette selection.")
        return
    choices = same_client.index.tolist()
    def label(index):
        row = same_client.loc[index]
        timestamp = row.get("timestamp")
        if pd.isna(timestamp) or not str(timestamp).strip():
            timestamp = "Date non renseignee"
        return f"{timestamp} / dossier {row['_record_number']}"
    selected = st.selectbox("Dossier / occurrence", choices, format_func=label)
    row = same_client.loc[selected].to_dict()
    if st.checkbox("Voir la ligne source", value=False):
        st.json({key: str(value) for key,value in row.items() if not key.startswith("_")})
    try:
        ctx = _report_context_from_row(row)
    except (ValueError, TypeError) as error:
        st.warning(f"Dossier incomplet ou invalide : {error}")
        return
    cfg = get_settings()
    ctx.update(primary_color=cfg.report_primary_color, footer_text=cfg.report_footer_text)
    logo = "assets/logo.png" if Path("assets/logo.png").is_file() else None
    ctx["logo_path"] = logo
    values = report_display_values(ctx)
    st.subheader("Resultats conserves")
    fields = [("pd_value", "PD"), ("score_1000", "Score (0-1000)"), ("rating", "Notation"),
              ("risk_level", "Niveau de risque"), ("decision", "Decision"), ("threshold", "Seuil utilise")]
    for column,(key,title) in zip(st.columns(6),fields):
        column.metric(title, values[key])
    for note in ctx["data_notes"]:
        st.caption(note)
    st.info("Aucune explication individuelle validee n'est disponible pour ce dossier. Les importances globales ne sont pas des contributions individuelles.")
    try:
        html_doc = render_html_report(ctx)
        pdf_bytes = build_pdf_bytes(ctx, df_exp=None, logo_path=logo)
    except (ValueError, TypeError, OSError) as error:
        st.error(f"Generation du rapport impossible : {error}")
        return
    name = _safe_filename(ctx["client_id"])
    st.download_button("Telecharger HTML", html_doc.encode("utf-8"), file_name=f"report_{name}.html", mime="text/html")
    st.download_button("Telecharger PDF", pdf_bytes, file_name=f"report_{name}.pdf", mime="application/pdf")
    if st.button("Enregistrer HTML et PDF"):
        try:
            folder = Path(REPORTS_DIR)
            folder.mkdir(parents=True, exist_ok=True)
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            html_path = folder / f"report_{name}_{suffix}.html"
            pdf_path = folder / f"report_{name}_{suffix}.pdf"
            with html_path.open("x", encoding="utf-8") as stream:
                stream.write(html_doc)
            with pdf_path.open("xb") as stream:
                stream.write(pdf_bytes)
            st.success("Rapports HTML et PDF enregistres.")
        except OSError as error:
            st.error(f"Enregistrement impossible : {error}")
    if st.button("Ouvrir le dossier des rapports"):
        try:
            folder = Path(REPORTS_DIR)
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(str(folder))
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])
        except OSError as error:
            st.warning(f"Ouverture impossible : {error}")


if __name__ == "__main__":
    main()
