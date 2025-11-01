# app/prediction.py
from __future__ import annotations
import json, csv, os, io, sys
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from utils.api_client import predict_credit_api, explain_credit_api, ApiError
from utils.explanations import heuristic_narrative, build_narrative_from_contrib
from utils.report_generator import build_report_context, render_html_report
from utils.reporting_pdf import build_pdf_bytes
from utils.settings import get_settings

# ====== Chemins / log ======
PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
LOG_PATH = os.path.join(DATA_DIR, "predictions_log.csv")
LOG_HEADER = ["timestamp","client_id","prob_default","pred_label","threshold","inputs_json","model_tag","rating","decision"]
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
if not os.path.exists(LOG_PATH):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(LOG_HEADER)

# ====== Libellés FR/EN ======
DISPLAY_NAMES = {
    "DTIRatio": ("Ratio d’endettement (DTI)", "Debt-to-Income Ratio"),
    "TrustScorePsychometric": ("Score psychométrique (confiance)", "Psychometric trust score"),
    "HouseholdSize": ("Taille du foyer", "Household size"),
    "NumCreditLines": ("Nombre de lignes de crédit actives", "Active credit lines"),
    "Income": ("Revenu principal déclaré", "Main declared income"),
    "CommunityGroupMember": ("Membre d’un groupe communautaire", "Community group member"),
    "HasMortgage": ("Possession d’un prêt immobilier", "Has a mortgage"),
    "MonthsEmployed": ("Ancienneté emploi (mois)", "Months in current employment"),
    "HasSocialAid": ("Bénéficie d’une aide sociale", "Receives social aid"),
    "MobileMoneyTransactions": ("Transactions Mobile Money", "Mobile money transactions"),
    "Age": ("Âge", "Age"),
    "InterestRate": ("Taux d’intérêt du prêt (%)", "Loan interest rate (%)"),
    "LoanTerm": ("Durée du prêt (mois)", "Loan term (months)"),
    "LoanAmount": ("Montant du prêt demandé", "Requested loan amount"),
    "InformalIncome": ("Revenu informel estimé", "Estimated informal income"),
}

# ====== UI / look ======
def try_load_logo():
    for p in ["assets/logo.png", os.path.join("assets", "logo.png")]:
        if os.path.exists(p):
            return p
    return None

def header():
    st.markdown(
        """
        <style>
        .kpi-card {border-radius:16px; padding:16px; background:#0b1220; color:#e8edf6; border:1px solid #1e2a44;}
        .kpi-title {font-size:12px; text-transform:uppercase; letter-spacing:.12em; color:#9fb3d1;}
        .kpi-value {font-size:28px; font-weight:700; margin-top:4px;}
        .subtle {color:#99a6bf;}
        .note {font-size:12px; color:#718198;}
        </style>
        """,
        unsafe_allow_html=True
    )
    cols = st.columns([1,3])
    with cols[0]:
        logo = try_load_logo()
        if logo: st.image(logo, width=90)
    with cols[1]:
        st.markdown("### Prédiction de Risque — **Scoring Crédit**")
        st.caption("Saisissez les 15 variables explicatives. Calcul : PD, score (0–1000), notation et décision (via API).")

def input_form(lang="FR") -> dict:
    disp = (lambda k: DISPLAY_NAMES[k][0]) if lang == "FR" else (lambda k: DISPLAY_NAMES[k][1])
    st.markdown("#### Données client & prêt")
    c1, c2, c3 = st.columns(3)
    with c1:
        dti   = st.number_input(disp("DTIRatio"), min_value=0.0, max_value=5.0, value=0.35, step=0.01)
        trust = st.number_input(disp("TrustScorePsychometric"), min_value=0.0, max_value=1.0, value=0.62, step=0.01)
        hh    = st.number_input(disp("HouseholdSize"), min_value=1, max_value=20, value=4, step=1)
        lines = st.number_input(disp("NumCreditLines"), min_value=0, max_value=50, value=2, step=1)
        income= st.number_input(disp("Income"), min_value=0.0, value=300000.0, step=1000.0)
    with c2:
        comm  = st.selectbox(disp("CommunityGroupMember"), ["Non","Oui"])
        mort  = st.selectbox(disp("HasMortgage"), ["Non","Oui"])
        months_emp = st.number_input(disp("MonthsEmployed"), min_value=0, max_value=600, value=36, step=1)
        aid   = st.selectbox(disp("HasSocialAid"), ["Non","Oui"])
        mm_tx = st.number_input(disp("MobileMoneyTransactions"), min_value=0, max_value=100000, value=120, step=1)
    with c3:
        age   = st.number_input(disp("Age"), min_value=18, max_value=90, value=32, step=1)
        ir    = st.number_input(disp("InterestRate"), min_value=0.0, max_value=100.0, value=12.0, step=0.1)
        term  = st.number_input(disp("LoanTerm"), min_value=1, max_value=360, value=24, step=1)
        amount= st.number_input(disp("LoanAmount"), min_value=0.0, value=800000.0, step=1000.0)
        informal = st.number_input(disp("InformalIncome"), min_value=0.0, value=50000.0, step=1000.0)

    return {
        "DTIRatio": float(dti), "TrustScorePsychometric": float(trust), "HouseholdSize": int(hh),
        "NumCreditLines": int(lines), "Income": float(income),
        "CommunityGroupMember": True if comm == "Oui" else False,
        "HasMortgage": True if mort == "Oui" else False,
        "MonthsEmployed": int(months_emp), "HasSocialAid": True if aid == "Oui" else False,
        "MobileMoneyTransactions": int(mm_tx), "Age": int(age), "InterestRate": float(ir),
        "LoanTerm": int(term), "LoanAmount": float(amount), "InformalIncome": float(informal),
    }

def kpi_cards(pd_val: float, score_1000: int, rating: str, risk_level: str, decision_txt: str):
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>PD (Probabilité de Défaut)</div><div class='kpi-value'>{pd_val*100:.2f}%</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Score (0-1000)</div><div class='kpi-value'>{score_1000}</div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Notation</div><div class='kpi-value'>{rating}</div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Niveau de risque</div><div class='kpi-value'>{risk_level}</div></div>", unsafe_allow_html=True)
    with c5:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Décision</div><div class='kpi-value'>{decision_txt}</div></div>", unsafe_allow_html=True)

def _plotly_pd_gauge(pd_val: float, threshold: float):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    v = float(max(0.0, min(1.0, pd_val))) * 100.0
    thr = float(max(0.0, min(1.0, threshold))) * 100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={'suffix': '%', 'font': {'size': 26}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#111827"},
               'steps': [{'range': [0,8], 'color':'#16a34a'},
                         {'range':[8,15],'color':'#f59e0b'},
                         {'range':[15,100],'color':'#dc2626'}],
               'threshold': {'line': {'color':'#0ea5e9', 'width':4}, 'thickness':0.75, 'value':thr}},
        title={'text': "PD (%) — seuil en bleu", 'font': {'size': 14}}
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=0))
    return fig

def _plotly_score_gauge(score_1000: int):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    v = int(max(0, min(1000, score_1000)))
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={'font': {'size': 26}},
        gauge={'axis': {'range': [0, 1000]}, 'bar': {'color': "#111827"},
               'steps': [{'range':[0,300],'color':'#dc2626'},
                         {'range':[300,700],'color':'#f59e0b'},
                         {'range':[700,1000],'color':'#16a34a'}]},
        title={'text': "Score (0–1000)", 'font': {'size': 14}}
    ))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=0))
    return fig

def gauge_row(pd_val: float, score_1000: int, threshold: float):
    c1, c2 = st.columns(2)
    fig1 = _plotly_pd_gauge(pd_val, threshold)
    fig2 = _plotly_score_gauge(score_1000)
    if fig1 is not None:
        with c1: st.plotly_chart(fig1, use_container_width=True)
    if fig2 is not None:
        with c2: st.plotly_chart(fig2, use_container_width=True)
    if fig1 is None and fig2 is None:
        st.info("Impossible d’afficher les jauges (Plotly indisponible).")

def rating_to_risk_level(rating: str) -> str:
    table = {"AAA":"Très faible","AA":"Très faible","A":"Faible","BBB":"Modéré",
             "BB":"Élevé","B":"Très élevé","CCC":"Critique","CC":"Critique","C":"Critique","D":"Défaut imminent"}
    return table.get(rating, "Inconnu")

def append_log(client_id: str, pd_val: float, label: int, threshold: float, inputs: dict, model_tag: str, rating: str, decision: str):
    new_row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "client_id": client_id,
        "prob_default": float(pd_val),
        "pred_label": int(label),
        "threshold": float(threshold),
        "inputs_json": json.dumps(inputs, ensure_ascii=False),
        "model_tag": model_tag,
        "rating": rating,
        "decision": decision,
    }
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        pd.DataFrame([new_row]).to_csv(f, header=not exists, index=False)

# ====== Explicabilité — helpers ======
def _parse_explain(exp_res: Dict[str, Any], inputs: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """Normalise /tools/explain -> DataFrame."""
    method = str(exp_res.get("method", "unknown"))
    items: List[Dict[str, Any]] = exp_res.get("items") or []
    raw_contrib = {}
    for it in items:
        feat = str(it.get("feature"))
        if "contribution" in it:
            raw_contrib[feat] = float(it.get("contribution", 0.0))
        elif "importance" in it:
            raw_contrib[feat] = float(it.get("importance", 0.0))
    for k in inputs.keys():
        raw_contrib.setdefault(k, 0.0)
    df = pd.DataFrame({"feature": list(raw_contrib.keys()), "contribution": list(raw_contrib.values())})
    df["value"] = df["feature"].map(lambda f: inputs.get(f))
    df["abs_contribution"] = df["contribution"].abs()
    df["sign"] = df["contribution"].map(lambda v: "+" if v >= 0 else "−")
    df = df.sort_values("abs_contribution", ascending=False).reset_index(drop=True)  # <-- fix
    df["rank"] = df.index + 1
    return df, method

def _waterfall_plot(df_top: pd.DataFrame, title: str = "Décomposition relative des contributions"):
    try:
        import plotly.graph_objects as go
    except Exception:
        return None
    top = df_top.copy().sort_values("contribution", ascending=True)
    measure = ["relative"] * len(top)
    x = [f"{r.feature}" for _, r in top.iterrows()]
    y = [float(r.contribution) for _, r in top.iterrows()]
    fig = go.Figure(go.Waterfall(orientation="v", measure=measure, x=x, text=[f"{v:+.3f}" for v in y], y=y, connector={"line":{"dash":"dot"}}))
    fig.update_layout(title=title, showlegend=False, height=420, margin=dict(l=10,r=10,t=50,b=20))
    return fig

def _download_bytes(df: pd.DataFrame, as_json: bool = False) -> bytes:
    if as_json:
        return df.to_json(orient="records", force_ascii=False).encode("utf-8")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")

# ================== MAIN ==================
def main():
    st.set_page_config(page_title="Prédiction Scoring", page_icon="📊", layout="wide")
    header()
    cfg = get_settings()

    st.markdown("---")
    left, right = st.columns([2,1])
    with left:
        inputs = input_form(lang="FR")
    with right:
        st.markdown("#### Paramètres décisionnels")
        threshold = st.slider("Seuil décisionnel interne (PD)", 0.01, 0.50, 0.10, 0.01)
        client_id = st.text_input("Identifiant client", "CLI-0001")
        use_api_explain = st.toggle("Explication avancée (SHAP si disponible)", value=True)
        run = st.button("⚡ Lancer la prédiction", type="primary", use_container_width=True)

    st.markdown("---")

    if run:
        payload = dict(inputs); payload["threshold"] = float(threshold)

        with st.expander("Diagnostics"):
            st.write("Mode prédiction :", "**API FastAPI** (aucun modèle local chargé)")
            st.json(payload)

        # 1) Prédiction via API
        try:
            res = predict_credit_api(payload)
        except ApiError as e:
            st.error(str(e)); return

        pd_val = float(res.get("pd", 0.5))
        score_1000 = int(res.get("score_1000", int(round((1 - pd_val) * 1000))))
        rating = str(res.get("rating", "N/A"))
        decision = str(res.get("decision", "N/A"))
        risk_level = rating_to_risk_level(rating)
        model_tag = str(res.get("model_tag", "pipeline_v1"))

        kpi_cards(pd_val, score_1000, rating, risk_level, decision)
        gauge_row(pd_val, score_1000, threshold)

        label_bin = 1 if pd_val >= threshold else 0
        append_log(client_id, pd_val, label_bin, threshold, inputs, model_tag, rating, decision)

        # 2) Explicabilité détaillée
        st.markdown("### Explicabilité détaillée")
        df_exp: pd.DataFrame | None = None
        method = "none"

        if use_api_explain:
            exp_res = explain_credit_api(dict(inputs), top_k=30)
            if exp_res and exp_res.get("ok"):
                df_exp, method = _parse_explain(exp_res, inputs)

        if df_exp is None:
            nar = heuristic_narrative(pd_val, threshold, pd.Series(inputs))
            tabs = st.tabs(["Client", "Analyste", "Admin", "Technique"])
            with tabs[0]: st.markdown(nar.client)
            with tabs[1]: st.markdown(nar.analyst)
            with tabs[2]: st.markdown(nar.admin)
            with tabs[3]:
                st.info("Aucune décomposition fine disponible depuis le modèle. Fallback heuristique affiché.")
        else:
            topN = st.slider("Nombre de facteurs à détailler (Top absolu)", 5, 30, 15, 1)
            df_top = df_exp.sort_values("abs_contribution", ascending=False).head(int(topN))

            cA, cB = st.columns([3,2])
            with cA:
                st.subheader("Bar chart — Importance absolue")
                try:
                    st.bar_chart(df_top.set_index("feature")["abs_contribution"])
                except Exception:
                    st.write(df_top[["feature","abs_contribution"]])

            with cB:
                st.subheader("Top positifs / négatifs")
                pos = df_top[df_top["contribution"] > 0].sort_values("contribution", ascending=False).head(5)
                neg = df_top[df_top["contribution"] < 0].sort_values("contribution", ascending=True).head(5)
                st.markdown("**Augmentent la PD (risque) :**")
                if len(pos):
                    for _, r in pos.iterrows():
                        st.write(f"- **{r.feature}** → +{r.contribution:.4f} (valeur : `{r.value}`)")
                else:
                    st.write("_aucun contributeur positif dominant_")
                st.markdown("**Réduisent la PD (protection) :**")
                if len(neg):
                    for _, r in neg.iterrows():
                        st.write(f"- **{r.feature}** → {r.contribution:.4f} (valeur : `{r.value}`)")
                else:
                    st.write("_aucun contributeur négatif dominant_")

            st.subheader("Waterfall — décomposition relative (top absolu)")
            fig = _waterfall_plot(df_top.head(10))
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waterfall indisponible (Plotly non installé).")

            try:
                contrib_series = pd.Series(df_exp.set_index("feature")["contribution"].to_dict(), dtype=float)
                nar = build_narrative_from_contrib(
                    pd_val=pd_val, threshold=threshold, contrib=contrib_series, X_row=pd.Series(inputs),
                )
                tabs = st.tabs(["Client", "Analyste", "Admin", "Technique"])
                with tabs[0]: st.markdown(nar.client)
                with tabs[1]: st.markdown(nar.analyst)
                with tabs[2]: st.markdown(nar.admin)
                with tabs[3]:
                    st.markdown(
                        "- Méthode d’explication : **{}**  \n"
                        "- Nombre de facteurs utilisés : **{}**  \n"
                        "- Seuil décisionnel interne : **{:.2%}**  \n"
                        "- PD prédite : **{:.2%}**"
                        .format("SHAP" if method=="shap" else "Fallback (coefficients/variance)",
                                len(df_exp), threshold, pd_val)
                    )
            except Exception:
                pass

        # 3) Export explications
        st.markdown("#### Export des explications")
        cdl1, cdl2 = st.columns(2)
        if use_api_explain and df_exp is not None:
            with cdl1:
                st.download_button(
                    "📥 Contributions (CSV)",
                    data=_download_bytes(df_exp, as_json=False),
                    file_name=f"explanations_{client_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with cdl2:
                st.download_button(
                    "📥 Contributions (JSON)",
                    data=_download_bytes(df_exp, as_json=True),
                    file_name=f"explanations_{client_id}.json",
                    mime="application/json",
                    use_container_width=True
                )

        # 4) Rapports — HTML & PDF
        st.markdown("### Génération de rapports (CA / Administrations)")
        logo_path = try_load_logo()
        ctx = build_report_context(
            client_id=client_id,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            pd_value=pd_val,
            threshold=threshold,
            inputs_json=json.dumps(inputs, ensure_ascii=False),
            model_tag=model_tag,
            logo_path=logo_path,
            primary_color=cfg.report_primary_color,
            footer_text=cfg.report_footer_text,
        )
        html = render_html_report(ctx)

        colH, colP = st.columns(2)
        with colH:
            st.download_button(
                "📄 Télécharger rapport (HTML structuré)",
                data=html.encode("utf-8"),
                file_name=f"report_{client_id}.html",
                mime="text/html",
                use_container_width=True,
            )
        with colP:
            pdf_bytes = build_pdf_bytes(
                ctx={
                    "client_id": client_id,
                    "timestamp": ctx["timestamp"],
                    "pd_value": pd_val,
                    "threshold": threshold,
                    "model_tag": model_tag,
                    "inputs": inputs,
                    "score_1000": score_1000,
                    "rating": rating,
                    "decision": decision,
                    "footer_text": cfg.report_footer_text,
                },
                df_exp=df_exp,
                logo_path=logo_path,
            )
            st.download_button(
                "🧾 Télécharger rapport (PDF corporate)",
                data=pdf_bytes,
                file_name=f"report_{client_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        st.markdown("#### Enregistrer dans /reports")
        try:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            c_save, c_open = st.columns([2,1])
            with c_save:
                if st.button("📂 Enregistrer HTML + PDF (horodaté)", use_container_width=True):
                    ts = datetime.now().isoformat(timespec="seconds").replace(":", "").replace(" ", "_")
                    html_path = os.path.join(REPORTS_DIR, f"report_{client_id}_{ts}.html")
                    pdf_path  = os.path.join(REPORTS_DIR, f"report_{client_id}_{ts}.pdf")
                    with open(html_path, "w", encoding="utf-8") as f: f.write(html)
                    pdf_bytes2 = build_pdf_bytes(
                        ctx={
                            "client_id": client_id,
                            "timestamp": ctx["timestamp"],
                            "pd_value": pd_val,
                            "threshold": threshold,
                            "model_tag": model_tag,
                            "inputs": inputs,
                            "score_1000": score_1000,
                            "rating": rating,
                            "decision": decision,
                            "footer_text": cfg.report_footer_text,
                        },
                        df_exp=df_exp,
                        logo_path=logo_path,
                    )
                    with open(pdf_path, "wb") as f: f.write(pdf_bytes2)
                    st.success(f"Rapports enregistrés : `{html_path}` et `{pdf_path}`")

            with c_open:
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

        with st.expander("Réponse brute de l'API /api/predict"):
            st.json(res)

        st.success("Prédiction effectuée **via l'API FastAPI**. Explicabilité détaillée & rapports (HTML/PDF) prêts.")

if __name__ == "__main__":
    main()
