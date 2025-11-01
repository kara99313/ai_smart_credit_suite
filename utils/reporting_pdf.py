# utils/reporting_pdf.py
from __future__ import annotations
import io
from typing import Dict, Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = A4

def _fmt_pct(x: float) -> str:
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "N/A"

def _fmt_int(x) -> str:
    try:
        return f"{int(x)}"
    except Exception:
        return "N/A"

def _centered_title(flow, title: str):
    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        "CenterH1", parent=styles["Heading1"], alignment=1, spaceAfter=6
    )
    flow.append(Paragraph(title, style))
    flow.append(Spacer(1, 6))

def _kpi_table(ctx: Dict[str, Any]) -> Table:
    pdv = ctx.get("pd_value", 0.0)
    score = ctx.get("score_1000", None)
    rating = ctx.get("rating", "N/A")
    thr = ctx.get("threshold", 0.1)
    decision = ctx.get("decision", "N/A")
    risk_level = ctx.get("risk_level", None)

    left = [
        ["PD", _fmt_pct(pdv)],
        ["Notation", str(rating)],
        ["Décision", str(decision)],
    ]
    right = [
        ["Score (0–1000)", _fmt_int(score if score is not None else (1.0 - float(pdv)) * 1000)],
        ["Seuil interne", _fmt_pct(thr)],
        ["Niveau de risque", str(risk_level) if risk_level else ""],
    ]

    data = [
        [left[0][0], left[0][1], right[0][0], right[0][1]],
        [left[1][0], left[1][1], right[1][0], right[1][1]],
        [left[2][0], left[2][1], right[2][0], right[2][1]],
    ]
    t = Table(data, colWidths=[45*mm, 55*mm, 45*mm, 55*mm])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.6, colors.black),
        ("BACKGROUND", (0,0), (-1,-1), colors.whitesmoke),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 11),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("ALIGN", (3,0), (3,-1), "RIGHT"),
    ]))
    return t

def _header_block(flow, ctx: Dict[str, Any], logo_path: Optional[str]):
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=11)

    # Logo (centré) si dispo
    if logo_path:
        try:
            flow.append(Image(logo_path, width=60*mm, height=60*mm))
            flow.append(Spacer(1, 4))
        except Exception:
            pass

    _centered_title(flow, "Rapport de Scoring Crédit")

    info = f"<b>Client</b> : {ctx.get('client_id','N/A')} &nbsp;&nbsp; | &nbsp;&nbsp; <b>Date</b> : {ctx.get('timestamp','')}"
    flow.append(Paragraph(info, small))
    flow.append(Spacer(1, 4))
    flow.append(Paragraph(f"<b>Modèle</b> : {ctx.get('model_tag','')}", small))
    flow.append(Spacer(1, 8))

    flow.append(_kpi_table(ctx))
    flow.append(Spacer(1, 12))

def _bar_chart_img(df_exp: pd.DataFrame) -> Image | None:
    try:
        top = df_exp.sort_values("abs_contribution", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=150)
        ax.barh(top["feature"][::-1], top["abs_contribution"][::-1])
        ax.set_xlabel("|Contribution| (relative)")
        ax.set_title("Principaux facteurs (importance absolue)", loc="left")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        img = Image(buf, width=170*mm, height=85*mm)
        return img
    except Exception:
        return None

def _contrib_table(df_exp: pd.DataFrame) -> Table:
    sub = df_exp.sort_values("abs_contribution", ascending=False).head(10).copy()
    sub["Contribution_str"] = sub["contribution"].map(lambda v: f"{v:+.4f}")
    data = [["Variable","Valeur","Contribution","|Contribution|"]]
    for _, r in sub.iterrows():
        data.append([str(r["feature"]), str(r["value"]), r["Contribution_str"], f"{float(r['abs_contribution']):.4f}"])
    t = Table(data, colWidths=[60*mm, 35*mm, 35*mm, 35*mm])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (1,1), (-1,-1), "RIGHT"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
    ]))
    return t

def _inputs_table(inputs: Dict[str, Any]) -> Table:
    data = [["Variable", "Valeur"]]
    for k, v in inputs.items():
        data.append([str(k), str(v)])
    t = Table(data, colWidths=[70*mm, 95*mm])
    t.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
    ]))
    return t

def _footer(canvas: Canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawRightString(PAGE_W - 15*mm, 10*mm, f"{doc.page}")
    canvas.restoreState()

def build_pdf_bytes(ctx: Dict[str, Any], df_exp: Optional[pd.DataFrame], logo_path: Optional[str]) -> bytes:
    """
    Génère un PDF bytes en suivant le format précédent : 
    - En-tête centré (logo + titre) + infos client + tableau KPI (incluant Niveau de risque)
    - Section '2) Résultats & Explicabilité' : bar chart + tableau Top facteurs
    - Section 'Annexes / Variables d’entrée' : tableau des inputs
    - Pagination simple, pas de page vide ni placeholder TOC.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], spaceBefore=6, spaceAfter=6)

    flow = []

    # Page 1 : en-tête + KPI
    _header_block(flow, ctx, logo_path)

    # 2) Résultats & Explicabilité
    flow.append(Paragraph("2) Résultats & Explicabilité", h2))

    if df_exp is not None and isinstance(df_exp, pd.DataFrame) and not df_exp.empty:
        img = _bar_chart_img(df_exp)
        if img: 
            flow.append(img)
            flow.append(Spacer(1, 6))
        flow.append(_contrib_table(df_exp))
        flow.append(Spacer(1, 10))
    else:
        flow.append(Paragraph("Aucune explication détaillée disponible (SHAP/coefficients non renvoyés).", styles["Normal"]))
        flow.append(Spacer(1, 10))

    # Commentaires/recommandations/conclusion (courts, intégrés, sans page vide)
    if ctx.get("pd_value") is not None:
        pdv = float(ctx["pd_value"])
        thr = float(ctx.get("threshold", 0.1))
        rating = str(ctx.get("rating", "N/A"))
        decision = str(ctx.get("decision", "N/A"))
        risk_text = str(ctx.get("risk_level", "")) if ctx.get("risk_level") else ""

        flow.append(Paragraph("<b>Commentaires analytiques</b>", styles["Heading3"]))
        flow.append(Paragraph(
            f"PD prédite : <b>{_fmt_pct(pdv)}</b> (seuil interne : <b>{_fmt_pct(thr)}</b>) ; "
            f"Score : <b>{_fmt_int(ctx.get('score_1000',''))}</b>/1000 ; "
            f"Notation : <b>{rating}</b> ; Niveau de risque : <b>{risk_text}</b> ; "
            f"Décision : <b>{decision}</b>.",
            styles["Normal"]
        ))
        flow.append(Spacer(1, 6))

        flow.append(Paragraph("<b>Recommandations</b>", styles["Heading3"]))
        flow.append(Paragraph(
            "• Ajuster le montant et/ou la durée si la relation est stratégique. "
            "• Renforcer la collecte documentaire (revenus, stabilité d’emploi). "
            "• Coaching budgétaire pour améliorer le DTI avant tout engagement. "
            "• Journaliser et archiver les explications pour conformité (BCBS/EBA/IFRS 9/AI Act).",
            styles["Normal"]
        ))
        flow.append(Spacer(1, 6))

        flow.append(Paragraph("<b>Conclusion</b>", styles["Heading3"]))
        flow.append(Paragraph(
            "La décision proposée est cohérente au vu des indicateurs et facteurs explicatifs. "
            "Un réexamen demeure possible si des éléments probants atténuent le risque "
            "(revenus récurrents prouvés, garanties, co-emprunteur solvable).",
            styles["Normal"]
        ))
        flow.append(Spacer(1, 8))

    # Annexes : Inputs
    flow.append(Paragraph("Annexes / Variables d’entrée", h2))
    inputs = ctx.get("inputs") or {}
    flow.append(_inputs_table(inputs))
    flow.append(Spacer(1, 4))
    ft = str(ctx.get("footer_text") or "Sous réserve des politiques internes de risque & conformité.")
    flow.append(Paragraph(ft, ParagraphStyle("foot", parent=styles["Normal"], fontSize=9, textColor=colors.grey)))

    doc.build(flow, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
