"""PDF rendering of the same validated source context as the HTML report."""
from __future__ import annotations
import io
import json
from html import escape
from typing import Any, Dict, Optional
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from utils.report_generator import normalize_report_context, report_display_values


def build_pdf_bytes(ctx: Dict[str, Any], df_exp: Optional[pd.DataFrame], logo_path: Optional[str]) -> bytes:
    ctx = normalize_report_context(ctx)
    values = report_display_values(ctx)
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    styles = getSampleStyleSheet()
    flow = []
    def paragraph(value, style="Normal"):
        return Paragraph(escape(str(value)), styles[style])
    def table(rows, widths):
        item = Table([[paragraph(cell) for cell in row] for row in rows],
                     colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("GRID", (0,0), (-1,-1), .5, colors.grey),
                                  ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#eef5f4")),
                                  ("VALIGN", (0,0), (-1,-1), "TOP"),
                                  ("LEFTPADDING", (0,0), (-1,-1), 6),
                                  ("RIGHTPADDING", (0,0), (-1,-1), 6)]))
        return item
    if logo_path:
        try:
            width, height = ImageReader(logo_path).getSize()
            ratio = min(45*mm/width, 20*mm/height)
            flow.append(Image(logo_path, width=width*ratio, height=height*ratio))
        except Exception:
            flow.append(paragraph("Logo indisponible."))
    flow.append(paragraph("Rapport de scoring credit", "Title"))
    for key, label in (("client_id", "Client"), ("timestamp", "Date du dossier"), ("model_tag", "Modele")):
        flow.append(paragraph(f"{label} : {values[key]}"))
    flow.append(Spacer(1, 8))
    fields = [("pd_value", "PD"), ("score_1000", "Score (0-1000)"),
              ("rating", "Notation"), ("risk_level", "Niveau de risque"),
              ("decision", "Decision"), ("threshold", "Seuil utilise")]
    flow.append(table([["Indicateur", "Valeur conservee"], *[[label, values[key]] for key,label in fields]], [60*mm,120*mm]))
    flow.append(paragraph("Disponibilite des donnees", "Heading2"))
    for note in ctx["data_notes"] or ["Valeurs fournies par la source du dossier."]:
        flow.append(paragraph(note))
    flow.append(paragraph("Explicabilite", "Heading2"))
    # A dataframe alone is not proof of validated individual attribution.
    # In particular global_importance is never converted into signed contributions.
    flow.append(paragraph("Aucune explication individuelle validee n'est disponible pour ce dossier."))
    flow.append(paragraph("Une importance globale du modele ne constitue pas une contribution individuelle au risque du client."))
    flow.append(paragraph("Limites de validation", "Heading2"))
    flow.append(paragraph("L'horizon du defaut, la calibration, le backtesting et la conformite reglementaire ne sont pas etablis par ce rapport."))
    flow.append(paragraph("Variables d'entree archivees", "Heading2"))
    rows = [["Variable", "Valeur"]]
    for name,value in ctx["inputs"].items():
        rows.append([name, json.dumps(value, ensure_ascii=False, default=str) if isinstance(value,(dict,list)) else str(value)])
    flow.append(table(rows,[65*mm,115*mm]))
    flow.append(Spacer(1,8))
    flow.append(paragraph(ctx.get("footer_text") or "Sous reserve des politiques internes de risque."))
    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica",9)
        canvas.drawRightString(A4[0]-15*mm,10*mm,str(document.page))
        canvas.restoreState()
    doc.build(flow,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()
