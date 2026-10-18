# utils/report_generator.py
from __future__ import annotations
import json
import math
from html import escape
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List

# =============== Contexte ===============
@dataclass
class ReportContext:
    client_id: str
    timestamp: str
    pd_value: float
    threshold: float
    model_tag: str
    inputs_json: str  # chaîne JSON
    logo_path: Optional[str] = None
    primary_color: str = "#0F766E"
    footer_text: str = "Sous réserve des politiques internes de risque & conformité."
    rating: Optional[str] = None
    score_1000: Optional[int] = None
    decision: Optional[str] = None

def _infer_rating(pd_value: float) -> str:
    pv = max(0.0, min(1.0, float(pd_value)))
    if pv < 0.02: return "AAA"
    if pv < 0.04: return "AA"
    if pv < 0.07: return "A"
    if pv < 0.10: return "BBB"
    if pv < 0.15: return "BB"
    if pv < 0.25: return "B"
    if pv < 0.40: return "CCC"
    if pv < 0.60: return "CC"
    if pv < 0.80: return "C"
    return "D"

def _risk_level_from_rating(r: str) -> str:
    table = {
        "AAA":"Très faible","AA":"Très faible","A":"Faible","BBB":"Modéré",
        "BB":"Élevé","B":"Très élevé","CCC":"Critique","CC":"Critique","C":"Critique","D":"Défaut imminent"
    }
    return table.get((r or "").upper(), "Inconnu")

def _missing(value):
    return value is None or str(value).strip().lower() in {"", "nan", "none", "null", "nat", "n/a", "<na>"}


def _optional_text(value):
    return None if _missing(value) else str(value).strip()


def normalize_report_context(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Validate source values; do not choose a rating or decision policy."""
    result = dict(ctx)
    notes = list(result.get("data_notes") or [])
    value = result.get("pd_value", result.get("prob_default"))
    try:
        if isinstance(value, bool) or _missing(value):
            raise ValueError()
        probability = float(value)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError()
    except (ValueError, TypeError, OverflowError):
        raise ValueError("PD absente ou invalide : generation du rapport impossible.")
    result["pd_value"] = probability
    threshold = result.get("threshold")
    if _missing(threshold):
        result["threshold"] = None
        notes.append("Seuil historique non renseigne.")
    else:
        try:
            threshold = float(threshold)
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError()
            result["threshold"] = threshold
        except (TypeError, ValueError):
            result["threshold"] = None
            notes.append("Seuil historique invalide ; aucune valeur par defaut appliquee.")
    score = result.get("score_1000")
    if _missing(score):
        result["score_1000"] = int(round((1.0 - probability) * 1000))
        result["score_source"] = "derived_from_pd"
        notes.append("Score non archive : reconstitue avec la formule existante round(1000 * (1 - PD)).")
    else:
        numeric_score = float(score)
        if isinstance(score, bool) or not math.isfinite(numeric_score) or not numeric_score.is_integer() or not 0 <= numeric_score <= 1000:
            raise ValueError("Score enregistre invalide.")
        result["score_1000"] = int(numeric_score)
        result.setdefault("score_source", "recorded")
    rating = _optional_text(result.get("rating"))
    if rating and rating.upper() not in {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"}:
        notes.append("Notation historique invalide ; aucune grille de remplacement appliquee.")
        rating = None
    if rating is None:
        notes.append("Notation non renseignee ; aucun rating recalcule.")
    result["rating"] = rating
    result["decision"] = _optional_text(result.get("decision"))
    if result["decision"] is None:
        notes.append("Decision historique non renseignee ; aucune politique choisie.")
    risk = _optional_text(result.get("risk_level"))
    if risk is None and rating:
        risk = _risk_level_from_rating(rating)
        notes.append("Niveau de risque non archive : correspondance existante de la notation conservee.")
    result["risk_level"] = risk
    for name in ("client_id", "timestamp", "model_tag"):
        result[name] = _optional_text(result.get(name))
    inputs = result.get("inputs")
    if not isinstance(inputs, dict):
        try:
            inputs = json.loads(result.get("inputs_json") or "{}")
            if not isinstance(inputs, dict):
                raise ValueError()
        except (ValueError, TypeError):
            inputs = {}
            notes.append("Variables d'entree historiques illisibles.")
    result["inputs"] = inputs
    result["inputs_json"] = json.dumps(inputs, ensure_ascii=False, default=str)
    result["data_notes"] = list(dict.fromkeys(notes))
    return result


def build_report_context(
    client_id: str, timestamp: str, pd_value: float, threshold: float,
    inputs_json: str, model_tag: str, logo_path: Optional[str] = None,
    primary_color: str = "#0F766E",
    footer_text: str = "Sous reserve des politiques internes de risque.",
    rating: Optional[str] = None, score_1000: Optional[int] = None,
    decision: Optional[str] = None, risk_level: Optional[str] = None,
    model_id: Optional[str] = None, input_id: Optional[str] = None,
) -> Dict[str, Any]:
    return normalize_report_context(dict(
        client_id=client_id, timestamp=timestamp, pd_value=pd_value, threshold=threshold,
        inputs_json=inputs_json, model_tag=model_tag, logo_path=logo_path,
        primary_color=primary_color, footer_text=footer_text, rating=rating,
        score_1000=score_1000, decision=decision, risk_level=risk_level,
        model_id=model_id, input_id=input_id,
    ))


def report_display_values(ctx):
    def text(value):
        return "Non renseigne" if _missing(value) else str(value)
    return {
        "client_id": text(ctx.get("client_id")),
        "timestamp": text(ctx.get("timestamp")),
        "model_tag": text(ctx.get("model_tag")),
        "pd_value": f"{ctx['pd_value']:.2%}",
        "score_1000": str(ctx["score_1000"]),
        "rating": text(ctx.get("rating")),
        "risk_level": text(ctx.get("risk_level")),
        "decision": text(ctx.get("decision")),
        "threshold": "Non renseigne" if ctx.get("threshold") is None else f"{ctx['threshold']:.2%}",
    }


def render_html_report(ctx: Dict[str, Any]) -> str:
    ctx = normalize_report_context(ctx)
    values = {key: escape(value) for key, value in report_display_values(ctx).items()}
    labels = [("pd_value", "PD"), ("score_1000", "Score (0-1000)"),
              ("rating", "Notation"), ("risk_level", "Niveau de risque"),
              ("decision", "Decision"), ("threshold", "Seuil utilise")]
    rows = "".join(f'<tr><th>{label}</th><td data-field="{key}">{values[key]}</td></tr>' for key, label in labels)
    notes = "".join(f"<li>{escape(note)}</li>" for note in ctx["data_notes"])
    inputs = escape(json.dumps(ctx["inputs"], ensure_ascii=False, indent=2, default=str))
    return f"""<!DOCTYPE html>
<html lang="fr"><head><meta charset="utf-8"><title>Rapport credit</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#0b1220}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:9px;text-align:left}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}h1,h2{{color:#0F766E}}</style>
</head><body><h1>Rapport de scoring credit</h1>
<p>Client : <b>{values['client_id']}</b> | Date du dossier : {values['timestamp']}<br>Modele : {values['model_tag']}</p>
<h2>Resultats conserves</h2><table>{rows}</table>
<h2>Disponibilite des donnees</h2><ul>{notes or '<li>Valeurs fournies par la source du dossier.</li>'}</ul>
<h2>Explicabilite</h2><p>Aucune explication individuelle validee n'est disponible pour ce dossier.
Une importance globale du modele ne constitue pas une contribution individuelle au risque du client.</p>
<h2>Limites de validation</h2><p>L'horizon du defaut, la calibration, le backtesting et la conformite reglementaire ne sont pas etablis par ce rapport.</p>
<h2>Variables d'entree archivees</h2><pre>{inputs}</pre>
<p>{escape(str(ctx.get('footer_text') or ''))}</p></body></html>"""


# =============== HTML -> PDF (basique via ReportLab) ===============
def html_to_pdf_bytes(html: str, title: str = "Rapport crédit", logo_path: Optional[str] = None) -> bytes:
    """
    Convertisseur léger (HTML limité) -> PDF avec ReportLab.
    Supporte : h1/h2/h3, p, ul/li, pre, tables simples (th/td).
    Pour des PDFs premium utiliser utils.reporting_pdf.build_pdf_bytes.
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, Preformatted
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title=title, leftMargin=36, rightMargin=36, topMargin=42, bottomMargin=36)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", fontSize=18, leading=22, spaceAfter=8, spaceBefore=10, textColor=colors.HexColor("#0b1220")))
    styles.add(ParagraphStyle(name="H2", fontSize=15, leading=19, spaceAfter=6, spaceBefore=10, textColor=colors.HexColor("#0b1220")))
    styles.add(ParagraphStyle(name="H3", fontSize=12.5, leading=16, spaceAfter=4, spaceBefore=8, textColor=colors.HexColor("#334155")))
    styles.add(ParagraphStyle(name="P", fontSize=10.5, leading=14))
    styles.add(ParagraphStyle(name="SMALL", fontSize=8.5, leading=11, textColor=colors.HexColor("#6b7280")))

    flow: List[Any] = []

    # En-tête
    if logo_path:
        try:
            flow.append(Image(logo_path, width=120, height=40))
            flow.append(Spacer(1, 6))
        except Exception:
            pass
    flow.append(Paragraph(title, styles["H1"]))
    flow.append(Spacer(1, 6))

    # Parsing ultra-simple des sections du HTML généré par render_html_report
    # On va chercher le tableau principal (Résultats chiffrés)
    try:
        # Récupérer les lignes <tr><th>..</th><td>..</td>...
        import re
        table_rows = re.findall(r"<tr>(.*?)</tr>", html, flags=re.S | re.I)
        if table_rows:
            data = []
            for row in table_rows:  # retain every result row
                cells_th = re.findall(r"<th>(.*?)</th>", row, flags=re.S | re.I)
                cells_td = re.findall(r"<td\b[^>]*>(.*?)</td>", row, flags=re.S | re.I)
                # alterner TH/TD
                row_data: List[str] = []
                for a, b in zip(cells_th, cells_td):
                    row_data.extend([_strip_html(a), _strip_html(b)])
                data.append(row_data)
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e5e7eb")),
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#f8fafc")),
                ('FONTNAME', (0,0), (-1,-1), "Helvetica"),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            flow.append(t)
            flow.append(Spacer(1, 8))
    except Exception:
        pass

    # Paragraphe “Résumé exécutif” extrait rapidement
    try:
        start = html.lower().find("<h2>1. résumé exécutif</h2>")
        if start != -1:
            pstart = html.find("<p>", start)
            pend = html.find("</p>", pstart)
            if pstart != -1 and pend != -1:
                flow.append(Paragraph(_strip_html(html[pstart:pend+4]), styles["P"]))
                flow.append(Spacer(1, 8))
    except Exception:
        pass

    # Annexes — Entrées JSON
    try:
        pre_start = html.lower().find("<pre>")
        pre_end = html.lower().find("</pre>", pre_start)
        if pre_start != -1 and pre_end != -1:
            code = html[pre_start+5:pre_end]
            flow.append(Paragraph("Entrées du client (JSON)", styles["H3"]))
            flow.append(Preformatted(code, styles["P"]))
            flow.append(Spacer(1, 6))
    except Exception:
        pass

    # Pied
    try:
        foot = re_search_group(html, r'<div class="footer">(.*?)</div>')
        if foot:
            flow.append(Paragraph(_strip_html(foot), styles["SMALL"]))
    except Exception:
        pass

    doc.build(flow)
    return buf.getvalue()

def _strip_html(s: str) -> str:
    import re
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<.*?>", "", s)
    return s.strip()

def re_search_group(text: str, pattern: str) -> Optional[str]:
    import re
    m = re.search(pattern, text, flags=re.S | re.I)
    return m.group(1).strip() if m else None
