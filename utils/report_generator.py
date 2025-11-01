# utils/report_generator.py
from __future__ import annotations
import json
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

def build_report_context(
    client_id: str,
    timestamp: str,
    pd_value: float,
    threshold: float,
    inputs_json: str,
    model_tag: str,
    logo_path: Optional[str] = None,
    primary_color: str = "#0F766E",
    footer_text: str = "Sous réserve des politiques internes de risque & conformité.",
    rating: Optional[str] = None,
    score_1000: Optional[int] = None,
    decision: Optional[str] = None,
) -> Dict[str, Any]:
    """Construit un dict propre utilisé par HTML & PDF."""
    if score_1000 is None:
        score_1000 = int(round((1.0 - max(0.0, min(pd_value, 1.0))) * 1000))
    if not rating:
        rating = _infer_rating(pd_value)
    if not decision:
        decision = "ACCEPT" if pd_value < threshold else "REVUE / REFUS"
    return asdict(ReportContext(
        client_id=client_id,
        timestamp=timestamp,
        pd_value=float(pd_value),
        threshold=float(threshold),
        model_tag=model_tag,
        inputs_json=inputs_json,
        logo_path=logo_path,
        primary_color=primary_color,
        footer_text=footer_text,
        rating=rating,
        score_1000=int(score_1000),
        decision=decision,
    ))

# =============== HTML ===============
def _pretty_json(js: str) -> str:
    try:
        obj = json.loads(js or "{}")
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return js or "{}"

def render_html_report(ctx: Dict[str, Any]) -> str:
    """
    Rend un rapport HTML (avec niveau de risque ajouté dans le tableau).
    Plan : Couverture → Résumé exécutif → Résultats → Explication → Conformité → Annexes → Conclusion.
    """
    color = ctx.get("primary_color", "#0F766E")
    logo = ctx.get("logo_path") or ""
    client_id = ctx.get("client_id", "N/A")
    ts = ctx.get("timestamp", "")
    pdv = float(ctx.get("pd_value", 0.0))
    thr = float(ctx.get("threshold", 0.1))
    rating = ctx.get("rating") or _infer_rating(pdv)
    score_1000 = int(ctx.get("score_1000", 0))
    decision = ctx.get("decision") or ("ACCEPT" if pdv < thr else "REVUE / REFUS")
    risk_level = _risk_level_from_rating(rating)
    inputs_pretty = _pretty_json(ctx.get("inputs_json", "{}"))
    model_tag = ctx.get("model_tag", "pipeline_v1")

    html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<title>Rapport crédit — {client_id}</title>
<style>
  body {{ font-family: Arial, sans-serif; color:#0b1220; margin:28px; }}
  .cover {{
    border:2px solid {color}; border-radius:12px; padding:18px; margin-bottom:20px;
    background: linear-gradient(180deg, {color}1A, #ffffff 60%);
  }}
  .brand {{ display:flex; align-items:center; gap:12px; }}
  .brand img {{ height:52px; }}
  h1 {{ margin:0; font-size:24px; }}
  h2 {{ margin-top:28px; font-size:20px; border-bottom:2px solid {color}; padding-bottom:6px; }}
  h3 {{ margin-top:16px; font-size:16px; color:#334155; }}
  .meta {{ color:#334155; }}
  table {{ border-collapse: collapse; width:100%; }}
  th, td {{ border:1px solid #e5e7eb; padding:10px; text-align:left; }}
  th {{ background:#f8fafc; }}
  .kpi {{ background:#0b1220; color:#e8edf6; border-radius:12px; padding:12px; margin:6px 0; }}
  .footer {{ margin-top:24px; color:#6b7280; font-size:12px; }}
  pre {{ background:#f7fafc; border:1px solid #e5e7eb; padding:12px; border-radius:10px; overflow:auto; }}
  .toc li {{ margin:4px 0; }}
</style>
</head>
<body>

<div class="cover">
  <div class="brand">
    {('<img src="'+logo+'" alt="logo">') if logo else ''}
    <div>
      <h1>Rapport de Scoring Crédit — {client_id}</h1>
      <div class="meta">Date : <b>{ts}</b> &nbsp;|&nbsp; Modèle : <b>{model_tag}</b></div>
    </div>
  </div>
</div>

<h2>Sommaire</h2>
<ol class="toc">
  <li>Résumé exécutif</li>
  <li>Résultats chiffrés</li>
  <li>Explication & facteurs clés</li>
  <li>Conformité & gouvernance</li>
  <li>Annexes (entrées & trace)</li>
  <li>Conclusion</li>
</ol>

<h2>1. Résumé exécutif</h2>
<p>Ce rapport présente la décision de crédit pour le client <b>{client_id}</b>, basée sur le modèle <b>{model_tag}</b>.
La probabilité de défaut (PD) estimée est de <b>{pdv:.2%}</b> pour un seuil interne de <b>{thr:.2%}</b>.
La notation calculée est <b>{rating}</b> (niveau de risque : <b>{risk_level}</b>) pour un score de <b>{score_1000}/1000</b>.
Décision proposée : <b>{decision}</b>.</p>

<h2>2. Résultats chiffrés</h2>
<table>
  <tr><th>PD</th><td>{pdv:.2%}</td><th>Score (0–1000)</th><td>{score_1000}</td></tr>
  <tr><th>Notation</th><td>{rating}</td><th>Seuil interne</th><td>{thr:.2%}</td></tr>
  <tr><th>Niveau de risque</th><td>{risk_level}</td><th>Décision</th><td>{decision}</td></tr>
</table>

<div class="kpi">Interprétation rapide : une PD de {pdv:.2%} au-dessus d’un seuil de {thr:.2%} indique un profil à surveiller.
La notation {rating} traduit un niveau de risque « {risk_level} ». La décision doit être confirmée au regard des politiques internes.</div>

<h2>3. Explication & facteurs clés</h2>
<p>Les facteurs explicatifs (SHAP/coefficients selon disponibilité) sont détaillés dans la page “Prédiction” de l’application.
Ils permettent d’identifier les variables qui augmentent ou réduisent le risque. Un suivi spécifique peut être recommandé
sur les variables les plus contributives.</p>

<h2>4. Conformité & gouvernance</h2>
<ul>
  <li><b>IFRS 9 :</b> calcul PD compatible ECL, documentation de la méthode, traçabilité.</li>
  <li><b>Bâle/BCBS 239 :</b> qualité des données, audit trail, agrégation et reporting.</li>
  <li><b>Éthique & IA Act :</b> transparence des modèles, explicabilité, non-discrimination.</li>
</ul>

<h2>5. Annexes — Entrées & trace</h2>
<h3>Entrées du client (JSON)</h3>
<pre>{inputs_pretty}</pre>

<h2>6. Conclusion</h2>
<p>Au vu des éléments ci-dessus, la décision proposée est : <b>{decision}</b>.
Les comités compétents peuvent exiger des justificatifs complémentaires (revenus, garanties, historique) avant validation finale.</p>

<div class="footer">{ctx.get("footer_text","")}</div>
</body>
</html>
""".strip()
    return html

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
            for row in table_rows[:3]:  # notre tableau principal a 3 lignes
                cells_th = re.findall(r"<th>(.*?)</th>", row, flags=re.S | re.I)
                cells_td = re.findall(r"<td>(.*?)</td>", row, flags=re.S | re.I)
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
