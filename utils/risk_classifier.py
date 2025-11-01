# utils/risk_classifier.py
def pd_to_risk_level(pd: float) -> str:
    if pd < 0.05: return "Très faible"
    if pd < 0.10: return "Faible"
    if pd < 0.20: return "Modéré"
    if pd < 0.35: return "Élevé"
    return "Très élevé"

def pd_to_grade(pd: float) -> str:
    # Exemple d’échelle simple (à adapter à ta politique)
    if pd < 0.02: return "AAA"
    if pd < 0.05: return "AA"
    if pd < 0.10: return "A"
    if pd < 0.15: return "BBB"
    if pd < 0.20: return "BB"
    if pd < 0.30: return "B"
    if pd < 0.40: return "CCC"
    if pd < 0.60: return "CC"
    return "C"

def pd_to_score(pd: float) -> int:
    # Score 300–900 (style crédit) mappé sur PD (0→900 ; 1→300)
    pd = max(0.0, min(1.0, pd))
    return int(round(900 - 600*pd))
