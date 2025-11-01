# utils/context_bridge.py
from __future__ import annotations
import csv, json, os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

LOG_PATH = os.path.join("data", "predictions_log.csv")

WANTED_KEYS = [
    "DTIRatio","TrustScorePsychometric","HouseholdSize","NumCreditLines",
    "Income","CommunityGroupMember","HasMortgage","MonthsEmployed",
    "HasSocialAid","MobileMoneyTransactions","Age","InterestRate",
    "LoanTerm","LoanAmount","InformalIncome"
]

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _shorten(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else (s[:n] + "…")

def _iter_rows() -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_PATH):
        return []
    rows: List[Dict[str, Any]] = []
    with open(LOG_PATH, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def _read_last_row() -> Optional[Dict[str, Any]]:
    rows = _iter_rows()
    return rows[-1] if rows else None

def _read_by_client(client_id: str) -> Optional[Dict[str, Any]]:
    hit = None
    for row in _iter_rows():
        if (row.get("client_id") or "").strip() == client_id.strip():
            hit = row  # garde la dernière occurrence
    return hit

def load_client_record(client_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Renvoie la ligne du log pour client_id (ou la dernière ligne si None)."""
    return _read_by_client(client_id) if client_id else _read_last_row()

def build_client_context(client_id: Optional[str] = None, max_chars: int = 900) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Construit un bloc texte concis injecté dans le RAG/agent.
    Retourne (bloc_contexte, dict_record) ; bloc vide si pas de log.
    """
    row = load_client_record(client_id)
    if not row:
        return ("", None)

    cid = row.get("client_id", "N/A")
    ts  = row.get("timestamp") or datetime.utcnow().isoformat()
    pdv = _safe_float(row.get("prob_default", row.get("pd", 0.0)))
    thr = _safe_float(row.get("threshold", 0.1))
    rating   = row.get("rating", row.get("pred_label","N/A"))
    decision = row.get("decision", "ACCEPT" if pdv < thr else "REVIEW/REJECT")
    model    = row.get("model_tag","pipeline_v1")

    # inputs_json -> dict -> on ne garde que quelques clés utiles
    raw_inputs = row.get("inputs_json", "{}")
    try:
        inp = json.loads(raw_inputs) if raw_inputs else {}
    except Exception:
        inp = {}
    kept = {k: inp.get(k) for k in WANTED_KEYS if k in inp}

    head = (
        f"[C] CONTEXTE CLIENT COURANT\n"
        f"- client_id: {cid}\n"
        f"- date: {ts}\n"
        f"- modèle: {model}\n"
        f"- PD: {pdv:.2%} | seuil: {thr:.2%} | rating: {rating} | décision: {decision}\n"
    )
    var_lines = [f"  · {k}: {v}" for k, v in kept.items()]
    vars_block = "Variables clés:\n" + "\n".join(var_lines) if var_lines else "Variables clés: n/a"

    block = _shorten(head + vars_block, max_chars)
    return (block, row)

def build_logs_summary(n: int = 10, max_chars: int = 900) -> str:
    """Résumé des N dernières prédictions (pour RAG / outils agent)."""
    rows = _iter_rows()
    if not rows:
        return ""
    last = rows[-n:] if len(rows) >= n else rows
    lines = []
    for r in reversed(last):
        pdv = _safe_float(r.get("prob_default", 0.0))
        score = round((1 - pdv) * 1000)
        thr = _safe_float(r.get("threshold", 0.1))
        decision = "ACCEPT" if pdv < thr else "REVIEW/REJECT"
        lines.append(
            f"- {r.get('timestamp','')} | {r.get('client_id','N/A')} | PD={pdv:.2%} | "
            f"Score={score}/1000 | Seuil={thr:.2%} | Décision={decision}"
        )
    txt = "[HISTORY] Dernières prédictions:\n" + "\n".join(lines)
    return _shorten(txt, max_chars)
