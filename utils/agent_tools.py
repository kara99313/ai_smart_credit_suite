# utils/agent_tools.py
# -*- coding: utf-8 -*-
"""
Outils métiers exposés aux agents IA.
Chaque fonction ici est "agent-safe" (retourne toujours une chaîne UTF-8).
Compatible LangChain 0.3.x et Groq / OpenAI / Ollama.
"""
from __future__ import annotations
import os, sys, json, base64
from utils.llm_runtime import explain
from utils.api_client import _validate_prediction_response
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple

# =========================
# Sécurité & encodage
# =========================
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _k in ("LC_ALL", "LANG", "LANGUAGE"):
    os.environ.setdefault(_k, "C.UTF-8")
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

def _utf8(s) -> str:
    """Force un str UTF-8 propre (pas d’erreur d’encodage)."""
    try:
        if isinstance(s, bytes):
            try:
                return s.decode("utf-8", errors="ignore")
            except Exception:
                return s.decode("latin1", errors="ignore")
        return str(s)
    except Exception:
        try:
            return repr(s)
        except Exception:
            return "<unrepresentable>"

# =========================
# Imports applicatifs
# =========================
from utils.api_client import predict_credit_api, explain_credit_api
from utils.rag_utils import build_or_load_faiss_docs, build_or_load_faiss_hybrid
from utils.web_search import search_web, build_web_context
from utils.context_bridge import build_client_context, load_client_record
from utils.reporting_pdf import build_pdf_bytes
from utils.report_generator import build_report_context  # conservé pour compat éventuelle

# =========================
# Helpers
# =========================
def _ctx_from_last_log() -> dict:
    """Récupère le dernier enregistrement du journal prédictions comme contexte par défaut."""
    row = load_client_record(None)
    if not row:
        return {}
    return {
        "client_id": row.get("client_id", "N/A"),
        "timestamp": row.get("timestamp", ""),
        "pd_value": float(row.get("prob_default", 0.0) or 0.0),
        "threshold": float(row.get("threshold", 0.1) or 0.1),
        "model_tag": row.get("model_tag", "pipeline_v1"),
        "inputs_json": row.get("inputs_json", "{}"),
        "rating": row.get("rating"),
        "decision": row.get("decision"),
    }

def _infer_rating(pd_value: float) -> str:
    """Infère une notation simple si absente (cohérente PDF/HTML)."""
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

# =========================
# OUTILS EXPOSÉS
# =========================
def predict_credit_tool(payload_json: str) -> str:
    """Prédit PD/score/rating/décision via /api/predict (backend FastAPI)."""
    try:
        payload: Dict[str, Any] = json.loads(payload_json or "{}")
    except Exception:
        return _utf8("Format JSON invalide pour predict_credit (vérifie les guillemets).")
    try:
        if not isinstance(payload, dict) or not payload:
            return "ERROR: Objet JSON client non vide requis."
        from utils.credit_agent_contract import payload_error
        error = payload_error(payload)
        if error:
            return error
        res = _validate_prediction_response(predict_credit_api(payload, timeout=30.0))
        pdv = float(res.get("pd", 0.0))
        score = int(res.get("score_1000", 0))
        rating = str(res.get("rating", "N/A"))
        decision = str(res.get("decision", "N/A"))
        return _utf8(
            "Résultat de prédiction :\n"
            f"• Probabilité de défaut (PD) = {pdv:.2%}\n"
            f"• Score = {score}/1000\n"
            f"• Notation = {rating}\n"
            f"• Décision = {decision}"
        )
    except Exception as e:
        return _utf8(f"❌ Échec d'appel /api/predict : {explain(e)}")

def rag_retrieve_tool(query: str, mode: str = "docs", k: int = 3, max_chars: int = 500) -> str:
    """Recherche sémantique interne (RAG) dans FAISS/Hybride."""
    try:
        if (mode or "").lower().startswith("doc"):
            vs, _ = build_or_load_faiss_docs(rebuild=False)
        else:
            vs, _ = build_or_load_faiss_hybrid(rebuild=False)
        if not vs:
            return _utf8("Aucun index RAG disponible (vérifie docs_rag/ ou data/predictions_log.csv).")
        hits = vs.similarity_search(query, k=int(k))
        if not hits:
            return _utf8("Aucun passage pertinent trouvé dans les documents internes.")
        parts = []
        for i, d in enumerate(hits, 1):
            src = _utf8(d.metadata.get("source", "inconnu"))
            page = d.metadata.get("page")
            page_txt = f" (page {page})" if page is not None else ""
            txt = _utf8(d.page_content or "").replace("\n", " ")
            if len(txt) > max_chars:
                txt = txt[:max_chars] + "…"
            parts.append(f"[D{i}] {src}{page_txt}\n{txt}")
        return _utf8("\n\n".join(parts))
    except Exception as e:
        return _utf8(f"⚠️ RAG indisponible : {explain(e)}")

def tool_web_search(query: str, k: int = 3, fetch: bool = False, max_chars: int = 900) -> str:
    """Recherche Web (DuckDuckGo)."""
    try:
        hits = search_web(query, max_results=int(k))
        if not hits:
            return _utf8("Aucun résultat Web trouvé.")
        ctx = build_web_context(hits, max_chars_total=max_chars, fetch_full=bool(fetch))
        return _utf8(ctx or "Aucun extrait Web exploitable.")
    except Exception as e:
        return _utf8(f"⚠️ Recherche Web indisponible : {explain(e)}")

def generate_report_tool(ctx_json: str | None = None, max_chars: int = 1200) -> str:
    """Rapport HTML complet depuis les archives ; ctx_json choisit client_id.

    Les valeurs de scoring proposees par le modele ne sont jamais utilisees.
    max_chars reste accepte pour compatibilite ; aucun HTML tronque.
    """
    try:
        request = json.loads(ctx_json) if ctx_json else {}
        if not isinstance(request, dict):
            raise ValueError()
        row = load_client_record(request.get("client_id") or None)
        if row is None:
            return "ERROR: Aucun resultat archive pour ce client ; rapport indisponible."
        from utils.report_generator import render_html_report
        ctx = dict(row, pd_value=row.get("prob_default"))
        return render_html_report(ctx)
    except Exception:
        return "ERROR: Contexte archive absent ou invalide ; rapport indisponible."


def now_tool(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Renvoie l'heure UTC actuelle formatée."""
    try:
        return _utf8(datetime.now(timezone.utc).strftime(fmt))
    except Exception:
        return _utf8(datetime.now(timezone.utc).isoformat())

def client_context_tool(client_id: str = "", max_chars: int = 700) -> str:
    """Contexte court du client (depuis predictions_log.csv)."""
    block, _ = build_client_context(client_id or None, max_chars=max_chars)
    return block or "Aucun contexte client disponible."

def generate_pdf_report_tool(ctx_json: str | None = None) -> str:
    """
    Construit un PDF 'corporate' (page de garde, KPIs, graphes, conformité).
    - ctx_json optionnel : si None → utilise le dernier log.
    - Retourne une data-URL base64 'data:application/pdf;base64,...'.
    Résilience : évite les attributs sur None et passe rating/score/decision au PDF.
    """
    # -------- Parse contexte robuste
    try:
        ctx_in = _ctx_from_last_log() if not ctx_json else json.loads(ctx_json or "{}")
    except Exception:
        return _utf8("❌ Contexte JSON invalide pour generate_pdf_report_tool.")

    client_id  = str(ctx_in.get("client_id", "N/A"))
    timestamp  = str(ctx_in.get("timestamp") or datetime.now(timezone.utc).isoformat())
    pd_value   = float(ctx_in.get("pd_value", 0.0))
    threshold  = float(ctx_in.get("threshold", 0.1))
    model_tag  = str(ctx_in.get("model_tag", "pipeline_v1"))
    logo_path  = ctx_in.get("logo_path") or None
    rating     = ctx_in.get("rating") or _infer_rating(pd_value)
    score_1000 = ctx_in.get("score_1000")
    if score_1000 is None:
        score_1000 = int(round((1.0 - max(0.0, min(pd_value, 1.0))) * 1000))
    decision   = ctx_in.get("decision") or ("ACCEPT" if pd_value < threshold else "REVUE / REFUS")

    # -------- inputs_json -> dict
    try:
        inputs = json.loads(ctx_in.get("inputs_json") or "{}")
    except Exception:
        inputs = {}

    # -------- contributions optionnelles (sécurisées)
    df_exp = None
    try:
        exp_res = explain_credit_api(inputs, top_k=20)
        if exp_res and isinstance(exp_res, dict) and exp_res.get("ok"):
            items = exp_res.get("items") or []
            contrib = {}
            for it in items:
                try:
                    f = str(it.get("feature"))
                    if "contribution" in it:
                        contrib[f] = float(it.get("contribution", 0.0))
                    elif "importance" in it:
                        contrib[f] = float(it.get("importance", 0.0))
                except Exception:
                    continue
            if contrib:
                import pandas as pd
                df_exp = pd.DataFrame({"feature": list(contrib.keys()), "contribution": list(contrib.values())})
                df_exp["value"] = df_exp["feature"].map(lambda f: inputs.get(f))
    except Exception:
        df_exp = None  # on continue sans SHAP

    # -------- Construction PDF
    pdf_bytes = build_pdf_bytes(
        ctx={
            "client_id": client_id,
            "timestamp": timestamp,
            "pd_value": pd_value,
            "threshold": threshold,
            "model_tag": model_tag,
            "inputs": inputs,
            "rating": rating,
            "score_1000": score_1000,
            "decision": decision,
        },
        df_exp=df_exp,
        logo_path=logo_path,
    )

    return "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii")

# =========================
# Aliases compat
# =========================
def tool_credit_score(payload_json: str) -> str:
    return predict_credit_tool(payload_json)

def search_web_tool(query: str, k: int = 3, fetch: bool = False) -> str:
    return tool_web_search(query, k=k, fetch=fetch)

# Alias historique exigé par certains prompts/outils (ex. "build_report")
def build_report(ctx_json: str | None = None) -> str:
    return generate_report_tool(ctx_json)
