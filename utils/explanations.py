# utils/explanations.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Quelques colonnes “économiques” utiles si aucune explication n'est dispo
ECON_FEATURES = {
    "LoanAmount": "montant du prêt",
    "Income": "revenus",
    "DTIRatio": "ratio d’endettement",
    "InterestRate": "taux d’intérêt",
    "LoanTerm": "durée du prêt",
}

@dataclass
class Narrative:
    client: str
    analyst: str
    admin: str

def _rank_contrib(contrib: pd.Series, top_k: int = 5) -> Tuple[pd.Series, pd.Series]:
    """Retourne (plus_positifs, plus_negatifs) triés, valeurs absolues décroissantes."""
    contrib = contrib.fillna(0.0)
    pos = contrib[contrib > 0].sort_values(ascending=False).head(top_k)
    neg = contrib[contrib < 0].sort_values(ascending=True).head(top_k)
    return pos, neg

def _fmt_feature_name(name: str) -> str:
    # Nettoyage minimal pour lisibilité
    return str(name)

def _bullets(s: pd.Series) -> str:
    if s.empty:
        return "• (aucun facteur majeur détecté)"
    lines = []
    for k, v in s.items():
        sign = "+" if v >= 0 else "−"
        lines.append(f"• {_fmt_feature_name(k)} ({sign}{abs(v):.2f})")
    return "\n".join(lines)

def build_narrative_from_contrib(
    pd_val: float,
    threshold: float,
    contrib: Optional[pd.Series],
    X_row: Optional[pd.Series] = None,
    locale: str = "fr",
) -> Narrative:
    """
    Construit une explication textuelle.
    - pd_val : probabilité de défaut (0–1)
    - threshold : seuil interne
    - contrib : contributions additives (alignées aux features d’entrée)
    - X_row : valeurs brutes alignées (facultatif, utile pour analyste/admin)
    """
    decision = "acceptée" if pd_val < threshold else "refusée"
    sentiment = "favorable" if pd_val < threshold else "défavorable"

    # Si on a des contributions -> top drivers
    top_pos, top_neg = (pd.Series(dtype=float), pd.Series(dtype=float))
    if contrib is not None and isinstance(contrib, pd.Series) and not contrib.empty:
        top_pos, top_neg = _rank_contrib(contrib, top_k=5)

    # 1) Message client – simple, non technique
    if contrib is not None and (not top_pos.empty or not top_neg.empty):
        main_driver = (top_pos if pd_val >= threshold else top_neg)
        main_txt = ""
        if not main_driver.empty:
            feat = _fmt_feature_name(main_driver.index[0])
            if feat.lower().startswith("loanamount"):
                main_txt = "le montant du prêt demandé est trop élevé par rapport à votre capacité de remboursement"
            elif feat.lower().startswith("dti"):
                main_txt = "votre ratio d’endettement est jugé trop élevé"
            elif feat.lower().startswith("income"):
                main_txt = "vos revenus déclarés sont jugés insuffisants"
            elif feat.lower().startswith("interestrate"):
                main_txt = "le taux d’intérêt demandé rend l’opération plus risquée"
            else:
                main_txt = f"le facteur « {feat} » pèse fortement dans l’évaluation"
        client_msg = (
            f"Votre demande est **{decision}**. "
            f"Le modèle estime une probabilité de défaut de **{pd_val:.0%}** (seuil interne **{threshold:.0%}**). "
            + (f"Décision principalement car {main_txt}." if main_txt else "")
        )
    else:
        # fallback simple
        client_msg = (
            f"Votre demande est **{decision}** (PD **{pd_val:.0%}**, seuil **{threshold:.0%}**). "
            "Les principaux critères sont le montant, vos revenus et votre ratio d’endettement."
        )

    # 2) Message analyste – avec drivers + valeurs clés (si X_row dispo)
    analyst_lines = [
        f"Décision **{decision}** ({'PD' if locale=='fr' else 'PD'} = {pd_val:.2%}, seuil = {threshold:.2%}).",
        f"Signal global **{sentiment}**.",
        "",
        "**Facteurs qui augmentent le risque (contributions +)**",
        _bullets(top_pos if not top_pos.empty else pd.Series(dtype=float)),
        "",
        "**Facteurs qui réduisent le risque (contributions −)**",
        _bullets(top_neg if not top_neg.empty else pd.Series(dtype=float)),
    ]
    if X_row is not None and isinstance(X_row, pd.Series):
        # Variables économiques faciles à lire
        econ_vals = []
        for col, fr_name in ECON_FEATURES.items():
            if col in X_row.index:
                try:
                    v = float(X_row[col])
                    econ_vals.append(f"- {fr_name} : {v:,.2f}".replace(",", " ").replace(".", ","))
                except Exception:
                    econ_vals.append(f"- {fr_name} : {X_row[col]}")
        if econ_vals:
            analyst_lines += ["", "**Valeurs économiques clés**"] + econ_vals
    analyst_msg = "\n".join(analyst_lines)

    # 3) Message admin – met en avant la politique (seuil) et la traçabilité
    admin_lines = [
        f"Décision **{decision}** (PD={pd_val:.4f}, seuil={threshold:.4f}).",
        "Politique appliquée :",
        f"- Règle binaire : `pd >= seuil` ⇒ Refus ; sinon Accord.",
        "- Conformité : KYC/AML/antifraude à vérifier en cas d'accord.",
        "",
        "Top drivers (absolu) :",
    ]
    if contrib is not None and isinstance(contrib, pd.Series) and not contrib.empty:
        abs_imp = contrib.abs().sort_values(ascending=False).head(10)
        for k, v in abs_imp.items():
            admin_lines.append(f"• {k}: {v:.4f}")
    else:
        admin_lines.append("• (aucun explainer disponible)")
    admin_msg = "\n".join(admin_lines)

    return Narrative(client=client_msg, analyst=analyst_msg, admin=admin_msg)

def heuristic_narrative(pd_val: float, threshold: float, X_row: Optional[pd.Series] = None) -> Narrative:
    """Fallback si aucune contribution n'est disponible."""
    decision = "acceptée" if pd_val < threshold else "refusée"

    # Petite heuristique lisible pour le client
    hint = "vos revenus et votre ratio d’endettement"  # défaut
    if X_row is not None:
        try:
            loan = float(X_row.get("LoanAmount", 0) or 0)
            inc  = float(X_row.get("Income", 0) or 0)
            dti  = float(X_row.get("DTIRatio", 0) or 0)
            if loan > 0 and inc > 0 and loan > 3.5 * inc:
                hint = "le montant du prêt demandé est très élevé par rapport à vos revenus"
            elif dti > 0.6:
                hint = "votre ratio d’endettement est jugé trop élevé"
            elif inc < 1e5:
                hint = "vos revenus sont jugés insuffisants"
        except Exception:
            pass

    client = (
        f"Votre demande est **{decision}** (PD **{pd_val:.0%}**, seuil **{threshold:.0%}**). "
        f"Décision principalement liée à {hint}."
    )

    analyst = (
        f"Décision **{decision}** (PD={pd_val:.2%}, seuil={threshold:.2%}). "
        "Aucune contribution modèle disponible — heuristique appliquée sur LoanAmount/Income/DTIRatio."
    )
    admin = (
        f"Décision **{decision}** (PD={pd_val:.4f}, seuil={threshold:.4f}). "
        "Explainer absent. Traçabilité : heuristique fondée sur LoanAmount, Income, DTIRatio."
    )

    return Narrative(client=client, analyst=analyst, admin=admin)
