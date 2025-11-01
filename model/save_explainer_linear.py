# save_explainer_linear.py
from __future__ import annotations
from pathlib import Path
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# -----------------------------
# Chemins
# -----------------------------
ROOT        = Path.cwd()
MODEL_DIR   = ROOT / "model"
OUTPUTS_DIR = MODEL_DIR / "outputs"
MODEL_PATH  = MODEL_DIR / "logistic_pipeline_best.pkl"

# ⚠️ Mets/ajuste ici le dossier des features d'entraînement
TRAIN_DIR = Path(r"D:\DOSSIER INSSEDS\Master's_thesis 2025\memoire_master_2025\MODELE_HYBRIDE_TEST\tables\feature_engineering")
CANDIDATES = [
    TRAIN_DIR / "X_hybride_feature_engineered_selected_20250729_1526.csv",
]

# Noms des colonnes catégorielles (celles qui ont 2 modalités)
CATEGORICAL_VARS = {"CommunityGroupMember", "HasMortgage", "HasSocialAid"}

# -----------------------------
# Utilitaires
# -----------------------------
def pick_training_csv() -> Path:
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "CSV d'entraînement introuvable. Mets à jour CANDIDATES dans save_explainer_linear.py."
    )

def load_pipeline_required() -> Pipeline:
    pipe = joblib.load(MODEL_PATH)
    assert isinstance(pipe, Pipeline) and len(pipe.steps) >= 2, \
        "Pipeline attendu: preprocess + modèle (sklearn Pipeline avec au moins 2 steps)."
    return pipe

def safe_feature_list(pipe: Pipeline, df: pd.DataFrame) -> list[str]:
    if hasattr(pipe, "feature_names_in_") and getattr(pipe, "feature_names_in_") is not None:
        feats = list(map(str, pipe.feature_names_in_))
    else:
        feats = list(map(str, df.columns))
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
    return feats

def decision_score(pipe: Pipeline, Xdf: pd.DataFrame) -> float:
    """
    Score additif f(X) :
    - priorité à decision_function (logit-like)
    - sinon logit(p1) via predict_proba
    - sinon fallback: y_pred (0/1)
    """
    if hasattr(pipe, "decision_function"):
        val = pipe.decision_function(Xdf)
        return float(np.asarray(val).ravel()[0])
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(Xdf)
        p1 = float(np.asarray(proba)[:, -1].ravel()[0])
        eps = 1e-9
        p1 = min(max(p1, eps), 1 - eps)
        return float(math.log(p1 / (1 - p1)))
    if hasattr(pipe, "predict"):
        return float(int(np.asarray(pipe.predict(Xdf)).ravel()[0]))
    return 0.0

def _two_categories_from_series(s: pd.Series):
    """
    Retourne (v0, v1) = deux catégories cohérentes avec le dataset.
    - Si présence de 'Yes'/'No' ou 'Oui'/'Non', on les utilise.
    - Sinon on prend les deux modalités les plus fréquentes.
    - Si une seule modalité, on duplique (le poids sera ~0).
    """
    vals = s.dropna()
    uniques = list(pd.unique(vals))
    # Normalise pour repérer oui/non
    lower = {str(u).strip().lower(): u for u in uniques}
    yes_keys = ["yes", "true", "1", "oui"]
    no_keys  = ["no", "false", "0", "non"]
    has_yes = any(k in lower for k in yes_keys)
    has_no  = any(k in lower for k in no_keys)
    if has_yes and has_no:
        v0 = next(lower[k] for k in no_keys  if k in lower)
        v1 = next(lower[k] for k in yes_keys if k in lower)
        return v0, v1
    # sinon: top-2 par fréquence
    vc = vals.value_counts(dropna=True)
    if len(vc.index) >= 2:
        v0, v1 = list(vc.index[:2])
        return v0, v1
    if len(vc.index) == 1:
        only = vc.index[0]
        # essaie de créer une "autre" valeur viable
        if isinstance(only, (int, float)):
            return only, (0 if only != 0 else 1)
        if isinstance(only, bool):
            return False, True
        # texte: inverse 'Yes'/'No' si possible
        if str(only).strip().lower() in ("yes","oui"):
            return "No", "Yes"
        if str(only).strip().lower() in ("no","non"):
            return "No", "Yes"
        # fallback: duplique (poids ~0)
        return only, only
    # aucun échantillon non-NaN
    return 0, 1

def build_reference_point(df: pd.DataFrame, feats: list[str]) -> pd.Series:
    """
    x0 :
    - catégorielles -> mode (conserve le TYPE d'origine !)
    - numériques    -> moyenne
    """
    x0 = {}
    for c in feats:
        s = df[c]
        if c in CATEGORICAL_VARS:
            # mode (garde le type original: str/bool/int…)
            try:
                mode_val = s.mode(dropna=True)
                val = mode_val.iloc[0] if not mode_val.empty else None
            except Exception:
                val = None
            if val is None:
                v0, _ = _two_categories_from_series(s)
                val = v0
            x0[c] = val
        else:
            m = pd.to_numeric(s, errors="coerce").mean()
            x0[c] = float(m) if pd.notna(m) else 0.0
    return pd.Series(x0, index=feats, dtype=object)

def finite_diff_weights(pipe: Pipeline, x0: pd.Series, df_stats: pd.DataFrame) -> dict[str, float]:
    """
    Poids par différences finies autour de x0.
    - Catégorielles: w = f(x0 avec v1) - f(x0 avec v0) (v0/v1: deux modalités du dataset)
    - Numériques:    w ≈ (f(x+Δ) - f(x-Δ)) / (2Δ), Δ = 0.1*std (fallback Δ=1)
    """
    weights: dict[str, float] = {}
    cols = list(x0.index)

    # std pour numériques
    std_map = {}
    for c in cols:
        col = pd.to_numeric(df_stats[c], errors="coerce")
        std = float(col.std(skipna=True))
        std_map[c] = 0.0 if not np.isfinite(std) else std

    for c in cols:
        if c in CATEGORICAL_VARS:
            s = df_stats[c]
            v0, v1 = _two_categories_from_series(s)
            x_a = x0.copy(); x_a[c] = v0
            x_b = x0.copy(); x_b[c] = v1
            f_a = decision_score(pipe, pd.DataFrame([x_a], columns=cols))
            f_b = decision_score(pipe, pd.DataFrame([x_b], columns=cols))
            weights[c] = float(f_b - f_a)
        else:
            std = std_map.get(c, 0.0)
            d = 0.1 * std
            if not np.isfinite(d) or d <= 0:
                d = 1.0
            x_plus  = x0.copy(); x_plus[c]  = float(x0[c]) + d
            x_minus = x0.copy(); x_minus[c] = float(x0[c]) - d
            f_plus  = decision_score(pipe, pd.DataFrame([x_plus],  columns=cols))
            f_minus = decision_score(pipe, pd.DataFrame([x_minus], columns=cols))
            weights[c] = float((f_plus - f_minus) / (2.0 * d))

    return weights

def compute_bias(pipe: Pipeline, x0: pd.Series, weights: dict[str, float]) -> float:
    """
    b = f(x0) - Σ w_j * g_j(x0)  (ici g_j = identité; les catégorielles contribuent via
    la valeur telle quelle — c’est une approximation linéaire locale)
    """
    f0 = decision_score(pipe, pd.DataFrame([x0], columns=list(x0.index)))
    linear = 0.0
    for c, w in weights.items():
        v = x0[c]
        try:
            v_num = float(v) if c not in CATEGORICAL_VARS else (1.0 if str(v).strip().lower() in ("yes","true","1","oui") else 0.0)
        except Exception:
            # si pas convertible, 1 pour la modalité de référence, 0 sinon (approx)
            v_num = 1.0
        linear += w * v_num
    return float(f0 - linear)

# ===========================
# Main
# ===========================
if __name__ == "__main__":
    print("🔄 Chargement du pipeline…")
    pipe = load_pipeline_required()

    print("🔄 Lecture du dataset…")
    csv_path = pick_training_csv()
    df = pd.read_csv(csv_path)

    feats = safe_feature_list(pipe, df)
    print(f"→ {len(feats)} features détectées.")

    # x0 avec types cohérents (très important pour OneHotEncoder)
    x0 = build_reference_point(df, feats)

    print("🔧 Calcul des poids (différences finies)…")
    weights = finite_diff_weights(pipe, x0, df[feats])

    # Biais
    bias = compute_bias(pipe, x0, weights)

    # Payload simple lu par prediction.py (etype = "linear")
    payload = {
        "type": "linear",
        "feature_names": feats,
        "weights": weights,     # dict {col: w}
        "bias": float(bias),
        "reference": {k: (str(v) if isinstance(v, (list, dict)) else v) for k, v in x0.to_dict().items()},
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "explainer.pkl"
    joblib.dump(payload, out_path)
    print(f"✅ Explainer sauvegardé → {out_path}")
    print(f"ℹ️ Exemple: nb_features={len(feats)} | biais={bias:.4f}")
