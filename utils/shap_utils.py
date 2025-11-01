# utils/shap_utils.py
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Sequence, Any

class LinearCoefExplainer:
    """
    Explainer déterministe pour modèles linéaires (ex: LogisticRegression)
    sur données *déjà transformées* par le preprocess du pipeline.

    SHAP approx = (X_t - mu_t) * coef  (valeurs additive feature-wise)
    expected_value = intercept + mu_t @ coef
    """
    def __init__(self, preprocess, coef: np.ndarray, intercept: float,
                 mu_transformed: np.ndarray, feature_names_in: Sequence[str]):
        self.preprocess = preprocess
        self.coef = np.asarray(coef).ravel()
        self.intercept = float(intercept)
        self.mu_t = np.asarray(mu_transformed).ravel()
        self.feature_names_in_ = list(feature_names_in)

        # expected value du logit
        self.expected_value = float(self.intercept + float(np.dot(self.mu_t, self.coef)))

    def shap_values(self, raw_df: pd.DataFrame) -> np.ndarray:
        X = raw_df.copy()
        # aligne colonnes (valeurs par défaut neutres)
        for c in self.feature_names_in_:
            if c not in X.columns:
                X[c] = 0
        X = X[self.feature_names_in_]

        # transforme via preprocess du pipeline
        Xt = self.preprocess.transform(X)
        if hasattr(Xt, "toarray"):
            Xt = Xt.toarray()
        Xt = np.asarray(Xt)

        contrib = (Xt - self.mu_t) * self.coef  # (n_samples, n_feats_transf)
        # On ramène à l’espace d’entrée : somme par feature d’origine (safe)
        # -> si ColumnTransformer a étendu des colonnes, on agrège par nom source quand dispo.
        # Dans ton use-case, on a 1:1 ou OHE limité: afficher l’importance globale (somme abs)
        shap_per_sample = contrib.sum(axis=1)  # scalaire additif (logit). Pour le graphe, on renverra par colonne input.
        return shap_per_sample

def load_explainer_from_file(path: Path | str) -> Any | None:
    try:
        return joblib.load(Path(path))
    except Exception:
        return None
