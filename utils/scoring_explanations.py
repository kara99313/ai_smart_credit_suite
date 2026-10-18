"""Validated global coefficient importances; no individual attribution is inferred."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression


def global_coefficient_importance(pipeline, normalized_df) -> dict:
    preprocessor = pipeline[:-1]
    estimator = pipeline.steps[-1][1]
    if not isinstance(estimator, LogisticRegression):
        raise ValueError("Global coefficient explanation requires logistic regression.")
    qualified = [str(name) for name in preprocessor.get_feature_names_out()]
    names = [name.split("__", 1)[-1] for name in qualified]
    coefficients = np.asarray(estimator.coef_, dtype=float)
    classes = np.asarray(estimator.classes_)
    if classes.shape != (2,) or np.count_nonzero(classes == 1) != 1:
        raise ValueError("Default class 1 is missing or ambiguous.")
    if len(set(names)) != len(names) or not names or coefficients.shape != (1, len(names)):
        raise ValueError("Coefficient and transformed-feature dimensions do not match.")
    transformed = preprocessor.transform(normalized_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    transformed = np.asarray(transformed, dtype=float)
    if transformed.shape != (1, len(names)):
        raise ValueError("Unexpected transformed-input dimensions.")
    if not np.isfinite(coefficients).all() or not np.isfinite(transformed).all():
        raise ValueError("Non-finite explanation data.")
    # Binary sklearn coefficients describe classes_[1]; orient the global sign to class 1.
    oriented = coefficients[0] if classes[1] == 1 else -coefficients[0]
    return {
        "ok": True,
        "method": "global_importance",
        "scope": "global",
        "space": "transformed_features",
        "coefficient_space": "log_odds",
        "class_label": 1,
        "feature_count": len(names),
        "fallback_reason": "individual_shap_not_validated",
        "items": [
            {"feature": name, "transformed_feature": full_name,
             "coefficient": float(coef), "importance": float(abs(coef))}
            for name, full_name, coef in zip(names, qualified, oriented)
        ],
    }
