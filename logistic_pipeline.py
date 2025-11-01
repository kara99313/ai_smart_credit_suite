
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, f1_score,
    balanced_accuracy_score, recall_score, precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os
import json

# --- Chemins & paramètres
DATA_PATH = "data/X_hybride_feature_engineered_selected_20250729_1526.csv"
MODEL_PATH = "model/logistic_pipeline_best.pkl"
FEATURES_PKL_PATH = "model/features.pkl"
FEATURES_CSV_PATH = "model/features.csv"
FEATURES_JSON_PATH = "model/features.json"
OUTPUTS_DIR = "model/outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

target_col = "Default"
categorical_cols = ["HasMortgage", "HasSocialAid", "CommunityGroupMember"]

# --- Chargement des données
df = pd.read_csv(DATA_PATH)
numeric_cols = [col for col in df.columns if col not in categorical_cols + [target_col]]
X = df[categorical_cols + numeric_cols]
y = df[target_col]

# --- Split stratifié train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- Préprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop='if_binary', dtype=int, handle_unknown="ignore"), categorical_cols)
    ]
)

# --- Pipeline 1 : class_weight balanced
pipe_weighted = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    ))
])
param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__penalty": ["l1", "l2"],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_weighted = GridSearchCV(
    pipe_weighted, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=2
)
grid_weighted.fit(X_train, y_train)

# --- Pipeline 2 : SMOTE
pipe_smote = ImbPipeline([
    ("preprocessing", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", LogisticRegression(
        solver="liblinear",
        class_weight=None,  # SMOTE équilibre déjà
        max_iter=1000,
        random_state=42
    ))
])
grid_smote = GridSearchCV(
    pipe_smote, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=2
)
grid_smote.fit(X_train, y_train)

# --- Evaluation & comparaison
def evaluate_model(grid, X_test, y_test, suffix):
    y_pred_prob = grid.predict_proba(X_test)[:, 1]
    y_pred = grid.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_prob)
    auc_pr = average_precision_score(y_test, y_pred_prob)
    f1 = f1_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    recall1 = recall_score(y_test, y_pred)
    precision1 = precision_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"AUC ROC = {auc_roc:.3f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Courbe ROC - {suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"roc_curve_{suffix}.png"))
    plt.close()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
    plt.figure(figsize=(7,5))
    plt.plot(recall, precision, label=f"AUC PR = {auc_pr:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Courbe Precision-Recall - {suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"pr_curve_{suffix}.png"))
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.title(f"Matrice de Confusion - {suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, f"confusion_matrix_{suffix}.png"))
    plt.close()

    # Feature importance (coefficients)
    classifier = (
        grid.best_estimator_.named_steps["classifier"]
        if "classifier" in grid.best_estimator_.named_steps
        else grid.best_estimator_.steps[-1][1]
    )
    coef = classifier.coef_[0]
    cat_features = grid.best_estimator_.named_steps["preprocessing"].transformers_[1][1].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_features)
    feat_importance = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coef
    }).sort_values(by="coefficient", key=abs, ascending=False)
    feat_importance.to_csv(os.path.join(OUTPUTS_DIR, f"feature_importance_{suffix}.csv"), index=False)

    # Save classification report
    with open(os.path.join(OUTPUTS_DIR, f"classification_report_{suffix}.txt"), "w") as f:
        f.write(report)

    metrics_dict = {
        "AUC_ROC": auc_roc,
        "AUC_PR": auc_pr,
        "F1": f1,
        "BalancedAccuracy": bal_acc,
        "Recall_PositiveClass": recall1,
        "Precision_PositiveClass": precision1
    }
    # --- SAUVEGARDE DES FEATURES (pour chaque pipeline, on le refait pour le "meilleur" plus bas)
    if suffix == "weighted" or suffix == "smote":
        features_list = feature_names
        # Sauvegarde pkl
        joblib.dump(features_list, f"model/features_{suffix}.pkl")
        # Sauvegarde csv
        pd.Series(features_list).to_csv(f"model/features_{suffix}.csv", index=False)
        # Sauvegarde json
        with open(f"model/features_{suffix}.json", "w", encoding="utf-8") as f_json:
            json.dump(features_list, f_json, ensure_ascii=False, indent=2)
    return metrics_dict, grid, report, feature_names

# --- Evaluate both
metrics_weighted, model_weighted, report_weighted, features_weighted = evaluate_model(grid_weighted, X_test, y_test, "weighted")
metrics_smote, model_smote, report_smote, features_smote = evaluate_model(grid_smote, X_test, y_test, "smote")

# --- Compare and choose best
metrics_df = pd.DataFrame([metrics_weighted, metrics_smote], index=["weighted", "smote"])
metrics_df.to_csv(os.path.join(OUTPUTS_DIR, "comparison_metrics.csv"))

# Critère de choix = meilleur AUC_ROC
if metrics_weighted["AUC_ROC"] >= metrics_smote["AUC_ROC"]:
    best_model = model_weighted.best_estimator_
    best_tag = "weighted"
    features_final = features_weighted
else:
    best_model = model_smote.best_estimator_
    best_tag = "smote"
    features_final = features_smote

# Save best pipeline
joblib.dump(best_model, MODEL_PATH)

# SAUVEGARDE DES FEATURES DU MEILLEUR PIPELINE
joblib.dump(features_final, FEATURES_PKL_PATH)
pd.Series(features_final).to_csv(FEATURES_CSV_PATH, index=False)
with open(FEATURES_JSON_PATH, "w", encoding="utf-8") as f_json:
    json.dump(features_final, f_json, ensure_ascii=False, indent=2)

# --- Rapport Markdown synthétique
rapport_md = f"""
# Modélisation — Benchmark Déséquilibre : class_weight vs SMOTE

**Modèle final retenu** : `{best_tag}`  
**AUC ROC :** {metrics_df.loc[best_tag, 'AUC_ROC']:.3f}  
**AUC PR  :** {metrics_df.loc[best_tag, 'AUC_PR']:.3f}  
**F1-score :** {metrics_df.loc[best_tag, 'F1']:.3f}  
**Recall (classe positive) :** {metrics_df.loc[best_tag, 'Recall_PositiveClass']:.3f}

## Features utilisées dans le pipeline final

```python
{features_final}
```

## Rapport classification détaillé

```
{report_weighted if best_tag=='weighted' else report_smote}
```

## Tableau comparatif des métriques (csv disponible)

| Metric                 | Weighted    | SMOTE     |
|------------------------|------------|-----------|
| AUC ROC                | {metrics_df.loc['weighted','AUC_ROC']:.3f} | {metrics_df.loc['smote','AUC_ROC']:.3f} |
| AUC PR                 | {metrics_df.loc['weighted','AUC_PR']:.3f}  | {metrics_df.loc['smote','AUC_PR']:.3f}  |
| F1-score               | {metrics_df.loc['weighted','F1']:.3f}      | {metrics_df.loc['smote','F1']:.3f}      |
| Balanced Accuracy      | {metrics_df.loc['weighted','BalancedAccuracy']:.3f} | {metrics_df.loc['smote','BalancedAccuracy']:.3f} |
| Recall PositiveClass   | {metrics_df.loc['weighted','Recall_PositiveClass']:.3f} | {metrics_df.loc['smote','Recall_PositiveClass']:.3f} |
| Precision PositiveClass| {metrics_df.loc['weighted','Precision_PositiveClass']:.3f} | {metrics_df.loc['smote','Precision_PositiveClass']:.3f} |

## Courbes et matrices

Les courbes ROC et Precision-Recall, la matrice de confusion, et l’importance des features sont enregistrées dans le dossier `model/outputs`.

Script et outputs générés automatiquement pour mémoire/projet : pipeline, features, métriques, visualisations.
"""

with open(os.path.join(OUTPUTS_DIR, "rapport_modele.md"), "w", encoding="utf-8") as f:
    f.write(rapport_md)

print("\n==== RAPPORT SYNTHÉTIQUE ====")
print(rapport_md)
print("\nTous les fichiers et outputs sont sauvegardés dans le dossier", OUTPUTS_DIR)
print("\n=== FIN DE L’ENTRAÎNEMENT ET DE L’EXPORT. ===\n")
