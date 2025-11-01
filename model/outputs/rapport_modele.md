
# Modélisation — Benchmark Déséquilibre : class_weight vs SMOTE

**Modèle final retenu** : `weighted`  
**AUC ROC :** 0.742  
**AUC PR  :** 0.291  
**F1-score :** 0.326  
**Recall (classe positive) :** 0.694

## Features utilisées dans le pipeline final

```python
['DTIRatio', 'TrustScorePsychometric', 'HouseholdSize', 'NumCreditLines', 'Income', 'MonthsEmployed', 'MobileMoneyTransactions', 'Age', 'InterestRate', 'LoanTerm', 'LoanAmount', 'InformalIncome', 'HasMortgage_Yes', 'HasSocialAid_Yes', 'CommunityGroupMember_Yes']
```

## Rapport classification détaillé

```
              precision    recall  f1-score   support

           0      0.943     0.663     0.778     45139
           1      0.213     0.694     0.326      5931

    accuracy                          0.666     51070
   macro avg      0.578     0.678     0.552     51070
weighted avg      0.858     0.666     0.726     51070

```

## Tableau comparatif des métriques (csv disponible)

| Metric                 | Weighted    | SMOTE     |
|------------------------|------------|-----------|
| AUC ROC                | 0.742 | 0.741 |
| AUC PR                 | 0.291  | 0.289  |
| F1-score               | 0.326      | 0.326      |
| Balanced Accuracy      | 0.678 | 0.678 |
| Recall PositiveClass   | 0.694 | 0.687 |
| Precision PositiveClass| 0.213 | 0.214 |

## Courbes et matrices

Les courbes ROC et Precision-Recall, la matrice de confusion, et l’importance des features sont enregistrées dans le dossier `model/outputs`.

Script et outputs générés automatiquement pour mémoire/projet : pipeline, features, métriques, visualisations.
