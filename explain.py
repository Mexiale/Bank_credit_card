# -*- coding: utf-8 -*-
"""
SHAP-based "why" explanations for a single client's churn prediction.

Uses shap.TreeExplainer (exact, near-instant) for tree-based winners and a
closed-form linear contribution for Logistic Regression. Gracefully returns
no reasons for any other estimator type (e.g. Stacking) rather than falling
back to a slow model-agnostic explainer.

Gender is deliberately excluded from the displayed reasons: it's used by the
model, but is not an appropriate or actionable "why" to hand a retention
agent.
"""

import numpy as np
import shap

TREE_ESTIMATORS = (
    'RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier',
    'CatBoostClassifier', 'DecisionTreeClassifier',
)

HIDDEN_FROM_REASONS = {'Gender'}


def _phrase(feature, value, high):
    if feature == 'Months_Inactive_12_mon':
        return (f"Inactif depuis {value:.0f} mois" if high
                else f"Client resté actif ({value:.0f} mois d'inactivité)")
    if feature == 'Total_Relationship_Count':
        return (f"Nombreux produits détenus ({value:.0f})" if high
                else f"Faible nombre de produits détenus ({value:.0f})")
    if feature == 'Avg_Utilization_Ratio':
        return (f"Utilisation de la carte élevée ({value * 100:.0f}%)" if high
                else f"Faible utilisation de la carte ({value * 100:.0f}%)")
    if feature == 'Total_Trans_Amt':
        return (f"Montant de transactions élevé ({value:,.0f})".replace(',', ' ') if high
                else f"Faible montant de transactions ({value:,.0f})".replace(',', ' '))
    if feature == 'Total_Revolving_Bal':
        return (f"Solde renouvelable élevé ({value:,.0f})".replace(',', ' ') if high
                else f"Faible solde renouvelable ({value:,.0f})".replace(',', ' '))
    if feature == 'Customer_Age':
        return f"Client plus âgé ({value:.0f} ans)" if high else f"Client plus jeune ({value:.0f} ans)"
    return feature


def build_explainer(pipeline):
    """Returns a (kind, explainer_or_estimator) tuple, kind in {'tree', 'linear', None}."""
    estimator = pipeline.named_steps['model']
    cls_name = type(estimator).__name__
    if cls_name in TREE_ESTIMATORS:
        return 'tree', shap.TreeExplainer(estimator)
    if cls_name == 'LogisticRegression':
        return 'linear', estimator
    return None, None


def explain(pipeline, explainer_kind, explainer, row_df, predicted_class, top_n=4):
    """Returns a list of {label, share} dicts for the top contributing features
    pushing toward the predicted class, or [] if this estimator isn't supported."""
    if explainer_kind is None:
        return []

    scaler = pipeline.named_steps['scaler']
    scaled = scaler.transform(row_df)

    if explainer_kind == 'tree':
        values = np.array(explainer(scaled).values)
        if values.ndim == 3:
            values = values[..., 1]
        shap_row = values[0]
    else:
        shap_row = explainer.coef_[0] * scaled[0]

    features = list(row_df.columns)
    contributions = [
        (f, shap_row[i]) for i, f in enumerate(features) if f not in HIDDEN_FROM_REASONS
    ]

    if predicted_class == 1:
        relevant = [(f, v) for f, v in contributions if v > 0]
    else:
        relevant = [(f, -v) for f, v in contributions if v < 0]

    if not relevant:
        return []

    total = sum(v for _, v in relevant)
    relevant.sort(key=lambda t: t[1], reverse=True)

    reasons = []
    for feature, value in relevant[:top_n]:
        raw_value = row_df.iloc[0][feature]
        high = scaled[0][features.index(feature)] > 0
        share = round(value / total * 100) if total else 0
        reasons.append({'label': _phrase(feature, raw_value, high), 'share': share})
    return reasons
