# -*- coding: utf-8 -*-
"""
Trains several churn-prediction candidates on Client_Bank, compares them
(including a Stacking ensemble), and saves the best pipeline as model.pkl.

Usage:
    env311/Scripts/python.exe train_model.py
"""

import json
import sqlite3
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

FEATURES = [
    'Gender', 'Customer_Age', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Total_Revolving_Bal', 'Total_Trans_Amt', 'Avg_Utilization_Ratio',
]
RANDOM_STATE = 42


def load_dataset():
    con = sqlite3.connect('Db.sqlite3')
    df = pd.read_sql(
        f"select {', '.join(FEATURES)}, Attrition_Flag from Client_Bank", con
    )
    con.close()
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})
    y = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
    X = df[FEATURES]
    return X, y


def build_candidates(scale_pos_weight):
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss',
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            is_unbalance=True, random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        ),
        'CatBoost': CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05,
            auto_class_weights='Balanced', random_state=RANDOM_STATE, verbose=False,
        ),
    }


def evaluate(name, pipeline, X_train, y_train, X_test, y_test, cv):
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1').mean()
    return {
        'model': name,
        'pipeline': pipeline,
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
        'roc_auc': roc_auc_score(y_test, proba),
        'cv_f1_mean': cv_f1,
    }


def main():
    warnings.filterwarnings('ignore')
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    candidates = build_candidates(scale_pos_weight)
    results = []
    for name, estimator in candidates.items():
        pipeline = Pipeline([('scaler', StandardScaler()), ('model', estimator)])
        results.append(evaluate(name, pipeline, X_train, y_train, X_test, y_test, cv))
        print(f"  trained {name}")

    stacking_candidates = build_candidates(scale_pos_weight)
    stacking = StackingClassifier(
        estimators=list(stacking_candidates.items()),
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5, n_jobs=-1,
    )
    stacking_pipeline = Pipeline([('scaler', StandardScaler()), ('model', stacking)])
    results.append(evaluate('Stacking', stacking_pipeline, X_train, y_train, X_test, y_test, cv))
    print("  trained Stacking")

    comparison = pd.DataFrame(results).drop(columns=['pipeline'])
    comparison = comparison.sort_values('f1', ascending=False).reset_index(drop=True)
    numeric_cols = comparison.columns.drop('model')
    comparison[numeric_cols] = comparison[numeric_cols].round(4)
    print("\n=== Comparaison des modèles (triée par F1 sur 'Attrited') ===")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    comparison.to_csv('model_comparison.csv', index=False)

    best_name = comparison.iloc[0]['model']
    best_result = next(r for r in results if r['model'] == best_name)
    best_pipeline = best_result['pipeline']
    print(f"\nMeilleur modèle : {best_name} (F1={best_result['f1']:.4f})")

    # Refit the winning pipeline on the full dataset before saving, to use all
    # available data for the model that will actually serve predictions.
    best_pipeline.fit(X, y)
    with open('model.pkl', 'wb') as f:
        import pickle
        pickle.dump(best_pipeline, f)

    metrics = {
        'model': best_name,
        'accuracy': round(best_result['accuracy'], 4),
        'precision': round(best_result['precision'], 4),
        'recall': round(best_result['recall'], 4),
        'f1': round(best_result['f1'], 4),
        'roc_auc': round(best_result['roc_auc'], 4),
        'trained_on_rows': int(len(X)),
    }
    with open('model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nmodel.pkl et model_metrics.json mis à jour ({best_name}).")


if __name__ == '__main__':
    main()
