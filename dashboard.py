# -*- coding: utf-8 -*-
"""
Aggregates Client_Bank + model predictions into the data the /dashboard
route needs. Kept out of app.py since it's a chunk of pandas logic, not
routing.

Deliberately excludes "churn by branch/region" and "monthly evolution":
Client_Bank has no branch, region, or date column, so those breakdowns
would have to be fabricated. Real churn dashboards should not show invented
dimensions next to real customer data.
"""

import sqlite3

import pandas as pd

FEATURES = [
    'Gender', 'Customer_Age', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Total_Revolving_Bal', 'Total_Trans_Amt', 'Avg_Utilization_Ratio',
]

AGE_BINS = [0, 30, 40, 50, 60, 200]
AGE_LABELS = ['< 30 ans', '30-39 ans', '40-49 ans', '50-59 ans', '60 ans et +']


def _load_clients():
    con = sqlite3.connect('Db.sqlite3')
    df = pd.read_sql(
        f"select CLIENTNUM, {', '.join(FEATURES)}, Attrition_Flag from Client_Bank", con,
    )
    con.close()
    return df


def build_dashboard_data(model, risk_level_fn):
    df = _load_clients()
    features = df[FEATURES].copy()
    features['Gender'] = features['Gender'].map({'M': 1, 'F': 0})

    df['Risk_Score'] = (model.predict_proba(features)[:, 1] * 100).round().astype(int)
    df['Risk_Level'] = df['Risk_Score'].apply(risk_level_fn)
    df['Is_Churned'] = df['Attrition_Flag'] == 'Attrited Customer'

    total_clients = len(df)
    churn_rate = round(df['Is_Churned'].mean() * 100, 1)
    avg_risk = round(df['Risk_Score'].mean(), 1)

    top10 = (
        df.sort_values('Risk_Score', ascending=False)
        .head(10)[['CLIENTNUM', 'Customer_Age', 'Gender', 'Risk_Score', 'Risk_Level']]
        .to_dict('records')
    )

    df['Age_Bucket'] = pd.cut(df['Customer_Age'], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    by_age = df.groupby('Age_Bucket', observed=True)['Is_Churned'].mean() * 100
    churn_by_age = {
        'labels': AGE_LABELS,
        'values': [round(by_age.get(label, 0), 1) for label in AGE_LABELS],
    }

    by_gender = df.groupby('Gender', observed=True)['Is_Churned'].mean() * 100
    churn_by_gender = {
        'labels': ['Féminin', 'Masculin'],
        'values': [round(by_gender.get('F', 0), 1), round(by_gender.get('M', 0), 1)],
    }

    return {
        'total_clients': total_clients,
        'churn_rate': churn_rate,
        'avg_risk': avg_risk,
        'top10': top10,
        'churn_by_age': churn_by_age,
        'churn_by_gender': churn_by_gender,
    }
