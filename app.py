# -*- coding: utf-8 -*-
"""
Author : Mexiale
"""

import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sqlite3 as sql

import explain
import dashboard

FEATURES = [
    'Gender', 'Customer_Age', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Total_Revolving_Bal', 'Total_Trans_Amt', 'Avg_Utilization_Ratio',
]


def risk_level(score):
    """Maps a 0-100 churn risk score to a 4-tier qualitative label."""
    if score >= 75:
        return 'Critique'
    if score >= 50:
        return 'Élevé'
    if score >= 25:
        return 'Moyen'
    return 'Faible'

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
EXPLAINER_KIND, EXPLAINER = explain.build_explainer(model)

try:
    with open('model_metrics.json', encoding='utf-8') as f:
        MODEL_METRICS = json.load(f)
except FileNotFoundError:
    MODEL_METRICS = None

@app.template_filter('thousands')
def thousands_filter(value):
    try:
        return "{:,.0f}".format(float(value)).replace(',', ' ')
    except (TypeError, ValueError):
        return value

@app.route('/')
def home():
    return render_template('index.html', metrics=MODEL_METRICS)

@app.route('/predict',methods=['POST'])
def Predict():

    values = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame([values], columns=FEATURES)
    prediction = model.predict(final_features)
    proba = model.predict_proba(final_features)[0]

    output = round(prediction[0], 2)
    reasons = explain.explain(model, EXPLAINER_KIND, EXPLAINER, final_features, output)
    churn_probability = round(proba[1] * 100)
    level = risk_level(churn_probability)

    if output == 1:
        return render_template(
            'Answers1.html', prediction_text='Le client va se désabonner',
            probability=churn_probability, level=level, reasons=reasons,
        )
    else:
        return render_template(
            'Answers0.html', prediction_text='Le client ne va pas se désabonner',
            probability=churn_probability, level=level, reasons=reasons,
        )

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force = True)
    final_features = pd.DataFrame([data])[FEATURES].astype(float)
    prediction = model.predict(final_features)

    output = prediction[0]
    return jsonify(int(output))


CLIENT_BANK_COLUMNS = FEATURES + ['Attrition_Flag']

@app.route('/search', methods=['POST', 'GET'])
def list():
    nb = 10
    if request.form.get('nombre'):
        try:
            nb = max(1, min(int(request.form['nombre']), 200))
        except ValueError:
            nb = 10

    genre = request.form.get('genre', '')
    statut = request.form.get('statut', '')

    where_clauses = []
    params = []
    if genre in ('M', 'F'):
        where_clauses.append('Gender = ?')
        params.append(genre)
    where_sql = ('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''

    # A predicted-status filter can only be applied after scoring, so pull a
    # larger pool up front to have enough matches left after filtering.
    fetch_n = min(nb * 20, 2000) if statut in ('stable', 'risque') else nb

    con = sql.connect("Db.sqlite3")
    con.row_factory = sql.Row
    cur = con.cursor()
    cur.execute(
        f"select Gender, Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon, "
        f"Total_Revolving_Bal, Total_Trans_Amt, Avg_Utilization_Ratio, Attrition_Flag "
        f"from Client_Bank {where_sql} ORDER BY RANDOM() LIMIT ?",
        (*params, fetch_n),
    )
    rows = cur.fetchall()
    con.close()

    context = {'nb': nb, 'genre': genre, 'statut': statut, 'datas': [], 'stats': None}
    if not rows:
        return render_template("Predic_BD.html", **context)

    datas = pd.DataFrame(np.array(rows), columns=CLIENT_BANK_COLUMNS)
    int_cols = ['Customer_Age', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Total_Revolving_Bal', 'Total_Trans_Amt']
    datas[int_cols] = datas[int_cols].astype(int)
    datas['Avg_Utilization_Ratio'] = datas['Avg_Utilization_Ratio'].astype(float)
    datas['Gender'] = datas['Gender'].replace({'M': 1, 'F': 0})
    features = datas.drop(['Attrition_Flag'], axis=1)
    datas['Prediction'] = model.predict(features)
    datas['Risk_Score'] = (model.predict_proba(features)[:, 1] * 100).round().astype(int)
    datas['Risk_Level'] = datas['Risk_Score'].apply(risk_level)
    datas['Gender'] = datas['Gender'].replace({1: 'M', 0: 'F'})
    datas['Prediction'] = datas['Prediction'].replace({0: 'Existing Customer', 1: 'Attrited Customer'})

    if statut == 'risque':
        datas = datas[datas['Prediction'] == 'Attrited Customer']
    elif statut == 'stable':
        datas = datas[datas['Prediction'] == 'Existing Customer']
    datas = datas.head(nb)

    total = len(datas)
    at_risk = int((datas['Prediction'] == 'Attrited Customer').sum())
    match = int((datas['Attrition_Flag'] == datas['Prediction']).sum())
    context['stats'] = {
        'total': total,
        'at_risk': at_risk,
        'at_risk_pct': round(at_risk / total * 100) if total else 0,
        'match': match,
        'match_pct': round(match / total * 100) if total else 0,
    }
    context['datas'] = datas.values.tolist()
    return render_template("Predic_BD.html", **context)

@app.route('/Predic_form')
def Predic_form():
    return render_template("Predic_form.html")

@app.route('/Predic_BD')
def Predic_BD():
    return render_template("Predic_BD.html")

@app.route('/dashboard')
def dashboard_view():
    data = dashboard.build_dashboard_data(model, risk_level)
    return render_template('Dashboard.html', **data)

''' if __name__ == "__main__":
    app.run(debug=True) '''