# -*- coding: utf-8 -*-
"""
Author : Mexiale
"""

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, g
import pickle
import sqlite3 as sql

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
DATABASE = 'Db.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sql.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def home():
    cur = get_db().cursor()
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def Predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    if output == 1:
        return render_template('Answers1.html', prediction_text = 'Le client va se désabonner')
    else:
        return render_template('Answers0.html', prediction_text='Le client ne va pas se désabonner')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force = True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


@app.route('/search', methods=['POST', 'GET'])
def list():
    nb= 1
    if request.form.get('nombre'):
        nb = request.form['nombre']
    con = sql.connect("Db.sqlite3")
    con.row_factory = sql.Row
   
    cur = con.cursor()
    cur.execute(f"select Gender, Customer_Age, Total_Relationship_Count, Months_Inactive_12_mon, Total_Revolving_Bal, Total_Trans_Amt, Avg_Utilization_Ratio, Attrition_Flag from Client_Bank ORDER BY RANDOM() LIMIT {nb}")
    rows = cur.fetchall();
    datas = np.array(rows)
    datas = pd.DataFrame(datas, columns=['Gender', 'Customer_Age', 'Total_Relationship_Count', 'Months_Inactive_12_mon', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Avg_Utilization_Ratio', 'Attrition_Flag'])
    datas['Gender'].replace({'M': 1, 'F': 0}, inplace=True)
    features = datas.drop(['Attrition_Flag'], axis=1)
    prediction = model.predict(features)
    ypred = prediction.reshape(int(nb), 1)
    ypred = pd.DataFrame(ypred, columns=['Prediction'])
    datas = pd.concat([datas, ypred], axis=1)
    datas['Gender'].replace({1: 'M', 0: 'F'}, inplace=True)
    datas['Prediction'].replace({0: 'Existing Customer', 1: 'Attrited Customer'}, inplace=True)
    return render_template("Predic_BD.html",datas = datas.values.tolist())

@app.route('/Predic_form')
def Predic_form():
    return render_template("Predic_form.html")

@app.route('/Predic_BD')
def Predic_BD():
    return render_template("Predic_BD.html")

''' if __name__ == "__main__":
    app.run(debug=True) '''