# -*- coding: utf-8 -*-
# 1. Library imports
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="Déployer le modèle scoring des clients")
df = pd.read_csv('./database/df_test3.csv')
model = pickle.load(open('./database/model_lgbm_3.pkl', 'rb'))

templates = Jinja2Templates(directory="templates")

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
def predict(id_client):
    ID = float(id_client)
    X = df[df['SK_ID_CURR'] == ID]

    ignore_features = ['SK_ID_CURR']
    relevant_features = [col for col in df.columns if col not in ignore_features]

    X = X[relevant_features]
    proba = model.predict_proba(X)

    prediction_dict = {}
    prediction_dict.update({
        'id_client': int(id_client),
        'probability': float(proba[0][1]),
    })

    return prediction_dict


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
