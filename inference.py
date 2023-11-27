import xgboost as xgb
import os
import numpy as np
import pandas as pd
import awswrangler as wr
import boto3
import uvicorn
import asyncio
import pickle
from fastapi import FastAPI, status, Request, Response
from fastapi.responses import JSONResponse
from typing import Union

best_threshold = 0.264033

boto3.setup_default_session(region_name='us-east-1')

app = FastAPI()

model_path = '/opt/ml/model'

print([os.path.join(dirpath, f) for (dirpath, _, filenames) in os.walk(model_path) for f in filenames])

with open('/opt/ml/model/cloned_user_detection.pkl', 'rb') as f:
    xgb_model_loaded = pickle.load(f)
print(type(xgb_model_loaded))
print("model loaded from aws provided")


async def feature_calculation(body):
    users_df = wr.athena.read_sql_query(sql=f"SELECT * FROM cloned_user_data where weekly_report >= TIMESTAMP '{body['start_date']} 00:00:00' and weekly_report <= TIMESTAMP '{body['end_date']} 23:59:59'", database="feature_stores")
    cats = users_df.select_dtypes(exclude=np.number).columns.tolist()

    for col in cats:
        if col.endswith('trading_amount') or col.endswith('per_transaction'):
            users_df[col] = users_df[col].astype('float32')
        else:
            print(col)
            users_df[col].fillna('TBD', inplace=True)
            users_df[col] = users_df[col].astype('category')

    X = users_df.drop(['weekly_report',
                       'monthly_report',
                       'is_cloned',
                       'created_at',
                       'status',
                       'username',
                       'label'], axis=1)
    
    return X


async def predict_output(body):
    user_features = await feature_calculation(body['users'])
    predicted_label = np.where(np.array([pred[1] for pred in xgb_model_loaded.predict_proba(user_features.drop(['user_id'], axis=1))]) >= best_threshold, 1, 0).tolist()
    return dict(zip(user_features['user_id'].tolist(), predicted_label))


@app.post('/invocations')
async def invocations(request: Request):
    # model() is a hypothetical function that gets the inference output:
    body = await request.json()
    model_resp = await predict_output(body)
    print(model_resp)

    response = JSONResponse(
        content=model_resp,
        status_code=status.HTTP_200_OK,
        media_type="application/json",
    )
    return response

@app.get('/ping', status_code=status.HTTP_200_OK)
async def ping():
    return {"message": "ok"}

if __name__ == "__main__":
    uvicorn.run("inference:app", port=8080, log_level="info")