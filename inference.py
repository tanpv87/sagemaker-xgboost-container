import xgboost as xgb
import os
import numpy as np
import pandas as pd
import awswrangler as wr
import boto3
import uvicorn
import asyncio
from fastapi import FastAPI, status, Request, Response
from typing import Union

best_threshold = 0.264033

boto3.setup_default_session(region_name='us-east-1')

app = FastAPI()

model_path = '/opt/ml/model'

print([os.path.join(dirpath, f) for (dirpath, _, filenames) in os.walk(model_path) for f in filenames])


booster = xgb.Booster()
xgb_model = booster.load_model('/src/cloned_user_detection.json')
print(type(xgb_model))
print("model loaded")


async def feature_calculation(users):
    users_df = wr.athena.read_sql_query(sql="SELECT * FROM cloned_user_data where weekly_report >= TIMESTAMP '2022-11-01 00:00:00' and weekly_report <= TIMESTAMP '2022-11-07 23:59:59'", database="feature_stores")
    cats = users_df.select_dtypes(exclude=np.number).columns.tolist()

    for col in cats:
        if col.endswith('trading_amount') or col.endswith('per_transaction'):
            users_df[col] = users_df[col].astype('float32')
        else:
            print(col)
            users_df[col].fillna('TBD', inplace=True)
            users_df[col] = users_df[col].astype('category')

    X = users_df.drop(['user_id',
                       'weekly_report',
                       'monthly_report',
                       'is_cloned',
                       'created_at',
                       'status',
                       'username',
                       'label'], axis=1)
    
    return xgb.DMatrix(X, enable_categorical=True)


async def predict_output(body):
    user_features = await feature_calculation(body['users'])
    predicted_label = np.where(np.array([pred for pred in booster.predict(user_features)]) >= best_threshold, 1, 0).tolist()
    return await zip(body['users'], predicted_label)


@app.post('/invocations')
async def invocations(request: Request):
    # model() is a hypothetical function that gets the inference output:
    (print(await request.json()))
    model_resp = await predict_output(await request.json())
    print(await model_resp)

    response = Response(
        content=model_resp,
        status_code=status.HTTP_200_OK,
        media_type="text/plain",
    )
    return response

@app.get('/ping', status_code=status.HTTP_200_OK)
async def ping():
    return {"message": "ok"}

if __name__ == "__main__":
    uvicorn.run("inference:app", port=8080, log_level="info")