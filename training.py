import numpy as np
# from sklearn.model_selection import train_test_split
import tarfile
import os
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
# from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_squared_error, roc_curve
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import boto3
import awswrangler as wr
import pickle
from botocore.exceptions import ClientError
from datetime import datetime


model_path_basename = '/src/ml_model'

start_date = os.environ.get('start_date', '2022-01-01')
print(start_date)

end_date = os.environ.get('start_date', '2022-12-31')
print(end_date)

s3_bucket = os.environ.get('s3_bucket', 'mlops-feature-stores')
print(s3_bucket)

s3_prefix = os.environ.get('s3_prefix', 'models/cloned-user-detection')
print(s3_prefix)

database = os.environ.get('database', 'feature_stores')
print(database)

model_basename = os.environ.get('model_basename', 'cloned-user-detection')
print(model_basename)

def get_aws_client(service):
    boto3.setup_default_session(region_name='us-east-1')
    return boto3.client(service)


def save_compressed_model_and_push_to_s3(s3_client, s3_bucket, s3_prefix, model, model_basename, model_path_basename):
    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S').replace(' ', 'T')
    model_artifact = model_basename + now + '.pkl'

    if not os.path.exists(model_path_basename):
        os.makedirs(model_path_basename)

    model_path = os.path.join(model_path_basename, model_artifact)

    output_filename = model_artifact + '.tar.gz'

    arcname = os.path.basename(model_path)

    # save model by pickle
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # compress model file
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(model_path, arcname=arcname)

    # push to s3
    try:
        response = s3_client.upload_file(output_filename, s3_bucket, os.path.join(s3_prefix, output_filename))
    except ClientError as e:
        print(e)


def feature_processing(start_date, end_date, database):
    data_prepared_df = wr.athena.read_sql_query(sql=f"select * from cloned_user_data where monthly_report >= TIMESTAMP '{start_date} 00:00:00' and monthly_report <= TIMESTAMP '{end_date} 23:59:59'", database=database)

    data_prepared_df = data_prepared_df[(data_prepared_df.created_at >= start_date)
                                        & (data_prepared_df.created_at <= end_date)
                                        & (data_prepared_df.status == 'active')]

    cats = data_prepared_df.select_dtypes(exclude=np.number).columns.tolist()
    print(cats)

    for col in cats:
        if col.endswith('trading_amount') or col.endswith('per_transaction'):
            data_prepared_df[col] = data_prepared_df[col].astype('float32')
        else:
            print(col)
            data_prepared_df[col].fillna('TBD', inplace=True)
            data_prepared_df[col] = data_prepared_df[col].astype('category')

    X, y = data_prepared_df.drop(['user_id',
                                'weekly_report',
                                'monthly_report',
                                'is_cloned',
                                'created_at',
                                'status',
                                'username',
                                'label'], axis=1), data_prepared_df[['label']]
    
    return X, y


def training_job():
    client = get_aws_client('s3')

    X, y = feature_processing(start_date, end_date, database)
    #Training data
    xgb_classifier = XGBClassifier(n_estimators=100,
                                   objective='binary:logistic',
                                   tree_method='hist',
                                   eta=0.1,
                                   max_depth=6,
                                   verbosity=3,
                                   n_jobs=-1,
                                   subsample=0.8,
                                   enable_categorical=True)
    
    xgb_classifier.fit(X, y)

    save_compressed_model_and_push_to_s3(client, s3_bucket, s3_prefix, xgb_classifier, model_basename, model_path_basename)


if __name__ == "__main__":
    training_job()
