import os
import shutil
from dotenv import load_dotenv

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import matplotlib.pyplot as plt
import seaborn as sns

import dagshub

RANDOM_STATE = 19

load_dotenv()

model_path = "artifacts/mlflow_model"
if os.path.exists(model_path):
    shutil.rmtree(model_path)

dagshub.init(
    repo_owner='ffauzan',
    repo_name='msml-crop',
    mlflow=True
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if MLFLOW_TRACKING_URI and MLFLOW_TRACKING_URI.startswith("https://dagshub.com"):
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    dagshub.auth.add_app_token(token)
else:
    MLFLOW_TRACKING_URI = "file:./mlruns"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

mlflow.set_experiment("CropModelv2Basic")


# Load the dataset
df_cleaned = pd.read_csv('crop_data_cleaned.csv')
X = df_cleaned.drop(columns=['label', 'label_encoded'])
y = df_cleaned['label_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Define model
model = XGBClassifier(
    objective='multi:softmax',
    num_class=8,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE
)

# Remove stale MLflow run context if it exists
if "MLFLOW_RUN_ID" in os.environ:
    del os.environ["MLFLOW_RUN_ID"]

# Start MLflow run
with mlflow.start_run() as run:
    print(f"Active run ID: {run.info.run_id}")
    # Fit the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log metrics
    mlflow.autolog()
    
    # Save the model locally
    os.makedirs("artifacts/mlflow_model", exist_ok=True)
    mlflow.sklearn.save_model(model, path="artifacts/mlflow_model")
    
    # Save run ID
    with open("run_id.txt", "w", encoding='utf8') as f:
        f.write(run.info.run_id)

    # Log training/test split data shapes
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])

    # Print results
    print("Test Accuracy:", acc)
