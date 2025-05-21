import dagshub.auth
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import dagshub

RANDOM_STATE = 19

load_dotenv()

dagshub.init(
    repo_owner='ffauzan',
    repo_name='msml-crop',
    mlflow=True
)

token = os.getenv("DAGSHUB_USER_TOKEN")
dagshub.auth.add_app_token(token)

mlflow.set_tracking_uri("https://dagshub.com/ffauzan/msml-crop.mlflow/")
mlflow.set_experiment("Crop Model v2")

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

# Grid search parameters
param_grid = {
    'max_depth': [5, 7],
    'learning_rate': [0.1, 0.2],
    'n_estimators': [100, 150],
    'subsample': [0.8, 0.9]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    n_jobs=-1,
    cv=cv,
    verbose=1
)

# Start MLflow run
with mlflow.start_run():
    # Fit the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # 1. Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # 2. Log metrics
    mlflow.log_metric("accuracy", acc)
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for m_name, m_value in metrics.items():
                mlflow.log_metric(f"{label}_{m_name}", m_value)

    # 3. Log confusion matrix as artifact
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/confusion_matrix.png")
    mlflow.log_artifact("artifacts/confusion_matrix.png")

    # 4. Log classification report as text
    report_text = classification_report(y_test, y_pred)
    with open("artifacts/classification_report.txt", "w", encoding='utf-8') as f:
        f.write(report_text)
    mlflow.log_artifact("artifacts/classification_report.txt")

    # 5. Log the model with signature
    signature = infer_signature(X_test, y_pred)
    mlflow.sklearn.log_model(best_model, "model", signature=signature)
    
    # 6. Save the model locally
    os.makedirs("artifacts/mlflow_model", exist_ok=True)
    mlflow.sklearn.save_model(best_model, path="artifacts/mlflow_model")

    # 7. (Optional) Log training/test split data shapes
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])

    # Print results
    print("Best Parameters:", grid_search.best_params_)
    print("Test Accuracy:", acc)
    print("Classification Report:\n", report_text)
