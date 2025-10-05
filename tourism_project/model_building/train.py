# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for creating a folder
import os
# for Hugging Face uploads
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# -------------------------
# MLflow experiment setup
# -------------------------
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("visit-with-us-production-training")

api = HfApi()

# -------------------------
# Data locations (HF Datasets)
# Replace these with your actual dataset repo/paths
# -------------------------
Xtrain_path = "hf://datasets/Yash0204/tourism-prediction-mlops/Xtrain.csv"
Xtest_path  = "hf://datasets/Yash0204/tourism-prediction-mlops/Xtest.csv"
ytrain_path = "hf://datasets/Yash0204/tourism-prediction-mlops/ytrain.csv"
ytest_path  = "hf://datasets/Yash0204/tourism-prediction-mlops/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest  = pd.read_csv(Xtest_path)
ytrain_df = pd.read_csv(ytrain_path)
ytest_df  = pd.read_csv(ytest_path)

# y can be a single-column DataFrame; make it a Series
def to_series(yframe):
    if isinstance(yframe, pd.Series):
        return yframe.astype(int)
    if yframe.shape[1] == 1:
        return yframe.iloc[:, 0].astype(int)
    # if multiple columns, prefer 'ProdTaken'
    return yframe['ProdTaken'].astype(int)

ytrain = to_series(ytrain_df)
ytest  = to_series(ytest_df)

# -------------------------
# Feature buckets (based on data dictionary; only keep those present)
# -------------------------
numeric_features = [c for c in [
    'Age',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch',
    'Passport',   # binary treated as numeric
    'OwnCar'      # binary treated as numeric
] if c in Xtrain.columns]

categorical_features = [c for c in [
    'TypeofContact',
    'CityTier',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched'
] if c in Xtrain.columns]

# -------------------------
# Class weight (handle imbalance): neg/pos on TRAIN ONLY
# -------------------------
value_counts = ytrain.value_counts()
neg = int(value_counts.get(0, 0))
pos = int(value_counts.get(1, 0))
if pos == 0:
    raise ValueError("No positive class in ytrain; cannot compute scale_pos_weight.")
class_weight = neg / pos

# -------------------------
# Preprocessing
# -------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# -------------------------
# XGBoost + Grid
# -------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    n_jobs=-1
)

param_grid = {
    'xgbclassifier__n_estimators': [25, 50, 75],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

# -------------------------
# MLflow run
# -------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        verbose=0
    )
    grid_search.fit(Xtrain, ytrain)

    # Log each param set & score as nested runs
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = float(results['mean_test_score'][i])
        std_score  = float(results['std_test_score'][i])

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

    # Best params in main run
    mlflow.log_params(grid_search.best_params_)

    # Evaluate best model
    best_model = grid_search.best_estimator_
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba  = best_model.predict_proba(Xtest)[:, 1]

    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test  = (y_pred_test_proba  >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
    test_report  = classification_report(ytest,  y_pred_test,  output_dict=True, zero_division=0)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall":    train_report['1']['recall'],
        "train_f1-score":  train_report['1']['f1-score'],
        "test_accuracy":   test_report['accuracy'],
        "test_precision":  test_report['1']['precision'],
        "test_recall":     test_report['1']['recall'],
        "test_f1-score":   test_report['1']['f1-score'],
        "threshold":       classification_threshold
    })

    # -------------------------
    # Save & log model
    # -------------------------
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/best_tourism_wellness_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # -------------------------
    # Upload to Hugging Face Hub (Model Repo)
    # -------------------------
    repo_id = "Yash0204/tourism-prediction-mlops"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' exists. Uploading artifact...")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating it...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=os.path.basename(model_path),
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Uploaded '{model_path}' to '{repo_id}'.")
