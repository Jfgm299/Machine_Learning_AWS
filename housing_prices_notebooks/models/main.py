
import subprocess
import sys

print("Working")

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Force install required packages
for pkg in ["lightgbm", "pandas", "numpy", "joblib", "scikit-learn", "pyarrow"]:
    try:
        __import__(pkg)
    except ImportError:
        print(f"Package '{pkg}' not found. Installing...")
        install(pkg)



import argparse
import os
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import r2_score

# Helper function to load data
def load_data(path, file_name):
    full_path = os.path.join(path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found at {full_path}")
    print(f"Loading data from {full_path}")
    return pd.read_parquet(full_path)


if __name__ == "__main__":

    # --- 1. Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))

    parser.add_argument("--num_leaves", type=int, default=31)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--n_estimators", type=int, default=200)

    args = parser.parse_args()

    print("--- Starting LightGBM Training ---")
    print(f"Arguments: {args}")

    # --- 2. Load Data ---
    df_train = load_data(args.train, "train.parquet")
    df_validation = load_data(args.validation, "validation.parquet")

    # --- 3. Prepare Data ---
    target_column = "price"

    categorical_features = list(df_train.select_dtypes(include=['object', 'category']).columns)
    numeric_features = [
        col for col in df_train.select_dtypes(include=[np.number]).columns
        if col != target_column
    ]

    categorical_features = [c for c in categorical_features if c in df_validation.columns]
    numeric_features = [c for c in numeric_features if c in df_validation.columns]
    all_features = categorical_features + numeric_features

    print("Categorical:", categorical_features)
    print("Numeric:", numeric_features)

    for col in categorical_features:
        df_train[col] = df_train[col].astype("category")
        df_validation[col] = df_validation[col].astype("category")

    X_train = df_train[all_features]
    y_train = df_train[target_column]
    X_validation = df_validation[all_features]
    y_validation = df_validation[target_column]

    # --- 4. Train LightGBM Model ---
    model = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        eval_metric="r2",
        callbacks=[lgb.early_stopping(50)],
        categorical_feature=categorical_features
    )

    # --- 5. Evaluate ---
    y_pred = model.predict(X_validation)
    r2 = r2_score(y_validation, y_pred)
    print(f"Validation R²: {r2:.4f}")

    # --- 6. Save Model ---
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("--- Training Complete ---")