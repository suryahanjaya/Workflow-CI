"""
modelling.py (Kriteria 3 — MLflow Project entry point)
Training Random Forest using MLflow autolog().
Designed to be run via: mlflow run MLProject/
Author: Surya Hanjaya
"""

import sys
import types
from unittest.mock import MagicMock

# Reconfigure stdout to use UTF-8 to prevent UnicodeEncodeError from MLflow emojis on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Mock torch to prevent DLL loading errors on Windows
dummy_torch = types.ModuleType("torch")
class DummyTensor:
    pass
dummy_torch.Tensor = DummyTensor
sys.modules["torch"] = dummy_torch

# Mock sklearn.frozen to prevent import errors in corrupted sklearn installations
sys.modules['sklearn.frozen'] = MagicMock()
sys.modules['sklearn.frozen._frozen'] = MagicMock()

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PARAMETERS (can be overridden via mlflow run -P)
# ============================================================
DATA_PATH    = os.environ.get("DATA_PATH",     "adult_income_preprocessed.csv")
N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", 100))
MAX_DEPTH_RAW= os.environ.get("MAX_DEPTH",    "None")
MAX_DEPTH    = None if MAX_DEPTH_RAW == "None" else int(MAX_DEPTH_RAW)
MIN_SPLIT    = int(os.environ.get("MIN_SAMPLES_SPLIT", 2))
MIN_LEAF     = int(os.environ.get("MIN_SAMPLES_LEAF",  1))
TEST_SIZE    = float(os.environ.get("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", 42))


def main():
    print(f"[INFO] Loading data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["income"])
    y = df["income"]
    feature_names = list(X.columns)
    print(f"[INFO] Shape: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    params = {
        "n_estimators"    : N_ESTIMATORS,
        "max_depth"       : MAX_DEPTH,
        "min_samples_split": MIN_SPLIT,
        "min_samples_leaf": MIN_LEAF,
        "max_features"    : "sqrt",
        "random_state"    : RANDOM_STATE,
        "n_jobs"          : -1,
    }
    print(f"[INFO] Params: {params}")

    # Enable MLflow Autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary")
        rec  = recall_score(y_test, y_pred, average="binary")
        f1   = f1_score(y_test, y_pred, average="binary")

        # Autolog is active, logging parameters and model automatically. No manual logging calls.

        print(f"\n[RESULTS]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"\n[SUCCESS] MLflow run logged. Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
