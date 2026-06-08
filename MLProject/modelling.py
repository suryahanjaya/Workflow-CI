"""
modelling.py (Kriteria 3 — MLflow Project entry point)
Training Random Forest with manual MLflow logging.
Designed to be run via: mlflow run MLProject/
Author: Surya Hanjaya
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import json

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
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


def plot_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["<=50K", ">50K"],
        yticklabels=["<=50K", ">50K"], ax=ax,
    )
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_feature_importance(model, feature_names, save_dir, top_n=20):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), top_importances[::-1], edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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

    with mlflow.start_run():
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="binary")
        rec  = recall_score(y_test, y_pred, average="binary")
        f1   = f1_score(y_test, y_pred, average="binary")

        # Manual MLflow logging
        for k, v in params.items():
            mlflow.log_param(k, str(v))
        mlflow.log_param("test_size",    TEST_SIZE)
        mlflow.log_param("n_features",   X_train.shape[1])
        mlflow.log_param("train_rows",   X_train.shape[0])
        mlflow.log_param("test_rows",    X_test.shape[0])

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)

        # Artifacts
        with tempfile.TemporaryDirectory() as tmp_dir:
            cm_path = plot_confusion_matrix(y_test, y_pred, tmp_dir)
            mlflow.log_artifact(cm_path, "plots")

            fi_path = plot_feature_importance(model, feature_names, tmp_dir)
            mlflow.log_artifact(fi_path, "plots")

            report = classification_report(
                y_test, y_pred,
                target_names=["<=50K", ">50K"], digits=4
            )
            report_path = os.path.join(tmp_dir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path, "reports")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name="AdultIncome-CI-RandomForest",
        )

        print(f"\n[RESULTS]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"\n[SUCCESS] MLflow run logged. Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
