"""
src/model.py - CreditRisk AI model pipeline (Phase 2)

Run one step at a time from the project root:

    python -m src.model cv     # T4: compare 3 models with 5-fold CV
    python -m src.model tune   # T5: tune XGBoost, test it, save model
    python -m src.model roi    # T6: cost-based threshold + business ROI
    python -m src.model shap   # T7: SHAP explainability
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.data import build_preprocessor, get_data


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

REPORTS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

MODEL_PATH = MODELS / "model.joblib"

SEED = 42

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)


# ---------------------------------------------------------
# Helper function
# ---------------------------------------------------------

def imbalance_ratio(y):
    """
    Calculate the ratio of good loans to defaulted loans.
    """

    return float(
        (y == 0).sum() /
        (y == 1).sum()
    )


# ---------------------------------------------------------
# Create preprocessing + model pipeline
# ---------------------------------------------------------

def make_pipeline(model):
    """
    Combine preprocessing and the machine-learning model.

    Keeping preprocessing inside the pipeline prevents
    data leakage during cross-validation.
    """

    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", model),
        ]
    )


# =========================================================
# T4: Compare 3 models with cross-validation
# =========================================================

def run_cv():

    # Load training and testing data.
    X_train, X_test, y_train, y_test = get_data()

    # Calculate class imbalance.
    ratio = imbalance_ratio(y_train)

    # Define the three models.
    candidates = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEED,
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=ratio,
                eval_metric="logloss",
                n_jobs=-1,
                random_state=SEED,
            ),
    }

    # Metrics used for comparison.
    metrics = [
        "roc_auc",
        "precision",
        "recall",
        "f1",
    ]

    rows = []

    # Train and evaluate each model.
    for name, model in candidates.items():

        print(f"Cross-validating {name} ...")

        scores = cross_validate(
            make_pipeline(model),
            X_train,
            y_train,
            cv=CV,
            scoring=metrics,
            n_jobs=1,
        )

        row = {
            "model": name
        }

        for metric in metrics:

            row[f"{metric}_mean"] = round(
                scores[f"test_{metric}"].mean(),
                4,
            )

            row[f"{metric}_std"] = round(
                scores[f"test_{metric}"].std(),
                4,
            )

        rows.append(row)

    # Create results table.
    table = (
        pd.DataFrame(rows)
        .sort_values(
            "roc_auc_mean",
            ascending=False,
        )
    )

    # Save results.
    table.to_csv(
        REPORTS / "cv_results.csv",
        index=False,
    )

    print("\n")
    print(table.to_string(index=False))

    # -----------------------------------------------------
    # Create comparison chart
    # -----------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    ax.bar(
        table["model"],
        table["roc_auc_mean"],
        yerr=table["roc_auc_std"],
        capsize=6,
    )

    ax.set_ylim(
        0.5,
        1.0,
    )

    ax.set_ylabel(
        "ROC-AUC (5-fold CV)"
    )

    ax.set_title(
        "Model comparison"
    )

    fig.tight_layout()

    fig.savefig(
        REPORTS / "cv_comparison.png",
        dpi=150,
    )

    plt.close(fig)


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    step = (
        sys.argv[1]
        if len(sys.argv) > 1
        else ""
    )

    fn = globals().get(
        f"run_{step}"
    )

    if fn is None:

        sys.exit(
            "Usage: python -m src.model [cv|tune|roi|shap]"
        )

    fn()