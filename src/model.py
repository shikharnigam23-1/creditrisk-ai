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

# ========== T5: tune XGBoost, evaluate on the untouched test set, save ==========
def run_tune():
    X_train, X_test, y_train, y_test = get_data()
    ratio = imbalance_ratio(y_train)
    search_space = {
        "model__n_estimators": [200, 300, 500, 700],    # number of trees
        "model__max_depth": [3, 4, 5, 6, 8],            # how deep each tree can grow
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],  # size of each correction step
        "model__subsample": [0.6, 0.8, 1.0],            # % of rows each tree sees
        "model__colsample_bytree": [0.6, 0.8, 1.0],     # % of features each tree sees
        "model__min_child_weight": [1, 3, 5],           # min data needed to split
        "model__scale_pos_weight": [1.0, ratio],        # extra weight on defaults
    }
    base = make_pipeline(XGBClassifier(eval_metric="logloss", n_jobs=1, random_state=SEED))
    search = RandomizedSearchCV(base, search_space, n_iter=25, scoring="roc_auc",
                                cv=CV, n_jobs=-1, random_state=SEED, verbose=1)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"\nBest CV ROC-AUC: {search.best_score_:.4f}")
    print("Best parameters:", search.best_params_)

    # Final exam: the 20% test set the model has never seen
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results = {
        "cv_best_roc_auc": round(search.best_score_, 4),
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "test_roc_auc": round(roc_auc_score(y_test, proba), 4),
        "test_precision_at_0.5": round(precision_score(y_test, pred), 4),
        "test_recall_at_0.5": round(recall_score(y_test, pred), 4),
        "test_f1_at_0.5": round(f1_score(y_test, pred), 4),
    }
    (REPORTS / "test_metrics.json").write_text(json.dumps(results, indent=2, default=float))
    joblib.dump(best, MODEL_PATH)
    print(json.dumps(results, indent=2, default=float))

    # Charts for the slides
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name="XGBoost (tuned)")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    fig.savefig(REPORTS / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Repaid", "Default"],
                                            cmap="Blues", ax=ax)
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Error analysis: how do MISSED defaulters differ from CAUGHT ones?
    y = y_test.to_numpy()
    missed = X_test[(y == 1) & (pred == 0)].mean(numeric_only=True)
    caught = X_test[(y == 1) & (pred == 1)].mean(numeric_only=True)
    pd.DataFrame({"missed_defaulters": missed, "caught_defaulters": caught}).round(2) \
        .to_csv(REPORTS / "error_analysis.csv")

    # ========== T6: cost-based threshold + business ROI ==========
LGD = 0.60            # assumption: lender loses 60% of the loan when a borrower defaults
COST_OF_FUNDS = 0.07  # assumption: lender's own borrowing cost; profit = interest rate - this


def _costs(X):
    loan = X["loan_amnt"].to_numpy(dtype=float)
    rate = X["loan_int_rate"].fillna(X["loan_int_rate"].median()).to_numpy(dtype=float) / 100
    loss_if_default = loan * LGD
    profit_if_repaid = loan * np.clip(rate - COST_OF_FUNDS, 0.01, None)
    return loss_if_default, profit_if_repaid


def _portfolio_cost(y, proba, threshold, loss, profit):
    """Approve if risk < threshold.
    Cost = money lost on approved defaulters + profit lost on rejected good customers."""
    y = np.asarray(y)
    approve = proba < threshold
    return loss[approve & (y == 1)].sum() + profit[~approve & (y == 0)].sum()


def run_roi():
    X_train, X_test, y_train, y_test = get_data()
    model = joblib.load(MODEL_PATH)
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)

    # 1) Choose the threshold on TRAINING data (out-of-fold predictions), never on the test set
    oof = cross_val_predict(model, X_train, y_train, cv=CV, method="predict_proba")[:, 1]
    loss_tr, profit_tr = _costs(X_train)
    train_costs = [_portfolio_cost(y_train, oof, t, loss_tr, profit_tr) for t in thresholds]
    best_t = float(thresholds[int(np.argmin(train_costs))])

    # 2) Measure the business result on the untouched TEST set
    proba = model.predict_proba(X_test)[:, 1]
    loss, profit = _costs(X_test)
    y = y_test.to_numpy()
    cost_model = _portfolio_cost(y, proba, best_t, loss, profit)
    cost_approve_all = _portfolio_cost(y, proba, 1.01, loss, profit)
    rule_reject = X_test["loan_grade"].isin(["D", "E", "F", "G"]).to_numpy()  # simple human rule
    cost_rule = loss[~rule_reject & (y == 1)].sum() + profit[rule_reject & (y == 0)].sum()

    # 3) Three decision bands around the optimal threshold
    approve_below = round(max(best_t - 0.10, 0.02), 2)
    reject_above = round(min(best_t + 0.10, 0.98), 2)
    bands = np.where(proba < approve_below, "APPROVE",
                     np.where(proba >= reject_above, "REJECT", "REVIEW"))
    band_table = pd.DataFrame({"band": bands, "default": y}).groupby("band")["default"] \
        .agg(applicants="size", actual_default_rate="mean")
    band_table["share_of_applicants"] = (band_table["applicants"] / len(y)).round(3)
    band_table.round(3).to_csv(REPORTS / "decision_bands.csv")

    n = len(y)
    roi = {
        "assumptions": {"loss_given_default": LGD, "cost_of_funds": COST_OF_FUNDS,
                        "currency": "dataset units (USD) - convert to INR for slides"},
        "optimal_threshold": best_t,
        "bands": {"approve_below": approve_below, "reject_above": reject_above},
        "test_loans": n,
        "cost_approve_everyone": round(float(cost_approve_all)),
        "cost_simple_grade_rule": round(float(cost_rule)),
        "cost_creditrisk_ai": round(float(cost_model)),
        "saving_vs_approve_everyone": round(float(cost_approve_all - cost_model)),
        "saving_vs_grade_rule": round(float(cost_rule - cost_model)),
        "saving_per_1000_applications": round(float((cost_approve_all - cost_model) / n * 1000)),
    }
    (REPORTS / "roi.json").write_text(json.dumps(roi, indent=2))
    print(json.dumps(roi, indent=2))
    print("\n", band_table)

    # Chart: total cost at every threshold
    test_costs = [_portfolio_cost(y, proba, t, loss, profit) for t in thresholds]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, np.array(test_costs) / 1e6, color="#2563eb")
    ax.axvline(best_t, linestyle="--", color="red", label=f"chosen threshold = {best_t}")
    ax.set_xlabel("Risk threshold (approve if below)")
    ax.set_ylabel("Total cost (millions)")
    ax.set_title("Picking the threshold that minimises business cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "threshold_cost_curve.png", dpi=150)
    plt.close(fig)

    # ========== T7: explainability (SHAP) ==========
def _readable(name):
    """'num__loan_percent_income' -> 'loan percent income'"""
    return name.split("__", 1)[-1].replace("_", " ")


def _transform(pipe, X):
    prep = pipe.named_steps["prep"]
    Xt = prep.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    names = [_readable(n) for n in prep.get_feature_names_out()]
    return pd.DataFrame(Xt, columns=names, index=X.index)


def _original_feature_map(prep):
    """Map each one-hot column (e.g. 'cat__loan_grade_D') back to its original feature ('loan_grade')."""
    originals = [c for _, _, cols in prep.transformers_ if not isinstance(cols, str)
                 for c in cols if isinstance(c, str)]
    mapping = []
    for name in prep.get_feature_names_out():
        short = name.split("__", 1)[-1]
        matches = [c for c in originals if short == c or short.startswith(c + "_")]
        mapping.append(max(matches, key=len) if matches else short)
    return mapping


def explain(applicant, pipe=None, top_n=3):
    """Top reasons for ONE applicant (a 1-row DataFrame of raw inputs). Used by app.py.
    Positive impact = pushes default risk UP."""
    import xgboost as xgb
    pipe = pipe or joblib.load(MODEL_PATH)
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(applicant)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    booster = model.get_booster()
    dm = xgb.DMatrix(np.asarray(Xt, dtype=float), feature_names=booster.feature_names)
    contribs = booster.predict(dm, pred_contribs=True)[0][:-1]  # last value = baseline
    per_feature = pd.Series(contribs, index=_original_feature_map(prep)).groupby(level=0).sum()
    top = per_feature.reindex(per_feature.abs().sort_values(ascending=False).index)[:top_n]
    row = applicant.iloc[0]
    return [{"feature": f, "value": row.get(f), "impact": round(float(v), 3),
             "effect": "raises risk" if v > 0 else "lowers risk"} for f, v in top.items()]


def run_shap():
    import shap
    X_train, X_test, y_train, y_test = get_data()
    pipe = joblib.load(MODEL_PATH)
    sample = X_test.sample(min(1000, len(X_test)), random_state=SEED)
    Xt = _transform(pipe, sample)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    sv = explainer(Xt)

    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False)
    plt.savefig(REPORTS / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.plots.bar(sv, max_display=12, show=False)
    plt.savefig(REPORTS / "shap_importance.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    i = int(np.argmax(pipe.predict_proba(sample)[:, 1]))  # riskiest applicant in the sample
    plt.figure()
    shap.plots.waterfall(sv[i], max_display=10, show=False)
    plt.savefig(REPORTS / "shap_example_applicant.png", dpi=150, bbox_inches="tight")
    plt.close("all")

    print("Top reasons for the riskiest sample applicant:")
    for r in explain(sample.iloc[[i]], pipe):
        print("  ", r)
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