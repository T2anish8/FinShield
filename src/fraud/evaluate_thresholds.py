"""
FinShield AI
Fraud Detection Threshold Analysis

Evaluates fraud models at multiple probability thresholds.

Purpose:
    Understand the tradeoff between:
        - Precision
        - Recall
        - F1
        - False positives
        - False negatives
        - Alert volume

This will later feed the FinShield risk engine
and Monte Carlo threshold simulation.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    RANDOM_SEED,
)


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "transactions_features.csv"
)

RF_MODEL_FILE = (
    MODEL_DIR /
    "fraud_random_forest.joblib"
)

XGB_MODEL_FILE = (
    MODEL_DIR /
    "fraud_xgboost.joblib"
)

FEATURE_FILE = (
    MODEL_DIR /
    "fraud_feature_columns.joblib"
)

OUTPUT_FILE = (
    MODEL_DIR /
    "fraud_threshold_analysis.csv"
)

TARGET = "is_fraud"

TEST_SIZE = 0.20

THRESHOLDS = [
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# LOAD DATA
# ============================================================

def load_test_data():

    print("\nLoading feature dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    feature_columns = joblib.load(
        FEATURE_FILE
    )

    X = df[feature_columns].copy()

    y = df[TARGET].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(
        f"Test samples: {len(X_test):,}"
    )

    return X_test, y_test


# ============================================================
# EVALUATE THRESHOLDS
# ============================================================

def evaluate_thresholds(
    model,
    model_name,
    X_test,
    y_test,
):

    print(
        f"\nAnalyzing thresholds — {model_name}"
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    rows = []

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = cm.ravel()

        total_alerts = (
            predictions == 1
        ).sum()

        alert_rate = (
            total_alerts /
            len(y_test)
        )

        false_positive_rate = (
            fp /
            max(fp + tn, 1)
        )

        rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
                "total_alerts": total_alerts,
                "alert_rate": alert_rate,
                "false_positive_rate": false_positive_rate,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINSHIELD AI — FRAUD THRESHOLD ANALYSIS")
    print("=" * 60)

    X_test, y_test = load_test_data()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print("\nLoading trained models...")

    random_forest = joblib.load(
        RF_MODEL_FILE
    )

    xgboost = joblib.load(
        XGB_MODEL_FILE
    )

    print("  ✓ Random Forest")
    print("  ✓ XGBoost")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    rf_results = evaluate_thresholds(
        random_forest,
        "Random Forest",
        X_test,
        y_test,
    )

    xgb_results = evaluate_thresholds(
        xgboost,
        "XGBoost",
        X_test,
        y_test,
    )

    results = pd.concat(
        [
            rf_results,
            xgb_results,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved threshold analysis:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Display best F1 threshold
    # --------------------------------------------------------

    print("\nBest F1 thresholds:")

    for model_name in [
        "Random Forest",
        "XGBoost",
    ]:

        model_results = results[
            results["model"] == model_name
        ]

        best = model_results.loc[
            model_results["f1_score"].idxmax()
        ]

        print(
            f"\n{model_name}"
        )

        print(
            f"Threshold : "
            f"{best['threshold']:.2f}"
        )

        print(
            f"Precision : "
            f"{best['precision']:.4f}"
        )

        print(
            f"Recall    : "
            f"{best['recall']:.4f}"
        )

        print(
            f"F1        : "
            f"{best['f1_score']:.4f}"
        )

        print(
            f"Alerts    : "
            f"{int(best['total_alerts']):,}"
        )

    # --------------------------------------------------------
    # Full table
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS")
    print("=" * 60)

    display_columns = [
        "model",
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "false_positives",
        "false_negatives",
        "total_alerts",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()