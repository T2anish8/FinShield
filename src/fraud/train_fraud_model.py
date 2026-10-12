"""
FinShield AI
Fraud Detection Model Training

Models:
    1. Random Forest
    2. XGBoost

Evaluation:
    Precision, Recall, F1, ROC-AUC, PR-AUC
    Confusion Matrix

Input:
    data/processed/transactions_features.csv

Outputs:
    models/fraud_random_forest.joblib
    models/fraud_xgboost.joblib
    models/fraud_feature_columns.joblib
    models/fraud_evaluation.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from src.config import PROCESSED_DATA_DIR, MODEL_DIR, RANDOM_SEED


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = PROCESSED_DATA_DIR / "transactions_features.csv"

TEST_SIZE = 0.20

TARGET = "is_fraud"


# ============================================================
# FEATURE SELECTION
# ============================================================

# These features can be calculated from the transaction
# itself without looking at future transactions.
#
# Historical account statistics and full-dataset statistics
# are deliberately excluded from this first experiment.

FEATURE_COLUMNS = [
    "amount",
    "amount_log",
    "transaction_hour",
    "transaction_day",
    "transaction_month",
    "transaction_day_of_week",
    "is_weekend",
    "is_night_transaction",
    "is_business_hour",
    "has_beneficiary",
    "is_self_transfer",
    "is_transfer",
    "is_withdrawal",
    "is_deposit",
    "is_purchase",
    "is_bill_payment",
    "is_mobile",
    "is_web",
    "is_atm",
    "is_pos",
    "large_amount_flag",
    "structuring_amount_flag",
    "night_activity_flag",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("\nLoading feature dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Run feature engineering first."
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df):,} transactions")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    print("\nPreparing training data...")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    X = df[FEATURE_COLUMNS].copy()

    y = df[TARGET].astype(int)

    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X = X.fillna(0)

    print(f"Selected features: {len(FEATURE_COLUMNS)}")

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    return X, y


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(X, y):

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples : {len(X_test):,}")

    return X_train, X_test, y_train, y_test


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(X_train, y_train):

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    print("Random Forest training complete")

    return model


# ============================================================
# XGBOOST
# ============================================================

def train_xgboost(X_train, y_train):

    print("\nTraining XGBoost...")

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    scale_pos_weight = (
        negative_count / max(positive_count, 1)
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(X_train, y_train)

    print("XGBoost training complete")

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    model_name,
    X_test,
    y_test,
):

    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
    }

    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric:12}: {value:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print(f"\nTrue Negatives : {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives : {tp}")

    return metrics


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def show_feature_importance(
    model,
    model_name,
):

    print(f"\nTop features — {model_name}")

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.head(15).to_string(
            index=False
        )
    )


# ============================================================
# SAVE MODELS
# ============================================================

def save_models(
    random_forest,
    xgboost,
    evaluation_results,
):

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        random_forest,
        MODEL_DIR / "fraud_random_forest.joblib",
    )

    joblib.dump(
        xgboost,
        MODEL_DIR / "fraud_xgboost.joblib",
    )

    joblib.dump(
        FEATURE_COLUMNS,
        MODEL_DIR / "fraud_feature_columns.joblib",
    )

    evaluation_df = pd.DataFrame(
        evaluation_results
    )

    evaluation_df.to_csv(
        MODEL_DIR / "fraud_evaluation.csv",
        index=False,
    )

    print("\nSaved model artifacts:")
    print("  ✓ fraud_random_forest.joblib")
    print("  ✓ fraud_xgboost.joblib")
    print("  ✓ fraud_feature_columns.joblib")
    print("  ✓ fraud_evaluation.csv")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINSHIELD AI — FRAUD MODEL TRAINING")
    print("=" * 60)

    df = load_dataset()

    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    random_forest = train_random_forest(
        X_train,
        y_train,
    )

    rf_metrics = evaluate_model(
        random_forest,
        "Random Forest",
        X_test,
        y_test,
    )

    show_feature_importance(
        random_forest,
        "Random Forest",
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgboost = train_xgboost(
        X_train,
        y_train,
    )

    xgb_metrics = evaluate_model(
        xgboost,
        "XGBoost",
        X_test,
        y_test,
    )

    show_feature_importance(
        xgboost,
        "XGBoost",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_models(
        random_forest,
        xgboost,
        [
            rf_metrics,
            xgb_metrics,
        ],
    )

    print("\n" + "=" * 60)
    print("FRAUD MODEL TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()