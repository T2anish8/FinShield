"""
FinShield AI - Anomaly Detection Engine

Unsupervised anomaly detection using Isolation Forest.
Works directly with the current FinShield transaction feature dataset.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = (
    MODEL_DIR
    / "anomaly_isolation_forest.joblib"
)

SCORED_FILE = (
    OUTPUT_DIR
    / "transactions_anomaly_scores.csv"
)

EVALUATION_FILE = (
    MODEL_DIR
    / "anomaly_evaluation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

CONTAMINATION = 0.03

N_ESTIMATORS = 300


# ============================================================
# ACTUAL FEATURES AVAILABLE IN FINSHIELD
# ============================================================

ANOMALY_FEATURES = [
    "amount",
    "amount_log",

    "transaction_hour",
    "transaction_day_of_week",
    "transaction_day",
    "transaction_month",

    "is_weekend",
    "is_night_transaction",
    "is_business_hour",

    "is_transfer",
    "is_withdrawal",
    "is_purchase",
    "is_deposit",
    "is_bill_payment",

    "is_mobile",
    "is_web",
    "is_atm",
    "is_pos",

    "has_beneficiary",

    "account_transaction_count",
    "account_10min_transaction_count",

    "is_high_velocity",

    "large_amount_flag",
    "structuring_amount_flag",
    "velocity_risk_flag",

    "device_anomaly_flag",
    "ip_anomaly_flag",

    "is_new_device_for_account",
    "is_new_ip_for_account",

    "unique_devices_per_account",
    "unique_ips_per_account",

    "device_transaction_count",
    "ip_transaction_count",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING FEATURE DATA")
    print("=" * 60)

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    df = pd.read_csv(FEATURE_FILE)

    print(f"Rows loaded       : {len(df):,}")
    print(f"Columns loaded    : {len(df.columns):,}")

    return df


# ============================================================
# VALIDATE FEATURES
# ============================================================

def validate_features(df):

    print("\n" + "=" * 60)
    print("VALIDATING ANOMALY FEATURES")
    print("=" * 60)

    missing = [
        feature
        for feature in ANOMALY_FEATURES
        if feature not in df.columns
    ]

    if missing:

        print("\nMissing features:")

        for feature in missing:
            print(f"  - {feature}")

        raise ValueError(
            "Required anomaly-detection features are missing."
        )

    print(
        f"Required features : "
        f"{len(ANOMALY_FEATURES)}"
    )

    print("All features found : YES")


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    print("\n" + "=" * 60)
    print("PREPARING FEATURES")
    print("=" * 60)

    X = df[ANOMALY_FEATURES].copy()

    # Convert values to numeric
    for column in X.columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values with median
    X = X.fillna(
        X.median()
    )

    print(
        f"Feature matrix    : "
        f"{X.shape[0]:,} rows × "
        f"{X.shape[1]} features"
    )

    print(
        f"Missing values     : "
        f"{X.isna().sum().sum()}"
    )

    return X


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

def train_model(X):

    print("\n" + "=" * 60)
    print("TRAINING ISOLATION FOREST")
    print("=" * 60)

    print(
        f"Estimators        : "
        f"{N_ESTIMATORS}"
    )

    print(
        f"Contamination     : "
        f"{CONTAMINATION}"
    )

    print(
        f"Random state      : "
        f"{RANDOM_STATE}"
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X)

    print(
        "\nIsolation Forest training complete."
    )

    return model


# ============================================================
# SCORE TRANSACTIONS
# ============================================================

def score_transactions(
    model,
    df,
    X
):

    print("\n" + "=" * 60)
    print("GENERATING ANOMALY SCORES")
    print("=" * 60)

    predictions = model.predict(X)

    decision_scores = model.decision_function(X)

    # Lower Isolation Forest decision score
    # means more anomalous.
    #
    # We reverse it so that:
    #
    # higher anomaly_score = more anomalous

    anomaly_score = -decision_scores

    scored_df = df.copy()

    scored_df["anomaly_score"] = (
        anomaly_score
    )

    scored_df["is_anomaly"] = (
        predictions == -1
    ).astype(int)

    # Convert anomaly score into percentile
    scored_df["anomaly_percentile"] = (
        scored_df["anomaly_score"]
        .rank(pct=True)
        * 100
    )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def assign_risk(percentile):

        if percentile >= 99:

            return "critical"

        elif percentile >= 95:

            return "high"

        elif percentile >= 90:

            return "medium"

        elif percentile >= 75:

            return "low"

        else:

            return "minimal"

    scored_df["anomaly_risk_level"] = (
        scored_df["anomaly_percentile"]
        .apply(assign_risk)
    )

    anomaly_count = int(
        scored_df["is_anomaly"].sum()
    )

    anomaly_rate = (
        scored_df["is_anomaly"].mean()
        * 100
    )

    print(
        f"Anomalies detected : "
        f"{anomaly_count:,}"
    )

    print(
        f"Anomaly rate       : "
        f"{anomaly_rate:.2f}%"
    )

    print("\nAnomaly risk distribution:")

    print(
        scored_df[
            "anomaly_risk_level"
        ]
        .value_counts()
        .sort_index()
    )

    return scored_df


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(scored_df):

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION EVALUATION")
    print("=" * 60)

    if "is_fraud" not in scored_df.columns:

        print(
            "Fraud ground truth not available."
        )

        return None

    y_true = scored_df["is_fraud"]

    y_pred = scored_df["is_anomaly"]

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nConfusion Matrix:")

    print(cm)

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    fraud_precision = report.get(
        "1",
        {}
    ).get(
        "precision",
        0
    )

    fraud_recall = report.get(
        "1",
        {}
    ).get(
        "recall",
        0
    )

    fraud_f1 = report.get(
        "1",
        {}
    ).get(
        "f1-score",
        0
    )

    print("\nFraud detection metrics:")

    print(
        f"Precision : "
        f"{fraud_precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{fraud_recall:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{fraud_f1:.4f}"
    )

    # ========================================================
    # ROC AUC
    # ========================================================

    try:

        roc_auc = roc_auc_score(
            y_true,
            scored_df["anomaly_score"]
        )

        print(
            f"ROC AUC   : "
            f"{roc_auc:.4f}"
        )

    except ValueError:

        roc_auc = np.nan

    # ========================================================
    # CONFUSION MATRIX VALUES
    # ========================================================

    tn, fp, fn, tp = cm.ravel()

    print("\nDetection summary:")

    print(
        f"True Negatives  : "
        f"{tn:,}"
    )

    print(
        f"False Positives  : "
        f"{fp:,}"
    )

    print(
        f"False Negatives  : "
        f"{fn:,}"
    )

    print(
        f"True Positives   : "
        f"{tp:,}"
    )

    # ========================================================
    # EVALUATION DATAFRAME
    # ========================================================

    evaluation = pd.DataFrame(
        [
            {
                "model": "IsolationForest",

                "contamination":
                    CONTAMINATION,

                "precision":
                    fraud_precision,

                "recall":
                    fraud_recall,

                "f1_score":
                    fraud_f1,

                "roc_auc":
                    roc_auc,

                "true_negatives":
                    tn,

                "false_positives":
                    fp,

                "false_negatives":
                    fn,

                "true_positives":
                    tp,

                "anomaly_count":
                    int(
                        scored_df[
                            "is_anomaly"
                        ].sum()
                    ),

                "anomaly_rate":
                    float(
                        scored_df[
                            "is_anomaly"
                        ].mean()
                    ),
            }
        ]
    )

    return evaluation


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    joblib.dump(
        model,
        MODEL_FILE
    )

    print(
        "Model saved to:"
    )

    print(MODEL_FILE)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    scored_df,
    evaluation
):

    print("\n" + "=" * 60)
    print("SAVING ANOMALY OUTPUTS")
    print("=" * 60)

    scored_df.to_csv(
        SCORED_FILE,
        index=False
    )

    print(
        "Scored transactions:"
    )

    print(SCORED_FILE)

    if evaluation is not None:

        evaluation.to_csv(
            EVALUATION_FILE,
            index=False
        )

        print(
            "\nEvaluation:"
        )

        print(EVALUATION_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("FINSHIELD AI")
    print("ANOMALY DETECTION ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Validate
    # --------------------------------------------------------

    validate_features(df)

    # --------------------------------------------------------
    # 3. Prepare
    # --------------------------------------------------------

    X = prepare_features(df)

    # --------------------------------------------------------
    # 4. Train
    # --------------------------------------------------------

    model = train_model(X)

    # --------------------------------------------------------
    # 5. Score
    # --------------------------------------------------------

    scored_df = score_transactions(
        model,
        df,
        X
    )

    # --------------------------------------------------------
    # 6. Evaluate
    # --------------------------------------------------------

    evaluation = evaluate_model(
        scored_df
    )

    # --------------------------------------------------------
    # 7. Save model
    # --------------------------------------------------------

    save_model(model)

    # --------------------------------------------------------
    # 8. Save outputs
    # --------------------------------------------------------

    save_outputs(
        scored_df,
        evaluation
    )

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ANOMALY DETECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()