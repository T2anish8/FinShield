from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = PROJECT_ROOT / "data" / "processed" / "transactions_features.csv"
AML_FILE = PROJECT_ROOT / "data" / "processed" / "aml_alerts.csv"
ANOMALY_FILE = PROJECT_ROOT / "data" / "processed" / "transactions_anomaly_scores.csv"
GRAPH_FILE = PROJECT_ROOT / "data" / "processed" / "graph_entity_risk.csv"

FRAUD_MODEL_FILE = PROJECT_ROOT / "models" / "fraud_xgboost.joblib"
FRAUD_FEATURES_FILE = PROJECT_ROOT / "models" / "fraud_feature_columns.joblib"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "unified_risk_scores.csv"
SUMMARY_FILE = OUTPUT_DIR / "unified_risk_summary.csv"


# ============================================================
# RISK WEIGHTS
# ============================================================

FRAUD_WEIGHT = 0.35
AML_WEIGHT = 0.30
ANOMALY_WEIGHT = 0.20
GRAPH_WEIGHT = 0.15

DEFAULT_FRAUD_FEATURES = [
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
    "is_high_velocity",
    "large_amount_flag",
    "structuring_amount_flag",
    "velocity_risk_flag",
]


# ============================================================
# SHARED VALIDATION
# ============================================================

def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def validate_unique_ids(df: pd.DataFrame, id_column: str, label: str) -> None:
    if df[id_column].isna().any():
        raise ValueError(f"{label} contains missing {id_column} values.")
    if df[id_column].duplicated().any():
        duplicates = int(df[id_column].duplicated().sum())
        raise ValueError(
            f"{label} contains {duplicates:,} duplicate {id_column} values."
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    print("=" * 60)
    print("LOADING RISK ENGINE DATA")
    print("=" * 60)

    required_files = [
        FEATURE_FILE,
        AML_FILE,
        ANOMALY_FILE,
        GRAPH_FILE,
        FRAUD_MODEL_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found:\n{file_path}")

    features = pd.read_csv(FEATURE_FILE)
    aml = pd.read_csv(AML_FILE)
    anomaly = pd.read_csv(ANOMALY_FILE)
    graph = pd.read_csv(GRAPH_FILE)

    require_columns(
        features,
        ["transaction_id", "account_id", "amount", "transaction_type", "channel"],
        "Feature dataset",
    )
    require_columns(aml, ["transaction_id", "aml_rule_score"], "AML output")
    require_columns(
        anomaly,
        ["transaction_id", "anomaly_score", "anomaly_percentile"],
        "Anomaly output",
    )
    require_columns(graph, ["account_id", "graph_risk_score"], "Graph output")

    print(f"Feature rows       : {len(features):,}")
    print(f"AML rows           : {len(aml):,}")
    print(f"Anomaly rows       : {len(anomaly):,}")
    print(f"Graph accounts     : {len(graph):,}")

    return features, aml, anomaly, graph


# ============================================================
# PREPARE AML SIGNAL
# ============================================================

def prepare_aml_signal(aml: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PREPARING AML RISK SIGNAL")
    print("=" * 60)

    require_columns(aml, ["transaction_id", "aml_rule_score"], "AML output")
    result = aml[["transaction_id", "aml_rule_score"]].copy()
    result = result.rename(columns={"aml_rule_score": "aml_raw_score"})

    validate_unique_ids(result, "transaction_id", "AML output")
    result["aml_raw_score"] = pd.to_numeric(result["aml_raw_score"], errors="coerce")
    if result["aml_raw_score"].isna().any():
        raise ValueError("AML output contains missing or invalid aml_rule_score values.")

    result["aml_score_normalized"] = result["aml_raw_score"].clip(0, 100)

    print("AML score column   : aml_rule_score")
    print(f"AML transactions   : {len(result):,}")
    print(f"Unique transaction IDs: {result['transaction_id'].nunique():,}")
    print(f"Maximum AML score  : {result['aml_raw_score'].max():.2f}")
    print(f"Mean AML score     : {result['aml_raw_score'].mean():.2f}")
    return result


# ============================================================
# PREPARE ANOMALY SIGNAL
# ============================================================

def prepare_anomaly_signal(anomaly: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PREPARING ANOMALY RISK SIGNAL")
    print("=" * 60)

    require_columns(
        anomaly,
        ["transaction_id", "anomaly_score", "anomaly_percentile"],
        "Anomaly output",
    )
    result = anomaly[["transaction_id", "anomaly_score", "anomaly_percentile"]].copy()
    validate_unique_ids(result, "transaction_id", "Anomaly output")

    for column in ["anomaly_score", "anomaly_percentile"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    if result[["anomaly_score", "anomaly_percentile"]].isna().any().any():
        raise ValueError("Anomaly output contains missing or invalid scores.")

    result["anomaly_score_normalized"] = result["anomaly_percentile"].clip(0, 100)
    print("Anomaly signal prepared.")
    return result


# ============================================================
# PREPARE GRAPH SIGNAL
# ============================================================

def prepare_graph_signal(graph: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("PREPARING GRAPH RISK SIGNAL")
    print("=" * 60)

    require_columns(graph, ["account_id", "graph_risk_score"], "Graph output")
    result = graph[["account_id", "graph_risk_score"]].copy()
    result["account_id"] = result["account_id"].astype(str).str.strip()
    validate_unique_ids(result, "account_id", "Graph output")

    result["graph_score_normalized"] = pd.to_numeric(
        result["graph_risk_score"], errors="coerce"
    )
    if result["graph_score_normalized"].isna().any():
        raise ValueError("Graph output contains missing or invalid graph_risk_score values.")
    result["graph_score_normalized"] = result["graph_score_normalized"].clip(0, 100)

    print("Graph signal prepared.")
    return result[["account_id", "graph_score_normalized"]]


# ============================================================
# LOAD FRAUD MODEL
# ============================================================

def load_fraud_model():
    print("\n" + "=" * 60)
    print("LOADING FRAUD MODEL")
    print("=" * 60)

    model = joblib.load(FRAUD_MODEL_FILE)
    if FRAUD_FEATURES_FILE.exists():
        feature_columns = joblib.load(FRAUD_FEATURES_FILE)
    else:
        feature_columns = DEFAULT_FRAUD_FEATURES

    print(f"Fraud model        : {type(model).__name__}")
    print(f"Fraud features     : {len(feature_columns)}")
    return model, list(feature_columns)


# ============================================================
# CALCULATE FRAUD PROBABILITY
# ============================================================

def calculate_fraud_probability(
    features: pd.DataFrame, model, feature_columns: list[str]
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("CALCULATING FRAUD PROBABILITY")
    print("=" * 60)

    require_columns(features, ["transaction_id"], "Feature dataset")
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"Fraud model features missing from feature dataset: {missing}")

    X = features[feature_columns].copy()
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Fraud model does not support predict_proba().")

    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 not in classes:
        raise ValueError(f"Fraud model classes do not contain positive class 1: {classes}")
    positive_index = classes.index(1)
    fraud_probability = probabilities[:, positive_index] * 100.0

    result = pd.DataFrame(
        {
            "transaction_id": features["transaction_id"].values,
            "fraud_probability": fraud_probability,
        }
    )
    validate_unique_ids(result, "transaction_id", "Fraud probability output")

    print(f"Mean fraud probability : {result['fraud_probability'].mean():.2f}")
    print(f"Maximum fraud probability: {result['fraud_probability'].max():.2f}")
    return result


# ============================================================
# MERGE SIGNALS
# ============================================================

def merge_risk_signals(
    features: pd.DataFrame,
    aml_signal: pd.DataFrame,
    anomaly_signal: pd.DataFrame,
    graph_signal: pd.DataFrame,
    fraud_signal: pd.DataFrame,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("MERGING RISK SIGNALS")
    print("=" * 60)

    base_columns = [
        "transaction_id",
        "account_id",
        "amount",
        "transaction_type",
        "channel",
    ]
    optional_labels = [column for column in ["is_fraud", "is_aml"] if column in features.columns]
    require_columns(features, base_columns, "Feature dataset")

    result = features[base_columns + optional_labels].copy()
    validate_unique_ids(result, "transaction_id", "Feature dataset")
    result["account_id"] = result["account_id"].astype(str).str.strip()

    # Join by stable IDs, never by row order.
    result = result.merge(
        aml_signal[["transaction_id", "aml_raw_score", "aml_score_normalized"]],
        on="transaction_id", how="left", validate="one_to_one", indicator="_aml_merge",
    )
    missing_aml = int((result["_aml_merge"] == "left_only").sum())
    result = result.drop(columns="_aml_merge")

    result = result.merge(
        anomaly_signal[["transaction_id", "anomaly_score", "anomaly_score_normalized"]],
        on="transaction_id", how="left", validate="one_to_one", indicator="_anomaly_merge",
    )
    missing_anomaly = int((result["_anomaly_merge"] == "left_only").sum())
    result = result.drop(columns="_anomaly_merge")

    result = result.merge(
        fraud_signal, on="transaction_id", how="left", validate="one_to_one",
        indicator="_fraud_merge",
    )
    missing_fraud = int((result["_fraud_merge"] == "left_only").sum())
    result = result.drop(columns="_fraud_merge")

    result = result.merge(
        graph_signal, on="account_id", how="left", validate="many_to_one",
        indicator="_graph_merge",
    )
    missing_graph = int((result["_graph_merge"] == "left_only").sum())
    result = result.drop(columns="_graph_merge")

    # Fail rather than silently assigning zero if a transaction-level signal is missing.
    print(f"Missing AML matches     : {missing_aml:,}")
    print(f"Missing anomaly matches : {missing_anomaly:,}")
    print(f"Missing fraud matches   : {missing_fraud:,}")
    print(f"Missing graph matches   : {missing_graph:,}")

    if missing_aml or missing_anomaly or missing_fraud:
        raise ValueError(
            "One or more transaction-level risk signals did not match every feature row. "
            "Check transaction IDs in the input files."
        )

    for column in ["aml_raw_score", "aml_score_normalized", "anomaly_score", 
                   "anomaly_score_normalized", "fraud_probability"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if result[column].isna().any():
            raise ValueError(f"Merged signal contains missing values in {column}.")

    # An account absent from graph output receives zero graph score, with count logged above.
    result["graph_score_normalized"] = pd.to_numeric(
        result["graph_score_normalized"], errors="coerce"
    ).fillna(0).clip(0, 100)

    if len(result) != len(features):
        raise ValueError(
            f"Row count changed during joins: features={len(features):,}, "
            f"merged={len(result):,}."
        )
    validate_unique_ids(result, "transaction_id", "Merged risk signals")
    print(f"Final transaction rows : {len(result):,}")
    return result


# ============================================================
# UNIFIED RISK SCORE
# ============================================================

def calculate_unified_score(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("CALCULATING UNIFIED RISK SCORE")
    print("=" * 60)

    df["unified_risk_score"] = (
        FRAUD_WEIGHT * df["fraud_probability"]
        + AML_WEIGHT * df["aml_score_normalized"]
        + ANOMALY_WEIGHT * df["anomaly_score_normalized"]
        + GRAPH_WEIGHT * df["graph_score_normalized"]
    ).clip(0, 100)

    def assign_risk(score: float) -> str:
        if score >= 80:
            return "critical"
        if score >= 60:
            return "high"
        if score >= 40:
            return "medium"
        if score >= 20:
            return "low"
        return "minimal"

    df["unified_risk_level"] = df["unified_risk_score"].apply(assign_risk)
    print("\nUnified risk distribution:")
    print(df["unified_risk_level"].value_counts().sort_index())
    return df


# ============================================================
# RISK EXPLANATION
# ============================================================

def generate_explanation(row: pd.Series) -> str:
    reasons = []
    if row["fraud_probability"] >= 70:
        reasons.append("high fraud probability")
    if row["aml_score_normalized"] >= 50:
        reasons.append("strong AML rule signals")
    if row["anomaly_score_normalized"] >= 95:
        reasons.append("high anomaly percentile")
    if row["graph_score_normalized"] >= 75:
        reasons.append("high-risk account network")
    if row["amount"] >= 50000:
        reasons.append("large transaction amount")
    return "; ".join(reasons) if reasons else "No dominant risk signal"


def add_explanations(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("GENERATING RISK EXPLANATIONS")
    print("=" * 60)
    df["risk_explanation"] = df.apply(generate_explanation, axis=1)
    return df


# ============================================================
# SUMMARY
# ============================================================

def create_summary(df: pd.DataFrame) -> pd.DataFrame:
    aggregations = {
        "transaction_count": ("transaction_id", "count"),
        "total_amount": ("amount", "sum"),
        "average_risk_score": ("unified_risk_score", "mean"),
    }
    if "is_fraud" in df.columns:
        aggregations["fraud_transactions"] = ("is_fraud", "sum")
    if "is_aml" in df.columns:
        aggregations["aml_transactions"] = ("is_aml", "sum")

    summary = (
        df.groupby("unified_risk_level", dropna=False)
        .agg(**aggregations)
        .reset_index()
        .sort_values("average_risk_score", ascending=False)
    )
    return summary


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SAVING UNIFIED RISK OUTPUTS")
    print("=" * 60)
    df.to_csv(OUTPUT_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    print(f"Unified risk file: {OUTPUT_FILE}")
    print(f"Risk summary     : {SUMMARY_FILE}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("\n" + "=" * 60)
    print("FINSHIELD AI")
    print("UNIFIED RISK ENGINE")
    print("=" * 60)

    features, aml, anomaly, graph = load_data()

    aml_signal = prepare_aml_signal(aml)
    anomaly_signal = prepare_anomaly_signal(anomaly)
    graph_signal = prepare_graph_signal(graph)

    fraud_model, fraud_features = load_fraud_model()
    fraud_signal = calculate_fraud_probability(features, fraud_model, fraud_features)

    result = merge_risk_signals(
        features=features,
        aml_signal=aml_signal,
        anomaly_signal=anomaly_signal,
        graph_signal=graph_signal,
        fraud_signal=fraud_signal,
    )
    result = calculate_unified_score(result)
    result = add_explanations(result)
    summary = create_summary(result)

    print("\n" + "=" * 60)
    print("RISK SUMMARY")
    print("=" * 60)
    print(summary.to_string(index=False))

    save_outputs(result, summary)
    print("\n" + "=" * 60)
    print("UNIFIED RISK ENGINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
