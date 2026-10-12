"""
FinShield AI
Transaction Feature Engineering

Transforms the AML/fraud transaction dataset into a machine-learning
ready feature dataset.

Input:
    data/synthetic/transactions_aml.csv

Output:
    data/processed/transactions_features.csv

Feature groups:
    1. Time features
    2. Amount features
    3. Account behavioral features
    4. Transaction velocity
    5. Device features
    6. IP features
    7. Beneficiary features
    8. Transaction-type features
    9. Channel features
    10. Risk-indicator features
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from src.config import (
    SYNTHETIC_DATA_DIR,
    PROCESSED_DATA_DIR,
)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = (
    SYNTHETIC_DATA_DIR
    / "transactions_aml.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "transactions_features.csv"
)


# ============================================================
# VALIDATION
# ============================================================

def validate_input(df: pd.DataFrame):
    """Validate required transaction columns."""

    required_columns = [
        "transaction_id",
        "timestamp",
        "account_id",
        "amount",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# 1. TIME FEATURES
# ============================================================

def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create transaction time-based features."""

    print("  ✓ Creating time features")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values found."
        )

    df["transaction_hour"] = (
        df["timestamp"].dt.hour
    )

    df["transaction_day"] = (
        df["timestamp"].dt.day
    )

    df["transaction_month"] = (
        df["timestamp"].dt.month
    )

    df["transaction_day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["transaction_day_of_week"] >= 5
    ).astype(int)

    # Transactions between midnight and 5 AM
    df["is_night_transaction"] = (
        df["transaction_hour"] < 5
    ).astype(int)

    # Working-hour indicator
    df["is_business_hour"] = (
        (df["transaction_hour"] >= 9)
        & (df["transaction_hour"] <= 18)
    ).astype(int)

    return df


# ============================================================
# 2. AMOUNT FEATURES
# ============================================================

def create_amount_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create transaction amount-related features."""

    print("  ✓ Creating amount features")

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    if df["amount"].isna().any():
        raise ValueError(
            "Invalid transaction amount values found."
        )

    # Log transformation reduces the effect of extreme values.
    df["amount_log"] = np.log1p(
        df["amount"]
    )

    # Global amount statistics
    global_mean = df["amount"].mean()
    global_std = df["amount"].std()

    if global_std == 0 or pd.isna(global_std):
        global_std = 1.0

    df["amount_zscore"] = (
        (df["amount"] - global_mean)
        / global_std
    )

    # High-value transaction indicator
    high_value_threshold = (
        df["amount"].quantile(0.95)
    )

    df["is_high_value"] = (
        df["amount"] >= high_value_threshold
    ).astype(int)

    # Very high-value transaction
    extreme_value_threshold = (
        df["amount"].quantile(0.99)
    )

    df["is_extreme_value"] = (
        df["amount"] >= extreme_value_threshold
    ).astype(int)

    return df


# ============================================================
# 3. ACCOUNT BEHAVIOR FEATURES
# ============================================================

def create_account_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create account-level behavioral features."""

    print("  ✓ Creating account behavioral features")

    account_group = df.groupby(
        "account_id"
    )

    # Number of transactions made by account
    df["account_transaction_count"] = (
        df["account_id"]
        .map(
            account_group.size()
        )
        .astype(int)
    )

    # Average transaction amount
    account_mean_amount = (
        account_group["amount"]
        .mean()
    )

    df["account_avg_amount"] = (
        df["account_id"]
        .map(account_mean_amount)
    )

    # Maximum transaction amount
    account_max_amount = (
        account_group["amount"]
        .max()
    )

    df["account_max_amount"] = (
        df["account_id"]
        .map(account_max_amount)
    )

    # Minimum transaction amount
    account_min_amount = (
        account_group["amount"]
        .min()
    )

    df["account_min_amount"] = (
        df["account_id"]
        .map(account_min_amount)
    )

    # Difference from account's average
    df["amount_vs_account_avg"] = (
        df["amount"]
        - df["account_avg_amount"]
    )

    # Ratio compared with account average
    df["amount_to_account_avg_ratio"] = (
        df["amount"]
        / df["account_avg_amount"].replace(
            0,
            np.nan,
        )
    )

    df["amount_to_account_avg_ratio"] = (
        df["amount_to_account_avg_ratio"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0)
    )

    return df


# ============================================================
# 4. TRANSACTION VELOCITY
# ============================================================

def create_velocity_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create transaction-frequency and velocity features.

    These are especially useful for:
        - velocity fraud
        - rapid fund movement
        - account takeover
        - AML behavior
    """

    print("  ✓ Creating transaction velocity features")

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Transaction count per account
    # --------------------------------------------------------

    df["account_transaction_count"] = (
        df.groupby("account_id")[
            "transaction_id"
        ].transform("count")
    )

    # --------------------------------------------------------
    # Transactions per account per day
    # --------------------------------------------------------

    transaction_date = (
        df["timestamp"].dt.date
    )

    df["_transaction_date"] = (
        transaction_date
    )

    df["account_daily_transaction_count"] = (
        df.groupby(
            [
                "account_id",
                "_transaction_date",
            ]
        )["transaction_id"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Transactions per account per hour
    # --------------------------------------------------------

    transaction_hour_bucket = (
        df["timestamp"].dt.floor("h")
    )

    df["_hour_bucket"] = (
        transaction_hour_bucket
    )

    df["account_hourly_transaction_count"] = (
        df.groupby(
            [
                "account_id",
                "_hour_bucket",
            ]
        )["transaction_id"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Transactions within 10-minute period
    # --------------------------------------------------------

    ten_minute_bucket = (
        df["timestamp"].dt.floor("10min")
    )

    df["_10min_bucket"] = (
        ten_minute_bucket
    )

    df["account_10min_transaction_count"] = (
        df.groupby(
            [
                "account_id",
                "_10min_bucket",
            ]
        )["transaction_id"]
        .transform("count")
    )

    # --------------------------------------------------------
    # Rapid transaction indicator
    # --------------------------------------------------------

    df["is_high_velocity"] = (
        df["account_10min_transaction_count"] >= 3
    ).astype(int)

    # --------------------------------------------------------
    # Clean temporary columns
    # --------------------------------------------------------

    df.drop(
        columns=[
            "_transaction_date",
            "_hour_bucket",
            "_10min_bucket",
        ],
        inplace=True,
    )

    return df


# ============================================================
# 5. DEVICE FEATURES
# ============================================================

def create_device_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create account-device behavioral features.

    IMPORTANT:
    Instead of treating every account with multiple devices
    as suspicious, this version detects whether a device is
    NEW for the account at the time of the transaction.
    """

    print("  ✓ Creating device features")

    if "device_id" not in df.columns:

        df["unique_devices_per_account"] = 0

        df["is_multi_device_account"] = 0

        df["is_new_device_for_account"] = 0

        df["device_transaction_count"] = 0

        return df

    # --------------------------------------------------------
    # Total unique devices used by each account
    #
    # This remains a descriptive feature.
    # It is NOT directly used as the anomaly flag.
    # --------------------------------------------------------

    device_count = (
        df.groupby("account_id")[
            "device_id"
        ].nunique()
    )

    df["unique_devices_per_account"] = (
        df["account_id"]
        .map(device_count)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Detect NEW DEVICE for account
    #
    # Because the dataframe was sorted chronologically in
    # create_velocity_features(), the first occurrence of an
    # account-device combination is considered new.
    # --------------------------------------------------------

    account_device_key = (
        df["account_id"].astype(str)
        + "_"
        + df["device_id"].astype(str)
    )

    previous_device_usage = (
        account_device_key
        .duplicated(
            keep="first"
        )
    )

    df["is_new_device_for_account"] = (
        ~previous_device_usage
    ).astype(int)

    # --------------------------------------------------------
    # Descriptive multi-device feature
    #
    # We retain this for analysis, but the anomaly flag
    # below is based on NEW device behavior.
    # --------------------------------------------------------

    df["is_multi_device_account"] = (
        df["unique_devices_per_account"] >= 3
    ).astype(int)

    # --------------------------------------------------------
    # Number of transactions from each device
    # --------------------------------------------------------

    device_transaction_count = (
        df.groupby("device_id")[
            "transaction_id"
        ].transform("count")
    )

    df["device_transaction_count"] = (
        device_transaction_count
    )

    return df


# ============================================================
# 6. IP FEATURES
# ============================================================

def create_ip_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create account-IP behavioral features.

    IMPORTANT:
    Instead of treating every account with multiple IPs
    as suspicious, this version detects whether an IP is
    NEW for the account at the time of the transaction.
    """

    print("  ✓ Creating IP features")

    if "ip_address" not in df.columns:

        df["unique_ips_per_account"] = 0

        df["is_multi_ip_account"] = 0

        df["is_new_ip_for_account"] = 0

        df["ip_transaction_count"] = 0

        return df

    # --------------------------------------------------------
    # Total unique IP addresses used by each account
    #
    # Descriptive feature only.
    # --------------------------------------------------------

    ip_count = (
        df.groupby("account_id")[
            "ip_address"
        ].nunique()
    )

    df["unique_ips_per_account"] = (
        df["account_id"]
        .map(ip_count)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Detect NEW IP for account
    # --------------------------------------------------------

    account_ip_key = (
        df["account_id"].astype(str)
        + "_"
        + df["ip_address"].astype(str)
    )

    previous_ip_usage = (
        account_ip_key
        .duplicated(
            keep="first"
        )
    )

    df["is_new_ip_for_account"] = (
        ~previous_ip_usage
    ).astype(int)

    # --------------------------------------------------------
    # Descriptive multi-IP feature
    # --------------------------------------------------------

    df["is_multi_ip_account"] = (
        df["unique_ips_per_account"] >= 3
    ).astype(int)

    # --------------------------------------------------------
    # Number of transactions from each IP
    # --------------------------------------------------------

    ip_transaction_count = (
        df.groupby("ip_address")[
            "transaction_id"
        ].transform("count")
    )

    df["ip_transaction_count"] = (
        ip_transaction_count
    )

    return df


# ============================================================
# 7. BENEFICIARY FEATURES
# ============================================================

def create_beneficiary_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create beneficiary relationship features."""

    print("  ✓ Creating beneficiary features")

    if "beneficiary_account_id" not in df.columns:

        df["has_beneficiary"] = 0

        df["unique_beneficiaries_per_account"] = 0

        df["is_self_transfer"] = 0

        return df

    beneficiary = (
        df["beneficiary_account_id"]
        .fillna("")
        .astype(str)
    )

    df["has_beneficiary"] = (
        beneficiary.str.len() > 0
    ).astype(int)

    # Number of unique beneficiaries used by account
    beneficiary_count = (
        df[
            df["has_beneficiary"] == 1
        ]
        .groupby("account_id")[
            "beneficiary_account_id"
        ]
        .nunique()
    )

    df["unique_beneficiaries_per_account"] = (
        df["account_id"]
        .map(beneficiary_count)
        .fillna(0)
        .astype(int)
    )

    # Self-transfer indicator
    df["is_self_transfer"] = (
        (
            df["account_id"].astype(str)
            == df["beneficiary_account_id"].astype(str)
        )
        & (
            df["has_beneficiary"] == 1
        )
    ).astype(int)

    return df


# ============================================================
# 8. TRANSACTION TYPE FEATURES
# ============================================================

def create_transaction_type_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create transaction-type indicators."""

    print("  ✓ Creating transaction-type features")

    if "transaction_type" not in df.columns:
        return df

    transaction_type = (
        df["transaction_type"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    df["is_transfer"] = (
        transaction_type == "transfer"
    ).astype(int)

    df["is_withdrawal"] = (
        transaction_type == "withdrawal"
    ).astype(int)

    df["is_deposit"] = (
        transaction_type == "deposit"
    ).astype(int)

    df["is_purchase"] = (
        transaction_type == "purchase"
    ).astype(int)

    df["is_bill_payment"] = (
        transaction_type == "bill_payment"
    ).astype(int)

    return df


# ============================================================
# 9. CHANNEL FEATURES
# ============================================================

def create_channel_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create transaction-channel indicators."""

    print("  ✓ Creating channel features")

    if "channel" not in df.columns:
        return df

    channel = (
        df["channel"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    df["is_mobile"] = (
        channel == "mobile"
    ).astype(int)

    df["is_web"] = (
        channel == "web"
    ).astype(int)

    df["is_atm"] = (
        channel == "atm"
    ).astype(int)

    df["is_pos"] = (
        channel == "pos"
    ).astype(int)

    return df


# ============================================================
# 10. SUSPICIOUS BEHAVIOR FEATURES
# ============================================================

def create_risk_indicator_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create simple behavioral indicators.

    These are NOT final risk scores.

    The actual FinShield risk engine will be built later.
    """

    print("  ✓ Creating suspicious-behavior indicators")

    # --------------------------------------------------------
    # Large transaction
    # --------------------------------------------------------

    df["large_amount_flag"] = (
        df["amount"] >= 50000
    ).astype(int)

    # --------------------------------------------------------
    # Structuring-like amount
    # --------------------------------------------------------

    df["structuring_amount_flag"] = (
        (df["amount"] >= 8000)
        & (df["amount"] < 10000)
    ).astype(int)

    # --------------------------------------------------------
    # High transaction velocity
    # --------------------------------------------------------

    if "account_10min_transaction_count" in df.columns:

        df["velocity_risk_flag"] = (
            df["account_10min_transaction_count"] >= 3
        ).astype(int)

    else:

        df["velocity_risk_flag"] = 0

    # --------------------------------------------------------
    # Unusual night transaction
    # --------------------------------------------------------

    if "is_night_transaction" in df.columns:

        df["night_activity_flag"] = (
            df["is_night_transaction"]
        )

    else:

        df["night_activity_flag"] = 0

    # --------------------------------------------------------
    # NEW DEVICE anomaly
    #
    # IMPORTANT:
    # Do NOT use is_multi_device_account here.
    # --------------------------------------------------------

    if "is_new_device_for_account" in df.columns:

        df["device_anomaly_flag"] = (
            df["is_new_device_for_account"]
        )

    else:

        df["device_anomaly_flag"] = 0

    # --------------------------------------------------------
    # NEW IP anomaly
    #
    # IMPORTANT:
    # Do NOT use is_multi_ip_account here.
    # --------------------------------------------------------

    if "is_new_ip_for_account" in df.columns:

        df["ip_anomaly_flag"] = (
            df["is_new_ip_for_account"]
        )

    else:

        df["ip_anomaly_flag"] = 0

    return df


# ============================================================
# 11. CLEAN DATA
# ============================================================

def clean_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean generated feature columns."""

    print("  ✓ Cleaning feature dataset")

    # Replace infinite values
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # Numeric columns
    numeric_columns = (
        df.select_dtypes(
            include=[np.number]
        ).columns
    )

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(0)
    )

    # Keep categorical missing values explicit
    categorical_columns = (
        df.select_dtypes(
            include=["object"]
        ).columns
    )

    df[categorical_columns] = (
        df[categorical_columns]
        .fillna("unknown")
    )

    return df


# ============================================================
# 12. FEATURE PIPELINE
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the complete feature-engineering pipeline."""

    print("\nCreating features...\n")

    df = create_time_features(df)

    df = create_amount_features(df)

    df = create_account_features(df)

    df = create_velocity_features(df)

    df = create_device_features(df)

    df = create_ip_features(df)

    df = create_beneficiary_features(df)

    df = create_transaction_type_features(df)

    df = create_channel_features(df)

    df = create_risk_indicator_features(df)

    df = clean_features(df)

    return df


# ============================================================
# REPORT
# ============================================================

def generate_report(
    df: pd.DataFrame,
):
    """Print feature-engineering results."""

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 60)

    print(
        f"\nRows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    if "is_fraud" in df.columns:

        print(
            f"\nFraud transactions : "
            f"{int(df['is_fraud'].sum()):,}"
        )

    if "is_aml" in df.columns:

        print(
            f"AML transactions   : "
            f"{int(df['is_aml'].sum()):,}"
        )

    # --------------------------------------------------------
    # Device/IP diagnostics
    # --------------------------------------------------------

    print("\nDevice/IP behavioral diagnostics:")

    diagnostic_columns = [
        "unique_devices_per_account",
        "is_multi_device_account",
        "is_new_device_for_account",
        "unique_ips_per_account",
        "is_multi_ip_account",
        "is_new_ip_for_account",
        "device_anomaly_flag",
        "ip_anomaly_flag",
    ]

    for column in diagnostic_columns:

        if column in df.columns:

            print(
                f"{column:<35} "
                f"{int(df[column].sum()):,}"
            )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    print("\nFeature columns:")

    for i, column in enumerate(
        df.columns,
        start=1,
    ):

        print(
            f"{i:>3}. {column}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing values:")

    missing = (
        df.isnull()
        .sum()
        .loc[lambda x: x > 0]
    )

    if len(missing) == 0:

        print("No missing values")

    else:

        print(
            missing.to_string()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINSHIELD AI — TRANSACTION FEATURE ENGINEERING")
    print("=" * 60)

    # --------------------------------------------------------
    # Check input
    # --------------------------------------------------------

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run AML injection first:\n"
            f"python -m src.data_generation.inject_aml"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df):,} transactions"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input(df)

    # --------------------------------------------------------
    # Build features
    # --------------------------------------------------------

    df = build_features(df)

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved feature dataset:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    generate_report(df)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()