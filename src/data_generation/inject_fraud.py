"""
FinShield AI
Fraud Scenario Injection Engine

Injects realistic fraud patterns into the synthetic transaction dataset.

Fraud scenarios:
1. Account takeover
2. Amount anomaly
3. Velocity fraud
4. Device anomaly
5. Location/IP anomaly
6. Merchant anomaly
"""

from pathlib import Path
import random

import numpy as np
import pandas as pd

from src.config import SYNTHETIC_DATA_DIR, RANDOM_SEED


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = SYNTHETIC_DATA_DIR / "transactions.csv"
OUTPUT_FILE = SYNTHETIC_DATA_DIR / "transactions_fraud.csv"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# Percentage of transactions affected by each scenario
FRAUD_SCENARIO_RATIOS = {
    "account_takeover": 0.006,
    "amount_anomaly": 0.006,
    "velocity_fraud": 0.006,
    "device_anomaly": 0.005,
    "location_anomaly": 0.004,
    "merchant_anomaly": 0.003,
}


# ============================================================
# VALIDATION
# ============================================================

def validate_input(df: pd.DataFrame):
    """Validate that the transaction dataset contains required columns."""

    required_columns = [
        "transaction_id",
        "account_id",
        "amount",
        "device_id",
        "ip_address",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# HELPER
# ============================================================

def select_indices(df, ratio):
    """Randomly select transaction indices."""

    count = max(1, int(len(df) * ratio))

    count = min(count, len(df))

    return np.random.choice(
        df.index,
        size=count,
        replace=False,
    )


def mark_fraud(
    df,
    indices,
    scenario,
):
    """Mark selected transactions as fraudulent."""

    df.loc[indices, "is_fraud"] = 1

    # Do not overwrite an existing scenario.
    # This allows combined fraud patterns to exist.
    mask = df.loc[indices, "scenario_type"] == "normal"

    normal_indices = df.loc[indices].index[mask]

    df.loc[normal_indices, "scenario_type"] = scenario

    return df


# ============================================================
# 1. ACCOUNT TAKEOVER
# ============================================================

def inject_account_takeover(df):
    """
    Simulates an attacker gaining control of a legitimate account.

    Characteristics:
    - unusual device
    - unusual IP
    - unusually large transaction
    """

    indices = select_indices(
        df,
        FRAUD_SCENARIO_RATIOS["account_takeover"],
    )

    for idx in indices:

        # Generate a suspicious device
        df.at[idx, "device_id"] = (
            f"fraud_device_{random.randint(10000, 99999)}"
        )

        # Generate a suspicious IP
        df.at[idx, "ip_address"] = (
            f"185.{random.randint(1, 250)}."
            f"{random.randint(1, 250)}."
            f"{random.randint(1, 250)}"
        )

        # Increase transaction amount
        df.at[idx, "amount"] = round(
            float(df.at[idx, "amount"])
            * random.uniform(3, 10),
            2,
        )

    df = mark_fraud(
        df,
        indices,
        "account_takeover",
    )

    return df


# ============================================================
# 2. AMOUNT ANOMALY
# ============================================================

def inject_amount_anomaly(df):
    """
    Simulates transactions significantly larger than
    the customer's normal transaction behavior.
    """

    indices = select_indices(
        df,
        FRAUD_SCENARIO_RATIOS["amount_anomaly"],
    )

    for idx in indices:

        original_amount = float(df.at[idx, "amount"])

        multiplier = random.uniform(10, 30)

        df.at[idx, "amount"] = round(
            max(original_amount * multiplier, 50000),
            2,
        )

    df = mark_fraud(
        df,
        indices,
        "amount_anomaly",
    )

    return df


# ============================================================
# 3. VELOCITY FRAUD
# ============================================================

def inject_velocity_fraud(df):
    """
    Simulates many transactions occurring rapidly
    from the same account.

    The transaction dataset is sorted by timestamp and
    suspicious transactions are clustered around one account.
    """

    indices = select_indices(
        df,
        FRAUD_SCENARIO_RATIOS["velocity_fraud"],
    )

    # Group selected transactions by account
    selected = df.loc[indices].copy()

    accounts = selected["account_id"].unique()

    for account in accounts:

        account_indices = selected[
            selected["account_id"] == account
        ].index

        if len(account_indices) < 2:
            continue

        # Create rapid transaction timestamps
        base_time = pd.to_datetime(
            df.loc[account_indices[0], "timestamp"]
        )

        for i, idx in enumerate(account_indices):

            df.at[idx, "timestamp"] = (
                base_time
                + pd.Timedelta(seconds=i * random.randint(10, 60))
            )

            df.at[idx, "amount"] = round(
                random.uniform(5000, 25000),
                2,
            )

    df = mark_fraud(
        df,
        indices,
        "velocity_fraud",
    )

    return df


# ============================================================
# 4. DEVICE ANOMALY
# ============================================================

def inject_device_anomaly(df):
    """
    Simulates transactions coming from previously unseen
    or suspicious devices.
    """

    indices = select_indices(
        df,
        FRAUD_SCENARIO_RATIOS["device_anomaly"],
    )

    for idx in indices:

        df.at[idx, "device_id"] = (
            f"unknown_device_{random.randint(100000, 999999)}"
        )

    df = mark_fraud(
        df,
        indices,
        "device_anomaly",
    )

    return df


# ============================================================
# 5. LOCATION / IP ANOMALY
# ============================================================

def inject_location_anomaly(df):
    """
    Simulates transactions originating from unusual
    IP addresses.
    """

    indices = select_indices(
        df,
        FRAUD_SCENARIO_RATIOS["location_anomaly"],
    )

    for idx in indices:

        # Reserved/test-like suspicious IP range
        df.at[idx, "ip_address"] = (
            f"103.{random.randint(1, 250)}."
            f"{random.randint(1, 250)}."
            f"{random.randint(1, 250)}"
        )

    df = mark_fraud(
        df,
        indices,
        "location_anomaly",
    )

    return df


# ============================================================
# 6. MERCHANT ANOMALY
# ============================================================

def inject_merchant_anomaly(df):
    """
    Simulates suspicious transactions involving unusual
    merchants.

    If merchant_id exists, unusual merchant IDs are created.
    """

    indices = select_indices(
        df,
        FRAUD_SCENARIO_RATIOS["merchant_anomaly"],
    )

    if "merchant_id" in df.columns:

        for idx in indices:

            df.at[idx, "merchant_id"] = (
                f"suspicious_merchant_{random.randint(1000, 9999)}"
            )

    df = mark_fraud(
        df,
        indices,
        "merchant_anomaly",
    )

    return df


# ============================================================
# MAIN FRAUD INJECTION PIPELINE
# ============================================================

def inject_fraud_scenarios(df):
    """Run all fraud injection scenarios."""

    print("\nInjecting fraud scenarios...")

    df = inject_account_takeover(df)
    print("  ✓ Account takeover")

    df = inject_amount_anomaly(df)
    print("  ✓ Amount anomaly")

    df = inject_velocity_fraud(df)
    print("  ✓ Velocity fraud")

    df = inject_device_anomaly(df)
    print("  ✓ Device anomaly")

    df = inject_location_anomaly(df)
    print("  ✓ Location anomaly")

    df = inject_merchant_anomaly(df)
    print("  ✓ Merchant anomaly")

    return df


# ============================================================
# VALIDATION REPORT
# ============================================================

def generate_report(df):
    """Print fraud distribution and validation statistics."""

    print("\n" + "=" * 60)
    print("FRAUD INJECTION REPORT")
    print("=" * 60)

    total = len(df)
    fraud_count = int(df["is_fraud"].sum())

    print(f"\nTotal transactions : {total:,}")
    print(f"Fraud transactions : {fraud_count:,}")
    print(
        f"Fraud percentage   : "
        f"{(fraud_count / total) * 100:.2f}%"
    )

    print("\nScenario distribution:")
    print(
        df["scenario_type"]
        .value_counts()
        .to_string()
    )

    print("\nFraud label distribution:")
    print(
        df["is_fraud"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nMissing values:")
    print(
        df.isnull()
        .sum()
        .loc[lambda x: x > 0]
        .to_string()
        if df.isnull().sum().sum() > 0
        else "No missing values"
    )

    print("\nDuplicate transaction IDs:")

    if "transaction_id" in df.columns:
        print(
            df["transaction_id"]
            .duplicated()
            .sum()
        )
    else:
        print("transaction_id column not found")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINSHIELD AI — FRAUD SCENARIO INJECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load transactions
    # --------------------------------------------------------

    print(f"\nLoading: {INPUT_FILE}")

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nTransaction dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run the transaction generator first:\n"
            f"python -m src.data_generation.generator"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df):,} transactions")

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input(df)

    # --------------------------------------------------------
    # Create ground-truth columns
    # --------------------------------------------------------

    df["is_fraud"] = 0
    df["is_aml"] = 0
    df["scenario_type"] = "normal"

    # --------------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

    # --------------------------------------------------------
    # Inject fraud
    # --------------------------------------------------------

    df = inject_fraud_scenarios(df)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df = df.sort_values(
            "timestamp"
        ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved fraud dataset:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    generate_report(df)

    print("\n" + "=" * 60)
    print("FRAUD INJECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()