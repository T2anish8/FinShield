"""
FinShield AI
AML Scenario Injection Engine

Injects realistic Anti-Money Laundering patterns:

1. Structuring / Smurfing
2. Mule Accounts
3. Layering
4. Circular Transfers
5. Dormant Account Activation
6. Rapid Fund Movement
"""

import random

import numpy as np
import pandas as pd

from src.config import SYNTHETIC_DATA_DIR, RANDOM_SEED


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = SYNTHETIC_DATA_DIR / "transactions_fraud.csv"
OUTPUT_FILE = SYNTHETIC_DATA_DIR / "transactions_aml.csv"

random.seed(RANDOM_SEED + 100)
np.random.seed(RANDOM_SEED + 100)


AML_SCENARIO_RATIOS = {
    "structuring": 0.004,
    "mule_account": 0.004,
    "layering": 0.004,
    "circular_transfer": 0.003,
    "dormant_activation": 0.003,
    "rapid_fund_movement": 0.003,
}


# ============================================================
# VALIDATION
# ============================================================

def validate_input(df: pd.DataFrame):
    """Validate required transaction columns."""

    required_columns = [
        "transaction_id",
        "account_id",
        "amount",
        "timestamp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


# ============================================================
# HELPERS
# ============================================================

def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare columns before AML injection.

    Important:
    beneficiary_account_id can be loaded by Pandas as float64
    when the source CSV contains only NaN values. AML injection
    needs to store string account IDs such as ACC_000935.
    """

    if "is_fraud" not in df.columns:
        df["is_fraud"] = 0

    df["is_fraud"] = (
        pd.to_numeric(
            df["is_fraud"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    df["is_aml"] = 0
    df["is_aml"] = df["is_aml"].astype(int)

    if "scenario_type" not in df.columns:
        df["scenario_type"] = "normal"

    df["scenario_type"] = (
        df["scenario_type"]
        .fillna("normal")
        .astype(str)
    )

    # Fix beneficiary account column dtype.
    if "beneficiary_account_id" not in df.columns:
        df["beneficiary_account_id"] = pd.Series(
            None,
            index=df.index,
            dtype="object",
        )
    else:
        df["beneficiary_account_id"] = (
            df["beneficiary_account_id"]
            .astype("object")
        )

    return df


def select_indices(df: pd.DataFrame, ratio: float):
    """Select unique transaction indices."""

    count = max(1, int(len(df) * ratio))
    count = min(count, len(df))

    return np.random.choice(
        df.index,
        size=count,
        replace=False,
    )


def mark_aml(
    df: pd.DataFrame,
    indices,
    scenario: str,
) -> pd.DataFrame:
    """Mark transactions as AML and assign a scenario label."""

    df.loc[indices, "is_aml"] = 1

    # Preserve the first scenario_type assigned to a transaction.
    # This allows fraud + AML overlap without losing the original
    # fraud scenario label.
    current = df.loc[indices, "scenario_type"]

    normal_indices = current.index[
        current == "normal"
    ]

    df.loc[
        normal_indices,
        "scenario_type",
    ] = scenario

    return df


# ============================================================
# 1. STRUCTURING / SMURFING
# ============================================================

def inject_structuring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate multiple transactions just below a suspicious
    reporting threshold.

    Example:
        9,800
        9,500
        9,700
        9,900
    """

    indices = select_indices(
        df,
        AML_SCENARIO_RATIOS["structuring"],
    )

    for idx in indices:
        df.at[idx, "amount"] = round(
            random.uniform(8000, 9900),
            2,
        )

        df.at[idx, "transaction_type"] = "transfer"

    return mark_aml(
        df,
        indices,
        "structuring",
    )


# ============================================================
# 2. MULE ACCOUNT
# ============================================================

def inject_mule_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate a small set of accounts receiving money from
    multiple unrelated sources.
    """

    indices = select_indices(
        df,
        AML_SCENARIO_RATIOS["mule_account"],
    )

    selected_accounts = (
        df.loc[indices, "account_id"]
        .dropna()
        .unique()
    )

    if len(selected_accounts) == 0:
        return df

    mule_accounts = random.sample(
        list(selected_accounts),
        min(5, len(selected_accounts)),
    )

    for idx in indices:
        mule_account = random.choice(mule_accounts)

        df.at[idx, "account_id"] = mule_account
        df.at[idx, "transaction_type"] = "transfer"
        df.at[idx, "amount"] = round(
            random.uniform(5000, 50000),
            2,
        )

    return mark_aml(
        df,
        indices,
        "mule_account",
    )


# ============================================================
# 3. LAYERING
# ============================================================

def inject_layering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate movement of money through multiple accounts.

    Example:

        Account A -> Account B
        Account B -> Account C
        Account C -> Account D
    """

    indices = select_indices(
        df,
        AML_SCENARIO_RATIOS["layering"],
    )

    accounts = (
        df["account_id"]
        .dropna()
        .unique()
    )

    if len(accounts) < 4:
        return df

    for idx in indices:
        chain = random.sample(
            list(accounts),
            4,
        )

        amount = round(
            random.uniform(10000, 75000),
            2,
        )

        position = random.randint(0, 3)

        source_account = chain[position]
        destination_account = chain[
            (position + 1) % 4
        ]

        df.at[idx, "account_id"] = source_account
        df.at[
            idx,
            "beneficiary_account_id",
        ] = destination_account
        df.at[idx, "transaction_type"] = "transfer"
        df.at[idx, "amount"] = amount

    return mark_aml(
        df,
        indices,
        "layering",
    )


# ============================================================
# 4. CIRCULAR TRANSFERS
# ============================================================

def inject_circular_transfers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate transactions participating in circular money
    movement.

    Example:

        A -> B
        B -> C
        C -> A
    """

    indices = select_indices(
        df,
        AML_SCENARIO_RATIOS["circular_transfer"],
    )

    accounts = (
        df["account_id"]
        .dropna()
        .unique()
    )

    if len(accounts) < 3:
        return df

    for idx in indices:
        account_a, account_b, account_c = random.sample(
            list(accounts),
            3,
        )

        pattern = random.randint(0, 2)

        if pattern == 0:
            source = account_a
            destination = account_b
        elif pattern == 1:
            source = account_b
            destination = account_c
        else:
            source = account_c
            destination = account_a

        df.at[idx, "account_id"] = source
        df.at[
            idx,
            "beneficiary_account_id",
        ] = destination
        df.at[idx, "transaction_type"] = "transfer"
        df.at[idx, "amount"] = round(
            random.uniform(15000, 100000),
            2,
        )

    return mark_aml(
        df,
        indices,
        "circular_transfer",
    )


# ============================================================
# 5. DORMANT ACCOUNT ACTIVATION
# ============================================================

def inject_dormant_activation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate a previously inactive account suddenly
    receiving a large transaction.
    """

    indices = select_indices(
        df,
        AML_SCENARIO_RATIOS["dormant_activation"],
    )

    for idx in indices:
        df.at[idx, "amount"] = round(
            random.uniform(25000, 150000),
            2,
        )

        df.at[
            idx,
            "transaction_type",
        ] = random.choice(
            [
                "deposit",
                "transfer",
            ]
        )

    return mark_aml(
        df,
        indices,
        "dormant_activation",
    )


# ============================================================
# 6. RAPID FUND MOVEMENT
# ============================================================

def inject_rapid_fund_movement(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simulate rapid movement of funds through an account.
    """

    indices = select_indices(
        df,
        AML_SCENARIO_RATIOS["rapid_fund_movement"],
    )

    accounts = (
        df["account_id"]
        .dropna()
        .unique()
    )

    if len(accounts) == 0:
        return df

    for idx in indices:
        account = random.choice(
            list(accounts)
        )

        df.at[idx, "account_id"] = account
        df.at[idx, "transaction_type"] = "transfer"
        df.at[idx, "amount"] = round(
            random.uniform(10000, 100000),
            2,
        )

        original_time = pd.to_datetime(
            df.at[idx, "timestamp"]
        )

        df.at[idx, "timestamp"] = (
            original_time
            + pd.Timedelta(
                seconds=random.randint(1, 300)
            )
        )

    return mark_aml(
        df,
        indices,
        "rapid_fund_movement",
    )


# ============================================================
# AML PIPELINE
# ============================================================

def inject_aml_scenarios(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print("\nInjecting AML scenarios...")

    df = inject_structuring(df)
    print("  ✓ Structuring / Smurfing")

    df = inject_mule_accounts(df)
    print("  ✓ Mule accounts")

    df = inject_layering(df)
    print("  ✓ Layering")

    df = inject_circular_transfers(df)
    print("  ✓ Circular transfers")

    df = inject_dormant_activation(df)
    print("  ✓ Dormant account activation")

    df = inject_rapid_fund_movement(df)
    print("  ✓ Rapid fund movement")

    return df


# ============================================================
# REPORT
# ============================================================

def generate_report(df: pd.DataFrame):

    print("\n" + "=" * 60)
    print("AML INJECTION REPORT")
    print("=" * 60)

    total = len(df)

    aml_count = int(
        df["is_aml"].sum()
    )

    fraud_count = int(
        df["is_fraud"].sum()
    )

    combined_count = int(
        (
            (df["is_aml"] == 1)
            & (df["is_fraud"] == 1)
        ).sum()
    )

    print(
        f"\nTotal transactions : {total:,}"
    )

    print(
        f"AML transactions   : {aml_count:,}"
    )

    print(
        f"AML percentage     : "
        f"{(aml_count / total) * 100:.2f}%"
    )

    print(
        f"Fraud transactions : {fraud_count:,}"
    )

    print(
        f"Fraud + AML        : {combined_count:,}"
    )

    print("\nScenario distribution:")

    print(
        df["scenario_type"]
        .value_counts()
        .to_string()
    )

    print("\nAML label distribution:")

    print(
        df["is_aml"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFraud × AML matrix:")

    print(
        pd.crosstab(
            df["is_fraud"],
            df["is_aml"],
            rownames=["is_fraud"],
            colnames=["is_aml"],
        )
    )

    print("\nMissing values:")

    missing = (
        df.isnull()
        .sum()
        .loc[lambda x: x > 0]
    )

    if len(missing) == 0:
        print("No missing values")
    else:
        print(missing.to_string())

    print("\nDuplicate transaction IDs:")

    print(
        df["transaction_id"]
        .duplicated()
        .sum()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINSHIELD AI — AML SCENARIO INJECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        f"\nLoading: {INPUT_FILE}"
    )

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run fraud injection first:\n"
            f"python -m src.data_generation.inject_fraud"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df):,} transactions"
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_input(df)

    # --------------------------------------------------------
    # Prepare labels and columns
    # --------------------------------------------------------

    df = prepare_columns(df)

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values found in transactions."
        )

    # --------------------------------------------------------
    # AML injection
    # --------------------------------------------------------

    df = inject_aml_scenarios(df)

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved AML dataset:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    generate_report(df)

    print("\n" + "=" * 60)
    print("AML INJECTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
