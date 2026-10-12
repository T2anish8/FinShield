"""
FinShield AI
AML Rule-Based Detection Engine

Purpose:
    Detect suspicious AML behavior using transparent,
    explainable rules.

Input:
    data/processed/transactions_features.csv

Output:
    data/processed/aml_alerts.csv
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR
    / "transactions_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR
    / "aml_alerts.csv"
)


# ============================================================
# RULE CONFIGURATION
# ============================================================

RULE_WEIGHTS = {
    "structuring": 20,
    "high_velocity": 20,
    "large_transaction": 10,
    "rapid_fund_movement": 20,
    "multiple_beneficiaries": 10,
    "multiple_devices": 5,
    "multiple_ips": 5,
    "night_activity": 5,
    "dormant_activation": 15,
    "high_amount_deviation": 10,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """Load transaction feature dataset."""

    print("\nLoading feature dataset...")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run:\n"
            f"python -m src.features.transaction_features"
        )

    df = pd.read_csv(INPUT_FILE)

    print(
        f"Loaded {len(df):,} transactions"
    )

    # --------------------------------------------------------
    # Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "transaction_id",
        "timestamp",
        "account_id",
        "amount",
        "is_transfer",
        "has_beneficiary",
        "account_10min_transaction_count",
        "unique_beneficiaries_per_account",
        "is_new_device_for_account",
        "is_new_ip_for_account",
        "is_night_transaction",
        "amount_to_account_avg_ratio",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required AML feature columns:\n"
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamp values found."
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = (
        df.sort_values(
            ["account_id", "timestamp"]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# RULE 1 — STRUCTURING
# ============================================================

def rule_structuring(df):
    """
    Detect transactions with amounts clustered
    below a reporting-like threshold.

    This is an indicator, not proof of structuring.
    """

    return (
        (df["amount"] >= 8000)
        & (df["amount"] < 10000)
    ).astype(int)


# ============================================================
# RULE 2 — HIGH VELOCITY
# ============================================================

def rule_high_velocity(df):
    """
    Detect multiple transactions from the same account
    within a short period.
    """

    return (
        df["account_10min_transaction_count"] >= 3
    ).astype(int)


# ============================================================
# RULE 3 — LARGE TRANSACTION
# ============================================================

def rule_large_transaction(df):
    """Detect unusually large transaction amounts."""

    return (
        df["amount"] >= 50000
    ).astype(int)


# ============================================================
# RULE 4 — RAPID FUND MOVEMENT
# ============================================================

def rule_rapid_fund_movement(df):
    """
    Detect a possible rapid movement of funds.

    Logic:
        A transfer transaction with a beneficiary occurring
        shortly after another transaction from the same account.

    This is an indicator, not proof of layering or laundering.
    """

    working = df[
        [
            "account_id",
            "timestamp",
            "transaction_id",
            "is_transfer",
            "has_beneficiary",
        ]
    ].copy()

    # --------------------------------------------------------
    # Previous transaction timestamp for each account
    # --------------------------------------------------------

    working["previous_timestamp"] = (
        working
        .groupby("account_id")["timestamp"]
        .shift(1)
    )

    # Time since previous account transaction
    working["seconds_since_previous"] = (
        working["timestamp"]
        - working["previous_timestamp"]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Rapid movement threshold
    #
    # 10 minutes or less between account transactions
    # --------------------------------------------------------

    rapid_sequence = (
        working["seconds_since_previous"].notna()
        & (working["seconds_since_previous"] <= 600)
    )

    transfer = (
        working["is_transfer"] == 1
    )

    beneficiary = (
        working["has_beneficiary"] == 1
    )

    result = (
        rapid_sequence
        & transfer
        & beneficiary
    ).astype(int)

    return result.reindex(df.index, fill_value=0)


# ============================================================
# RULE 5 — MULTIPLE BENEFICIARIES
# ============================================================

def rule_multiple_beneficiaries(df):
    """
    Detect accounts interacting with many
    different beneficiary accounts.
    """

    return (
        df["unique_beneficiaries_per_account"] >= 3
    ).astype(int)


# ============================================================
# RULE 6 — NEW / MULTIPLE DEVICES
# ============================================================

def rule_multiple_devices(df):
    """
    Detect a new device being used by an account.

    A legitimate account may have multiple devices,
    so the rule focuses on NEW device behavior rather
    than simply counting the number of devices.
    """

    return (
        df["is_new_device_for_account"] == 1
    ).astype(int)


# ============================================================
# RULE 7 — NEW / MULTIPLE IPS
# ============================================================

def rule_multiple_ips(df):
    """
    Detect a new IP address being used by an account.

    The rule focuses on NEW IP behavior instead of
    treating every multi-IP account as suspicious.
    """

    return (
        df["is_new_ip_for_account"] == 1
    ).astype(int)


# ============================================================
# RULE 8 — NIGHT ACTIVITY
# ============================================================

def rule_night_activity(df):
    """Detect transactions occurring during unusual hours."""

    return (
        df["is_night_transaction"] == 1
    ).astype(int)


# ============================================================
# RULE 9 — DORMANT ACCOUNT ACTIVATION
# ============================================================

def rule_dormant_activation(df):
    """
    Detect a transaction occurring after a prolonged
    period of account inactivity.

    Logic:
        - Look at the previous transaction for the account.
        - Calculate the inactivity period.
        - Flag the transaction if the account was inactive
          for at least 30 days.
        - Require a meaningful transaction amount.

    This is a behavioral indicator, not proof of AML.
    """

    working = df[
        [
            "account_id",
            "timestamp",
            "amount",
        ]
    ].copy()

    working["previous_timestamp"] = (
        working
        .groupby("account_id")["timestamp"]
        .shift(1)
    )

    working["days_since_previous"] = (
        working["timestamp"]
        - working["previous_timestamp"]
    ).dt.total_seconds() / 86400.0

    dormant_period = (
        working["days_since_previous"] >= 30
    )

    meaningful_amount = (
        working["amount"] >= 25000
    )

    result = (
        dormant_period
        & meaningful_amount
    ).astype(int)

    return result.reindex(df.index, fill_value=0)


# ============================================================
# RULE 10 — AMOUNT DEVIATION
# ============================================================

def rule_high_amount_deviation(df):
    """
    Detect transactions significantly larger than
    the account's normal transaction amount.
    """

    return (
        df["amount_to_account_avg_ratio"] >= 5
    ).astype(int)


# ============================================================
# APPLY RULES
# ============================================================

def apply_rules(df):
    """Apply all AML detection rules."""

    print("\nApplying AML rules...\n")

    # --------------------------------------------------------
    # Rule 1
    # --------------------------------------------------------

    df["rule_structuring"] = (
        rule_structuring(df)
    )

    print("  ✓ Structuring")

    # --------------------------------------------------------
    # Rule 2
    # --------------------------------------------------------

    df["rule_high_velocity"] = (
        rule_high_velocity(df)
    )

    print("  ✓ High velocity")

    # --------------------------------------------------------
    # Rule 3
    # --------------------------------------------------------

    df["rule_large_transaction"] = (
        rule_large_transaction(df)
    )

    print("  ✓ Large transaction")

    # --------------------------------------------------------
    # Rule 4
    # --------------------------------------------------------

    df["rule_rapid_fund_movement"] = (
        rule_rapid_fund_movement(df)
    )

    print("  ✓ Rapid fund movement")

    # --------------------------------------------------------
    # Rule 5
    # --------------------------------------------------------

    df["rule_multiple_beneficiaries"] = (
        rule_multiple_beneficiaries(df)
    )

    print("  ✓ Multiple beneficiaries")

    # --------------------------------------------------------
    # Rule 6
    # --------------------------------------------------------

    df["rule_multiple_devices"] = (
        rule_multiple_devices(df)
    )

    print("  ✓ New device")

    # --------------------------------------------------------
    # Rule 7
    # --------------------------------------------------------

    df["rule_multiple_ips"] = (
        rule_multiple_ips(df)
    )

    print("  ✓ New IP")

    # --------------------------------------------------------
    # Rule 8
    # --------------------------------------------------------

    df["rule_night_activity"] = (
        rule_night_activity(df)
    )

    print("  ✓ Night activity")

    # --------------------------------------------------------
    # Rule 9
    # --------------------------------------------------------

    df["rule_dormant_activation"] = (
        rule_dormant_activation(df)
    )

    print("  ✓ Dormant activation")

    # --------------------------------------------------------
    # Rule 10
    # --------------------------------------------------------

    df["rule_high_amount_deviation"] = (
        rule_high_amount_deviation(df)
    )

    print("  ✓ Amount deviation")

    return df


# ============================================================
# CALCULATE AML RISK SCORE
# ============================================================

def calculate_aml_score(df):
    """Calculate weighted AML rule score."""

    print(
        "\nCalculating AML rule score..."
    )

    df["aml_rule_score"] = 0.0

    for rule_name, weight in RULE_WEIGHTS.items():

        column = (
            f"rule_{rule_name}"
        )

        if column in df.columns:

            df["aml_rule_score"] += (
                df[column] * weight
            )

    # --------------------------------------------------------
    # Cap score
    # --------------------------------------------------------

    df["aml_rule_score"] = (
        df["aml_rule_score"]
        .clip(upper=100)
    )

    return df


# ============================================================
# AML RISK LEVEL
# ============================================================

def assign_risk_level(df):
    """Assign AML risk category based on rule score."""

    print(
        "Assigning AML risk levels..."
    )

    df["aml_risk_level"] = np.select(
        [
            df["aml_rule_score"] >= 70,
            df["aml_rule_score"] >= 50,
            df["aml_rule_score"] >= 30,
            df["aml_rule_score"] >= 15,
        ],
        [
            "critical",
            "high",
            "medium",
            "low",
        ],
        default="minimal",
    )

    return df


# ============================================================
# EXPLANATION
# ============================================================

def generate_explanations(df):
    """Generate human-readable AML explanations."""

    print(
        "Generating AML explanations..."
    )

    rule_columns = [
        (
            "rule_structuring",
            "Structuring-like transaction amount",
        ),
        (
            "rule_high_velocity",
            "High transaction velocity",
        ),
        (
            "rule_large_transaction",
            "Large transaction amount",
        ),
        (
            "rule_rapid_fund_movement",
            "Rapid movement of funds",
        ),
        (
            "rule_multiple_beneficiaries",
            "Multiple beneficiaries",
        ),
        (
            "rule_multiple_devices",
            "New device used by account",
        ),
        (
            "rule_multiple_ips",
            "New IP address used by account",
        ),
        (
            "rule_night_activity",
            "Unusual night-time activity",
        ),
        (
            "rule_dormant_activation",
            "Possible dormant-account activation",
        ),
        (
            "rule_high_amount_deviation",
            "Transaction significantly above account average",
        ),
    ]

    explanations = []

    for _, row in df.iterrows():

        reasons = []

        for column, description in rule_columns:

            if row[column] == 1:
                reasons.append(description)

        if reasons:

            explanations.append(
                "; ".join(reasons)
            )

        else:

            explanations.append(
                "No AML rule triggered"
            )

    df["aml_explanation"] = explanations

    return df


# ============================================================
# AML ALERT
# ============================================================

def assign_alert_flag(df):
    """
    Generate AML alert when the combined rule score
    reaches the initial alert threshold.
    """

    df["aml_alert"] = (
        df["aml_rule_score"] >= 30
    ).astype(int)

    return df


# ============================================================
# REPORT
# ============================================================

def generate_report(df):
    """Print AML engine evaluation report."""

    print("\n" + "=" * 60)
    print("AML RULE ENGINE REPORT")
    print("=" * 60)

    total = len(df)

    alerts = int(
        df["aml_alert"].sum()
    )

    print(
        f"\nTotal transactions : {total:,}"
    )

    print(
        f"AML alerts         : {alerts:,}"
    )

    print(
        f"Alert rate         : "
        f"{alerts / total * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print(
        "\nRisk-level distribution:"
    )

    print(
        df["aml_risk_level"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Rule trigger counts
    # --------------------------------------------------------

    print(
        "\nRule trigger counts:"
    )

    rule_columns = [
        column
        for column in df.columns
        if column.startswith("rule_")
    ]

    for column in rule_columns:

        print(
            f"{column:<40}"
            f"{int(df[column].sum()):>8,}"
        )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    if "is_aml" in df.columns:

        print(
            "\nGround-truth AML comparison:"
        )

        true_aml = int(
            df["is_aml"].sum()
        )

        detected_aml = int(
            (
                (df["aml_alert"] == 1)
                & (df["is_aml"] == 1)
            ).sum()
        )

        false_alerts = int(
            (
                (df["aml_alert"] == 1)
                & (df["is_aml"] == 0)
            ).sum()
        )

        missed_aml = int(
            (
                (df["aml_alert"] == 0)
                & (df["is_aml"] == 1)
            ).sum()
        )

        print(
            f"True AML cases       : {true_aml:,}"
        )

        print(
            f"Detected AML cases   : {detected_aml:,}"
        )

        print(
            f"False AML alerts     : {false_alerts:,}"
        )

        print(
            f"Missed AML cases     : {missed_aml:,}"
        )

        if true_aml > 0:

            detection_rate = (
                detected_aml
                / true_aml
                * 100
            )

            print(
                f"AML detection rate   : "
                f"{detection_rate:.2f}%"
            )

        # ----------------------------------------------------
        # Additional evaluation metrics
        # ----------------------------------------------------

        if alerts > 0:

            precision = (
                detected_aml
                / alerts
                * 100
            )

        else:

            precision = 0.0

        print(
            f"AML alert precision  : "
            f"{precision:.2f}%"
        )

    # --------------------------------------------------------
    # Score statistics
    # --------------------------------------------------------

    print(
        "\nAML score statistics:"
    )

    print(
        df["aml_rule_score"]
        .describe()
        .round(2)
        .to_string()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FINSHIELD AI — AML RULE ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Apply rules
    # --------------------------------------------------------

    df = apply_rules(df)

    # --------------------------------------------------------
    # Calculate score
    # --------------------------------------------------------

    df = calculate_aml_score(df)

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    df = assign_risk_level(df)

    # --------------------------------------------------------
    # Explanations
    # --------------------------------------------------------

    df = generate_explanations(df)

    # --------------------------------------------------------
    # Alert
    # --------------------------------------------------------

    df = assign_alert_flag(df)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved AML alerts dataset:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    generate_report(df)

    print("\n" + "=" * 60)
    print("AML RULE ENGINE COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()