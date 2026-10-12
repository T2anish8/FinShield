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

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR


# ============================================================
# FILES
# ============================================================

INPUT_FILE = (
    PROCESSED_DATA_DIR /
    "transactions_features.csv"
)

OUTPUT_FILE = (
    PROCESSED_DATA_DIR /
    "aml_alerts.csv"
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

    return df


# ============================================================
# RULE 1 — STRUCTURING
# ============================================================

def rule_structuring(df):

    """
    Detect transactions with amounts clustered below
    a reporting-like threshold.

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
    Approximate rapid fund movement using:
        - transfer activity
        - high transaction velocity
        - beneficiary relationship
    """

    transfer = (
        df["is_transfer"] == 1
    )

    velocity = (
        df["account_10min_transaction_count"] >= 2
    )

    beneficiary = (
        df["has_beneficiary"] == 1
    )

    return (
        transfer
        & velocity
        & beneficiary
    ).astype(int)


# ============================================================
# RULE 5 — MULTIPLE BENEFICIARIES
# ============================================================

def rule_multiple_beneficiaries(df):

    """Detect accounts sending money to many beneficiaries."""

    return (
        df["unique_beneficiaries_per_account"] >= 3
    ).astype(int)


# ============================================================
# RULE 6 — MULTIPLE DEVICES
# ============================================================

def rule_multiple_devices(df):

    """Detect accounts using multiple devices."""

    return (
        df["unique_devices_per_account"] >= 3
    ).astype(int)


# ============================================================
# RULE 7 — MULTIPLE IPS
# ============================================================

def rule_multiple_ips(df):

    """Detect accounts using multiple IP addresses."""

    return (
        df["unique_ips_per_account"] >= 3
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
# RULE 9 — DORMANT ACTIVATION
# ============================================================

def rule_dormant_activation(df):

    """
    Approximate dormant-account activation.

    A transaction is suspicious when the account has very
    few historical transactions but suddenly performs a
    large transaction.

    This is a simplified baseline. Later we'll implement
    proper chronological account inactivity windows.
    """

    low_activity = (
        df["account_transaction_count"] <= 5
    )

    large_amount = (
        df["amount"] >= 25000
    )

    return (
        low_activity
        & large_amount
    ).astype(int)


# ============================================================
# RULE 10 — AMOUNT DEVIATION
# ============================================================

def rule_high_amount_deviation(df):

    """
    Detect transactions significantly larger than the
    account's normal transaction amount.
    """

    return (
        df["amount_to_account_avg_ratio"] >= 5
    ).astype(int)


# ============================================================
# APPLY RULES
# ============================================================

def apply_rules(df):

    print("\nApplying AML rules...\n")

    # --------------------------------------------------------
    # Individual rule outputs
    # --------------------------------------------------------

    df["rule_structuring"] = (
        rule_structuring(df)
    )

    print(
        "  ✓ Structuring"
    )

    df["rule_high_velocity"] = (
        rule_high_velocity(df)
    )

    print(
        "  ✓ High velocity"
    )

    df["rule_large_transaction"] = (
        rule_large_transaction(df)
    )

    print(
        "  ✓ Large transaction"
    )

    df["rule_rapid_fund_movement"] = (
        rule_rapid_fund_movement(df)
    )

    print(
        "  ✓ Rapid fund movement"
    )

    df["rule_multiple_beneficiaries"] = (
        rule_multiple_beneficiaries(df)
    )

    print(
        "  ✓ Multiple beneficiaries"
    )

    df["rule_multiple_devices"] = (
        rule_multiple_devices(df)
    )

    print(
        "  ✓ Multiple devices"
    )

    df["rule_multiple_ips"] = (
        rule_multiple_ips(df)
    )

    print(
        "  ✓ Multiple IPs"
    )

    df["rule_night_activity"] = (
        rule_night_activity(df)
    )

    print(
        "  ✓ Night activity"
    )

    df["rule_dormant_activation"] = (
        rule_dormant_activation(df)
    )

    print(
        "  ✓ Dormant activation"
    )

    df["rule_high_amount_deviation"] = (
        rule_high_amount_deviation(df)
    )

    print(
        "  ✓ Amount deviation"
    )

    return df


# ============================================================
# CALCULATE AML RISK SCORE
# ============================================================

def calculate_aml_score(df):

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

    # Cap score at 100
    df["aml_rule_score"] = (
        df["aml_rule_score"]
        .clip(upper=100)
    )

    return df


# ============================================================
# AML RISK LEVEL
# ============================================================

def assign_risk_level(df):

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
            "Multiple devices used",
        ),
        (
            "rule_multiple_ips",
            "Multiple IP addresses used",
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
    Generate an AML alert when at least one meaningful
    suspicious pattern is detected.

    A score of 30+ is treated as an alert in this
    initial baseline.
    """

    df["aml_alert"] = (
        df["aml_rule_score"] >= 30
    ).astype(int)

    return df


# ============================================================
# REPORT
# ============================================================

def generate_report(df):

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

    print("\nRisk-level distribution:")

    print(
        df["aml_risk_level"]
        .value_counts()
        .to_string()
    )

    print("\nRule trigger counts:")

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
    # Ground-truth comparison
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

            print(
                f"AML detection rate   : "
                f"{detected_aml / true_aml * 100:.2f}%"
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
    # Score
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


if __name__ == "__main__":
    main()