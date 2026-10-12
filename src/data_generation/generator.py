"""
FinShield AI
Synthetic Financial Transaction Data Generator

Generates realistic synthetic entities and transactions for:

    - Fraud detection
    - AML detection
    - Behavioral anomaly detection
    - Graph-based financial crime analysis
    - Risk scoring
    - Monte Carlo risk simulation

Important design principle:

Each account receives a normal behavioral profile:

    Account
       |
       +-- Primary device
       +-- Optional secondary device
       +-- Normal IP pool
       +-- Normal merchants

Fraud/AML injection modules can later introduce:

    - New devices
    - New IP addresses
    - Unusual locations
    - Velocity anomalies
    - Account takeover
    - Merchant anomalies
    - Structuring
    - Layering
    - Mule accounts

This creates a much more realistic foundation for FinShield.
"""

# ============================================================
# IMPORTS
# ============================================================

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from src.config import (
    NUM_CUSTOMERS,
    NUM_ACCOUNTS,
    NUM_MERCHANTS,
    NUM_DEVICES,
    NUM_TRANSACTIONS,
    RANDOM_SEED,
    TRANSACTION_TYPES,
    CHANNELS,
    CURRENCY,
    SYNTHETIC_DATA_DIR,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

fake = Faker("en_IN")
fake.seed_instance(RANDOM_SEED)


# ============================================================
# CUSTOMER GENERATOR
# ============================================================

def generate_customers() -> pd.DataFrame:
    """
    Generate synthetic customer records.
    """

    customers = []

    occupations = [
        "Software Engineer",
        "Data Analyst",
        "Business Owner",
        "Student",
        "Doctor",
        "Teacher",
        "Consultant",
        "Government Employee",
        "Finance Professional",
        "Self Employed",
    ]

    cities = [
        "Bengaluru",
        "Mumbai",
        "Delhi",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Kolkata",
        "Ahmedabad",
        "Jaipur",
        "Lucknow",
    ]

    for i in range(NUM_CUSTOMERS):

        customers.append(
            {
                "customer_id": f"CUST_{i + 1:06d}",
                "name": fake.name(),
                "age": random.randint(18, 75),
                "occupation": random.choice(
                    occupations
                ),
                "city": random.choice(
                    cities
                ),
                "customer_since": fake.date_between(
                    start_date="-10y",
                    end_date="-30d",
                ),
            }
        )

    return pd.DataFrame(customers)


# ============================================================
# ACCOUNT GENERATOR
# ============================================================

def generate_accounts(
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate synthetic bank accounts.
    """

    accounts = []

    account_types = [
        "savings",
        "current",
        "salary",
    ]

    statuses = [
        "active",
        "active",
        "active",
        "dormant",
    ]

    customer_ids = (
        customers["customer_id"].tolist()
    )

    for i in range(NUM_ACCOUNTS):

        accounts.append(
            {
                "account_id": f"ACC_{i + 1:06d}",

                "customer_id": random.choice(
                    customer_ids
                ),

                "account_type": random.choice(
                    account_types
                ),

                "balance": round(
                    np.random.lognormal(
                        mean=9,
                        sigma=1.2,
                    ),
                    2,
                ),

                "status": random.choice(
                    statuses
                ),

                "created_at": fake.date_between(
                    start_date="-10y",
                    end_date="-30d",
                ),
            }
        )

    return pd.DataFrame(accounts)


# ============================================================
# MERCHANT GENERATOR
# ============================================================

def generate_merchants() -> pd.DataFrame:
    """
    Generate synthetic merchant records.
    """

    merchants = []

    categories = [
        "Grocery",
        "Restaurant",
        "Fuel",
        "Travel",
        "Electronics",
        "Clothing",
        "Healthcare",
        "Entertainment",
        "Online Services",
        "Jewelry",
    ]

    cities = [
        "Bengaluru",
        "Mumbai",
        "Delhi",
        "Hyderabad",
        "Chennai",
        "Pune",
        "Kolkata",
        "Ahmedabad",
        "Jaipur",
        "Lucknow",
    ]

    for i in range(NUM_MERCHANTS):

        merchants.append(
            {
                "merchant_id": f"MER_{i + 1:05d}",

                "merchant_name": fake.company(),

                "merchant_category": random.choice(
                    categories
                ),

                "city": random.choice(
                    cities
                ),

                "risk_level": random.choices(
                    [
                        "low",
                        "medium",
                        "high",
                    ],
                    weights=[
                        70,
                        25,
                        5,
                    ],
                    k=1,
                )[0],
            }
        )

    return pd.DataFrame(merchants)


# ============================================================
# DEVICE GENERATOR
# ============================================================

def generate_devices() -> pd.DataFrame:
    """
    Generate synthetic devices.

    Devices are generated globally and later assigned
    primarily to accounts.
    """

    devices = []

    device_types = [
        "mobile",
        "tablet",
        "laptop",
    ]

    operating_systems = [
        "Android",
        "iOS",
        "Windows",
        "macOS",
    ]

    for i in range(NUM_DEVICES):

        devices.append(
            {
                "device_id": f"DEV_{i + 1:05d}",

                "device_type": random.choice(
                    device_types
                ),

                "os": random.choice(
                    operating_systems
                ),

                "first_seen": fake.date_between(
                    start_date="-5y",
                    end_date="today",
                ),
            }
        )

    return pd.DataFrame(devices)


# ============================================================
# ACCOUNT DEVICE PROFILE
# ============================================================

def generate_account_device_profiles(
    accounts: pd.DataFrame,
    devices: pd.DataFrame,
) -> dict:
    """
    Assign a small number of normal devices to each account.

    Most accounts receive one primary device.
    Some accounts receive a secondary device.

    Returns:

        {
            account_id: [
                device_id,
                device_id,
            ]
        }
    """

    device_ids = (
        devices["device_id"].tolist()
    )

    profiles = {}

    for account_id in accounts[
        "account_id"
    ]:

        # Primary device
        primary_device = random.choice(
            device_ids
        )

        account_devices = [
            primary_device
        ]

        # Approximately 20% of accounts
        # have a second legitimate device.
        if random.random() < 0.20:

            secondary_device = random.choice(
                device_ids
            )

            # Avoid duplicate device
            if (
                secondary_device
                not in account_devices
            ):
                account_devices.append(
                    secondary_device
                )

        profiles[account_id] = (
            account_devices
        )

    return profiles


# ============================================================
# IP GENERATOR
# ============================================================

def generate_ip_address() -> str:
    """
    Generate a synthetic IPv4 address.
    """

    return fake.ipv4()


# ============================================================
# ACCOUNT IP PROFILE
# ============================================================

def generate_account_ip_profiles(
    accounts: pd.DataFrame,
) -> dict:
    """
    Generate a small stable IP pool for each account.

    Most accounts receive one primary IP.
    Some accounts receive a second legitimate IP.

    This prevents every transaction from appearing
    to originate from a brand-new IP.
    """

    profiles = {}

    for account_id in accounts[
        "account_id"
    ]:

        primary_ip = (
            generate_ip_address()
        )

        account_ips = [
            primary_ip
        ]

        # Approximately 15% of accounts
        # have a second legitimate IP.
        if random.random() < 0.15:

            secondary_ip = (
                generate_ip_address()
            )

            if (
                secondary_ip
                not in account_ips
            ):
                account_ips.append(
                    secondary_ip
                )

        profiles[account_id] = (
            account_ips
        )

    return profiles


# ============================================================
# TRANSACTION TIMESTAMP
# ============================================================

def generate_transaction_timestamp() -> datetime:
    """
    Generate a random transaction timestamp
    from approximately the previous year.
    """

    start_date = (
        datetime.now()
        - timedelta(days=365)
    )

    random_days = random.randint(
        0,
        364,
    )

    random_seconds = random.randint(
        0,
        86_399,
    )

    return (
        start_date
        + timedelta(
            days=random_days,
            seconds=random_seconds,
        )
    )


# ============================================================
# TRANSACTION AMOUNT
# ============================================================

def generate_normal_amount() -> float:
    """
    Generate a realistic-looking normal
    transaction amount.

    Log-normal distribution is used because
    financial transaction amounts are generally
    right-skewed.
    """

    amount = np.random.lognormal(
        mean=7.5,
        sigma=1.0,
    )

    amount = min(
        max(
            amount,
            50,
        ),
        50_000,
    )

    return round(
        amount,
        2,
    )


# ============================================================
# NORMAL TRANSACTION
# ============================================================

def generate_normal_transaction(
    transaction_id: str,
    account_id: str,
    merchant_id: str,
    device_id: str,
    ip_address: str,
) -> dict:
    """
    Generate one normal financial transaction.

    Device and IP are selected from the account's
    established behavioral profile.
    """

    return {
        "transaction_id": transaction_id,

        "timestamp": generate_transaction_timestamp(),

        "account_id": account_id,

        "beneficiary_account_id": None,

        "merchant_id": merchant_id,

        "device_id": device_id,

        "amount": generate_normal_amount(),

        "currency": CURRENCY,

        "transaction_type": random.choice(
            TRANSACTION_TYPES
        ),

        "channel": random.choice(
            CHANNELS
        ),

        "ip_address": ip_address,

        "is_fraud": 0,

        "is_aml": 0,

        "scenario_type": "normal",
    }


# ============================================================
# BASE TRANSACTION GENERATOR
# ============================================================

def generate_base_transactions(
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate the initial population of normal transactions.

    Each account uses devices and IP addresses from
    its own behavioral profile.

    This is critical for later anomaly detection.
    """

    transactions = []

    account_ids = (
        accounts["account_id"].tolist()
    )

    merchant_ids = (
        merchants["merchant_id"].tolist()
    )

    # --------------------------------------------------------
    # Build normal account profiles
    # --------------------------------------------------------

    account_device_profiles = (
        generate_account_device_profiles(
            accounts,
            devices,
        )
    )

    account_ip_profiles = (
        generate_account_ip_profiles(
            accounts,
        )
    )

    # --------------------------------------------------------
    # Generate transactions
    # --------------------------------------------------------

    for i in range(
        NUM_TRANSACTIONS
    ):

        transaction_id = (
            f"TXN_{i + 1:08d}"
        )

        # Select account first
        account_id = random.choice(
            account_ids
        )

        # Select a device belonging to
        # this account
        device_id = random.choice(
            account_device_profiles[
                account_id
            ]
        )

        # Select a normal IP belonging
        # to this account
        ip_address = random.choice(
            account_ip_profiles[
                account_id
            ]
        )

        # Select merchant
        merchant_id = random.choice(
            merchant_ids
        )

        transaction = (
            generate_normal_transaction(
                transaction_id=transaction_id,
                account_id=account_id,
                merchant_id=merchant_id,
                device_id=device_id,
                ip_address=ip_address,
            )
        )

        transactions.append(
            transaction
        )

    return pd.DataFrame(
        transactions
    )


# ============================================================
# SAVE DATASETS
# ============================================================

def save_datasets(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    """
    Save all generated datasets to data/synthetic/.
    """

    SYNTHETIC_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    customers.to_csv(
        SYNTHETIC_DATA_DIR
        / "customers.csv",
        index=False,
    )

    accounts.to_csv(
        SYNTHETIC_DATA_DIR
        / "accounts.csv",
        index=False,
    )

    merchants.to_csv(
        SYNTHETIC_DATA_DIR
        / "merchants.csv",
        index=False,
    )

    devices.to_csv(
        SYNTHETIC_DATA_DIR
        / "devices.csv",
        index=False,
    )

    transactions.to_csv(
        SYNTHETIC_DATA_DIR
        / "transactions.csv",
        index=False,
    )


# ============================================================
# DATA QUALITY CHECK
# ============================================================

def validate_datasets(
    customers: pd.DataFrame,
    accounts: pd.DataFrame,
    merchants: pd.DataFrame,
    devices: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    """
    Perform basic data-quality checks.
    """

    print("\n")
    print("=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)

    datasets = {
        "Customers": customers,
        "Accounts": accounts,
        "Merchants": merchants,
        "Devices": devices,
        "Transactions": transactions,
    }

    for name, dataframe in datasets.items():

        print(f"\n{name}")

        print(
            f"Rows       : "
            f"{len(dataframe):,}"
        )

        print(
            f"Columns    : "
            f"{len(dataframe.columns)}"
        )

        print(
            f"Missing    : "
            f"{dataframe.isna().sum().sum():,}"
        )

        print(
            f"Duplicates : "
            f"{dataframe.duplicated().sum():,}"
        )

    # --------------------------------------------------------
    # Transaction-specific checks
    # --------------------------------------------------------

    if (
        "transaction_id"
        in transactions.columns
    ):

        duplicate_ids = (
            transactions[
                "transaction_id"
            ]
            .duplicated()
            .sum()
        )

        print(
            "\nDuplicate transaction IDs : "
            f"{duplicate_ids:,}"
        )

    # --------------------------------------------------------
    # Account-device behavior check
    # --------------------------------------------------------

    if {
        "account_id",
        "device_id",
    }.issubset(
        transactions.columns
    ):

        device_counts = (
            transactions.groupby(
                "account_id"
            )["device_id"]
            .nunique()
        )

        print(
            "\nAccount device profile:"
        )

        print(
            f"Accounts with 1 device : "
            f"{(device_counts == 1).sum():,}"
        )

        print(
            f"Accounts with 2 devices : "
            f"{(device_counts == 2).sum():,}"
        )

        print(
            f"Accounts with 3+ devices: "
            f"{(device_counts >= 3).sum():,}"
        )

    # --------------------------------------------------------
    # Account-IP behavior check
    # --------------------------------------------------------

    if {
        "account_id",
        "ip_address",
    }.issubset(
        transactions.columns
    ):

        ip_counts = (
            transactions.groupby(
                "account_id"
            )["ip_address"]
            .nunique()
        )

        print(
            "\nAccount IP profile:"
        )

        print(
            f"Accounts with 1 IP : "
            f"{(ip_counts == 1).sum():,}"
        )

        print(
            f"Accounts with 2 IPs: "
            f"{(ip_counts == 2).sum():,}"
        )

        print(
            f"Accounts with 3+ IPs: "
            f"{(ip_counts >= 3).sum():,}"
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    """
    Main synthetic data generation pipeline.
    """

    print("=" * 60)
    print("FINSHIELD AI")
    print(
        "Financial Crime, Fraud & AML "
        "Intelligence Platform"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Customers
    # --------------------------------------------------------

    print("\nGenerating customers...")

    customers = (
        generate_customers()
    )

    # --------------------------------------------------------
    # Accounts
    # --------------------------------------------------------

    print("Generating accounts...")

    accounts = (
        generate_accounts(
            customers
        )
    )

    # --------------------------------------------------------
    # Merchants
    # --------------------------------------------------------

    print("Generating merchants...")

    merchants = (
        generate_merchants()
    )

    # --------------------------------------------------------
    # Devices
    # --------------------------------------------------------

    print("Generating devices...")

    devices = (
        generate_devices()
    )

    # --------------------------------------------------------
    # Transactions
    # --------------------------------------------------------

    print("Generating transactions...")

    transactions = (
        generate_base_transactions(
            accounts=accounts,
            merchants=merchants,
            devices=devices,
        )
    )

    # --------------------------------------------------------
    # Display sample data
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CUSTOMERS")
    print("=" * 60)

    print(
        customers.head()
    )

    print("\n")
    print("=" * 60)
    print("ACCOUNTS")
    print("=" * 60)

    print(
        accounts.head()
    )

    print("\n")
    print("=" * 60)
    print("MERCHANTS")
    print("=" * 60)

    print(
        merchants.head()
    )

    print("\n")
    print("=" * 60)
    print("DEVICES")
    print("=" * 60)

    print(
        devices.head()
    )

    print("\n")
    print("=" * 60)
    print("TRANSACTIONS")
    print("=" * 60)

    print(
        transactions.head()
    )

    # --------------------------------------------------------
    # Dataset sizes
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("DATASET SIZES")
    print("=" * 60)

    print(
        f"Customers    : "
        f"{len(customers):,}"
    )

    print(
        f"Accounts     : "
        f"{len(accounts):,}"
    )

    print(
        f"Merchants    : "
        f"{len(merchants):,}"
    )

    print(
        f"Devices      : "
        f"{len(devices):,}"
    )

    print(
        f"Transactions : "
        f"{len(transactions):,}"
    )

    # --------------------------------------------------------
    # Transaction labels
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("TRANSACTION LABELS")
    print("=" * 60)

    print(
        transactions[
            "scenario_type"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    print("\nSaving datasets...")

    save_datasets(
        customers=customers,
        accounts=accounts,
        merchants=merchants,
        devices=devices,
        transactions=transactions,
    )

    print(
        f"\nDatasets saved to:\n"
        f"{SYNTHETIC_DATA_DIR}"
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_datasets(
        customers=customers,
        accounts=accounts,
        merchants=merchants,
        devices=devices,
        transactions=transactions,
    )

    print("\n")
    print("=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()