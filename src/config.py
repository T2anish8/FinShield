from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"

MODEL_DIR = PROJECT_ROOT / "models"


# ============================================================
# SYNTHETIC DATASET CONFIGURATION
# ============================================================

NUM_CUSTOMERS = 1_000
NUM_ACCOUNTS = 1_200
NUM_MERCHANTS = 200
NUM_DEVICES = 800
NUM_TRANSACTIONS = 50_000


# ============================================================
# RANDOM SEED
# ============================================================

RANDOM_SEED = 42


# ============================================================
# TRANSACTION CONFIGURATION
# ============================================================

TRANSACTION_TYPES = [
    "purchase",
    "transfer",
    "withdrawal",
    "deposit",
    "bill_payment",
]

CHANNELS = [
    "mobile",
    "web",
    "atm",
    "pos",
]

CURRENCY = "INR"


# ============================================================
# FUTURE FRAUD / AML DISTRIBUTION
# ============================================================

NORMAL_TRANSACTION_RATIO = 0.94
FRAUD_TRANSACTION_RATIO = 0.03
AML_TRANSACTION_RATIO = 0.025
COMBINED_TRANSACTION_RATIO = 0.005