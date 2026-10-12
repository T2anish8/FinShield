
import random
import uuid
from datetime import datetime, timezone

from faker import Faker

fake = Faker("en_IN")

TRANSACTION_TYPES = [
    "purchase",
    "transfer",
    "withdrawal",
    "deposit",
    "bill_payment",
]

CHANNELS = ["mobile", "web", "atm", "pos"]

NORMAL_WEIGHTS = [50, 20, 10, 10, 10]


def generate_transaction() -> dict:
    """
    Generate one synthetic transaction for dashboard demos.

    These transactions are simulated and are not scored by
    the trained FinShield fraud or AML models.
    """

    transaction_type = random.choices(
        TRANSACTION_TYPES,
        weights=NORMAL_WEIGHTS,
        k=1,
    )[0]

    channel = random.choice(CHANNELS)
    amount = round(random.uniform(100, 15000), 2)

    # Occasionally generate an intentionally unusual event.
    # This is a simulation label, not a model prediction.
    is_demo_suspicious = random.random() < 0.08

    if is_demo_suspicious:
        scenario = random.choice([
            "large_amount_demo",
            "unusual_hour_demo",
            "high_value_transfer_demo",
        ])

        if scenario == "large_amount_demo":
            amount = round(random.uniform(50000, 150000), 2)

        elif scenario == "unusual_hour_demo":
            transaction_type = "withdrawal"
            channel = "atm"

        else:
            transaction_type = "transfer"
            amount = round(random.uniform(30000, 100000), 2)
    else:
        scenario = "normal_demo"

    # Illustrative demo score only; not a model output.
    if scenario == "large_amount_demo":
        risk_score = random.uniform(65, 95)
    elif scenario == "high_value_transfer_demo":
        risk_score = random.uniform(60, 90)
    elif scenario == "unusual_hour_demo":
        risk_score = random.uniform(40, 70)
    else:
        risk_score = random.uniform(1, 45)

    if risk_score >= 80:
        risk_level = "critical"
    elif risk_score >= 60:
        risk_level = "high"
    elif risk_score >= 40:
        risk_level = "medium"
    elif risk_score >= 20:
        risk_level = "low"
    else:
        risk_level = "minimal"

    return {
        "transaction_id": f"SIM_{uuid.uuid4().hex[:12].upper()}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "account_id": f"ACC_{random.randint(1000, 2199)}",
        "amount": amount,
        "currency": "INR",
        "transaction_type": transaction_type,
        "channel": channel,
        "scenario_type": scenario,
        "is_demo_suspicious": int(is_demo_suspicious),
        "unified_risk_score": round(risk_score, 2),
        "unified_risk_level": risk_level,
        "simulation": True,
    }


if __name__ == "__main__":
    for _ in range(5):
        print(generate_transaction())
