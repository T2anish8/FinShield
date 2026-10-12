
from pathlib import Path
import asyncio
from datetime import datetime, timezone

import pandas as pd
from fastapi import (
    FastAPI,
    Query,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware

from backend.transaction_simulator import generate_transaction


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SCORES_FILE = DATA_DIR / "unified_risk_scores.csv"
SUMMARY_FILE = DATA_DIR / "unified_risk_summary.csv"


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="FinShield AI API",
    description="Financial crime risk intelligence API",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATA HELPERS
# ============================================================

def load_scores() -> pd.DataFrame:
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Unified risk data not found: {SCORES_FILE}"
        )

    df = pd.read_csv(SCORES_FILE)

    required = [
        "transaction_id",
        "unified_risk_score",
        "unified_risk_level",
        "amount",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing score columns: {missing}"
        )

    return df


def clean_records(df: pd.DataFrame) -> list[dict]:
    """Convert pandas missing values to JSON-safe values."""
    safe_df = df.astype(object).where(pd.notna(df), None)
    return safe_df.to_dict(orient="records")


# ============================================================
# REST API
# ============================================================

@app.get("/")
def root():
    return {
        "name": "FinShield AI",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health():
    df = load_scores()

    return {
        "status": "healthy",
        "transactions_loaded": len(df),
        "data_source": str(SCORES_FILE),
    }


@app.get("/api/summary")
def summary():
    df = load_scores()

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    ).fillna(0)

    df["unified_risk_score"] = pd.to_numeric(
        df["unified_risk_score"], errors="coerce"
    ).fillna(0)

    levels = (
        df["unified_risk_level"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    counts = levels.value_counts()

    risk_distribution = [
        {
            "risk_level": str(level),
            "count": int(count),
        }
        for level, count in counts.items()
    ]

    return {
        "total_transactions": int(len(df)),
        "total_amount": round(float(df["amount"].sum()), 2),
        "average_risk_score": round(
            float(df["unified_risk_score"].mean()), 2
        ),
        "high_risk": int((levels == "high").sum()),
        "critical_risk": int((levels == "critical").sum()),
        "medium_risk": int((levels == "medium").sum()),
        "low_risk": int((levels == "low").sum()),
        "minimal_risk": int((levels == "minimal").sum()),
        "risk_distribution": risk_distribution,
    }


@app.get("/api/transactions")
def transactions(
    risk_level: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    df = load_scores()

    if risk_level:
        df = df[
            df["unified_risk_level"]
            .fillna("")
            .astype(str)
            .str.lower()
            == risk_level.lower()
        ]

    if search:
        search = search.strip().lower()

        searchable = [
            column
            for column in [
                "transaction_id",
                "account_id",
                "transaction_type",
                "channel",
            ]
            if column in df.columns
        ]

        if searchable:
            mask = pd.Series(False, index=df.index)

            for column in searchable:
                mask |= (
                    df[column]
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.contains(search, regex=False)
                )

            df = df[mask]

    total = len(df)

    df["unified_risk_score"] = pd.to_numeric(
        df["unified_risk_score"],
        errors="coerce",
    ).fillna(0)

    df = df.sort_values(
        "unified_risk_score",
        ascending=False,
    )

    page = df.iloc[offset:offset + limit]

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "transactions": clean_records(page),
    }


@app.get("/api/transaction/{transaction_id}")
def transaction_detail(transaction_id: str):
    df = load_scores()

    match = df[
        df["transaction_id"].astype(str) == transaction_id
    ]

    if match.empty:
        raise HTTPException(
            status_code=404,
            detail="Transaction not found",
        )

    return clean_records(match.iloc[[0]])[0]


# ============================================================
# LIVE SYNTHETIC TRANSACTION WEBSOCKET
# ============================================================

@app.websocket("/ws/transactions")
async def transaction_stream(websocket: WebSocket):
    await websocket.accept()

    print("Dashboard connected to synthetic transaction stream.")

    try:
        await websocket.send_json({
            "type": "status",
            "stream_mode": "synthetic_generation",
            "message": "Synthetic transaction generator connected.",
        })

        while True:
            # Generate a brand-new synthetic transaction.
            transaction = generate_transaction()

            await websocket.send_json({
                "type": "transaction",
                "stream_mode": "synthetic_generation",
                "stream_timestamp": datetime.now(
                    timezone.utc
                ).isoformat(),
                "data": transaction,
            })

            # Send approximately one event every 1.5 seconds.
            await asyncio.sleep(1.5)

    except WebSocketDisconnect:
        print("Dashboard disconnected from transaction stream.")

    except Exception as exc:
        print(f"Transaction stream error: {exc}")

        try:
            await websocket.close(code=1011)
        except Exception:
            pass


# ============================================================
# GRAPH INTELLIGENCE API
# ============================================================

GRAPH_ENTITY_FILE = DATA_DIR / "graph_entity_risk.csv"
GRAPH_SHARED_FILE = DATA_DIR / "graph_shared_entities.csv"
GRAPH_EDGES_FILE = DATA_DIR / "graph_suspicious_edges.csv"
GRAPH_SUMMARY_FILE = DATA_DIR / "graph_summary.csv"


def load_graph_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Graph data file not found: {path.name}",
        )
    return pd.read_csv(path)


@app.get("/api/graph/summary")
def graph_summary():
    """Return graph-level metrics and risk counts."""
    summary_df = load_graph_csv(GRAPH_SUMMARY_FILE)
    entity_df = load_graph_csv(GRAPH_ENTITY_FILE)

    summary_data = summary_df.iloc[0].to_dict()

    risk_levels = (
        entity_df["graph_risk_level"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
    )

    return {
        "nodes": int(summary_data.get("nodes", 0)),
        "edges": int(summary_data.get("edges", 0)),
        "accounts": int(summary_data.get("accounts", 0)),
        "shared_devices": int(summary_data.get("shared_devices", 0)),
        "shared_ips": int(summary_data.get("shared_ips", 0)),
        "high_risk_accounts": int(
            summary_data.get("high_risk_accounts", 0)
        ),
        "critical_accounts": int(
            summary_data.get("critical_accounts", 0)
        ),
        "suspicious_transfer_edges": int(
            summary_data.get("suspicious_transfer_edges", 0)
        ),
        "risk_distribution": {
            str(level): int(count)
            for level, count in risk_levels.value_counts().items()
        },
    }


@app.get("/api/graph/accounts")
def graph_accounts(
    risk_level: str | None = Query(default=None),
    search: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Return account-level graph risk records."""
    df = load_graph_csv(GRAPH_ENTITY_FILE)

    if risk_level:
        df = df[
            df["graph_risk_level"]
            .fillna("")
            .astype(str)
            .str.lower()
            == risk_level.lower()
        ]

    if search:
        term = search.strip().lower()
        df = df[
            df["account_id"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(term, regex=False)
        ]

    df["graph_risk_score"] = pd.to_numeric(
        df["graph_risk_score"], errors="coerce"
    ).fillna(0)

    df = df.sort_values(
        "graph_risk_score", ascending=False
    )

    total = len(df)
    page = df.iloc[offset:offset + limit]

    return {
        "total": int(total),
        "limit": limit,
        "offset": offset,
        "accounts": clean_records(page),
    }


@app.get("/api/graph/shared-entities")
def graph_shared_entities(
    entity_type: str | None = Query(default=None),
    min_accounts: int = Query(default=2, ge=2, le=1200),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Return devices/IPs shared by multiple accounts."""
    df = load_graph_csv(GRAPH_SHARED_FILE)

    df["account_count"] = pd.to_numeric(
        df["account_count"], errors="coerce"
    ).fillna(0)

    df = df[df["account_count"] >= min_accounts]

    if entity_type:
        df = df[
            df["entity_type"]
            .fillna("")
            .astype(str)
            .str.lower()
            == entity_type.lower()
        ]

    df = df.sort_values(
        "account_count", ascending=False
    ).head(limit)

    return {
        "total_returned": int(len(df)),
        "entities": clean_records(df),
    }


@app.get("/api/graph/suspicious-edges")
def graph_suspicious_edges(
    limit: int = Query(default=100, ge=1, le=500),
    min_amount: float = Query(default=0, ge=0),
):
    """Return flagged account-to-account transfers."""
    df = load_graph_csv(GRAPH_EDGES_FILE)

    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    ).fillna(0)

    df = df[df["amount"] >= min_amount]

    if "suspicious_edge" in df.columns:
        df = df[
            pd.to_numeric(
                df["suspicious_edge"], errors="coerce"
            ).fillna(0) == 1
        ]

    df = df.sort_values(
        "amount", ascending=False
    ).head(limit)

    return {
        "total_returned": int(len(df)),
        "edges": clean_records(df),
    }
