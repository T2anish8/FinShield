"""
FinShield AI - Graph Intelligence Engine

Builds a financial transaction graph from the actual
transactions_aml.csv schema.

Entities:
    Account
    Beneficiary Account
    Device
    IP
    Merchant

Relationships:
    Account -> Beneficiary Account
    Account -> Device
    Account -> IP
    Account -> Merchant

The module calculates:
    - account network connectivity
    - shared devices
    - shared IPs
    - merchant connectivity
    - money-flow relationships
    - suspicious transfer edges
    - graph-based account risk
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "transactions_aml.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRAPH_METRICS_FILE = (
    OUTPUT_DIR
    / "graph_entity_risk.csv"
)

SUSPICIOUS_EDGES_FILE = (
    OUTPUT_DIR
    / "graph_suspicious_edges.csv"
)

GRAPH_SUMMARY_FILE = (
    OUTPUT_DIR
    / "graph_summary.csv"
)

SHARED_ENTITIES_FILE = (
    OUTPUT_DIR
    / "graph_shared_entities.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING TRANSACTION DATA")
    print("=" * 60)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Transactions loaded : "
        f"{len(df):,}"
    )

    print(
        f"Columns loaded      : "
        f"{len(df.columns):,}"
    )

    required_columns = [
        "transaction_id",
        "timestamp",
        "account_id",
        "beneficiary_account_id",
        "merchant_id",
        "device_id",
        "amount",
        "currency",
        "transaction_type",
        "channel",
        "ip_address",
        "is_fraud",
        "is_aml",
        "scenario_type",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        print("\nMissing columns:")

        for column in missing:
            print(f"  - {column}")

        raise ValueError(
            "Required graph columns are missing."
        )

    print(
        "Schema validation    : PASSED"
    )

    return df


# ============================================================
# CLEAN DATA
# ============================================================

def clean_data(df):

    print("\n" + "=" * 60)
    print("CLEANING GRAPH DATA")
    print("=" * 60)

    identifier_columns = [
        "transaction_id",
        "account_id",
        "beneficiary_account_id",
        "merchant_id",
        "device_id",
        "ip_address",
    ]

    for column in identifier_columns:

        df[column] = (
            df[column]
            .astype("string")
        )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce"
    ).fillna(0)

    df["is_fraud"] = pd.to_numeric(
        df["is_fraud"],
        errors="coerce"
    ).fillna(0).astype(int)

    df["is_aml"] = pd.to_numeric(
        df["is_aml"],
        errors="coerce"
    ).fillna(0).astype(int)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    print(
        f"Missing timestamps : "
        f"{df['timestamp'].isna().sum():,}"
    )

    print(
        f"Fraud transactions : "
        f"{df['is_fraud'].sum():,}"
    )

    print(
        f"AML transactions   : "
        f"{df['is_aml'].sum():,}"
    )

    return df


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(df):

    print("\n" + "=" * 60)
    print("BUILDING FINANCIAL NETWORK GRAPH")
    print("=" * 60)

    graph = nx.MultiDiGraph()

    for row in df.itertuples(
        index=False
    ):

        account = (
            f"ACCOUNT:{row.account_id}"
        )

        merchant = (
            f"MERCHANT:{row.merchant_id}"
        )

        device = (
            f"DEVICE:{row.device_id}"
        )

        ip = (
            f"IP:{row.ip_address}"
        )

        # ----------------------------------------------------
        # ACCOUNT
        # ----------------------------------------------------

        graph.add_node(
            account,
            entity_type="account"
        )

        # ----------------------------------------------------
        # DEVICE
        # ----------------------------------------------------

        graph.add_node(
            device,
            entity_type="device"
        )

        graph.add_edge(
            account,
            device,
            relationship="uses_device",
            transaction_id=str(
                row.transaction_id
            )
        )

        # ----------------------------------------------------
        # IP
        # ----------------------------------------------------

        graph.add_node(
            ip,
            entity_type="ip"
        )

        graph.add_edge(
            account,
            ip,
            relationship="uses_ip",
            transaction_id=str(
                row.transaction_id
            )
        )

        # ----------------------------------------------------
        # MERCHANT
        # ----------------------------------------------------

        graph.add_node(
            merchant,
            entity_type="merchant"
        )

        graph.add_edge(
            account,
            merchant,
            relationship="transacts_with",
            transaction_id=str(
                row.transaction_id
            ),
            amount=float(
                row.amount
            )
        )

        # ----------------------------------------------------
        # BENEFICIARY ACCOUNT
        # ----------------------------------------------------

        beneficiary = (
            row.beneficiary_account_id
        )

        if (
            pd.notna(beneficiary)
            and str(beneficiary).lower()
            not in {
                "",
                "nan",
                "none",
                "<na>",
            }
        ):

            beneficiary_account = (
                f"ACCOUNT:{beneficiary}"
            )

            graph.add_node(
                beneficiary_account,
                entity_type="account"
            )

            graph.add_edge(
                account,
                beneficiary_account,
                relationship="transfers_to",
                transaction_id=str(
                    row.transaction_id
                ),
                amount=float(
                    row.amount
                ),
                transaction_type=str(
                    row.transaction_type
                ),
                is_fraud=int(
                    row.is_fraud
                ),
                is_aml=int(
                    row.is_aml
                ),
            )

    print(
        f"Graph nodes         : "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges         : "
        f"{graph.number_of_edges():,}"
    )

    return graph


# ============================================================
# ACCOUNT METRICS
# ============================================================

def calculate_account_metrics(
    graph,
    df
):

    print("\n" + "=" * 60)
    print("CALCULATING ACCOUNT NETWORK METRICS")
    print("=" * 60)

    accounts = [
        node
        for node, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "entity_type"
        ) == "account"
    ]

    records = []

    for account in accounts:

        account_id = (
            account.replace(
                "ACCOUNT:",
                ""
            )
        )

        # ----------------------------------------------------
        # GRAPH DEGREE
        # ----------------------------------------------------

        degree = graph.degree(
            account
        )

        in_degree = graph.in_degree(
            account
        )

        out_degree = graph.out_degree(
            account
        )

        # ----------------------------------------------------
        # NEIGHBORS
        # ----------------------------------------------------

        neighbors = set(
            graph.predecessors(
                account
            )
        ).union(
            set(
                graph.successors(
                    account
                )
            )
        )

        # ----------------------------------------------------
        # ENTITY CONNECTIONS
        # ----------------------------------------------------

        connected_devices = len(
            [
                node
                for node in neighbors
                if node.startswith(
                    "DEVICE:"
                )
            ]
        )

        connected_ips = len(
            [
                node
                for node in neighbors
                if node.startswith(
                    "IP:"
                )
            ]
        )

        connected_merchants = len(
            [
                node
                for node in neighbors
                if node.startswith(
                    "MERCHANT:"
                )
            ]
        )

        connected_accounts = len(
            [
                node
                for node in neighbors
                if node.startswith(
                    "ACCOUNT:"
                )
                and node != account
            ]
        )

        # ----------------------------------------------------
        # TRANSACTION DATA
        # ----------------------------------------------------

        account_transactions = df[
            df["account_id"].astype(str)
            == account_id
        ]

        transaction_count = len(
            account_transactions
        )

        total_amount = (
            account_transactions[
                "amount"
            ].sum()
        )

        fraud_count = int(
            account_transactions[
                "is_fraud"
            ].sum()
        )

        aml_count = int(
            account_transactions[
                "is_aml"
            ].sum()
        )

        fraud_rate = (
            fraud_count
            / transaction_count
            if transaction_count > 0
            else 0
        )

        aml_rate = (
            aml_count
            / transaction_count
            if transaction_count > 0
            else 0
        )

        # ----------------------------------------------------
        # RECORD
        # ----------------------------------------------------

        records.append(
            {
                "account_id":
                    account_id,

                "graph_degree":
                    degree,

                "in_degree":
                    in_degree,

                "out_degree":
                    out_degree,

                "connected_accounts":
                    connected_accounts,

                "connected_devices":
                    connected_devices,

                "connected_ips":
                    connected_ips,

                "connected_merchants":
                    connected_merchants,

                "transaction_count":
                    transaction_count,

                "total_transaction_amount":
                    total_amount,

                "fraud_transaction_count":
                    fraud_count,

                "aml_transaction_count":
                    aml_count,

                "fraud_rate":
                    fraud_rate,

                "aml_rate":
                    aml_rate,
            }
        )

    metrics = pd.DataFrame(
        records
    )

    print(
        f"Accounts analyzed : "
        f"{len(metrics):,}"
    )

    return metrics


# ============================================================
# SHARED DEVICE / IP ANALYSIS
# ============================================================

def calculate_shared_entities(df):

    print("\n" + "=" * 60)
    print("ANALYZING SHARED DEVICES AND IPS")
    print("=" * 60)

    # --------------------------------------------------------
    # DEVICE SHARING
    # --------------------------------------------------------

    device_accounts = (
        df.groupby(
            "device_id"
        )[
            "account_id"
        ]
        .nunique()
    )

    shared_devices = (
        device_accounts
        .reset_index(
            name="account_count"
        )
    )

    shared_devices = (
        shared_devices[
            shared_devices[
                "account_count"
            ] > 1
        ]
        .copy()
    )

    shared_devices[
        "entity_type"
    ] = "device"

    # --------------------------------------------------------
    # IP SHARING
    # --------------------------------------------------------

    ip_accounts = (
        df.groupby(
            "ip_address"
        )[
            "account_id"
        ]
        .nunique()
    )

    shared_ips = (
        ip_accounts
        .reset_index(
            name="account_count"
        )
    )

    shared_ips = (
        shared_ips[
            shared_ips[
                "account_count"
            ] > 1
        ]
        .copy()
    )

    shared_ips[
        "entity_type"
    ] = "ip"

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    shared_devices = (
        shared_devices.rename(
            columns={
                "device_id":
                    "entity_id"
            }
        )
    )

    shared_ips = (
        shared_ips.rename(
            columns={
                "ip_address":
                    "entity_id"
            }
        )
    )

    shared = pd.concat(
        [
            shared_devices[
                [
                    "entity_type",
                    "entity_id",
                    "account_count",
                ]
            ],

            shared_ips[
                [
                    "entity_type",
                    "entity_id",
                    "account_count",
                ]
            ],
        ],
        ignore_index=True
    )

    print(
        f"Shared devices : "
        f"{len(shared_devices):,}"
    )

    print(
        f"Shared IPs     : "
        f"{len(shared_ips):,}"
    )

    return shared


# ============================================================
# GRAPH RISK SCORE
# ============================================================

def calculate_graph_risk(
    metrics
):

    print("\n" + "=" * 60)
    print("CALCULATING GRAPH RISK")
    print("=" * 60)

    result = metrics.copy()

    # --------------------------------------------------------
    # Percentile components
    # --------------------------------------------------------

    degree_score = (
        result[
            "graph_degree"
        ]
        .rank(pct=True)
        * 100
    )

    connection_score = (
        result[
            "connected_accounts"
        ]
        .rank(pct=True)
        * 100
    )

    device_score = (
        result[
            "connected_devices"
        ]
        .rank(pct=True)
        * 100
    )

    ip_score = (
        result[
            "connected_ips"
        ]
        .rank(pct=True)
        * 100
    )

    fraud_score = (
        result[
            "fraud_rate"
        ]
        .rank(pct=True)
        * 100
    )

    aml_score = (
        result[
            "aml_rate"
        ]
        .rank(pct=True)
        * 100
    )

    # --------------------------------------------------------
    # Combined score
    # --------------------------------------------------------

    result[
        "graph_risk_score"
    ] = (
        0.20 * degree_score
        + 0.20 * connection_score
        + 0.15 * device_score
        + 0.15 * ip_score
        + 0.15 * fraud_score
        + 0.15 * aml_score
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    def assign_risk(score):

        if score >= 90:
            return "critical"

        if score >= 75:
            return "high"

        if score >= 50:
            return "medium"

        if score >= 25:
            return "low"

        return "minimal"

    result[
        "graph_risk_level"
    ] = (
        result[
            "graph_risk_score"
        ]
        .apply(assign_risk)
    )

    print(
        "\nGraph risk distribution:"
    )

    print(
        result[
            "graph_risk_level"
        ]
        .value_counts()
        .sort_index()
    )

    return result


# ============================================================
# SUSPICIOUS TRANSFER EDGES
# ============================================================

def identify_suspicious_edges(
    graph
):

    print("\n" + "=" * 60)
    print("ANALYZING ACCOUNT-TO-ACCOUNT TRANSFERS")
    print("=" * 60)

    records = []

    for source, target, data in (
        graph.edges(
            data=True
        )
    ):

        if (
            data.get(
                "relationship"
            )
            != "transfers_to"
        ):
            continue

        source_account = (
            str(source)
            .replace(
                "ACCOUNT:",
                ""
            )
        )

        target_account = (
            str(target)
            .replace(
                "ACCOUNT:",
                ""
            )
        )

        records.append(
            {
                "source_account":
                    source_account,

                "target_account":
                    target_account,

                "amount":
                    float(
                        data.get(
                            "amount",
                            0
                        )
                    ),

                "transaction_id":
                    data.get(
                        "transaction_id"
                    ),

                "transaction_type":
                    data.get(
                        "transaction_type"
                    ),

                "is_fraud":
                    int(
                        data.get(
                            "is_fraud",
                            0
                        )
                    ),

                "is_aml":
                    int(
                        data.get(
                            "is_aml",
                            0
                        )
                    ),
            }
        )

    edges = pd.DataFrame(
        records
    )

    if edges.empty:

        print(
            "No account-to-account "
            "transfers found."
        )

        return edges

    # --------------------------------------------------------
    # Large transfer threshold
    # --------------------------------------------------------

    amount_threshold = (
        edges[
            "amount"
        ]
        .quantile(
            0.95
        )
    )

    edges[
        "large_transfer_flag"
    ] = (
        edges[
            "amount"
        ]
        >= amount_threshold
    ).astype(int)

    # --------------------------------------------------------
    # Suspicion logic
    # --------------------------------------------------------

    edges[
        "suspicious_edge"
    ] = (
        (
            edges[
                "is_fraud"
            ] == 1
        )
        |
        (
            edges[
                "is_aml"
            ] == 1
        )
        |
        (
            edges[
                "large_transfer_flag"
            ] == 1
        )
    ).astype(int)

    suspicious = (
        edges[
            edges[
                "suspicious_edge"
            ] == 1
        ]
        .copy()
    )

    print(
        f"Transfer edges     : "
        f"{len(edges):,}"
    )

    print(
        f"Large transfers    : "
        f"{edges['large_transfer_flag'].sum():,}"
    )

    print(
        f"Suspicious edges   : "
        f"{len(suspicious):,}"
    )

    return suspicious


# ============================================================
# GRAPH SUMMARY
# ============================================================

def create_summary(
    graph,
    metrics,
    shared_entities,
    suspicious_edges
):

    print("\n" + "=" * 60)
    print("GRAPH SUMMARY")
    print("=" * 60)

    high_risk_count = int(
        metrics[
            "graph_risk_level"
        ]
        .isin(
            [
                "high",
                "critical",
            ]
        )
        .sum()
    )

    critical_count = int(
        (
            metrics[
                "graph_risk_level"
            ]
            == "critical"
        )
        .sum()
    )

    shared_device_count = int(
        (
            shared_entities[
                "entity_type"
            ]
            == "device"
        ).sum()
    )

    shared_ip_count = int(
        (
            shared_entities[
                "entity_type"
            ]
            == "ip"
        ).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "nodes":
                    graph.number_of_nodes(),

                "edges":
                    graph.number_of_edges(),

                "accounts":
                    len(metrics),

                "shared_devices":
                    shared_device_count,

                "shared_ips":
                    shared_ip_count,

                "high_risk_accounts":
                    high_risk_count,

                "critical_accounts":
                    critical_count,

                "suspicious_transfer_edges":
                    len(
                        suspicious_edges
                    ),
            }
        ]
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return summary


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    metrics,
    shared_entities,
    suspicious_edges,
    summary
):

    print("\n" + "=" * 60)
    print("SAVING GRAPH OUTPUTS")
    print("=" * 60)

    metrics.to_csv(
        GRAPH_METRICS_FILE,
        index=False
    )

    shared_entities.to_csv(
        SHARED_ENTITIES_FILE,
        index=False
    )

    suspicious_edges.to_csv(
        SUSPICIOUS_EDGES_FILE,
        index=False
    )

    summary.to_csv(
        GRAPH_SUMMARY_FILE,
        index=False
    )

    print(
        f"Entity risk:"
    )

    print(
        GRAPH_METRICS_FILE
    )

    print(
        "\nShared entities:"
    )

    print(
        SHARED_ENTITIES_FILE
    )

    print(
        "\nSuspicious edges:"
    )

    print(
        SUSPICIOUS_EDGES_FILE
    )

    print(
        "\nGraph summary:"
    )

    print(
        GRAPH_SUMMARY_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 60)
    print("FINSHIELD AI")
    print("GRAPH INTELLIGENCE ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Clean
    # --------------------------------------------------------

    df = clean_data(
        df
    )

    # --------------------------------------------------------
    # 3. Build graph
    # --------------------------------------------------------

    graph = build_graph(
        df
    )

    # --------------------------------------------------------
    # 4. Account metrics
    # --------------------------------------------------------

    metrics = (
        calculate_account_metrics(
            graph,
            df
        )
    )

    # --------------------------------------------------------
    # 5. Shared entities
    # --------------------------------------------------------

    shared_entities = (
        calculate_shared_entities(
            df
        )
    )

    # --------------------------------------------------------
    # 6. Graph risk
    # --------------------------------------------------------

    metrics = (
        calculate_graph_risk(
            metrics
        )
    )

    # --------------------------------------------------------
    # 7. Suspicious edges
    # --------------------------------------------------------

    suspicious_edges = (
        identify_suspicious_edges(
            graph
        )
    )

    # --------------------------------------------------------
    # 8. Summary
    # --------------------------------------------------------

    summary = create_summary(
        graph,
        metrics,
        shared_entities,
        suspicious_edges
    )

    # --------------------------------------------------------
    # 9. Save
    # --------------------------------------------------------

    save_outputs(
        metrics,
        shared_entities,
        suspicious_edges,
        summary
    )

    print("\n" + "=" * 60)
    print("GRAPH INTELLIGENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()