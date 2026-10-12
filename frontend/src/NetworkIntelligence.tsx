
import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Network,
  RefreshCw,
  Search,
  ShieldAlert,
  X,
} from "lucide-react";

const API = "http://127.0.0.1:8000";

type GraphSummary = {
  nodes: number;
  edges: number;
  accounts: number;
  shared_devices: number;
  shared_ips: number;
  high_risk_accounts: number;
  critical_accounts: number;
  suspicious_transfer_edges: number;
  risk_distribution?: Record<string, number>;
};

type Account = {
  account_id: string;
  graph_degree: number;
  in_degree: number;
  out_degree: number;
  connected_accounts: number;
  connected_devices: number;
  connected_ips: number;
  connected_merchants: number;
  transaction_count: number;
  total_transaction_amount: number;
  fraud_transaction_count: number;
  aml_transaction_count: number;
  fraud_rate: number;
  aml_rate: number;
  graph_risk_score: number;
  graph_risk_level: string;
};

type Edge = {
  source_account: string;
  target_account: string;
  amount: number;
  transaction_id: string;
  transaction_type: string;
  is_fraud: number;
  is_aml: number;
  large_transfer_flag: number;
  suspicious_edge: number;
};

type SharedEntity = {
  entity_type: string;
  entity_id: string;
  account_count: number;
};

const money = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(Number(value) || 0);

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);

  if (!response.ok) {
    throw new Error(`${path} returned HTTP ${response.status}`);
  }

  return response.json() as Promise<T>;
}

function riskNodeColor(level: string): string {
  switch (level?.toLowerCase()) {
    case "critical":
      return "#fb7185";
    case "high":
      return "#fb923c";
    case "medium":
      return "#fbbf24";
    case "low":
      return "#38bdf8";
    default:
      return "#4ade80";
  }
}

export default function NetworkIntelligence() {
  const [summary, setSummary] = useState<GraphSummary | null>(null);
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [shared, setShared] = useState<SharedEntity[]>([]);
  const [selected, setSelected] = useState<Account | null>(null);
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  async function loadGraph() {
    setLoading(true);
    setError("");

    try {
      const [summaryData, accountData, edgeData, sharedData] =
        await Promise.all([
          getJson<GraphSummary>("/api/graph/summary"),
          getJson<{ total: number; accounts: Account[] }>(
            "/api/graph/accounts?limit=250&offset=0"
          ),
          getJson<
            { total_returned?: number; edges?: Edge[]; suspicious_edges?: Edge[] }
            | Edge[]
          >("/api/graph/suspicious-edges?limit=500"),
          getJson<
            { total_returned?: number; entities?: SharedEntity[]; shared_entities?: SharedEntity[] }
            | SharedEntity[]
          >("/api/graph/shared-entities?limit=100"),
        ]);

      setSummary(summaryData);
      setAccounts(Array.isArray(accountData.accounts) ? accountData.accounts : []);

      setEdges(
        Array.isArray(edgeData)
          ? edgeData
          : edgeData.edges ?? edgeData.suspicious_edges ?? []
      );

      setShared(
        Array.isArray(sharedData)
          ? sharedData
          : sharedData.entities ?? sharedData.shared_entities ?? []
      );
    } catch (err) {
      console.error("Graph API error:", err);
      setError(
        "Could not load graph data. Check that the FastAPI graph endpoints exist and the backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadGraph();
  }, []);

  const filtered = useMemo(
    () =>
      accounts
        .filter((account) => {
          const matchesQuery = account.account_id
            .toLowerCase()
            .includes(query.toLowerCase());

          const matchesLevel =
            filter === "all" ||
            account.graph_risk_level?.toLowerCase() === filter;

          return matchesQuery && matchesLevel;
        })
        .slice(0, 80),
    [accounts, query, filter]
  );

  const graph = useMemo(() => {
    const chosen = filtered.slice(0, 48);
    const cx = 480;
    const cy = 285;
    const rx = 390;
    const ry = 220;

    const positions = new Map<string, { x: number; y: number }>();

    chosen.forEach((account, index) => {
      const angle =
        (2 * Math.PI * index) / Math.max(chosen.length, 1) -
        Math.PI / 2;

      positions.set(account.account_id, {
        x: cx + Math.cos(angle) * rx,
        y: cy + Math.sin(angle) * ry,
      });
    });

    const visibleEdges = edges
      .filter(
        (edge) =>
          positions.has(edge.source_account) &&
          positions.has(edge.target_account)
      )
      .slice(0, 180);

    return { chosen, positions, visibleEdges };
  }, [filtered, edges]);

  const selectedEdges = selected
    ? edges
        .filter(
          (edge) =>
            edge.source_account === selected.account_id ||
            edge.target_account === selected.account_id
        )
        .slice(0, 12)
    : [];

  return (
    <div className="space-y-5 text-slate-100">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">
            Graph-based investigation
          </p>
          <h3 className="mt-2 text-xl font-semibold">
            Financial entity network
          </h3>
          <p className="mt-1 max-w-3xl text-sm text-slate-400">
            Explore account connections, shared devices and IPs, and
            suspicious transfers. Select an account node to inspect its
            risk profile.
          </p>
        </div>

        <button
          onClick={() => void loadGraph()}
          disabled={loading}
          className="flex items-center gap-2 rounded-lg border border-white/10 px-4 py-2 text-sm text-slate-300 hover:bg-white/5 disabled:opacity-50"
        >
          <RefreshCw
            size={15}
            className={loading ? "animate-spin" : ""}
          />
          Refresh graph
        </button>
      </div>

      {error && (
        <div className="rounded-xl border border-amber-300/20 bg-amber-300/5 p-4 text-sm text-amber-100">
          <div className="flex gap-2">
            <AlertTriangle size={18} className="shrink-0" />
            <span>{error}</span>
          </div>
        </div>
      )}

      {loading && !summary && (
        <p className="text-sm text-slate-400">
          Loading graph intelligence…
        </p>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ["Graph nodes", summary?.nodes],
          ["Graph relationships", summary?.edges],
          ["High-risk accounts", summary?.high_risk_accounts],
          ["Suspicious transfers", summary?.suspicious_transfer_edges],
        ].map(([label, value]) => (
          <article
            key={String(label)}
            className="rounded-xl border border-white/[0.08] bg-[#101725] p-4"
          >
            <p className="text-sm text-slate-400">{label}</p>
            <p className="mt-3 text-2xl font-semibold">
              {typeof value === "number"
                ? value.toLocaleString("en-IN")
                : value ?? "—"}
            </p>
          </article>
        ))}
      </div>

      <div className="grid gap-5 2xl:grid-cols-[minmax(0,1fr)_360px]">
        <section className="overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0b1220]">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.08] p-4">
            <div className="flex items-center gap-2">
              <Network size={18} className="text-cyan-300" />
              <h4 className="font-semibold">Relationship map</h4>
              <span className="rounded-full bg-white/5 px-2 py-1 text-xs text-slate-400">
                {graph.chosen.length} account nodes
              </span>
            </div>

            <div className="flex flex-wrap gap-2">
              <label className="flex items-center gap-2 rounded-lg border border-white/10 px-3">
                <Search size={14} className="text-slate-500" />
                <input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Find account…"
                  className="w-32 bg-transparent py-2 text-xs outline-none"
                />
              </label>

              <select
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
                className="rounded-lg border border-white/10 bg-[#101725] px-2 py-2 text-xs"
              >
                <option value="all">All risk levels</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="medium">Medium</option>
                <option value="low">Low</option>
                <option value="minimal">Minimal</option>
              </select>
            </div>
          </div>

          <div className="overflow-x-auto">
            <svg
              viewBox="0 0 960 570"
              role="img"
              aria-label="Interactive account relationship graph"
              className="min-w-[720px] w-full"
              style={{
                background:
                  "radial-gradient(ellipse at center, #111c2d 0%, #080e19 75%)",
              }}
            >
              <defs>
                <pattern
                  id="network-grid"
                  width="28"
                  height="28"
                  patternUnits="userSpaceOnUse"
                >
                  <path
                    d="M 28 0 L 0 0 0 28"
                    fill="none"
                    stroke="#243044"
                    strokeWidth="0.6"
                    opacity="0.45"
                  />
                </pattern>
              </defs>

              <rect
                width="960"
                height="570"
                fill="url(#network-grid)"
              />

              {graph.visibleEdges.map((edge, index) => {
                const start = graph.positions.get(edge.source_account)!;
                const end = graph.positions.get(edge.target_account)!;

                const isFlagged = Boolean(edge.is_fraud || edge.is_aml);

                return (
                  <line
                    key={`${edge.transaction_id}-${index}`}
                    x1={start.x}
                    y1={start.y}
                    x2={end.x}
                    y2={end.y}
                    stroke={
                      edge.is_fraud
                        ? "#fb7185"
                        : edge.is_aml
                          ? "#f59e0b"
                          : "#4c7188"
                    }
                    strokeOpacity={isFlagged ? 0.72 : 0.25}
                    strokeWidth={isFlagged ? 1.7 : 0.8}
                  />
                );
              })}

              {graph.chosen.map((account) => {
                const point = graph.positions.get(account.account_id)!;
                const isSelected =
                  selected?.account_id === account.account_id;
                const color = riskNodeColor(account.graph_risk_level);
                const radius = isSelected
                  ? 12
                  : Math.min(10, 5 + (account.graph_degree || 0) / 65);

                return (
                  <g
                    key={account.account_id}
                    role="button"
                    tabIndex={0}
                    aria-label={`Inspect ${account.account_id}`}
                    onClick={() => setSelected(account)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        setSelected(account);
                      }
                    }}
                    className="cursor-pointer"
                  >
                    {isSelected && (
                      <circle
                        cx={point.x}
                        cy={point.y}
                        r={radius + 7}
                        fill="none"
                        stroke="#67e8f9"
                        strokeWidth="2"
                      />
                    )}

                    <circle
                      cx={point.x}
                      cy={point.y}
                      r={radius}
                      fill={color}
                      fillOpacity="0.92"
                      stroke={isSelected ? "#cffafe" : "#dbeafe"}
                      strokeWidth={isSelected ? 2 : 0.5}
                    />

                    <title>
                      {`${account.account_id} • risk ${Number(
                        account.graph_risk_score
                      ).toFixed(1)} • ${account.graph_risk_level}`}
                    </title>

                    {(isSelected || graph.chosen.length < 25) && (
                      <text
                        x={point.x}
                        y={point.y + radius + 14}
                        textAnchor="middle"
                        fill="#cbd5e1"
                        fontSize="10"
                      >
                        {account.account_id}
                      </text>
                    )}
                  </g>
                );
              })}

              {graph.chosen.length === 0 && (
                <text
                  x="480"
                  y="285"
                  fill="#94a3b8"
                  fontSize="16"
                  textAnchor="middle"
                >
                  No accounts match these filters
                </text>
              )}
            </svg>
          </div>

          <div className="flex flex-wrap gap-x-5 gap-y-2 border-t border-white/[0.08] p-4 text-xs text-slate-400">
            {[
              ["Minimal", "bg-emerald-400"],
              ["Low", "bg-sky-400"],
              ["Medium", "bg-amber-300"],
              ["High", "bg-orange-400"],
              ["Critical", "bg-rose-400"],
            ].map(([label, color]) => (
              <span key={label} className="flex items-center gap-2">
                <i className={`h-2.5 w-2.5 rounded-full ${color}`} />
                {label}
              </span>
            ))}
            <span className="ml-auto">
              Edge: amber = AML · rose = fraud
            </span>
          </div>
        </section>

        <aside className="space-y-5">
          <section className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
            <div className="mb-4 flex items-center justify-between">
              <h4 className="font-semibold">Account details</h4>
              {selected && (
                <button
                  onClick={() => setSelected(null)}
                  aria-label="Close account details"
                  className="rounded-lg p-1 text-slate-400 hover:bg-white/5"
                >
                  <X size={16} />
                </button>
              )}
            </div>

            {selected ? (
              <>
                <p className="break-all text-sm font-semibold text-cyan-200">
                  {selected.account_id}
                </p>

                <div className="mt-3 inline-flex rounded-full bg-white/5 px-2.5 py-1 text-xs capitalize text-slate-200">
                  {selected.graph_risk_level} risk
                </div>

                <div className="mt-4">
                  <div className="mb-1 flex justify-between text-xs text-slate-400">
                    <span>Graph risk score</span>
                    <span>
                      {Number(selected.graph_risk_score).toFixed(1)}/100
                    </span>
                  </div>
                  <div className="h-2 rounded-full bg-white/5">
                    <div
                      className="h-2 rounded-full bg-cyan-400"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(
                            100,
                            Number(selected.graph_risk_score) || 0
                          )
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="mt-5 grid grid-cols-2 gap-3">
                  {[
                    ["Degree", selected.graph_degree],
                    ["Transactions", selected.transaction_count],
                    ["Devices", selected.connected_devices],
                    ["IP addresses", selected.connected_ips],
                    ["Merchants", selected.connected_merchants],
                    ["Connected accounts", selected.connected_accounts],
                    ["Fraud labels", selected.fraud_transaction_count],
                    ["AML labels", selected.aml_transaction_count],
                  ].map(([label, value]) => (
                    <div
                      key={String(label)}
                      className="rounded-lg bg-white/[0.03] p-3"
                    >
                      <p className="text-xs text-slate-500">{label}</p>
                      <p className="mt-1 font-semibold">
                        {Number(value ?? 0).toLocaleString("en-IN")}
                      </p>
                    </div>
                  ))}
                </div>

                <div className="mt-4 rounded-lg bg-white/[0.03] p-3">
                  <p className="text-xs text-slate-500">
                    Total transaction value
                  </p>
                  <p className="mt-1 font-semibold">
                    {money(selected.total_transaction_amount)}
                  </p>
                  <p className="mt-2 text-xs text-slate-500">
                    Fraud rate {(selected.fraud_rate * 100).toFixed(1)}% ·
                    AML rate {(selected.aml_rate * 100).toFixed(1)}%
                  </p>
                </div>

                <h5 className="mb-2 mt-5 text-sm font-medium">
                  Related suspicious transfers
                </h5>

                {selectedEdges.length ? (
                  <div className="space-y-2">
                    {selectedEdges.map((edge, index) => (
                      <div
                        key={`${edge.transaction_id}-${index}`}
                        className="rounded-lg border border-white/[0.06] p-3"
                      >
                        <div className="flex items-start justify-between gap-2">
                          <span className="text-xs text-slate-300">
                            {edge.transaction_id}
                          </span>
                          <span className="text-xs text-amber-200">
                            {money(edge.amount)}
                          </span>
                        </div>

                        <p className="mt-1 break-all text-[11px] text-slate-500">
                          {edge.source_account} → {edge.target_account}
                        </p>

                        <p className="mt-1 text-[11px] text-slate-500">
                          {edge.is_fraud
                            ? "Fraud-labelled"
                            : edge.is_aml
                              ? "AML-labelled"
                              : "Flagged transfer"}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500">
                    No returned suspicious transfer edge for this account.
                  </p>
                )}
              </>
            ) : (
              <div className="rounded-xl border border-dashed border-white/10 p-6 text-center">
                <ShieldAlert
                  className="mx-auto text-slate-500"
                  size={24}
                />
                <p className="mt-3 text-sm text-slate-300">
                  Select a node to investigate an account
                </p>
                <p className="mt-1 text-xs leading-5 text-slate-500">
                  Risk score, connected entities, activity and matching
                  suspicious transfers will appear here.
                </p>
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
            <h4 className="font-semibold">Shared entities</h4>
            <p className="mt-1 text-xs text-slate-500">
              Devices or IPs associated with multiple accounts
            </p>

            <div className="mt-4 space-y-2">
              {shared.slice(0, 8).map((entity, index) => (
                <div
                  key={`${entity.entity_type}-${entity.entity_id}-${index}`}
                  className="flex items-center justify-between gap-3 rounded-lg bg-white/[0.03] p-3"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm">{entity.entity_id}</p>
                    <p className="mt-1 text-xs capitalize text-slate-500">
                      {entity.entity_type}
                    </p>
                  </div>

                  <span className="shrink-0 rounded-full bg-violet-300/10 px-2 py-1 text-xs text-violet-200">
                    {entity.account_count} accounts
                  </span>
                </div>
              ))}

              {shared.length === 0 && (
                <p className="text-sm text-slate-500">
                  No shared-entity data returned.
                </p>
              )}
            </div>
          </section>
        </aside>
      </div>

      <p className="text-xs leading-5 text-slate-500">
        This visualization displays graph-engine risk scores and synthetic
        fraud/AML labels. These are investigation signals, not proof of
        financial crime. The map displays up to 48 account nodes at a time.
      </p>
    </div>
  );
}
