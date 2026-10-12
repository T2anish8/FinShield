import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CircleDollarSign,
  Search,
  ShieldCheck,
  ShieldAlert,
  RefreshCw,
  TrendingUp,
  Zap,
  Network,
} from "lucide-react";
import NetworkIntelligence from "./NetworkIntelligence";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  LineChart,
  Line,
  Legend,
} from "recharts";

const API = "http://127.0.0.1:8000";

// ============================================================
// TYPES
// ============================================================

type Summary = {
  total_transactions: number;
  total_amount: number;
  average_risk_score: number;
  high_risk: number;
  critical_risk: number;
  medium_risk: number;
  low_risk: number;
  minimal_risk: number;
  risk_distribution: {
    risk_level: string;
    count: number;
  }[];
};

type Transaction = {
  transaction_id: string;
  timestamp?: string;
  account_id?: string;
  amount: number;
  transaction_type?: string;
  channel?: string;
  unified_risk_score: number;
  unified_risk_level: string;
  is_demo_suspicious?: number;
  simulation?: boolean;
  scenario_type?: string;
};

type StreamStatus = "connecting" | "connected" | "disconnected";

type ActivityPoint = {
  time: string;
  amount: number;
  count: number;
};

// ============================================================
// HELPERS
// ============================================================

const money = (value: number) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);

function riskColor(level: string) {
  switch (level?.toLowerCase()) {
    case "critical":
      return "text-red-400 bg-red-400/10";
    case "high":
      return "text-orange-400 bg-orange-400/10";
    case "medium":
      return "text-amber-300 bg-amber-300/10";
    case "low":
      return "text-sky-300 bg-sky-300/10";
    default:
      return "text-emerald-300 bg-emerald-300/10";
  }
}

// ============================================================
// APPLICATION
// ============================================================

function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [search, setSearch] = useState("");
  const [risk, setRisk] = useState("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState<"overview" | "network">("overview");

  // WebSocket state
  const [liveTransactions, setLiveTransactions] = useState<Transaction[]>([]);
  const [streamStatus, setStreamStatus] =
    useState<StreamStatus>("connecting");
  const [eventsReceived, setEventsReceived] = useState(0);

  // Live session analytics
  const [liveSessionAmount, setLiveSessionAmount] = useState(0);
  const [suspiciousEvents, setSuspiciousEvents] = useState(0);
  const [highRiskEvents, setHighRiskEvents] = useState(0);
  const [recentActivity, setRecentActivity] = useState<ActivityPoint[]>([]);
  const [eventTimestamps, setEventTimestamps] = useState<number[]>([]);

  // ==========================================================
  // LOAD HISTORICAL DASHBOARD DATA
  // ==========================================================

  async function loadDashboard() {
    setLoading(true);
    setError("");

    try {
      const [summaryResponse, transactionResponse] = await Promise.all([
        fetch(`${API}/api/summary`),
        fetch(`${API}/api/transactions?limit=100&offset=0`),
      ]);

      if (!summaryResponse.ok || !transactionResponse.ok) {
        throw new Error("The API returned an error.");
      }

      const summaryData = await summaryResponse.json();
      const transactionData = await transactionResponse.json();

      setSummary(summaryData);
      setTransactions(transactionData.transactions);
    } catch (err) {
      console.error("Dashboard API error:", err);
      setError(
        "Cannot connect to FinShield API. Check that the FastAPI backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void loadDashboard();
  }, []);

  // ==========================================================
  // LIVE WEBSOCKET STREAM
  // ==========================================================

  useEffect(() => {
    let disposed = false;
    let retryTimer: number | undefined;
    let socket: WebSocket | null = null;

    function connect() {
      if (disposed) return;

      setStreamStatus("connecting");

      socket = new WebSocket(
        `${API.replace(/^http/, "ws")}/ws/transactions`
      );

      socket.onopen = () => {
        if (!disposed) {
          setStreamStatus("connected");
        }
      };

      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);

          if (message.type !== "transaction" || !message.data) {
            return;
          }

          const transaction = message.data as Transaction;

          if (disposed) return;

          const amount = Number(transaction.amount) || 0;
          const level =
            transaction.unified_risk_level?.toLowerCase() ?? "minimal";

          const receivedAt = new Date();
          const receivedAtMs = receivedAt.getTime();

          // Maintain timestamps for the rolling 60-second event rate.
          setEventTimestamps((previous) => [
            ...previous.filter(
              (timestamp) => receivedAtMs - timestamp < 60_000
            ),
            receivedAtMs,
          ]);

          // Keep the latest 10 live transactions.
          setLiveTransactions((previous) =>
            [transaction, ...previous].slice(0, 10)
          );

          // Update session counters.
          setEventsReceived((count) => count + 1);
          setLiveSessionAmount((total) => total + amount);

          if (Number(transaction.is_demo_suspicious) === 1) {
            setSuspiciousEvents((count) => count + 1);
          }

          if (["high", "critical"].includes(level)) {
            setHighRiskEvents((count) => count + 1);
          }

          // Maintain the latest 20 transaction amounts.
          setRecentActivity((previous) =>
            [
              ...previous,
              {
                time: receivedAt.toLocaleTimeString("en-IN", {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                }),
                amount,
                count: 1,
              },
            ].slice(-20)
          );
        } catch (err) {
          console.error("Invalid WebSocket message:", err);
        }
      };

      socket.onerror = () => {
        if (!disposed) {
          setStreamStatus("disconnected");
        }
      };

      socket.onclose = () => {
        if (disposed) return;

        setStreamStatus("disconnected");
        retryTimer = window.setTimeout(connect, 3000);
      };
    }

    connect();

    return () => {
      disposed = true;

      if (retryTimer !== undefined) {
        window.clearTimeout(retryTimer);
      }

      socket?.close();
    };
  }, []);

  // ==========================================================
  // LIVE ANALYTICS
  // ==========================================================

  const transactionsPerMinute = eventTimestamps.filter(
    (timestamp) => Date.now() - timestamp < 60_000
  ).length;

  const suspiciousFeed = liveTransactions
    .filter(
      (item) =>
        Number(item.is_demo_suspicious) === 1 ||
        ["high", "critical"].includes(
          String(item.unified_risk_level).toLowerCase()
        )
    )
    .slice(0, 5);

  // ==========================================================
  // FILTER HISTORICAL TRANSACTIONS
  // ==========================================================

  const filteredTransactions = transactions.filter((item) => {
    const matchesRisk =
      risk === "all" || item.unified_risk_level?.toLowerCase() === risk;

    const term = search.toLowerCase();

    const matchesSearch = [
      item.transaction_id,
      item.account_id,
      item.transaction_type,
      item.channel,
    ].some((value) =>
      String(value ?? "").toLowerCase().includes(term)
    );

    return matchesRisk && matchesSearch;
  });

  // ==========================================================
  // HISTORICAL SUMMARY CARDS
  // ==========================================================

  const cards = summary
    ? [
        {
          title: "Transactions analyzed",
          value: summary.total_transactions.toLocaleString("en-IN"),
          icon: Activity,
          detail: "Unified risk engine",
          color: "text-sky-300",
        },
        {
          title: "High-risk transactions",
          value: summary.high_risk.toLocaleString("en-IN"),
          icon: ShieldAlert,
          detail: `${summary.critical_risk} critical`,
          color: "text-orange-300",
        },
        {
          title: "Medium risk",
          value: summary.medium_risk.toLocaleString("en-IN"),
          icon: AlertTriangle,
          detail: "Requires review",
          color: "text-amber-300",
        },
        {
          title: "Transaction value",
          value: money(summary.total_amount),
          icon: CircleDollarSign,
          detail: `Average score ${summary.average_risk_score}/100`,
          color: "text-emerald-300",
        },
      ]
    : [];

  // ==========================================================
  // UI
  // ==========================================================

  return (
    <div className="min-h-screen bg-[#080c14] text-slate-100">
      {/* SIDEBAR */}
      <aside className="fixed inset-y-0 left-0 hidden w-64 border-r border-white/10 bg-[#0b101b] p-6 lg:block">
        <div className="flex items-center gap-3">
          <div className="rounded-xl bg-cyan-400/10 p-3 text-cyan-300">
            <ShieldCheck size={26} />
          </div>

          <div>
            <h1 className="text-lg font-bold tracking-wide">FinShield AI</h1>
            <p className="text-xs text-slate-500">Risk intelligence</p>
          </div>
        </div>

        <p className="mb-3 mt-10 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
          Workspace
        </p>

        <button
          onClick={() => setActiveView("overview")}
          className={`flex w-full items-center gap-3 rounded-xl border px-4 py-3 text-left ${activeView === "overview" ? "border-cyan-300/15 bg-cyan-300/10 text-cyan-200" : "border-transparent text-slate-400 hover:bg-white/5"}`}
        >
          <Activity size={18} />
          <span className="text-sm font-medium">Overview</span>
        </button>

        <button
          onClick={() => setActiveView("network")}
          className={`mt-3 flex w-full items-center gap-3 rounded-xl border px-4 py-3 text-left ${activeView === "network" ? "border-cyan-300/15 bg-cyan-300/10 text-cyan-200" : "border-transparent text-slate-400 hover:bg-white/5"}`}
        >
          <Network size={18} />
          <span className="text-sm font-medium">Network intelligence</span>
        </button>

        <div className="mt-3 flex items-center gap-3 px-4 py-3 text-slate-400">
          <ShieldAlert size={18} />
          <span className="text-sm">Risk monitoring</span>
        </div>

        <div className="mt-3 flex items-center gap-3 px-4 py-3 text-slate-400">
          <CircleDollarSign size={18} />
          <span className="text-sm">Exposure analytics</span>
        </div>

        <div className="absolute bottom-6 left-6 right-6 rounded-xl border border-white/10 p-4">
          <div className="flex items-center gap-2 text-sm text-emerald-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Risk engine outputs
          </div>
          <p className="mt-2 text-xs text-slate-500">
            Historical CSV-backed data
          </p>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="lg:ml-64">
        <header className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-4 border-b border-white/10 bg-[#080c14]/90 px-5 py-5 backdrop-blur-xl md:px-8">
          <div>
            <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">
              Financial crime intelligence
            </p>
            <h2 className="mt-1 text-2xl font-semibold">{activeView === "network" ? "Network Intelligence" : "Risk Overview"}</h2>
          </div>

          <button
            onClick={() => void loadDashboard()}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg border border-white/10 px-4 py-2 text-sm text-slate-300 hover:bg-white/5 disabled:opacity-50"
          >
            <RefreshCw size={15} className={loading ? "animate-spin" : ""} />
            Refresh data
          </button>
        </header>

        {activeView === "network" ? (
          <div className="p-5 md:p-8">
            <NetworkIntelligence />
          </div>
        ) : (
        <div className="space-y-7 p-5 md:p-8">
          {error && (
            <div className="rounded-xl border border-red-400/20 bg-red-400/10 p-4 text-sm text-red-200">
              {error}
            </div>
          )}

          {loading && !summary && (
            <p className="text-sm text-slate-400">Loading risk data…</p>
          )}

          {/* HISTORICAL SUMMARY CARDS */}
          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {cards.map((card) => {
              const Icon = card.icon;

              return (
                <article
                  key={card.title}
                  className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5"
                >
                  <div className="flex items-center justify-between">
                    <p className="text-sm text-slate-400">{card.title}</p>
                    <Icon size={20} className={card.color} />
                  </div>

                  <p className="mt-5 break-words text-2xl font-semibold tracking-tight">
                    {card.value}
                  </p>

                  <div className="mt-3 flex items-center gap-2 text-xs text-slate-500">
                    <ArrowUpRight size={14} />
                    {card.detail}
                  </div>
                </article>
              );
            })}
          </section>

          {/* HISTORICAL RISK DISTRIBUTION AND SIGNALS */}
          <section className="grid gap-5 xl:grid-cols-5">
            <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5 xl:col-span-3">
              <div className="mb-6">
                <h3 className="font-semibold">Risk distribution</h3>
                <p className="mt-1 text-sm text-slate-500">
                  Transactions grouped by unified risk level
                </p>
              </div>

              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={summary?.risk_distribution ?? []}
                    margin={{ top: 8, right: 8, left: -18, bottom: 4 }}
                  >
                    <CartesianGrid
                      stroke="#263244"
                      strokeDasharray="3 3"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="risk_level"
                      stroke="#8794a8"
                      tick={{ fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      stroke="#8794a8"
                      tick={{ fontSize: 11 }}
                      tickLine={false}
                      axisLine={false}
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#101725",
                        border: "1px solid #293447",
                        borderRadius: 12,
                        color: "#e5e7eb",
                      }}
                    />
                    <Bar
                      dataKey="count"
                      name="Transactions"
                      fill="#38bdf8"
                      radius={[6, 6, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </article>

            <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5 xl:col-span-2">
              <h3 className="font-semibold">Risk signal coverage</h3>
              <p className="mt-1 text-sm text-slate-500">
                Inputs to the unified risk engine
              </p>

              <div className="mt-6 space-y-5">
                {[
                  ["Fraud model", "35%", "bg-sky-400", 35],
                  ["AML rule engine", "30%", "bg-violet-400", 30],
                  ["Anomaly detection", "20%", "bg-amber-400", 20],
                  ["Graph intelligence", "15%", "bg-emerald-400", 15],
                ].map(([label, weight, color, width]) => (
                  <div key={String(label)}>
                    <div className="mb-2 flex justify-between gap-3 text-sm">
                      <span className="text-slate-300">{label}</span>
                      <span className="text-slate-400">{weight}</span>
                    </div>

                    <div className="h-2 overflow-hidden rounded-full bg-white/5">
                      <div
                        className={`h-full rounded-full ${color}`}
                        style={{ width: `${width}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-7 rounded-xl border border-emerald-300/10 bg-emerald-300/5 p-4">
                <div className="flex items-center gap-2 text-sm text-emerald-300">
                  <ShieldCheck size={17} />
                  Unified scoring configured
                </div>
                <p className="mt-2 text-xs leading-5 text-slate-400">
                  Historical scores combine model probabilities, AML rules,
                  anomaly percentiles, and account graph risk.
                </p>
              </div>
            </article>
          </section>

          {/* LIVE MONITORING ANALYTICS */}
          <section className="space-y-5">
            <div>
              <div className="flex items-center gap-2">
                <TrendingUp size={20} className="text-cyan-300" />
                <h3 className="text-lg font-semibold">
                  Live monitoring analytics
                </h3>
              </div>

              <p className="mt-1 text-sm text-slate-500">
                Metrics for the current synthetic simulation session
              </p>
            </div>

            {/* LIVE METRIC CARDS */}
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
              <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-400">Events received</p>
                  <Activity size={19} className="text-sky-300" />
                </div>
                <p className="mt-4 text-2xl font-semibold">
                  {eventsReceived.toLocaleString("en-IN")}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  Current browser session
                </p>
              </article>

              <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-400">
                    Simulated transaction value
                  </p>
                  <CircleDollarSign size={19} className="text-emerald-300" />
                </div>
                <p className="mt-4 break-words text-2xl font-semibold">
                  {money(liveSessionAmount)}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  Cumulative session value
                </p>
              </article>

              <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-400">
                    Suspicious demo events
                  </p>
                  <Zap size={19} className="text-orange-300" />
                </div>
                <p className="mt-4 text-2xl font-semibold text-orange-300">
                  {suspiciousEvents.toLocaleString("en-IN")}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  Simulator-provided labels
                </p>
              </article>

              <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-400">High-risk events</p>
                  <ShieldAlert size={19} className="text-red-300" />
                </div>
                <p className="mt-4 text-2xl font-semibold text-red-300">
                  {highRiskEvents.toLocaleString("en-IN")}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  High or critical demo scores
                </p>
              </article>

              <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
                <div className="flex items-center justify-between">
                  <p className="text-sm text-slate-400">Transactions/min</p>
                  <TrendingUp size={19} className="text-cyan-300" />
                </div>
                <p className="mt-4 text-2xl font-semibold text-cyan-300">
                  {transactionsPerMinute.toLocaleString("en-IN")}
                </p>
                <p className="mt-2 text-xs text-slate-500">
                  Rolling 60-second WebSocket rate
                </p>
              </article>
            </div>

            {/* LIVE ACTIVITY CHART */}
            <article className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
              <div className="mb-5">
                <h3 className="font-semibold">Recent transaction activity</h3>
                <p className="mt-1 text-sm text-slate-500">
                  Amount per event across the latest 20 stream events
                </p>
              </div>

              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={recentActivity}
                    margin={{ top: 8, right: 12, left: 0, bottom: 4 }}
                  >
                    <CartesianGrid
                      stroke="#263244"
                      strokeDasharray="3 3"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="time"
                      stroke="#8794a8"
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      minTickGap={24}
                    />
                    <YAxis
                      stroke="#8794a8"
                      tick={{ fontSize: 10 }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(value: number) =>
                        new Intl.NumberFormat("en-IN", {
                          notation: "compact",
                          maximumFractionDigits: 1,
                        }).format(value)
                      }
                    />
                    <Tooltip
                      contentStyle={{
                        background: "#101725",
                        border: "1px solid #293447",
                        borderRadius: 12,
                        color: "#e5e7eb",
                      }}
                      formatter={(value, name) => [
                        name === "Amount (INR)"
                          ? money(Number(value))
                          : value,
                        name,
                      ]}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="amount"
                      name="Amount (INR)"
                      stroke="#38bdf8"
                      strokeWidth={2}
                      dot={false}
                      activeDot={{ r: 4 }}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {recentActivity.length === 0 && (
                <p className="mt-2 text-center text-sm text-slate-500">
                  Waiting for incoming transactions…
                </p>
              )}
            </article>
          </section>

          {/* RECENT SUSPICIOUS / HIGH-RISK EVENTS */}
          <section className="rounded-2xl border border-orange-300/10 bg-[#101725] p-5">
            <div className="mb-4 flex items-center gap-2">
              <AlertTriangle size={19} className="text-orange-300" />
              <h3 className="font-semibold">
                Recent suspicious / high-risk events
              </h3>
            </div>

            {suspiciousFeed.length > 0 ? (
              <div className="space-y-2">
                {suspiciousFeed.map((item, index) => (
                  <div
                    key={`${item.transaction_id}-alert-${index}`}
                    className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-orange-300/10 bg-orange-300/[0.03] p-3"
                  >
                    <div>
                      <p className="text-sm font-medium">
                        {item.transaction_id}
                      </p>
                      <p className="mt-1 text-xs text-slate-500">
                        {item.scenario_type ?? "Synthetic event"} ·{" "}
                        {item.transaction_type ?? "Transaction"}
                        {Number(item.is_demo_suspicious) === 1
                          ? " · Simulator flagged"
                          : ""}
                      </p>
                    </div>

                    <div className="text-right">
                      <p className="text-sm">
                        {money(Number(item.amount) || 0)}
                      </p>
                      <span
                        className={`mt-1 inline-block rounded-full px-2 py-1 text-xs capitalize ${riskColor(
                          item.unified_risk_level
                        )}`}
                      >
                        {item.unified_risk_level} ·{" "}
                        {Number(item.unified_risk_score).toFixed(1)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-500">
                No suspicious or high-risk events in the latest 10 received
                transactions.
              </p>
            )}

            <p className="mt-3 text-xs text-slate-500">
              Alerts use simulator-provided labels or a high/critical demo risk
              level.
            </p>
          </section>

          {/* LIVE TRANSACTION STREAM */}
          <section className="rounded-2xl border border-white/[0.08] bg-[#101725] p-5">
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h3 className="font-semibold">Live transaction stream</h3>
                <p className="mt-1 text-sm text-slate-500">
                  Newly generated synthetic transactions
                </p>
              </div>

              <div className="flex items-center gap-2 text-sm">
                <span
                  className={`h-2 w-2 rounded-full ${
                    streamStatus === "connected"
                      ? "bg-emerald-400"
                      : streamStatus === "connecting"
                        ? "bg-amber-400"
                        : "bg-red-400"
                  }`}
                />
                <span className="capitalize">{streamStatus}</span>
              </div>
            </div>

            <div className="mb-5 grid grid-cols-2 gap-3 sm:grid-cols-3">
              <div className="rounded-xl bg-white/[0.03] p-3">
                <p className="text-xs text-slate-500">Events received</p>
                <p className="mt-1 text-xl font-semibold">
                  {eventsReceived.toLocaleString("en-IN")}
                </p>
              </div>

              <div className="rounded-xl bg-white/[0.03] p-3">
                <p className="text-xs text-slate-500">
                  High-risk events in session
                </p>
                <p className="mt-1 text-xl font-semibold text-orange-300">
                  {highRiskEvents.toLocaleString("en-IN")}
                </p>
              </div>

              <div className="col-span-2 rounded-xl bg-white/[0.03] p-3 sm:col-span-1">
                <p className="text-xs text-slate-500">Event interval</p>
                <p className="mt-1 text-xl font-semibold">1.5 seconds</p>
              </div>
            </div>

            <div className="space-y-2">
              {liveTransactions.map((item, index) => (
                <div
                  key={`${item.transaction_id}-${index}`}
                  className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/[0.06] p-3"
                >
                  <div>
                    <p className="text-sm font-medium">
                      {item.transaction_id}
                    </p>
                    <p className="mt-1 text-xs capitalize text-slate-500">
                      {item.transaction_type ?? "Transaction"} ·{" "}
                      {item.channel ?? "Unknown channel"}
                    </p>
                  </div>

                  <div className="text-right">
                    <p className="text-sm">
                      {money(Number(item.amount) || 0)}
                    </p>
                    <span
                      className={`mt-1 inline-block rounded-full px-2 py-1 text-xs capitalize ${riskColor(
                        item.unified_risk_level
                      )}`}
                    >
                      {item.unified_risk_level} ·{" "}
                      {Number(item.unified_risk_score).toFixed(1)}
                    </span>
                  </div>
                </div>
              ))}

              {liveTransactions.length === 0 && (
                <p className="py-6 text-center text-sm text-slate-500">
                  {streamStatus === "connected"
                    ? "Waiting for the first transaction…"
                    : "Connecting to the transaction stream…"}
                </p>
              )}
            </div>

            <p className="mt-4 text-xs leading-5 text-slate-500">
              Demo mode: the simulator generates synthetic events with
              illustrative risk scores. These scores are not fresh predictions
              from the trained fraud model or the complete AML pipeline.
            </p>
          </section>

          {/* HISTORICAL TRANSACTION MONITORING */}
          <section className="overflow-hidden rounded-2xl border border-white/[0.08] bg-[#101725]">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.08] p-5">
              <div>
                <h3 className="font-semibold">Transaction monitoring</h3>
                <p className="mt-1 text-sm text-slate-500">
                  Highest-scoring transactions from the historical dataset
                </p>
              </div>

              <div className="flex flex-wrap gap-3">
                <div className="flex items-center gap-2 rounded-lg border border-white/10 px-3">
                  <Search size={16} className="text-slate-500" />
                  <input
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search transactions…"
                    className="w-40 bg-transparent py-2 text-sm outline-none placeholder:text-slate-600"
                  />
                </div>

                <select
                  value={risk}
                  onChange={(event) => setRisk(event.target.value)}
                  className="rounded-lg border border-white/10 bg-[#101725] px-3 py-2 text-sm outline-none"
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
              <table className="w-full min-w-[760px] text-left text-sm">
                <thead className="bg-white/[0.025] text-xs uppercase tracking-wider text-slate-500">
                  <tr>
                    <th className="px-5 py-4">Transaction</th>
                    <th className="px-5 py-4">Account</th>
                    <th className="px-5 py-4">Type</th>
                    <th className="px-5 py-4">Amount</th>
                    <th className="px-5 py-4">Risk score</th>
                    <th className="px-5 py-4">Risk level</th>
                  </tr>
                </thead>

                <tbody className="divide-y divide-white/[0.06]">
                  {filteredTransactions.map((item) => (
                    <tr
                      key={item.transaction_id}
                      className="hover:bg-white/[0.025]"
                    >
                      <td className="px-5 py-4 font-medium text-slate-200">
                        {item.transaction_id}
                      </td>

                      <td className="px-5 py-4 text-slate-400">
                        {item.account_id ?? "—"}
                      </td>

                      <td className="px-5 py-4 capitalize text-slate-400">
                        {item.transaction_type ?? "—"}
                      </td>

                      <td className="px-5 py-4">
                        {money(Number(item.amount) || 0)}
                      </td>

                      <td className="px-5 py-4 font-medium">
                        {Number(item.unified_risk_score).toFixed(1)}
                      </td>

                      <td className="px-5 py-4">
                        <span
                          className={`rounded-full px-2.5 py-1 text-xs font-medium capitalize ${riskColor(
                            item.unified_risk_level
                          )}`}
                        >
                          {item.unified_risk_level}
                        </span>
                      </td>
                    </tr>
                  ))}

                  {!loading && filteredTransactions.length === 0 && (
                    <tr>
                      <td
                        colSpan={6}
                        className="px-5 py-12 text-center text-slate-500"
                      >
                        No matching transactions on this page. Try another
                        search or risk level.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>

            <div className="border-t border-white/[0.08] px-5 py-4 text-xs text-slate-500">
              Showing matching records among the 100 highest-risk API results.
            </div>
          </section>
        </div>
        )}
      </main>
    </div>
  );
}

export default App;