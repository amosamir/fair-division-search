import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("/home/claude/fair_division/results.csv")
df["astar_converged"] = df["astar_converged"].astype(str) == "True"
df["astar_improved"] = df["astar_improved"].astype(str) == "True"
df["astar_is_ef1"] = df["astar_is_ef1"].astype(str) == "True"
df["nash_gain"] = df["astar_nash_log"] - df["rr_nash_log"]
df["utilitarian_gain_pct"] = 100 * (df["astar_utilitarian"] - df["rr_utilitarian"]) / df["rr_utilitarian"]
df["egalitarian_gain_pct"] = 100 * (df["astar_egalitarian"] - df["rr_egalitarian"]) / df["rr_egalitarian"].replace(0, 1)

summary = df.groupby("m").agg(
    n_runs=("seed", "count"),
    rr_ef1_rate=("rr_nash_log", lambda s: 1.0),  # RR is always EF1 by construction
    astar_ef1_rate=("astar_is_ef1", "mean"),
    astar_converged_rate=("astar_converged", "mean"),
    astar_improved_rate=("astar_improved", "mean"),
    rr_nash_mean=("rr_nash_log", "mean"),
    astar_nash_mean=("astar_nash_log", "mean"),
    nash_gain_mean=("nash_gain", "mean"),
    utilitarian_gain_pct_mean=("utilitarian_gain_pct", "mean"),
    egalitarian_gain_pct_mean=("egalitarian_gain_pct", "mean"),
    rr_time_mean=("rr_time", "mean"),
    astar_time_mean=("astar_time", "mean"),
    astar_time_median=("astar_time", "median"),
    astar_nodes_mean=("astar_nodes", "mean"),
    astar_nodes_median=("astar_nodes", "median"),
).round(4)

print(summary.to_string())
summary.to_csv("/home/claude/fair_division/summary.csv")

# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

ms = summary.index.values

ax = axes[0, 0]
ax.plot(ms, summary["astar_converged_rate"] * 100, marker="o", label="A* converged (proved optimal)")
ax.plot(ms, summary["astar_improved_rate"] * 100, marker="s", label="A* improved over Round-Robin")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("% of the 50 runs")
ax.set_title("Convergence & improvement rate vs. M")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(ms, summary["rr_nash_mean"], marker="o", label="Round-Robin")
ax.plot(ms, summary["astar_nash_mean"], marker="s", label="A*")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("Mean Nash log-welfare")
ax.set_title("Nash welfare: Round-Robin vs A*")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(ms, summary["astar_time_mean"], marker="o", label="mean")
ax.plot(ms, summary["astar_time_median"], marker="s", label="median")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("A* runtime (seconds, capped)")
ax.set_title("A* runtime vs. M (time-capped runs included)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(ms, summary["astar_nodes_mean"], marker="o", label="mean")
ax.plot(ms, summary["astar_nodes_median"], marker="s", label="median")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("Nodes expanded")
ax.set_title("A* search-tree size vs. M")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/fair_division/summary_plots.png", dpi=150)
print("\nSaved plots to summary_plots.png")
