import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/claude/fair_division/results_v2.csv")
for c in ["rr_is_ef1", "rr_is_ef", "astar_is_ef1", "astar_is_ef", "astar_converged", "astar_improved"]:
    df[c] = df[c].astype(str) == "True"

df["nash_gain"] = df["astar_nash_log"] - df["rr_nash_log"]
df["utilitarian_gain_pct"] = 100 * (df["astar_utilitarian"] - df["rr_utilitarian"]) / df["rr_utilitarian"]
df["egalitarian_gain_pct"] = 100 * (df["astar_egalitarian"] - df["rr_egalitarian"]) / df["rr_egalitarian"].replace(0, 1)
df["astar_ef_but_not_rr"] = df["astar_is_ef"] & ~df["rr_is_ef"]
df["rr_ef_but_not_astar"] = df["rr_is_ef"] & ~df["astar_is_ef"]

summary = df.groupby("m").agg(
    n_runs=("seed", "count"),
    rr_ef1_rate=("rr_is_ef1", "mean"),
    rr_ef_rate=("rr_is_ef", "mean"),
    astar_ef1_rate=("astar_is_ef1", "mean"),
    astar_ef_rate=("astar_is_ef", "mean"),
    astar_ef_but_not_rr=("astar_ef_but_not_rr", "sum"),
    astar_converged_rate=("astar_converged", "mean"),
    astar_improved_rate=("astar_improved", "mean"),
    rr_nash_mean=("rr_nash_log", "mean"),
    astar_nash_mean=("astar_nash_log", "mean"),
    nash_gain_mean=("nash_gain", "mean"),
    utilitarian_gain_pct_mean=("utilitarian_gain_pct", "mean"),
    egalitarian_gain_pct_mean=("egalitarian_gain_pct", "mean"),
    rr_time_mean=("rr_time", "mean"),
    astar_time_median=("astar_time", "median"),
    astar_nodes_median=("astar_nodes", "median"),
).round(4)

print(summary.to_string())
summary.to_csv("/home/claude/fair_division/summary_v2.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
ms = summary.index.values

ax = axes[0, 0]
ax.plot(ms, summary["rr_ef_rate"] * 100, marker="o", label="Round-Robin is EF")
ax.plot(ms, summary["astar_ef_rate"] * 100, marker="s", label="A* is EF")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("% of the 50 runs")
ax.set_title("Full envy-freeness (EF) rate: RR vs A*\n(both are EF1 in ~100% of runs)")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.bar(ms, summary["astar_ef_but_not_rr"], width=3, color="tab:green")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("# of 50 runs")
ax.set_title("Runs where A* found an EF allocation\nbut Round-Robin's was not EF")
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(ms, summary["astar_converged_rate"] * 100, marker="o", label="A* converged (proved optimal)")
ax.plot(ms, summary["astar_improved_rate"] * 100, marker="s", label="A* improved Nash welfare")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("% of the 50 runs")
ax.set_title("Convergence & improvement rate vs. M")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(ms, summary["rr_nash_mean"], marker="o", label="Round-Robin")
ax.plot(ms, summary["astar_nash_mean"], marker="s", label="A*")
ax.set_xlabel("Number of items (M)")
ax.set_ylabel("Mean Nash log-welfare")
ax.set_title("Nash welfare: Round-Robin vs A*")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("/home/claude/fair_division/summary_plots_v2.png", dpi=150)
print("\nSaved plots to summary_plots_v2.png")
