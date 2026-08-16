import sys, csv, time, os
sys.path.insert(0, "/home/claude/fair_division")
from fair_division import (
    generate_values, round_robin, nash_log_welfare, utilitarian_welfare,
    egalitarian_welfare, bundle_utilities, a_star_fair_division, is_EF1,
)

N_PLAYERS = 5
N_RUNS = 50
WALL_BUDGET_SECONDS = 85  # stay safely under this tool's per-call ceiling

# (M, time_limit_seconds per trial, node_limit per trial)
CONFIGS = [
    (10, 10, 2_000_000),
    (20, 10, 2_000_000),
    (30, 15, 2_000_000),
    (40, 15, 2_000_000),
    (50, 20, 2_000_000),
]

OUT_PATH = "/home/claude/fair_division/results.csv"

FIELDS = [
    "m", "seed",
    "rr_nash_log", "rr_utilitarian", "rr_egalitarian", "rr_time",
    "astar_nash_log", "astar_utilitarian", "astar_egalitarian", "astar_time",
    "astar_nodes", "astar_leaves", "astar_solutions",
    "astar_converged", "astar_improved", "astar_is_ef1",
]


def already_done():
    done = set()
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["m"]), int(row["seed"])))
    return done


def main():
    start = time.time()
    done = already_done()
    file_exists = os.path.exists(OUT_PATH)

    with open(OUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()

        for m, tlimit, nlimit in CONFIGS:
            for seed in range(N_RUNS):
                if (m, seed) in done:
                    continue
                if time.time() - start > WALL_BUDGET_SECONDS:
                    print(f"PAUSING for wall-clock budget (next up: m={m} seed={seed})", flush=True)
                    return

                values = generate_values(m, N_PLAYERS, seed=1000 * m + seed)

                t0 = time.time()
                rr = round_robin(values, N_PLAYERS, seed=1000 * m + seed)
                rr_time = time.time() - t0
                rr_utils = bundle_utilities(rr, values, N_PLAYERS)
                rr_nash = nash_log_welfare(rr_utils)

                res = a_star_fair_division(
                    values, N_PLAYERS,
                    initial_bound_log_score=rr_nash,
                    initial_allocation=rr,
                    node_limit=nlimit,
                    time_limit_seconds=tlimit,
                )

                if res.allocation is not None:
                    a_utils = bundle_utilities(res.allocation, values, N_PLAYERS)
                else:
                    a_utils = rr_utils  # A* found nothing better than the RR seed

                row = {
                    "m": m, "seed": seed,
                    "rr_nash_log": rr_nash,
                    "rr_utilitarian": utilitarian_welfare(rr_utils),
                    "rr_egalitarian": egalitarian_welfare(rr_utils),
                    "rr_time": rr_time,
                    "astar_nash_log": res.nash_log_score,
                    "astar_utilitarian": utilitarian_welfare(a_utils),
                    "astar_egalitarian": egalitarian_welfare(a_utils),
                    "astar_time": res.runtime_seconds,
                    "astar_nodes": res.nodes_expanded,
                    "astar_leaves": res.leaves_reached,
                    "astar_solutions": res.final_solutions_reached,
                    "astar_converged": not res.stopped_early,
                    "astar_improved": res.improved_over_round_robin,
                    "astar_is_ef1": is_EF1(res.allocation, values, N_PLAYERS) if res.allocation else True,
                }
                writer.writerow(row)
                f.flush()
                print(f"m={m:3d} seed={seed:2d}  RR={rr_nash:8.4f}  A*={res.nash_log_score:8.4f}  "
                      f"converged={not res.stopped_early}  improved={res.improved_over_round_robin}  "
                      f"time={res.runtime_seconds:6.2f}s", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()

