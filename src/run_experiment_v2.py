import sys, csv, time, os
sys.path.insert(0, "/home/claude/fair_division")
from fair_division import (
    generate_values, round_robin, nash_log_welfare, utilitarian_welfare,
    egalitarian_welfare, bundle_utilities, a_star_fair_division, is_EF1, is_EF,
)

N_PLAYERS = 5
N_RUNS = 50
WALL_BUDGET_SECONDS = 85

CONFIGS = [
    (10, 10, 2_000_000),
    (20, 10, 2_000_000),
    (30, 15, 2_000_000),
    (40, 15, 2_000_000),
    (50, 20, 2_000_000),
]

SUMMARY_PATH = "/home/claude/fair_division/results_v2.csv"
ITEM_DETAILS_PATH = "/home/claude/fair_division/item_details.csv"

SUMMARY_FIELDS = [
    "m", "seed",
    "rr_nash_log", "rr_utilitarian", "rr_egalitarian", "rr_time",
    "rr_is_ef1", "rr_is_ef",
    "astar_nash_log", "astar_utilitarian", "astar_egalitarian", "astar_time",
    "astar_nodes", "astar_leaves", "astar_solutions",
    "astar_converged", "astar_improved", "astar_is_ef1", "astar_is_ef",
]

ITEM_FIELDS = [
    "m", "run", "item",
    "value_player0", "value_player1", "value_player2", "value_player3", "value_player4",
    "rr_allocation", "astar_allocation",
]


def already_done():
    done = set()
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["m"]), int(row["seed"])))
    return done


def main():
    start = time.time()
    done = already_done()
    summary_exists = os.path.exists(SUMMARY_PATH)
    items_exists = os.path.exists(ITEM_DETAILS_PATH)

    with open(SUMMARY_PATH, "a", newline="") as sf, open(ITEM_DETAILS_PATH, "a", newline="") as itf:
        swriter = csv.DictWriter(sf, fieldnames=SUMMARY_FIELDS)
        iwriter = csv.DictWriter(itf, fieldnames=ITEM_FIELDS)
        if not summary_exists:
            swriter.writeheader(); sf.flush()
        if not items_exists:
            iwriter.writeheader(); itf.flush()

        for m, tlimit, nlimit in CONFIGS:
            for seed in range(N_RUNS):
                if (m, seed) in done:
                    continue
                if time.time() - start > WALL_BUDGET_SECONDS:
                    print(f"PAUSING (next up: m={m} seed={seed})", flush=True)
                    return

                values = generate_values(m, N_PLAYERS, seed=1000 * m + seed)

                t0 = time.time()
                rr = round_robin(values, N_PLAYERS, seed=1000 * m + seed)
                rr_time = time.time() - t0
                rr_utils = bundle_utilities(rr, values, N_PLAYERS)
                rr_nash = nash_log_welfare(rr_utils)
                rr_ef1 = is_EF1(rr, values, N_PLAYERS)
                rr_ef = is_EF(rr, values, N_PLAYERS)

                res = a_star_fair_division(
                    values, N_PLAYERS,
                    initial_bound_log_score=rr_nash,
                    initial_allocation=rr,
                    node_limit=nlimit,
                    time_limit_seconds=tlimit,
                )
                astar_alloc = res.allocation if res.allocation is not None else rr
                a_utils = bundle_utilities(astar_alloc, values, N_PLAYERS)
                astar_ef1 = is_EF1(astar_alloc, values, N_PLAYERS)
                astar_ef = is_EF(astar_alloc, values, N_PLAYERS)

                swriter.writerow({
                    "m": m, "seed": seed,
                    "rr_nash_log": rr_nash,
                    "rr_utilitarian": utilitarian_welfare(rr_utils),
                    "rr_egalitarian": egalitarian_welfare(rr_utils),
                    "rr_time": rr_time,
                    "rr_is_ef1": rr_ef1,
                    "rr_is_ef": rr_ef,
                    "astar_nash_log": res.nash_log_score,
                    "astar_utilitarian": utilitarian_welfare(a_utils),
                    "astar_egalitarian": egalitarian_welfare(a_utils),
                    "astar_time": res.runtime_seconds,
                    "astar_nodes": res.nodes_expanded,
                    "astar_leaves": res.leaves_reached,
                    "astar_solutions": res.final_solutions_reached,
                    "astar_converged": not res.stopped_early,
                    "astar_improved": res.improved_over_round_robin,
                    "astar_is_ef1": astar_ef1,
                    "astar_is_ef": astar_ef,
                })
                sf.flush()

                for j in range(m):
                    iwriter.writerow({
                        "m": m, "run": seed, "item": j,
                        "value_player0": values[0][j],
                        "value_player1": values[1][j],
                        "value_player2": values[2][j],
                        "value_player3": values[3][j],
                        "value_player4": values[4][j],
                        "rr_allocation": rr[j],
                        "astar_allocation": astar_alloc[j],
                    })
                itf.flush()

                print(f"m={m:3d} seed={seed:2d}  RR_EF={rr_ef} RR_EF1={rr_ef1}  "
                      f"A*_EF={astar_ef} A*_EF1={astar_ef1}  converged={not res.stopped_early}  "
                      f"time={res.runtime_seconds:6.2f}s", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
