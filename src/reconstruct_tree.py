"""
Reconstructs the A* search tree for one specific run, exactly as it happened
inside fair_division.a_star_fair_division (same item ordering, same water-
filling heuristic, same EF1 / dynamic-bound pruning, same tie-breaking) -
but this version records EVERY node instead of just returning the answer.

Usage:
    python3 reconstruct_tree.py --m 10 --seed 42
    python3 reconstruct_tree.py --m 20 --run 5      # uses the experiment's
                                                      # seed formula 1000*m+run

Produces, under --outdir (default: ./tree_out):
    nodes.json          - full structured data for every node
    tree_outline.txt     - TAB-indented outline, paste-ready into a Word
                            SmartArt "Hierarchy" / Organization-Chart text pane
    tree_table.csv        - one row per node, matching the outline
"""
import sys, os, json, math, heapq, argparse, csv
sys.path.insert(0, "/home/claude/fair_division")
from fair_division import generate_values, round_robin, nash_log_welfare, bundle_utilities


def reconstruct(m, n=5, seed=None, run=None, node_limit=2_000_000, time_limit_seconds=None):
    if seed is None:
        assert run is not None, "צריך לתת seed מפורש או m+run מהניסוי"
        seed = 1000 * m + run

    values = generate_values(m, n, seed=seed)
    rr = round_robin(values, n, seed=seed)
    rr_utils = bundle_utilities(rr, values, n)
    rr_nash = nash_log_welfare(rr_utils)

    mean_value = [sum(values[i][j] for i in range(n)) / n for j in range(m)]
    item_order = sorted(range(m), key=lambda j: -mean_value[j])

    suffix_total = [[0] * (m + 1) for _ in range(n)]
    suffix_best_sum = [0] * (m + 1)
    for k in range(m - 1, -1, -1):
        j = item_order[k]
        best_here = max(values[i][j] for i in range(n))
        suffix_best_sum[k] = suffix_best_sum[k + 1] + best_here
        for i in range(n):
            suffix_total[i][k] = suffix_total[i][k + 1] + values[i][j]

    caps_by_level = [tuple(suffix_total[i][k] for i in range(n)) for k in range(m + 1)]
    sum_caps_by_level = [sum(c) for c in caps_by_level]
    WATERFILL_ITERS = 30

    def f_score(utilities, level):
        caps = caps_by_level[level]
        total_cap = suffix_best_sum[level]
        if total_cap >= sum_caps_by_level[level]:
            return sum(math.log(utilities[i] + caps[i] + 1) for i in range(n))
        lo, hi = float(min(utilities)), float(max(utilities[i] + caps[i] for i in range(n)))
        for _ in range(WATERFILL_ITERS):
            w = (lo + hi) / 2.0
            spent = sum(min(max(w - utilities[i], 0.0), caps[i]) for i in range(n))
            if spent < total_cap:
                lo = w
            else:
                hi = w
        w = hi
        return sum(math.log(utilities[i] + min(max(w - utilities[i], 0.0), caps[i]) + 1) for i in range(n))

    def g_score(utilities):
        return sum(math.log(u + 1) for u in utilities)

    # ---- records, keyed by assignment tuple ----
    records = {}

    def make_record(assignment, level, utilities, cross, maxb, parent):
        g = g_score(utilities)
        f = f_score(utilities, level)
        h = f - g
        # allocation_vector is indexed by REAL item index (0..m-1), not by search
        # position - item_order[k] tells us which real item was decided at search
        # step k, so we scatter assignment[k] into that real position.
        alloc_full = [None] * m
        for k in range(level):
            alloc_full[item_order[k]] = assignment[k]
        log_utils = [math.log(u + 1) for u in utilities]
        rec = {
            "assignment": assignment,
            "parent": parent,
            "level": level,
            "g": g, "h": h, "f": f,
            "utilities": list(utilities),
            "log_utilities": log_utils,
            "allocation_vector": alloc_full,   # item -> player, in ORIGINAL item index order
            "cross": cross, "maxb": maxb,
            "label": None,        # filled in later: "N3" / "L2" / "S7" / "P2"
            "pop_order": None,    # set when popped
            "is_terminal_pop": None,  # True = popped but generated no children (leaf or stop-node)
            "pruned": False,
            "prune_reason": None,
            "children": [],
        }
        records[assignment] = rec
        return rec

    root_utils = tuple([0] * n)
    root = make_record((), 0, root_utils, tuple([0] * (n * n)), tuple([0] * (n * n)), None)

    heap = [(-root["f"], 0, 0, ())]
    counter = 0
    bound = rr_nash
    best_score = rr_nash
    best_allocation = rr
    have_baseline = True

    pop_order = 0
    nodes_expanded = 0
    stopped_early = False
    import time as _time
    start_time = _time.time()

    while heap:
        neg_f, _, _, assignment = heapq.heappop(heap)
        cur_f = -neg_f
        rec = records[assignment]

        pop_order += 1
        rec["pop_order"] = pop_order

        if cur_f <= bound + 1e-12 and have_baseline:
            rec["stop_here"] = True
            rec["is_terminal_pop"] = True
            break

        if node_limit is not None and nodes_expanded >= node_limit:
            stopped_early = True
            break
        if time_limit_seconds is not None and (_time.time() - start_time) > time_limit_seconds:
            stopped_early = True
            break

        level = rec["level"]
        if level == m:
            score = g_score(rec["utilities"])
            rec["is_leaf"] = True
            rec["is_terminal_pop"] = True
            rec["leaf_score"] = score
            if score > best_score:
                best_score = score
                bound = score
                real_alloc = [None] * m
                for k, p in enumerate(rec["assignment"]):
                    real_alloc[item_order[k]] = p
                best_allocation = real_alloc
            continue

        nodes_expanded += 1
        rec["is_terminal_pop"] = False
        item = item_order[level]
        item_values = [values[i][item] for i in range(n)]

        for p in range(n):
            new_utilities = list(rec["utilities"])
            new_utilities[p] += item_values[p]
            new_cross = list(rec["cross"])
            new_max = list(rec["maxb"])
            for i in range(n):
                idx = i * n + p
                new_cross[idx] += item_values[i]
                if item_values[i] > new_max[idx]:
                    new_max[idx] = item_values[i]

            infeasible = False
            for i in range(n):
                u_i_max_possible = new_utilities[i] + suffix_total[i][level + 1]
                base = i * n
                for j in range(n):
                    if i == j:
                        continue
                    cij = new_cross[base + j]
                    if cij == 0:
                        continue
                    if u_i_max_possible < cij - new_max[base + j]:
                        infeasible = True
                        break
                if infeasible:
                    break

            child_assignment = assignment + (p,)
            child_rec = make_record(child_assignment, level + 1, tuple(new_utilities),
                                     tuple(new_cross), tuple(new_max), assignment)
            rec["children"].append(child_assignment)

            if infeasible:
                child_rec["pruned"] = True
                child_rec["prune_reason"] = "אי-היתכנות EF1"
                continue

            child_f = child_rec["f"]
            if child_f <= bound + 1e-12 and have_baseline:
                child_rec["pruned"] = True
                child_rec["prune_reason"] = "חסם דינמי (f \u2264 bound)"
                continue

            counter += 1
            heapq.heappush(heap, (-child_f, -(level + 1), counter, child_assignment))

    # ---- label popped nodes: N = true expansion (generated children, matches
    # the experiment's `astar_nodes` / nodes_expanded counter exactly), L =
    # terminal pop that generated no children (a leaf, or the final node that
    # triggered the stop-on-bound condition) ----
    n_counter = 0
    l_counter = 0
    popped = sorted((r for r in records.values() if r["pop_order"] is not None),
                     key=lambda r: r["pop_order"])
    for rec in popped:
        if rec["is_terminal_pop"]:
            l_counter += 1
            rec["label"] = f"L{l_counter}"
        else:
            n_counter += 1
            rec["label"] = f"N{n_counter}"

    # ---- label remaining never-popped, non-pruned children as S ----
    s_counter = 0
    for assignment, rec in records.items():
        if rec["pop_order"] is None and not rec["pruned"] and assignment != ():
            s_counter += 1
            rec["label"] = f"S{s_counter}"
    # ---- label pruned children as P ----
    p_counter = 0
    for assignment, rec in records.items():
        if rec["pruned"]:
            p_counter += 1
            rec["label"] = f"P{p_counter}"

    return {
        "m": m, "n": n, "seed": seed, "run": run,
        "values": values,
        "item_order": item_order,
        "rr_allocation": rr, "rr_nash": rr_nash,
        "best_allocation": best_allocation, "best_score": best_score,
        "records": records,
        "root_assignment": (),
        "stopped_early": stopped_early,
    }


def build_outline(result):
    """TAB-indented outline text, paste-ready into a Word SmartArt hierarchy text pane."""
    records = result["records"]
    lines = []

    def visit(assignment, depth):
        rec = records[assignment]
        lines.append("\t" * depth + rec["label"])
        for child in rec["children"]:
            visit(child, depth + 1)

    visit((), 0)
    return "\n".join(lines)


def build_table_rows(result):
    records = result["records"]
    m = result["m"]
    rows = []

    def visit(assignment):
        rec = records[assignment]
        rows.append({
            "label": rec["label"],
            "level": rec["level"],
            "F": round(rec["f"], 4),
            "G": round(rec["g"], 4),
            "H": round(rec["h"], 4),
            "allocation_vector": rec["allocation_vector"],
            "utilities_vector": rec["utilities"],
            "log_utilities_vector": [round(x, 4) for x in rec["log_utilities"]],
            "pruned": rec["pruned"],
            "prune_reason": rec["prune_reason"],
        })
        for child in rec["children"]:
            visit(child)

    visit(())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--run", type=int, default=None, help="run index within the experiment (uses seed=1000*m+run)")
    ap.add_argument("--seed", type=int, default=None, help="explicit seed override")
    ap.add_argument("--outdir", type=str, default="/home/claude/fair_division/tree_out")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    result = reconstruct(args.m, seed=args.seed, run=args.run)

    outline = build_outline(result)
    with open(os.path.join(args.outdir, "tree_outline.txt"), "w", encoding="utf-8") as f:
        f.write(outline)

    rows = build_table_rows(result)
    with open(os.path.join(args.outdir, "tree_table.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # dump full structured data (assignment tuples -> strings for JSON)
    json_records = {}
    for assignment, rec in result["records"].items():
        key = ",".join(map(str, assignment)) if assignment else "ROOT"
        rec2 = dict(rec)
        rec2["assignment"] = list(assignment)
        rec2["parent"] = list(rec["parent"]) if rec["parent"] is not None else None
        rec2["children"] = [list(c) for c in rec["children"]]
        json_records[key] = rec2

    with open(os.path.join(args.outdir, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump({
            "m": result["m"], "n": result["n"], "seed": result["seed"],
            "values": result["values"], "item_order": result["item_order"],
            "rr_allocation": result["rr_allocation"], "rr_nash": result["rr_nash"],
            "best_allocation": result["best_allocation"], "best_score": result["best_score"],
            "records": json_records,
        }, f, ensure_ascii=False, indent=1)

    n_total = len(result["records"])
    n_true_expansions = sum(1 for r in result["records"].values()
                             if r["label"] is not None and r["label"].startswith("N"))
    n_terminal = sum(1 for r in result["records"].values()
                      if r["label"] is not None and r["label"].startswith("L"))
    n_pruned = sum(1 for r in result["records"].values() if r["pruned"])
    n_frontier = n_total - n_true_expansions - n_terminal - n_pruned
    print(f"m={result['m']} seed={result['seed']}: סה'כ צמתים בעץ = {n_total}")
    print(f"  N (הרחבות אמיתיות, = astar_nodes בקובץ הניסוי): {n_true_expansions}")
    print(f"  L (עלים/עצירה - נשלפו אך לא הורחבו): {n_terminal}")
    print(f"  S (בתור, מעולם לא נשלפו): {n_frontier}")
    print(f"  P (נגזמו): {n_pruned}")
    print(f"נכתב אל: {args.outdir}/tree_outline.txt , tree_table.csv , nodes.json")


if __name__ == "__main__":
    main()
