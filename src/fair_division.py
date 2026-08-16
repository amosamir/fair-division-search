"""
Fair division of indivisible items - simulation engine.

Implements:
  - Random value-function generation (4 value tiers, as in the paper).
  - Round-Robin algorithm (fast EF1-guaranteeing baseline).
  - Exact EF1 verification for a complete allocation.
  - A* search over the allocation state-space tree, using:
        f(n) = g(n) + h(n),  g/h defined via sum of log(u_i + 1)
        (an admissible heuristic: each remaining item is optimistically
        given to whichever player values it the most)
      with two pruning rules:
        1) EF1-infeasibility pruning (irrecoverable envy detection)
        2) Dynamic lower-bound pruning, seeded from the Round-Robin score
           and tightened whenever a better EF1 solution is found.

Author: simulation code for the course paper on search methods in AI.
"""

from __future__ import annotations
import math
import heapq
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# --------------------------------------------------------------------------
# Value-function generation
# --------------------------------------------------------------------------

# (fraction_of_items, value_low, value_high) - matches the paper's 4 tiers
VALUE_TIERS = [
    (0.30, 10, 50),
    (0.30, 40, 120),
    (0.25, 100, 200),
    (0.15, 150, 400),
]


def item_tier_sizes(m: int) -> List[int]:
    """Split m items into 4 tiers using the paper's proportions (30/30/25/15)."""
    sizes = [round(frac * m) for frac, _, _ in VALUE_TIERS]
    # fix rounding drift so sizes sum exactly to m
    diff = m - sum(sizes)
    sizes[-1] += diff
    return sizes


def generate_values(m: int, n: int, seed: Optional[int] = None) -> List[List[int]]:
    """
    Returns values[i][j] = utility of player i for item j, for i in [0,n), j in [0,m).
    Each player independently draws a random integer value for every item within
    that item's tier range (so players broadly agree on "expensive vs cheap" items,
    while differing individually - matching the inheritance-division scenario).
    """
    rng = random.Random(seed)
    sizes = item_tier_sizes(m)
    tier_of_item = []
    for tier_idx, size in enumerate(sizes):
        tier_of_item.extend([tier_idx] * size)
    rng.shuffle(tier_of_item)  # so tiers aren't laid out in contiguous blocks

    values = [[0] * m for _ in range(n)]
    for j in range(m):
        _, lo, hi = VALUE_TIERS[tier_of_item[j]]
        for i in range(n):
            values[i][j] = rng.randint(lo, hi)
    return values


# --------------------------------------------------------------------------
# Utility / welfare helpers
# --------------------------------------------------------------------------

def bundle_utilities(allocation: List[int], values: List[List[int]], n: int) -> List[int]:
    """utilities[i] = sum of values[i][j] for every item j allocated to player i."""
    utilities = [0] * n
    for j, p in enumerate(allocation):
        utilities[p] += values[p][j]
    return utilities


def nash_log_welfare(utilities: List[int]) -> float:
    """sum(ln(u_i + 1)) - the objective A* maximizes."""
    return sum(math.log(u + 1) for u in utilities)


def utilitarian_welfare(utilities: List[int]) -> int:
    return sum(utilities)


def egalitarian_welfare(utilities: List[int]) -> int:
    return min(utilities)


def is_EF1(allocation: List[int], values: List[List[int]], n: int) -> bool:
    """
    Exact EF1 check on a COMPLETE allocation.
    For every ordered pair (i, j): i must not envy j once the single item
    in A_j most valuable to i (if any) is hypothetically removed.
    """
    m = len(allocation)
    bundles = [[] for _ in range(n)]
    for j, p in enumerate(allocation):
        bundles[p].append(j)

    for i in range(n):
        u_i_own = sum(values[i][j] for j in bundles[i])
        for j in range(n):
            if i == j:
                continue
            if not bundles[j]:
                continue  # empty bundle -> no envy possible
            u_i_of_j = sum(values[i][k] for k in bundles[j])
            best_item_value = max(values[i][k] for k in bundles[j])
            if u_i_own < u_i_of_j - best_item_value:
                return False
    return True


def is_EF(allocation: List[int], values: List[List[int]], n: int) -> bool:
    """
    Exact (full) Envy-Freeness check on a COMPLETE allocation - the strict
    version of EF1, with NO "remove one item" leniency: every player must
    value their own bundle at least as much as every other player's bundle.
    """
    m = len(allocation)
    bundles = [[] for _ in range(n)]
    for j, p in enumerate(allocation):
        bundles[p].append(j)

    for i in range(n):
        u_i_own = sum(values[i][j] for j in bundles[i])
        for j in range(n):
            if i == j:
                continue
            u_i_of_j = sum(values[i][k] for k in bundles[j])
            if u_i_own < u_i_of_j:
                return False
    return True


# --------------------------------------------------------------------------
# Round-Robin
# --------------------------------------------------------------------------

def round_robin(values: List[List[int]], n: int, seed: Optional[int] = None) -> List[int]:
    """
    Returns allocation[j] = player who receives item j.
    Player order is randomized once, then fixed for the whole run.
    """
    rng = random.Random(seed)
    m = len(values[0])
    order = list(range(n))
    rng.shuffle(order)

    remaining = set(range(m))
    allocation = [-1] * m
    turn = 0
    while remaining:
        player = order[turn % n]
        # pick the remaining item this player values most
        best_item = max(remaining, key=lambda j: values[player][j])
        allocation[best_item] = player
        remaining.remove(best_item)
        turn += 1
    return allocation


# --------------------------------------------------------------------------
# A* search over the allocation tree
# --------------------------------------------------------------------------

@dataclass
class AStarResult:
    allocation: Optional[List[int]]
    nash_log_score: float
    nodes_expanded: int
    leaves_reached: int
    final_solutions_reached: int
    runtime_seconds: float
    stopped_early: bool  # True if we hit a node/time cap before proving optimality
    improved_over_round_robin: bool


class _Node:
    __slots__ = ("level", "utilities", "assignment", "cross_util", "max_item_in_bundle")

    def __init__(self, level, utilities, assignment, cross_util, max_item_in_bundle):
        self.level = level
        self.utilities = utilities              # utilities[i]: player i's own utility so far
        self.assignment = assignment             # tuple of player-ids, length == level
        self.cross_util = cross_util             # cross_util[i][j]: i's valuation of j's bundle so far
        self.max_item_in_bundle = max_item_in_bundle  # [i][j]: max value_i(item) among items currently in A_j


def a_star_fair_division(
    values: List[List[int]],
    n: int,
    initial_bound_log_score: float,
    initial_allocation: Optional[List[int]] = None,
    node_limit: int = 2_000_000,
    time_limit_seconds: Optional[float] = None,
) -> AStarResult:
    m = len(values[0])
    start_time = time.time()

    # Item ordering: most valuable first (by mean value across players) - as in the paper.
    mean_value = [sum(values[i][j] for i in range(n)) / n for j in range(m)]
    item_order = sorted(range(m), key=lambda j: -mean_value[j])

    # Suffix sums (index k = "from level k to the end")
    # suffix_total[i][k]     = sum of value_i(item_order[t]) for t in [k, m)
    #                          -> the loosest admissible per-player cap
    #                             ("I personally receive every remaining item"),
    #                             also used for EF1-infeasibility pruning.
    # suffix_best_sum[k]     = sum over t in [k, m) of max_i value_i(item_order[t])
    #                          -> total pie cap: no matter how remaining items are
    #                             split, the SUM of utility gained across all
    #                             players can never exceed this (each item's best
    #                             case is going to whoever values it most).
    suffix_total = [[0] * (m + 1) for _ in range(n)]
    suffix_best_sum = [0] * (m + 1)
    for k in range(m - 1, -1, -1):
        j = item_order[k]
        best_here = max(values[i][j] for i in range(n))
        suffix_best_sum[k] = suffix_best_sum[k + 1] + best_here
        for i in range(n):
            suffix_total[i][k] = suffix_total[i][k + 1] + values[i][j]

    # Precompute, per level, the per-player caps (as a tuple) and their sum, so
    # f_score doesn't rebuild these lists on every single call (this is the
    # hottest inner loop in the whole search).
    caps_by_level = [tuple(suffix_total[i][k] for i in range(n)) for k in range(m + 1)]
    sum_caps_by_level = [sum(c) for c in caps_by_level]

    WATERFILL_ITERS = 30  # plenty of precision; fewer iterations = faster search

    def f_score(utilities, level) -> float:
        """
        Admissible upper bound on the best achievable sum(log(u_i+1)) from this
        node onward. We solve a concave relaxation:
            maximize  sum_i log(u_i + x_i + 1)
            subject to  0 <= x_i <= suffix_total[i][level]      (per-player cap:
                          i alone could not get more value than every remaining
                          item is worth to i)
                        sum_i x_i <= suffix_best_sum[level]      (pie cap: total
                          utility handed out cannot exceed every item going to
                          whoever values it most)
        Any real, integral completion of the search satisfies both constraints,
        so this relaxation's optimum is a valid upper bound (log is concave,
        so water-filling - give first to whoever is currently worst off - is
        exactly optimal for the relaxation).
        """
        caps = caps_by_level[level]
        total_cap = suffix_best_sum[level]
        if total_cap >= sum_caps_by_level[level]:
            # Budget doesn't bind - everyone can reach their individual cap.
            total = 0.0
            for i in range(n):
                total += math.log(utilities[i] + caps[i] + 1)
            return total

        lo = float(min(utilities))
        hi = float(max(utilities[i] + caps[i] for i in range(n)))
        for _ in range(WATERFILL_ITERS):
            w = (lo + hi) / 2.0
            spent = 0.0
            for i in range(n):
                d = w - utilities[i]
                if d > 0:
                    spent += d if d < caps[i] else caps[i]
            if spent < total_cap:
                lo = w
            else:
                hi = w
        w = hi  # slightly over-spends the budget if anything -> stays a valid upper bound
        total = 0.0
        for i in range(n):
            d = w - utilities[i]
            x = 0.0 if d <= 0 else (d if d < caps[i] else caps[i])
            total += math.log(utilities[i] + x + 1)
        return total

    root = _Node(
        level=0,
        utilities=tuple([0] * n),
        assignment=tuple(),
        cross_util=tuple([0] * (n * n)),          # flat: index i*n+j
        max_item_in_bundle=tuple([0] * (n * n)),  # flat: index i*n+j
    )

    # Priority key: (-f, -level, counter).
    # Primary order is still strictly by f(n) descending (this is what makes it A*
    # and guarantees the stopping condition below is sound). Early in the search,
    # the admissible-but-loose heuristic ("best case: I get every remaining item")
    # ties or near-ties across huge numbers of nodes, since it barely depends on
    # the specific items already assigned. Breaking those ties in favor of DEEPER
    # nodes (closer to a complete, checkable allocation) turns what would otherwise
    # be a near-breadth-first crawl into a search that reaches real leaves quickly -
    # giving useful anytime improvements over Round-Robin well before (if ever) the
    # full tree is exhausted. This tie-break choice cannot affect correctness: A*'s
    # optimality guarantee only relies on expanding nodes in non-increasing f order.
    heap: List[Tuple[float, int, int, _Node]] = []
    counter = 0  # final tie-breaker so heap never compares _Node objects
    heapq.heappush(heap, (-f_score(root.utilities, root.level), -root.level, counter, root))

    bound = initial_bound_log_score
    # Round-Robin's allocation is a *guaranteed valid EF1 solution*, so its score
    # is usable as a real pruning threshold from node zero - not just once A*
    # happens to stumble on its own first leaf. (Previously the code withheld
    # dynamic-bound pruning until `best_allocation is not None`, which meant no
    # pruning at all took place until a leaf was reached - on anything beyond
    # trivially small instances that made the search effectively unpruned.)
    best_allocation: Optional[List[int]] = initial_allocation
    best_score = initial_bound_log_score
    have_baseline = True  # Round-Robin's score is always a legitimate bound

    nodes_expanded = 0
    leaves_reached = 0
    final_solutions_reached = 0
    stopped_early = False

    while heap:
        neg_f, _, _, node = heapq.heappop(heap)
        cur_f = -neg_f

        if cur_f <= bound + 1e-12 and have_baseline:
            # Best-first search: nothing left in the heap can beat the current bound.
            break

        if node_limit is not None and nodes_expanded >= node_limit:
            stopped_early = True
            break
        if time_limit_seconds is not None and (time.time() - start_time) > time_limit_seconds:
            stopped_early = True
            break

        if node.level == m:
            leaves_reached += 1
            allocation = [0] * m
            for k, p in enumerate(node.assignment):
                allocation[item_order[k]] = p
            if is_EF1(allocation, values, n):
                final_solutions_reached += 1
                score = nash_log_welfare(node.utilities)
                if score > best_score:
                    best_score = score
                    best_allocation = allocation
                    bound = score
            continue

        nodes_expanded += 1
        item = item_order[node.level]
        item_values = [values[i][item] for i in range(n)]  # precompute once per node, not per child

        for p in range(n):
            new_utilities = list(node.utilities)
            new_utilities[p] += item_values[p]

            # Flat (i*n+j) copies - one list allocation each instead of n nested ones.
            new_cross = list(node.cross_util)
            new_max = list(node.max_item_in_bundle)
            infeasible = False
            for i in range(n):
                idx = i * n + p
                new_cross[idx] += item_values[i]
                if item_values[i] > new_max[idx]:
                    new_max[idx] = item_values[i]

            for i in range(n):
                u_i_max_possible = new_utilities[i] + suffix_total[i][node.level + 1]
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
            if infeasible:
                continue

            child = _Node(
                level=node.level + 1,
                utilities=tuple(new_utilities),
                assignment=node.assignment + (p,),
                cross_util=tuple(new_cross),
                max_item_in_bundle=tuple(new_max),
            )
            child_f = f_score(child.utilities, child.level)
            if child_f <= bound + 1e-12 and have_baseline:
                continue  # dynamic-bound pruning
            counter += 1
            heapq.heappush(heap, (-child_f, -child.level, counter, child))

    runtime = time.time() - start_time
    return AStarResult(
        allocation=best_allocation,
        nash_log_score=best_score,
        nodes_expanded=nodes_expanded,
        leaves_reached=leaves_reached,
        final_solutions_reached=final_solutions_reached,
        runtime_seconds=runtime,
        stopped_early=stopped_early,
        improved_over_round_robin=(best_score > initial_bound_log_score + 1e-9),
    )
