import sys, math, heapq, time
sys.path.insert(0, "/home/claude/fair_division")
from fair_division import generate_values, round_robin, nash_log_welfare, bundle_utilities

M, N, SEED = 10, 5, 42
values = generate_values(M, N, seed=SEED)
n, m = N, M
rr = round_robin(values, n, seed=SEED)
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


class Node:
    __slots__ = ("level", "utilities", "assignment", "cross", "maxb")
    def __init__(self, level, utilities, assignment, cross, maxb):
        self.level, self.utilities, self.assignment, self.cross, self.maxb = level, utilities, assignment, cross, maxb


root = Node(0, tuple([0]*n), tuple(), tuple([0]*(n*n)), tuple([0]*(n*n)))
heap = [(-f_score(root.utilities, 0), 0, 0, root)]
counter = 0
bound = rr_nash
best_score = rr_nash
print(f"{'#pop':>5} {'level':>5} {'f(n)':>10} {'is_leaf':>8}   assignment-so-far")

popcount = 0
while heap:
    neg_f, _, _, node = heapq.heappop(heap)
    cur_f = -neg_f
    popcount += 1
    is_leaf = node.level == m
    print(f"{popcount:>5} {node.level:>5} {cur_f:>10.4f} {str(is_leaf):>8}   {node.assignment}")
    if cur_f <= bound + 1e-12:
        print("  -> עצירה: שום דבר בתור הפתוח לא יכול לעבור את החסם הנוכחי")
        break
    if is_leaf:
        score = sum(math.log(u+1) for u in node.utilities)
        print(f"     [עלה! ניקוד אמיתי = {score:.4f}]")
        if score > best_score:
            best_score = score
            bound = score
        continue

    item = item_order[node.level]
    for p in range(n):
        nu = list(node.utilities); nu[p] += values[p][item]
        nc = list(node.cross); nb = list(node.maxb)
        for i in range(n):
            idx = i*n+p
            nc[idx] += values[i][item]
            if values[i][item] > nb[idx]:
                nb[idx] = values[i][item]
        child = Node(node.level+1, tuple(nu), node.assignment+(p,), tuple(nc), tuple(nb))
        cf = f_score(child.utilities, child.level)
        if cf <= bound + 1e-12:
            continue
        counter += 1
        heapq.heappush(heap, (-cf, -child.level, counter, child))

print(f"\nסה'כ הוצאו מהתור {popcount} צמתים.")
