import sys, random
sys.path.insert(0, "/home/claude/fair_division")
from fair_division import (
    generate_values, round_robin, is_EF1, nash_log_welfare,
    bundle_utilities, utilitarian_welfare, egalitarian_welfare,
    a_star_fair_division,
)

M, N, SEED = 10, 5, 42
values = generate_values(M, N, seed=SEED)

print("=" * 90)
print(f"טבלת ערכים: values[i][j] = כמה שחקן i מעריך פריט j   (M={M} פריטים, N={N} שחקנים)")
print("=" * 90)
header = "שחקן\\פריט".rjust(10) + "".join(f"{j:>7d}" for j in range(M))
print(header)
for i in range(N):
    row = f"שחקן {i}".rjust(10) + "".join(f"{values[i][j]:>7d}" for j in range(M))
    print(row)

# ---------------------------------------------------------------------
# Round-Robin, step by step
# ---------------------------------------------------------------------
print()
print("=" * 90)
print("סבב (Round-Robin) - שלב אחר שלב")
print("=" * 90)
rng = random.Random(SEED)
order = list(range(N))
rng.shuffle(order)
print(f"סדר השחקנים שהוגרל: {order}")

remaining = set(range(M))
allocation_rr = [-1] * M
turn = 0
while remaining:
    player = order[turn % N]
    best_item = max(remaining, key=lambda j: values[player][j])
    allocation_rr[best_item] = player
    remaining.remove(best_item)
    print(f"  תור {turn+1}: שחקן {player} בוחר פריט {best_item} "
          f"(ששווה לו {values[player][best_item]}) מתוך הנותרים {sorted(remaining | {best_item})}")
    turn += 1

print()
print("חלוקת הסבב הסופית (פריט -> שחקן):")
for j in range(M):
    print(f"  פריט {j}: שחקן {allocation_rr[j]}")

rr_utils = bundle_utilities(allocation_rr, values, N)
print()
print("תועלות (הסבב):")
for i in range(N):
    items_i = [j for j in range(M) if allocation_rr[j] == i]
    print(f"  שחקן {i}: פריטים {items_i}  ->  תועלת = {rr_utils[i]}")
print(f"  EF1? {is_EF1(allocation_rr, values, N)}")
print(f"  רווחת-נאש (log-sum) = {nash_log_welfare(rr_utils):.4f}")
print(f"  רווחה אוטיליטרית (סכום) = {utilitarian_welfare(rr_utils)}")
print(f"  רווחה שוויונית (מינימום) = {egalitarian_welfare(rr_utils)}")

# ---------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------
print()
print("=" * 90)
print("חיפוש A*")
print("=" * 90)
res = a_star_fair_division(values, N, initial_bound_log_score=nash_log_welfare(rr_utils),
                            initial_allocation=allocation_rr, node_limit=2_000_000, time_limit_seconds=30)

print(f"נצפו {res.nodes_expanded} צמתים, הגיעו ל-{res.leaves_reached} עלים, "
      f"מתוכם {res.final_solutions_reached} חוקיים (EF1). זמן ריצה: {res.runtime_seconds:.4f} שניות. "
      f"הוכחה שהפתרון אופטימלי הושלמה: {not res.stopped_early}")

allocation_astar = res.allocation
print()
print("חלוקת A* הסופית (פריט -> שחקן):")
for j in range(M):
    print(f"  פריט {j}: שחקן {allocation_astar[j]}")

a_utils = bundle_utilities(allocation_astar, values, N)
print()
print("תועלות (A*):")
for i in range(N):
    items_i = [j for j in range(M) if allocation_astar[j] == i]
    print(f"  שחקן {i}: פריטים {items_i}  ->  תועלת = {a_utils[i]}")
print(f"  EF1? {is_EF1(allocation_astar, values, N)}")
print(f"  רווחת-נאש (log-sum) = {nash_log_welfare(a_utils):.4f}")
print(f"  רווחה אוטיליטרית (סכום) = {utilitarian_welfare(a_utils)}")
print(f"  רווחה שוויונית (מינימום) = {egalitarian_welfare(a_utils)}")

print()
print("=" * 90)
print("סיכום השוואה")
print("=" * 90)
print(f"  רווחת-נאש:   סבב = {nash_log_welfare(rr_utils):.4f}   A* = {nash_log_welfare(a_utils):.4f}   "
      f"שיפור = {nash_log_welfare(a_utils) - nash_log_welfare(rr_utils):.4f}")
print(f"  אוטיליטרי:   סבב = {utilitarian_welfare(rr_utils)}   A* = {utilitarian_welfare(a_utils)}")
print(f"  שוויוני (מינימום): סבב = {egalitarian_welfare(rr_utils)}   A* = {egalitarian_welfare(a_utils)}")

# ---------------------------------------------------------------------
# Manual EF1 audit trail for the A* allocation - print the actual envy check
# for every ordered pair so it can be verified by hand.
# ---------------------------------------------------------------------
print()
print("=" * 90)
print("בדיקת EF1 ידנית עבור חלוקת A* (לכל זוג שחקנים i,j)")
print("=" * 90)
bundles = [[j for j in range(M) if allocation_astar[j] == i] for i in range(N)]
for i in range(N):
    u_i_own = sum(values[i][j] for j in bundles[i])
    for j in range(N):
        if i == j or not bundles[j]:
            continue
        u_i_of_j = sum(values[i][k] for k in bundles[j])
        best_item_val = max(values[i][k] for k in bundles[j])
        best_item = max(bundles[j], key=lambda k: values[i][k])
        ok = u_i_own >= u_i_of_j - best_item_val
        print(f"  שחקן {i} מול חבילת שחקן {j}: u_{i}(own)={u_i_own}, "
              f"u_{i}(bundle {j})={u_i_of_j}, הפריט הכי-יקר בעיני {i} בחבילה {j} = פריט {best_item} (שווה {best_item_val}) "
              f"-> {u_i_own} >= {u_i_of_j}-{best_item_val}={u_i_of_j-best_item_val} ? {ok}")
