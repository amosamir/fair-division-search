import sys, random, pandas as pd
sys.path.insert(0, "/home/claude/fair_division")

def player_order(seed, n=5):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    return order

for path in ["/home/claude/fair_division/results.csv", "/home/claude/fair_division/results_v2.csv"]:
    df = pd.read_csv(path)
    df["rr_player_order"] = df.apply(lambda r: str(player_order(1000*int(r["m"])+int(r["seed"]))), axis=1)
    out_path = path.replace(".csv", "_with_order.csv")
    df.to_csv(out_path, index=False)
    print(f"{path} -> {out_path}  ({len(df)} rows)")
    print(df[["m","seed","rr_player_order"]].head(3).to_string())
