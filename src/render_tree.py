import json, argparse, os
import graphviz

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", type=str, required=True, help="directory containing nodes.json (from reconstruct_tree.py)")
args = ap.parse_args()

with open(os.path.join(args.outdir, "nodes.json"), encoding="utf-8") as f:
    data = json.load(f)

records = data["records"]

dot = graphviz.Digraph(
    "search_tree",
    graph_attr={"rankdir": "TB", "splines": "ortho", "nodesep": "0.15", "ranksep": "0.6", "bgcolor": "white", "ordering": "out"},
    node_attr={"shape": "box", "style": "filled", "fontname": "Arial", "fontsize": "10", "width": "0.5", "height": "0.28"},
    edge_attr={"arrowsize": "0.6", "color": "#888888"},
)

COLORS = {
    "N": "#BBDEFB",
    "L": "#C8E6C9",
    "S": "#E0E0E0",
    "P": "#FFCDD2",
}

for key, rec in records.items():
    label = rec["label"]
    kind = label[0]
    color = COLORS.get(kind, "#FFFFFF")
    dot.node(key, label=label, fillcolor=color)

for key, rec in records.items():
    for child in rec["children"]:
        child_key = ",".join(map(str, child)) if child else "ROOT"
        dot.edge(key, child_key)

by_level = {}
for key, rec in records.items():
    by_level.setdefault(rec["level"], []).append(key)
for level, keys in by_level.items():
    with dot.subgraph(name=f"level_{level}") as s:
        s.attr(rank="same")
        for k in keys:
            s.node(k)

out_path = os.path.join(args.outdir, "search_tree")
dot.render(out_path, format="png", cleanup=True)
print("wrote", out_path + ".png")
