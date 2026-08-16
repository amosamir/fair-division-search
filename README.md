# Fair Division of Indivisible Items — Round-Robin vs. A* Search

Simulation code and experiment results for a course project on search methods
in AI: comparing the **Round-Robin** algorithm (a fast baseline that
guarantees EF1) against an **A\*** state-space search that maximizes Nash
social welfare subject to EF1, for the problem of dividing indivisible items
among agents with private valuations.

## Repository layout

```
src/          all Python source code
results/      raw experiment output (CSV / JSON)
figures/      generated plots and search-tree diagrams (PNG)
docs/         write-ups: the "Experiments and Results" chapter (docx),
              worked examples, and SmartArt-ready tree outlines
```

### `src/`
| File | Purpose |
|---|---|
| `fair_division.py` | Core library: value generation, Round-Robin, exact EF/EF1 checks, A* search with the admissible water-filling heuristic and EF1/dynamic-bound pruning |
| `run_experiment.py` / `run_experiment_v2.py` | Run the full 50-trial × 5-size (M=10..50) experiment; v2 adds EF (not just EF1) tracking and per-item detail export |
| `analyze_results.py` / `analyze_results_v2.py` | Compute summary statistics and generate the summary plots |
| `reconstruct_tree.py` | **Generic** tool: given `--m` and `--run` (or `--seed`), exactly replays the A* search for that one instance and records every node (N=expanded, L=leaf/terminal pop, S=frontier, P=pruned) |
| `render_tree.py` | Renders a reconstructed search tree (from `reconstruct_tree.py`'s output) as a PNG diagram |
| `detailed_example.py` / `trace_search.py` | Standalone illustrative walkthroughs used while developing/debugging the search |
| `add_player_order.py` | Adds the Round-Robin player draw order to the results CSVs |

### `results/`
- `results.csv` / `results_v2.csv` — one row per trial (250 rows: 50 × {M=10,20,30,40,50}) with Nash/utilitarian/egalitarian welfare, convergence, EF/EF1 flags, timing, node counts.
- `*_with_order.csv` — same, plus the Round-Robin player draw order.
- `item_details*.csv` — one row per (run, item): every player's valuation, the tier/value-range it was drawn from, and both algorithms' allocation decision for that item.
- `tree_table*.csv`, `nodes.json` — full per-node data for the reconstructed search-tree examples.

### `docs/`
- `experiments_and_results.docx` — the full "Experiments and Results" write-up (Hebrew), with summary tables, plots, and discussion of the EF/EF1 gap and the computational wall.
- `search_tree_example.docx`, `m10run5_tree_example.docx` — worked search-tree examples (diagram + full node table).
- `tree_outline*.txt` — tab-indented outlines, paste-ready into Word's SmartArt hierarchy/org-chart text pane.

## Reproducing a specific run

Every trial is fully deterministic from `(M, run)` via `seed = 1000*M + run`
(run ∈ 0..49). To replay and inspect any single trial from the experiment:

```bash
cd src
python3 reconstruct_tree.py --m 10 --run 5 --outdir ../out_m10run5
python3 render_tree.py --outdir ../out_m10run5
```

This regenerates `nodes.json`, `tree_outline.txt`, `tree_table.csv`, and
`search_tree.png` for that exact instance, matching the corresponding row in
`results_v2.csv` node-for-node.

## Key definitions

- **Nash welfare (as reported here)**: $\sum_i \log(u_i + 1)$ — the "+1" is a
  smoothing term required because intermediate search nodes (and the root)
  routinely have $u_i=0$ for some player, which would make $\log(u_i)$
  undefined. This differs from the textbook $\sum_i \log(u_i)$ by a constant
  offset per run; see `docs/experiments_and_results.docx` for details.
- **EF1** vs **EF**: Round-Robin guarantees EF1 (envy-freeness up to one item)
  by construction; A* is EF1-constrained by the search itself. Full EF (no
  envy at all) is checked separately and is *not* guaranteed by either
  method — see the write-up for how often each achieves it in practice.
