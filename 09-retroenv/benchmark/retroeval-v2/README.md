# retroeval-v2

1,000 route-planning tasks mined from PaRoutes v2 (Zenodo 7341155, CC-BY-4.0). Every leaf is in the n1 stock (13,432 molecules), and every route has at most 3 steps.

| Split | Tasks | From v1 | Two-route | 2 / 3 steps | Easy / medium / hard |
|---|---:|---:|---:|---:|---:|
| train | 600 | 37 | 50 | 327 / 273 | 344 / 190 / 66 |
| dev | 100 | 18 | 22 | 56 / 44 | 55 / 24 / 21 |
| eval | 150 | 16 | 18 | 84 / 66 | 83 / 44 / 23 |
| stress | 150 | 17 | 20 | 79 / 71 | 89 / 34 / 27 |

The 88 v1 tasks whose targets are not themselves in stock keep their v1 split, so all 16 surviving v1 eval tasks are in this eval split.

## What is in git

| Committed | Kept out of git (rebuild it) |
|---|---|
| `tasks-public/`: target, budgets and public difficulty, with no references | `tasks-private/`: the same tasks with their reference routes |
| `stocks/paroutes-v2-n1.smi` | `normalized-routes.jsonl` |
| `manifest.json`, `audit.json`, `difficulty.summary.json`, `baselines/report.json` | `difficulty.jsonl`: per-task heuristic tiers and first-step families, which are hints |
| `checksums.json`: SHA-256 of every private file | `baselines/*.jsonl`: oracle submissions |
| `results/`: the eval board | |

PaRoutes is public, so anyone can rebuild the answers. They stay out of git and out of the Docker image so they don't end up in crawled training data. During an episode, the guard is the rollout boundary: there is no web access, and references never leave the server.

## Rebuild the private files

From `09-retroenv`, after downloading `all_loaded_routes.json.gz` and `stock_n1.txt` into `data/raw/paroutes-v2-benchmark/` (see `dataset/README.md`):

```bash
uv run python dataset/mine_route_pool.py            # 28,788 verified targets, about 1.5 min
uv run python dataset/build_benchmark_v2.py --tasks 1000 --output-dir benchmark/retroeval-v2
uv run python dataset/label_difficulty.py --benchmark-dir benchmark/retroeval-v2
uv run python dataset/audit_benchmark.py --benchmark-dir benchmark/retroeval-v2 \
  --expected-eval-tasks 150 --exact-single-ring-scaffolds
```

The build is deterministic. Compare the result with `checksums.json`; `prepare.py` in the OpenEnv folder checks it at server startup.

## How the tasks were chosen

1. **Mine.** Keep PaRoutes routes with 1–3 reactions, one reaction per molecule, and every leaf in the stock. Take up to five routes per target with distinct first steps, replay each through the serving verifier, and drop targets that are themselves in stock (191, including 12 v1 tasks).
2. **Select.** Take every target with two distinct first steps (110), then single-route targets in stable-hash order. At most one task per patent and one per multi-ring scaffold; single-ring and acyclic scaffolds are capped at their share of the pool; 2- and 3-step routes are split evenly.
3. **Split.** Group tasks that share a target, scaffold, route, intermediate, reaction or patent, or whose molecules have Morgan Tanimoto ≥ 0.90. Each group goes to one split. Single-ring and acyclic molecules group by exact structure rather than scaffold, because benzene alone would merge 254 tasks and leave eval with no simple benzenes.

The audit passes: no shared target, scaffold, intermediate, reaction, patent or near-duplicate across splits, and every reference route and oracle graph replays. The oracle scores 1.000 and an empty graph 0.050.
