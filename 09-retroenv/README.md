# 09 · RetroEnv

Plan a retrosynthesis: given a target molecule, work back to molecules you can
buy. The agent searches a fixed stock and training precedents, checks
disconnections, and submits molecule/reaction trees. A deterministic verifier
scores them against patent routes it never shows.

```text
licensed route trees → normalized references → leak-audited split → hidden tasks
→ bounded multi-turn tools → route graphs → dense reward → eval
```

RDKit and the hidden records establish structural consistency and dataset
support. They do not establish that a synthesis works experimentally, so results
say `dataset_supported`, never `experimentally_validated`.

## Layout

| Path | What it is |
|---|---|
| [`envs/retro_route/core`](envs/retro_route/core) | The `retroenv` package: chemistry, stock search, tasks, verifier, tool schemas, and the pure-Python session every front-end shares |
| [`envs/retro_route/openenv`](envs/retro_route/openenv) | The OpenEnv server, client, agent loops, playground and Docker/Space deployment |
| [`dataset/`](dataset) | Mining, task building, difficulty labels, and the audit |
| [`eval/`](eval) | `run_eval.py` through the server, plus the v1 board tools |
| [`benchmark/`](benchmark) | `retroeval-v1` (100 tasks, committed with references) and [`retroeval-v2`](benchmark/retroeval-v2) (1,000 tasks) |
| [`train/`](train) | The TRL GRPO smoke recipe |
| [`web/`](web) | Static explorer for the v1 board trajectories |

## Benchmarks

`benchmark/retroeval-v2` is the current set: 1,000 tasks, at most 3 steps, every
leaf in the frozen 13,432-molecule n1 stock. Its private references are rebuilt
rather than committed; see [its README](benchmark/retroeval-v2/README.md) for the
splits, the selection rules and the rebuild.

| Set | Tasks | train / dev / eval / stress | References in git |
|---|---:|---:|---|
| `retroeval-v1` | 100 | 40 / 20 / 20 / 20 | Yes |
| `retroeval-v2` | 1,000 | 600 / 100 / 150 / 150 | No, rebuilt from the scripts |

Both pass `dataset/audit_benchmark.py`: no target, scaffold, intermediate,
reaction, patent or near-duplicate at Morgan Tanimoto ≥ 0.90 crosses a split,
and every reference route and oracle graph replays through the serving verifier.

## Run it

```bash
uv sync --extra dev --extra eval
uv run pytest

RETROENV_BENCHMARK_DIR=benchmark/retroeval-v2 uv run uvicorn retroenv_openenv.server:app --port 8000
```

Open <http://localhost:8000/web/> to try a task by hand; the API is at `/docs`.
One model episode against that server:

```bash
uv run python envs/retro_route/openenv/rollout.py --server http://127.0.0.1:8000 \
  --split eval --index 0 --provider anthropic --model claude-opus-5-5
```

A whole split, with a server started for you:

```bash
uv run python eval/run_eval.py --provider anthropic --model claude-opus-5-5 \
  --split eval --output runs/v2-eval/claude-opus-5-5
uv run python eval/summarize.py runs/v2-eval/* --output benchmark/retroeval-v2/results
```

See [`envs/retro_route/openenv/README.md`](envs/retro_route/openenv/README.md) for
the server, Docker and Space deployment, and [`eval/README.md`](eval/README.md)
for providers, resuming and cost caps.

## Harness

`reset(split=..., index=...)` is deterministic and returns only the target,
budgets, stock ID, prompt, and tool names. It never returns a reference route,
reference count, patent ID, or answer-bearing metadata. Stock is not embedded
in the prompt. Tools are MCP calls, capped at 32 per episode
(`RETROENV_MAX_TOOL_CALLS`); `emit_routes` always stays available, and the step
that calls it carries `done` and the reward.

`RETROENV_TOOLSET=unaided` removes `validate_disconnection` and
`reaction_class_lookup`, and stops `reaction_conditions_search` answering from a
matched hidden record. Those three are the only tools that consult this task's
references, so the unaided surface measures retrosynthesis without an answer
oracle. Scores from the two toolsets are not comparable.

| Tool | Deterministic rollout behavior |
|---|---|
| `inspect_molecule` | RDKit formula, scaffold, rings, charge, and stereo |
| `pubchem_lookup` | Canonicalize SMILES locally; name/CAS requires a frozen cache |
| `stock_retrieve` | Only stock access; exact/InChIKey/class/SMARTS/similarity, cap 20 |
| `reaction_precedent_search` | Capped analogues from the **train split only** |
| `validate_disconnection` | Check an agent-supplied cut without returning a route |
| `reaction_class_lookup` | Class attached to a supported supplied cut, or `unclassified` |
| `reaction_conditions_search` | Frozen conditions from evidence/analogues |
| `search_literature` | Frozen patent/citation metadata; never live web during rollout |
| `emit_routes` | The only terminal action; parse, verify, and score 1–5 trees |

Live web and supplier-price search are intentionally absent from the current
rollout boundary. Network changes would make episodes non-replayable. They can
be used during dataset hydration and snapshotted before a future release.
Name/CAS support follows that rule: run
`python dataset/hydrate_pubchem.py queries.txt cache/pubchem.json`, then set
`RETROENV_PUBCHEM_CACHE=cache/pubchem.json` when serving.

The final submission is always renderable graph JSON:

```json
{
  "schema_version": "retro-route-graph-v1",
  "routes": [
    {
      "type": "mol",
      "smiles": "CCOC(C)=O",
      "in_stock": false,
      "children": [
        {
          "type": "reaction",
          "is_reaction": true,
          "metadata": {
            "explanation": "Dataset-supported ester disconnection.",
            "reaction_class": "esterification",
            "confidence": 0.9,
            "literature": [],
            "precursor_roles": {"CCO": "alcohol", "CC(=O)O": "acid"}
          },
          "children": [
            {"type": "mol", "smiles": "CCO", "in_stock": true, "children": []},
            {"type": "mol", "smiles": "CC(=O)O", "in_stock": true, "children": []}
          ]
        }
      ]
    }
  ]
}
```

A leaf claim is rewarded only when it agrees with exact stock membership. The
route score shown per tree is its weakest verified step, while the episode
reward covers the whole submitted route set.

## Reward

The terminal verifier is tolerant enough to train weak policies: malformed or
partial outputs do not collapse all groups to the same value. Passing status is
still strict—route-count compliance and at least `min_routes` fully supported,
stock-closed routes with distinct first cuts are required.

| Component | Weight |
|---|---:|
| JSON parse validity | 0.05 |
| Valid RDKit molecules | 0.10 |
| Alternating, connected target-rooted graph | 0.10 |
| Supported/atom-conserving steps | 0.20 |
| Truthful, complete stock leaves | 0.10 |
| Best reference-route similarity | 0.10 |
| Exact match to any reference | 0.10 |
| Verified distinct first cuts | 0.10 |
| Required route-set cardinality | 0.15 |

The legacy flat-route verifier remains hard-gated for compatibility. New
training uses `emit_routes` and the dense graph score. Unsupported chemistry,
valid-looking SMILES, or fluent explanations cannot earn the high-value
correctness components.

Sanity baselines are generated from private references only to test plumbing:

```bash
uv run python eval/run_baselines.py
```

Expected result: oracle ceiling `1.000/pass`, one-route ablation `0.788/fail`,
and empty graph floor `0.050/fail`. These are not model baselines.

## The v1 board

The v1 board (20 tasks, run in-process over OpenRouter) used `eval/run_model.py`:

```bash
uv run python eval/run_model.py \
  --endpoint http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen3.5-4B \
  --split dev --attempts 4 \
  --output sample/model-runs/qwen-dev.jsonl
```

The runner records raw transcripts separately, stores submissions, writes a
hash-pinned run manifest, checkpoints each paid attempt, and can resume after a
provider failure. It forces the final turn to expose only `emit_routes`, then
recomputes Pass@1/Pass@k and every component offline against private truth.
Reports include Wilson 95% intervals for pass rates. To score an existing
prediction file:

```bash
uv run retroenv-eval \
  --tasks-dir sample/tasks-private \
  --stocks-dir sample/stocks \
  --predictions predictions.jsonl \
  --split eval --k 1 4 8
```

The current 20-task model board is in
[`benchmark/retroeval-v1/model-runs/board-v1/RESULTS.md`](benchmark/retroeval-v1/model-runs/board-v1/RESULTS.md).

The static benchmark explainer and trajectory explorer is under `web/`. It
contains all 160 completed episodes, model/task selectors, compact tool traces,
and interactive submitted-route DAGs without exposing private references:

```bash
uv run python web/build_data.py
uv run python -m http.server 8080 --directory web
```

Open <http://localhost:8080> or open `web/index.html` directly.

For the first GRPO experiment, overfit a handful of training tasks before scaling:
use deterministic `(split, index)` reset, 4–8 rollouts per group, and monitor
within-task reward standard deviation—not only mean reward. The environment
server used for training must be the same image and private task bundle used for
evaluation. A trainer should persist raw tool episodes so graph/reward failures
can be replayed.

## Full data pipeline

The workspace already has about 6.5 GB of raw artifacts covering ORD, Lowe
USPTO, CRD, USPTO-LLM, CREED/CREED-CCV, FREA/RxnVerif, PaRoutes, and SynRXN.
The registry distinguishes approved, non-commercial, research-only, and pending
sources; “download everything” never means scraping proprietary or
license-unknown data.

```bash
python dataset/download_raw.py --include-research-only
python dataset/inventory_raw.py

retroenv-prepare-source paroutes_v2 \
  data/raw/paroutes-v2-benchmark build/paroutes-n1.routes.jsonl --release n1

retroenv-build-tasks \
  --routes build/paroutes-n1.routes.jsonl \
  --stock-file data/raw/paroutes-v2-benchmark/stock_n1.txt \
  --stock-id paroutes-v2-n1 \
  --output-dir build/paroutes-n1-tasks
```

The larger dry run produced 7,123 provenance-backed 1–3 step tasks with zero
audited crossings. See `dataset/README.md` for normalization, deduplication,
evidence labels, licensing, and why flat reaction rows are never globally
joined into invented multi-step routes.

## Design basis

The implementation follows the data-first lessons in the
[FineEnvs GeoGuesser environment article](https://huggingface.co/spaces/FineEnvs/geoguesser-article):
freeze a small eval early, define contamination at the correct grouping unit,
make reset indexable, keep truth inside the environment, provide continuous
reward, simulate full rollouts before GPU training, overfit a few tasks first,
use the same hosted environment for train/eval, and preserve raw episodes.

See `RESEARCH.md` for source and verifier evidence, and `dataset/README.md` for
the corpus contract.
