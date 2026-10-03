# 09 · RetroEnv

Plan a retrosynthesis: given a target molecule, work back to molecules you can buy. The
agent searches a fixed stock and training precedents, checks disconnections, and submits
molecule/reaction trees. A deterministic verifier scores them against patent routes it
never shows.

RDKit and the hidden records establish structural consistency and dataset support. They do
not establish that a synthesis works experimentally, so results say `dataset_supported`,
never `experimentally_validated`.

| Artifact | Where |
|---|---|
| RL tasks: the v3 benchmark (28,039 tasks), v2 and its model board | [AdithyaSK/RetroEnv-RL](https://huggingface.co/datasets/AdithyaSK/RetroEnv-RL) |
| SFT trajectories for the v3 train split | [AdithyaSK/RetroEnv-SFT](https://huggingface.co/datasets/AdithyaSK/RetroEnv-SFT) |
| Explainer video (83 s) | [`media/retroenv-explainer.mp4`](https://huggingface.co/datasets/AdithyaSK/RetroEnv-RL/blob/main/media/retroenv-explainer.mp4) |
| Local explorer for tasks, SFT rows and model runs | [`explorer/`](explorer) |
| Visual guide to one RL task and one SFT row | [`docs/training-data.html`](docs/training-data.html) (open in a browser) |
| What was built, and what is next | [`HANDOFF.md`](HANDOFF.md) |

## Quick start

```bash
uv sync --extra dev --extra eval
uv run pytest

# Serve v3 from the Hub (also the Docker image's default)
RETROENV_TASKS_REPO=AdithyaSK/RetroEnv-RL uv run bash envs/retro_route/openenv/start.sh
```

The server is at <http://localhost:8000>: a hand-play UI at `/web/`, the API at `/docs`. One
model episode, then a whole split:

```bash
uv run python envs/retro_route/openenv/rollout.py --server http://127.0.0.1:8000 \
  --split eval --index 0 --provider anthropic --model claude-opus-5-5
uv run python eval/run_eval.py --provider anthropic --model claude-opus-5-5 \
  --server http://127.0.0.1:8000 --split eval --output runs/v3-eval/claude-opus-5-5
```

To browse everything, download the benchmark and start the explorer:

```bash
hf download AdithyaSK/RetroEnv-RL --repo-type dataset --local-dir benchmark/retroeval-v3 \
  --exclude "retroeval-v2/*" "runs/*" "media/*"
uv run --extra eval python explorer/server.py      # http://127.0.0.1:8050
```

## Layout

| Path | What it is |
|---|---|
| [`envs/retro_route/core`](envs/retro_route/core) | The `retroenv` package: chemistry, stock search, tasks, verifier, tool schemas, the disconnection library, and the session every front-end shares |
| [`envs/retro_route/openenv`](envs/retro_route/openenv) | The OpenEnv server, client, agent loops (including the scripted chemist), hand-play UI and Docker/Space deployment |
| [`dataset/`](dataset) | Mining, the v2 and v3 builders, difficulty labels, the audit, and the SFT exporter |
| [`eval/`](eval) | `run_eval.py` through the server, baselines and summaries |
| [`train/`](train) | SFT and GRPO recipes |
| [`benchmark/`](benchmark) | `retroeval-v1` (100 tasks, with references), [`retroeval-v2`](benchmark/retroeval-v2) and [`retroeval-v3`](benchmark/retroeval-v3); answers come from the Hub |
| [`explorer/`](explorer) | Local browser for benchmarks, SFT datasets and model runs, with live play |
| [`sample/`](sample) | A six-task bundle for smoke tests |

## Benchmarks

| Set | Tasks | train / dev / eval / stress | Routes | Status |
|---|---:|---:|---|---|
| [`retroeval-v3`](benchmark/retroeval-v3) | 28,039 | 27,489 / 150 / 250 / 150 | 2–5 steps | current; no model board yet |
| [`retroeval-v2`](benchmark/retroeval-v2) | 1,000 | 600 / 100 / 150 / 150 | 2–3 steps | has the six-model board |
| `retroeval-v1` | 100 | 40 / 20 / 20 / 20 | — | first board, kept for reference |

v3's eval split was designed rather than sampled. It has 70 / 70 / 60 / 50 tasks with 2 / 3 /
4 / 5 steps, all ten first-step reaction families and four molecule-size bins. No held-out
task shares a target, scaffold, intermediate, reaction, patent or near-duplicate with train
or with another held-out task. Every set passes `dataset/audit_benchmark.py`.

## The environment

`reset(split=..., index=...)` is deterministic and returns only the target, the budgets, the
stock ID, the prompt and the tool names: never a reference route, its patent or its count.
The stock is not in the prompt. An episode allows 32 tool calls (`RETROENV_MAX_TOOL_CALLS`)
and ends with `emit_routes`, whose step carries `done` and the reward.

| Tool | What it does |
|---|---|
| `inspect_molecule` | RDKit formula, scaffold, rings, charge and stereo |
| `pubchem_lookup` | Canonicalize SMILES locally; names and CAS numbers need a frozen cache |
| `stock_retrieve` | The only stock access: exact, InChIKey, class, SMARTS or similarity, at most 20 results |
| `reaction_precedent_search` | Analogues from the train split, minus anything sharing a leakage group with the task |
| `validate_disconnection` | Check a proposed cut against the hidden evidence, without revealing the route |
| `reaction_class_lookup` | The class of a supported cut |
| `reaction_conditions_search` | Reported conditions for a supported cut, or from analogues |
| `search_literature` | Frozen citation metadata from training precedents |
| `emit_routes` | Submit one to five route trees; terminal |

`RETROENV_TOOLSET=unaided` removes `validate_disconnection` and `reaction_class_lookup`, and
stops `reaction_conditions_search` answering from the hidden record, so the agent has no
answer oracle. Scores from the two toolsets are not comparable.

A submission is a list of molecule → reaction → molecule trees. Each reaction carries
metadata: an explanation, the reaction class, a confidence, literature and precursor roles.
A leaf earns credit for `in_stock: true` only if it is in the stock.

## Reward

The verifier is graded, so weak policies still see a signal, but passing is strict: the
right number of routes, each fully supported and stock-closed, with distinct first cuts.

| Component | Weight |
|---|---:|
| JSON parse validity | 0.05 |
| Valid RDKit molecules | 0.10 |
| Alternating, connected, target-rooted graph | 0.10 |
| Supported, atom-conserving steps | 0.20 |
| Truthful, complete stock leaves | 0.10 |
| Similarity to the closest reference route | 0.10 |
| Exact match to a reference route | 0.10 |
| Verified distinct first cuts | 0.10 |
| Required number of routes | 0.15 |

The oracle scores 1.000, and an empty submission scores 0.050 (`eval/run_baselines.py`).

## Training data

- **RL:** the 27,489 v3 train tasks. A train task's precedent search hides what the split hides
  from an eval task (its own reactions, patent, scaffolds and near-duplicates), so RL cannot
  learn to copy answers. `MAX_ROUTE_STEPS` in `train/grpo_smoke.py` sets a curriculum by
  route length.
- **SFT:** episodes from a scripted chemist that knows each patent route and acts through the
  real server. It inspects the target and searches precedents, sometimes tries a plausible
  wrong cut and recovers, checks every piece against the stock, and writes up each reaction
  with its evidence. Every row passed the verifier. The `chemist` config has 32,220 rows over
  all train tasks, 46% of them recovering from a rejected cut. `train/sft_smoke.py` trains on
  it, with loss on the assistant turns only.

[`train/README.md`](train/README.md) has both recipes, and the dataset card explains how the
rows were made.

## Results

The v2 board covers 150 eval tasks, one attempt each, with the full toolset. Pass@1: Claude
Opus 5.5 0.560, Claude Sonnet 5.5 0.300, GPT-5.6 Sol 0.280, DeepSeek V4.1 Flash 0.107, GPT-5.6
Luna 0.073 and Qwen3.8 27B 0.013. Every model drops on 3-step routes. Opus refused 16% of
episodes, which score the floor. See
[`benchmark/retroeval-v2/results/RESULTS.md`](benchmark/retroeval-v2/results/RESULTS.md).

## Rebuilding the data

```bash
uv run python dataset/mine_route_pool.py --max-steps 5 --output .local/pool/pool-n1-s5.jsonl
uv run python dataset/build_benchmark_v3.py --output-dir benchmark/retroeval-v3
uv run python dataset/label_difficulty.py --benchmark-dir benchmark/retroeval-v3
uv run python dataset/audit_benchmark.py --benchmark-dir benchmark/retroeval-v3 \
  --expected-eval-tasks 250 --manifest-rules
```

Each build is deterministic and checked against `checksums.json`. The SFT data then comes
from `eval/run_eval.py --provider reference` and `dataset/build_sft.py`; see
[`train/README.md`](train/README.md). [`dataset/README.md`](dataset/README.md) covers the raw
sources, their licences and the corpus contract. [`RESEARCH.md`](RESEARCH.md) has the source
and verifier evidence.

## Design basis

This follows the data-first lessons of the
[FineEnvs GeoGuesser article](https://huggingface.co/spaces/FineEnvs/geoguesser-article):

- freeze a small eval early, and define contamination at the right grouping unit;
- make reset indexable, keep the truth inside the environment, and give a continuous reward;
- simulate full rollouts before GPU training, and overfit a few tasks first;
- use the same hosted environment for training and evaluation, and keep the raw episodes.
