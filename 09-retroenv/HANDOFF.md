# RetroEnv handoff

RetroEnv is a multi-turn tool-use environment for retrosynthesis. Given a target molecule, the
agent plans a route back to purchasable building blocks, and a deterministic verifier scores
it against the route in the target's patent. This page covers what exists, where it lives, how
to run it, what was decided and why, and what to do next.

## What exists

| Piece | Where | State |
|---|---|---|
| Environment core: chemistry, stock search, verifier, tools, session | `envs/retro_route/core/retroenv/` | Done, tested |
| OpenEnv server (HTTP/MCP), client, hand-play UI, Docker and Space deploy | `envs/retro_route/openenv/` | Done; loads its tasks from the Hub |
| v3 benchmark: 27,489 train, 150 dev, 250 eval, 150 stress | [AdithyaSK/RetroEnv-RL](https://huggingface.co/datasets/AdithyaSK/RetroEnv-RL), built by `dataset/build_benchmark_v3.py` | Done, audited, no model board yet |
| v2 benchmark (1,000 tasks) and its six-model board | The same Hub repo under `retroeval-v2/` and `runs/v2-eval/` | Done |
| SFT data for the v3 train split: `chemist` (32,220 rows) and `plain` (27,489) | [AdithyaSK/RetroEnv-SFT](https://huggingface.co/datasets/AdithyaSK/RetroEnv-SFT) | Done; no model trained yet |
| Evaluation harness: Claude, OpenAI, HF router, OpenRouter, any OpenAI-compatible endpoint, the scripted chemist | `eval/run_eval.py` | Done |
| Training recipes: SFT (TRL, assistant-only loss) and GRPO smoke | `train/sft_smoke.py`, `train/grpo_smoke.py` | Smoke-tested only |
| Local explorer: tasks, answer keys, live play, SFT rows, model runs | `explorer/` | Done |
| Explainer video | `media/retroenv-explainer.mp4` in the RL repo | Done |
| Visual guide to one RL task and one SFT row, with dataset statistics | `docs/training-data.html` (open in a browser) | Done |

## Run it

```bash
uv sync --extra dev --extra eval && uv run pytest

# Serve v3 from the Hub (the Docker image's default)
RETROENV_TASKS_REPO=AdithyaSK/RetroEnv-RL uv run bash envs/retro_route/openenv/start.sh

# Evaluate a model on v3 eval through that server
uv run python eval/run_eval.py --provider anthropic --model claude-opus-5-5 \
  --server http://127.0.0.1:8000 --split eval --output runs/v3-eval/claude-opus-5-5

# Browse everything locally
hf download AdithyaSK/RetroEnv-RL --repo-type dataset --local-dir benchmark/retroeval-v3 \
  --exclude "retroeval-v2/*" "runs/*" "media/*"
hf download AdithyaSK/RetroEnv-SFT --repo-type dataset --local-dir .local/sft
uv run --extra eval python explorer/server.py

# Train
uv run --with 'trl>=1.12,<1.13' --with datasets --with peft python train/sft_smoke.py
RETROENV_SERVER=http://127.0.0.1:8000 MAX_ROUTE_STEPS=3 uv run --with 'trl[vllm]>=1.12,<1.13' \
  --with datasets --with peft python train/grpo_smoke.py
```

`train/README.md`, `eval/README.md` and `envs/retro_route/openenv/README.md` have the details.

## Decisions and why

- **v3 eval was designed, not sampled.**
  - The held-out splits were drawn first, against quotas: 70 / 70 / 60 / 50 tasks with 2 / 3 / 4 /
    5 steps, all ten first-step reaction families, four size bins, and 10% two-route tasks.
  - Held-out tasks share nothing with train or with each other: no target, scaffold,
    intermediate, reaction, patent or near-duplicate.
  - v2's eval had only 2- and 3-step routes and was half alkylation and FGI.
- **33 generic scaffolds group by exact structure.** Biphenyl, indole and other scaffolds found
  in at least 100 pool tasks are treated like benzene. Grouping on them would have cut train
  from 27,489 to 20,979, since one eval biphenyl barred 719 train tasks. Patent, intermediate,
  reaction and near-duplicate grouping still apply.
- **Precedent search hides from a train task what the split hides from an eval task**: its own
  reactions, patent, scaffolds and near-duplicates (`PrecedentIndex.hidden`). Without this, RL
  learns to copy answers that evaluation never offers. Nothing is hidden from a held-out task,
  so the v2 board is unaffected.
- **The answers are public.** PaRoutes is public, so anyone can rebuild them. Note this when
  reporting a model trained on USPTO-derived data.
- **SFT comes from a scripted chemist, not a model teacher.**
  - It is free and has no licence restrictions; Anthropic's and OpenAI's terms restrict
    training other models on their outputs.
  - It uses the real server, so every row is graded like an evaluation episode.
  - It plans like a chemist: inspects the target, searches precedents, tries rule-library cuts
    (wrong in 46% of rows, then recovers), checks the stock, and writes up each reaction with
    evidence. All 54,978 generated episodes passed; 4- and 5-step tasks keep up to two.
  - Its text states only what the tools returned or what the structures show. It never writes
    the task's own patent or reagents.
- **`plain` is kept as a baseline.** It is the first expert: always right, four fixed sentences,
  no recoveries. Training on both and comparing on dev measures whether the richer behaviour
  helps.
- **SFT rows store tool-call arguments and `tools` as JSON strings.** Arrow would merge
  different tools' arguments into one struct and render the null gaps into the training text.
  `sft_smoke.py` decodes them and tokenizes with assistant-only masks, using TRL's Qwen3.5
  training template.

## Caveats

- **No model has been evaluated on v3** and no model has been trained. The v2 board's best is
  Claude Opus 5.5 at 0.560 pass@1. Six models on v3 eval should cost about $120.
- **The chemist's mistakes are textbook alternatives** from eight bond rules, not the errors a
  model makes, and its text is templated. Self-generated or teacher episodes would add real
  variety.
- **The full toolset includes two oracle tools.** `validate_disconnection` and
  `reaction_class_lookup` answer from the hidden route. Rows built with the full toolset teach
  a policy to rely on them; `--toolset unaided` builds rows without them, and its scores are
  not comparable.
- **5-step routes leave little slack.** A perfect solver uses 14 of the 32 tool calls on them.
- **The reaction-family classifier is coarse.** About 15% of first steps land in "other".
- **Only the patent's route scores 1.0.** A different valid route still passes, but scores
  below 1.0 on similarity and exact match.
- **Refusals score the floor.** Opus 5.5 refused 16% of v2 episodes on safety grounds.

## Next steps

1. **Run the v3 board** on dev and eval. This is the evaluation owner's call; `eval/README.md`
   has the commands.
2. **Fine-tune Qwen3.5-4B** on `chemist`, and separately on `plain`. Compare dev pass@k and
   whether GRPO groups stop being flat.
3. **Let the SFT model generate its own data.** Sample it about 8 times per train task, keep the
   passing episodes, retrain, then run GRPO with a `MAX_ROUTE_STEPS` curriculum on tasks it
   passes only sometimes.
4. **Add a non-LLM baseline**, such as AiZynthFinder, on v3 eval. It shows whether the benchmark
   rewards planning or recall.
5. **Optional:** a small teacher-model slice for hard tasks (check the terms), an `unaided` SFT
   variant, and Harbor support.

## Rebuild from scratch

The PaRoutes archive and stock come from `dataset/download_raw.py`.

```bash
uv run python dataset/mine_route_pool.py --max-steps 5 --output .local/pool/pool-n1-s5.jsonl
uv run python dataset/build_benchmark_v3.py --output-dir benchmark/retroeval-v3
uv run python dataset/label_difficulty.py --benchmark-dir benchmark/retroeval-v3
uv run python dataset/audit_benchmark.py --benchmark-dir benchmark/retroeval-v3 --expected-eval-tasks 250 --manifest-rules
uv run python eval/run_eval.py --provider reference --split train --attempts 2 --output runs/sft/chemist-v3
uv run python dataset/build_sft.py --run runs/sft/chemist-v3 --guard-dir benchmark/retroeval-v3 \
  --train-tasks benchmark/retroeval-v3/tasks-private/train.jsonl --per-task-by-steps 4=2,5=2 \
  --output .local/sft/chemist --tokenizer Qwen/Qwen3.5-4B
```

Generating the trajectories takes a few hours on one server process, and every build is
deterministic.
