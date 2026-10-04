# Training handoff

RetroEnv is an OpenEnv HTTP/MCP environment. Start it against
`sample/tasks-private` and `sample/stocks`, then give the trainer deterministic
`reset(split="train", index=i)` rows. The six-task sample is intended for a
rollout and overfit test, not a headline checkpoint.

Before spending GPU time:

1. Run `uv run pytest` and `uv run python eval/run_baselines.py`.
2. Use `eval/run_model.py` on the base model and inspect its `.episodes.jsonl`.
3. Verify every task has nonzero within-group reward variance across 4–8
   samples; flat groups produce no GRPO advantage.
4. Overfit 2–4 fixed task indices and confirm reward, valid graphs, exact stock
   claims, and `emit_routes` completion improve together.
5. Train against the same server image and private bundle used for evaluation.

The policy sees tool outputs and the public task only. It must never load
`tasks-private`, `normalized-routes.jsonl`, baseline oracle files, or the stock
file directly. `stock_retrieve` is the sole stock surface.

`grpo_smoke.py` is the minimal TRL 1.12 recipe. From a current TRL environment:

```bash
uv run --with 'trl[vllm]>=1.12,<1.13' --with datasets --with peft \
  python train/grpo_smoke.py
```

To train against the OpenEnv server instead, the same one evaluation uses, start it
and point the script at it. Each rollout then holds one WebSocket session:

```bash
RETROENV_BENCHMARK_DIR=benchmark/retroeval-v3 uv run uvicorn retroenv_openenv.server:app --port 8000 &
RETROENV_SERVER=http://127.0.0.1:8000 MAX_TASKS=8 uv run --with 'trl[vllm]>=1.12,<1.13' \
  --with datasets --with peft python train/grpo_smoke.py
```

`MAX_TASKS` limits the train split for an overfit run. In-process, it uses
`RetroRouteTrainingEnv` as `environment_factory`, four generations per
indexed task, Dr. GRPO, ten optimizer steps, and optional JSONL traces via
`RETROENV_TRACE_PATH=outputs/episodes.jsonl`. Treat it as a wiring/overfit run;
do not publish its six-task metric as model quality.

## Scaling RL on v3

`benchmark/retroeval-v3` has 27,489 train tasks: 13,376 / 8,878 / 3,724 / 1,511 with 2 /
3 / 4 / 5 steps, and 17,568 / 7,132 / 2,789 easy / medium / hard by the heuristic tier.
Train RL against it, watch the 150-task dev split, and report on eval.

- **Curriculum.** `MAX_ROUTE_STEPS` keeps only tasks whose step budget is at most that
  value (a public field). Start at 2 or 3 and raise it once groups stop being flat.
- **Precedents mirror evaluation.** The server's precedent index holds every train
  task, but a train task's search hides its own reactions, its patent, its scaffolds and
  its near-duplicates (`PrecedentIndex.hidden`). The split withholds exactly those from
  an eval task, so RL cannot learn to copy an answer that evaluation never offers. Nothing
  is hidden from a dev, eval or stress task.
- **Flat groups.** A base model that never passes gets no GRPO signal; warm-start it with
  SFT first.

## SFT warm start

GRPO needs reward variance within each group, and small models rarely pass at all
(Qwen3.8 27B passes 1.3% of v2 eval tasks), so a supervised warm start comes first.

1. **Generate trajectories.** `--provider reference` runs no model. A scripted chemist
   (`retroenv_openenv/agent_reference.py`) knows each task's patent route and acts
   through the server and the same agent loop, so every episode is graded like an
   evaluation episode. It plans like a chemist:
   - It looks first, inspecting the target and searching the training precedents.
   - It proposes strategic disconnections from `retroenv.disconnections`. In 46% of
     episodes at least one is a plausible cut the patent did not make; the validator
     rejects it and the chemist recovers.
   - It checks every piece against the stock and disconnects what is missing.
   - It writes up each reaction with its type, reagent roles and evidence.

   Choices are seeded by the episode ID, so two attempts per task give two different
   trajectories. It needs no key.

   ```bash
   uv run --extra eval python eval/run_eval.py --provider reference --split train \
     --benchmark-dir benchmark/retroeval-v3 --attempts 2 --concurrency 4 --output runs/sft/chemist-v3
   ```

   A teacher model's train-split run, from any provider, can be exported alongside
   it. Check that provider's terms before training on its outputs.

2. **Export.** `dataset/build_sft.py` keeps episodes that passed, cuts each one after its
   `emit_routes` call, holds out 1% of tasks for validation, and refuses any run or task
   from a held-out split. It then re-audits the train tasks used against the guard
   benchmark's held-out splits, under that benchmark's scaffold rules. Keeping up to two
   trajectories for 4- and 5-step tasks moves the mix toward the eval split:

   ```bash
   uv run python dataset/build_sft.py --run runs/sft/chemist-v3 \
     --train-tasks benchmark/retroeval-v3/tasks-private/train.jsonl \
     --guard-dir benchmark/retroeval-v3 --per-task-by-steps 4=2,5=2 \
     --output .local/sft/chemist --tokenizer Qwen/Qwen3.5-4B
   ```

   `manifest.json` reports how often rows recover from a rejected cut, which tools they
   use and how varied their text is.

Rows are OpenAI-style `messages` plus `tools`. Tool-call arguments and `tools` are JSON
strings; the generated dataset card explains why and how to decode them.

The published export is [AdithyaSK/RetroEnv-SFT](https://huggingface.co/datasets/AdithyaSK/RetroEnv-SFT):
config `chemist` (this recipe) and `plain` (the earlier always-right expert, kept as a
baseline).

| | `chemist` | `plain` |
|---|---:|---:|
| Rows (train / validation) | 31,897 / 323 | 27,222 / 267 |
| Rows with 4- or 5-step routes | 31% | 19% |
| Rows that recover from a rejected cut | 46% | 0% |
| Tools used | 5 | 3 |
| Mean tool calls per row | 11.8 | 8.6 |
| Distinct assistant texts | 112,902 | 31,736 |
| Tokens per row, p50 / p95 (Qwen3.5) | 5,473 / 7,778 | 4,142 / 5,858 |

Both cover all 27,489 v3 train tasks, and every row passed the verifier.

`sft_smoke.py` is the matching TRL recipe: LoRA on Qwen3.5-4B, loss on assistant turns
only, and a merged checkpoint that `grpo_smoke.py` can start from
(`MODEL=outputs/retroenv-sft-smoke/merged`). It reads the Hub dataset by default
(`SFT_CONFIG=chemist` or `plain`), or a local export with `SFT_DATA=.local/sft/chemist`:

```bash
uv run --with 'trl>=1.12,<1.13' --with datasets --with peft python train/sft_smoke.py
```

Build SFT data from the same benchmark you evaluate on. Many tasks that are train in one
version are held out in another; 108 of the 250 v3 eval tasks were train tasks in v2's
widened split.
