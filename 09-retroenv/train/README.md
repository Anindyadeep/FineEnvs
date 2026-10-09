# Training handoff

RetroEnv is an OpenEnv HTTP/MCP environment. Start it against a release
(`data/release/RetroEnv-RL`), then give the trainer deterministic
`reset(split="train", index=i)` rows. Download the release with
`uv run hf download LiteFold/RetroEnv --repo-type dataset --local-dir data/release/RetroEnv-RL`.
It is fully open and carries every split's known routes, which the environment needs: they
feed the reward and hide a train task's own reactions from the tools.

Before spending GPU time:

1. Run `uv run pytest` and `uv run python -m dataset.audit_release`.
2. Run `eval/run_eval.py` on the base model for a few dev tasks and read its episodes.
3. Verify every task has nonzero within-group reward variance across 4–8
   samples; flat groups produce no GRPO advantage.
4. Overfit 2–4 fixed task indices and confirm reward, valid graphs, exact stock
   claims, and `emit_routes` completion improve together.
5. Train against the same server image and private bundle used for evaluation.

The policy sees tool outputs and the public task only. It must never load
`tasks-private`, the `library/` files, or the stock file directly. `stock_retrieve` is the sole stock surface.

`grpo_smoke.py` is the minimal TRL 1.12 recipe. From a current TRL environment:

```bash
uv run --with 'trl[vllm]>=1.12,<1.13' --with datasets --with peft \
  python train/grpo_smoke.py
```

To train against the OpenEnv server instead, the same one evaluation uses, start it
and point the script at it. Each rollout then holds one WebSocket session:

```bash
RETROENV_BENCHMARK_DIR=data/release/RetroEnv-RL uv run uvicorn retroenv_openenv.server:app --port 8000 &
RETROENV_SERVER=http://127.0.0.1:8000 MAX_TASKS=8 uv run --with 'trl[vllm]>=1.12,<1.13' \
  --with datasets --with peft python train/grpo_smoke.py
```

`MAX_TASKS` limits the train split for an overfit run. In-process, it uses
`RetroRouteTrainingEnv` as `environment_factory`, four generations per
indexed task, Dr. GRPO, ten optimizer steps, and optional JSONL traces via
`RETROENV_TRACE_PATH=outputs/episodes.jsonl`. Treat it as a wiring/overfit run;
do not publish its metric as model quality.

## GRPO on HF Jobs

Two developer scripts train the same experiment, LoRA on Qwen3.8-27B (or Qwen3.6-35B-A3B)
against the RetroEnv server, with the evaluation board's system prompt and non-thinking chat
template. Read their numbered sections in order:

| Script | Trainer | Batch |
|---|---|---|
| [sync_grpo.py](sync_grpo.py) | TRL `GRPOTrainer`: generate a batch, then one optimizer step | 16 episodes per step (2 tasks × 8) |
| [async_grpo.py](async_grpo.py) | TRL `AsyncGRPOTrainer`: generation keeps running while it trains | the same, at most 4 policy versions stale |

Both use `RemoteRetroRouteEnv` as `environment_factory`: its public methods are the tools, each
call goes to the server over one WebSocket session per rollout, and the `emit_routes` grade is
the reward. An episode that never submits scores 0, as in evaluation; one whose session broke
scores NaN and stays out of its group's baseline. The default curriculum is the 51,358 standard
train tasks with a depth budget of 4 (`--max-depth`).

Evaluation tells the model its turns left and forces `emit_routes` on the last one; TRL's tool
loop does neither. Without a substitute, most rollouts researched until the loop ended and never
submitted, so whole groups tied at 0 and gave no gradient. So every tool result carries the
calls left in the server's 32-call budget and, with four left, a reminder to submit; the prompt
keeps the board's 16 turns and training allows 20, so an overrun can still submit. Each graded
episode is appended to `episodes.jsonl` in the run directory (task, reward, submitted, tool calls).

[jobs/](jobs) runs them on HF Jobs the way `05-multi-harness-rl/jobs` does, on `h200x2`: vLLM
serves the model on one GPU, the trainer runs on the other, and the RetroEnv server runs on CPU
from the project's locked environment, preparing its serving snapshot from the public bucket
mounted at `/data`. Checkpoints, logs and boards are written to an output bucket as they happen.

```bash
B=FineEnvs/retroenv-rl-runs
python train/jobs/hf_job.py check --name check --bucket $B --submit            # CPU: installs and setup checks
python train/jobs/hf_job.py smoke --mode sync --name sync-smoke --bucket $B --submit
python train/jobs/hf_job.py train --mode async --name async-100 --bucket $B --steps 100 --submit
python train/jobs/hf_job.py eval --name async-100-eval --bucket $B \
    --checkpoint /outputs/async-100/checkpoint-100 --submit                       # the core30 board
python train/jobs/hf_job.py eval --name base-eval --bucket $B --submit           # the base model, same harness
```

Without `--submit` the launcher prints the job plan. `check_setup.py` runs first in every job and
fails before the GPUs load anything if TRL cannot train through the model's chat template, a
tool call does not round-trip through it, or a tool's docstring does not become a schema. The
smoke test runs two updates of four rollouts each, requires a non-zero gradient and episode
traces, checks both checkpoints, then merges checkpoint 2 and evaluates it on two tasks. Evaluation merges the adapter into the base weights (`jobs/merge_lora.py`),
serves them with vLLM and runs `eval/evaluate.py` with the board's protocol, so its RESULTS.md
lines up with `runs/core30-nothink`.

## Scaling RL

The release's train split holds every eligible target that shares no leakage key with a
held-out task (see `manifest.json` for counts by shortest-route depth and tier), plus
constraint variants. Train against it, watch dev, and report on test_id and test_hard.

- **Curriculum.** `MAX_ROUTE_STEPS` keeps only tasks whose depth budget is at most that
  value (a public field). Start at 3 or 4 and raise it once groups stop being flat.
- **Precedents mirror evaluation.** Tools search only train-visible corpus reactions, and
  a train task's search also hides its own reactions, patents, scaffolds and
  near-duplicates (`PrecedentIndex.hidden`), so RL cannot learn to copy an answer that
  evaluation never offers. `validate_disconnection` likewise never confirms a train task's
  own reactions by corpus lookup.
- **Flat groups.** A base model that never passes gets no GRPO signal; warm-start it with
  SFT first.

## SFT warm start

The scripted chemist (`retroenv_openenv/agent_reference.py`, `--provider reference` in
`eval/run_eval.py`) replays each task's compliant known routes through the server, so its
episodes are graded like evaluation episodes. The previous SFT exporter and the v3 SFT
datasets are archived with the old benchmarks; regenerate trajectories from this release's
train split before training on them.

`sft_smoke.py` is the matching TRL recipe: LoRA on Qwen3.5-4B, loss on assistant turns
only, and a merged checkpoint that `grpo_smoke.py` can start from
(`MODEL=outputs/retroenv-sft-smoke/merged`).
