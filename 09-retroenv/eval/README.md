# Evaluate a model

`run_eval.py` runs a model on a split through the OpenEnv server. Every tool call and the reward come from the server, the same one used for training and the playground. Without `--server` it starts a local server for `--benchmark-dir` (default `benchmark/retroeval-v3`; download it from [AdithyaSK/RetroEnv-RL](https://huggingface.co/datasets/AdithyaSK/RetroEnv-RL) or point `--server` at one started with `RETROENV_TASKS_REPO`). The published board is on v2 (`--benchmark-dir benchmark/retroeval-v2`); v3 has none yet.

```bash
uv sync --extra dev --extra eval

# Claude, on the Messages API
uv run python eval/run_eval.py --provider anthropic --model claude-opus-5-5 \
  --split eval --output runs/v3-eval/claude-opus-5-5

# GPT-5.6, on the Responses API (chat completions rejects tools with reasoning)
uv run python eval/run_eval.py --provider openai --model gpt-5.6-sol --output runs/v3-eval/gpt-5.6-sol

# Open models on the HF router; pin a provider so prices are stable
uv run python eval/run_eval.py --provider hf --model "deepseek-ai/DeepSeek-V4.1-Flash:novita" \
  --output runs/v3-eval/deepseek-v4.1-flash

# A local vLLM server or any OpenAI-compatible endpoint
uv run python eval/run_eval.py --provider custom --endpoint http://127.0.0.1:8001/v1 \
  --model Qwen/Qwen3.5-4B --api-key-env VLLM_KEY --output runs/v3-eval/qwen3.5-4b

# No model: a scripted chemist replays the private reference routes (SFT data; see train/README.md)
uv run python eval/run_eval.py --provider reference --split train --attempts 2 --output runs/sft/chemist-v3

# Build the table from finished runs
uv run python eval/summarize.py runs/v3-eval/* --output benchmark/retroeval-v3/results
```

On v2, the six-model board cost $62.5 for 150 eval tasks. v3's 250 eval tasks include 4- and 5-step routes, so expect about twice that.

| Option | Default | Notes |
|---|---|---|
| `--split`, `--start`, `--tasks` | `eval`, 0, all | A stable slice of the split |
| `--attempts` | 1 | Above 1, the summary adds unbiased pass@k |
| `--max-turns` | 16 | The last turn exposes only `emit_routes` |
| `--toolset` | `full` | `unaided` starts the local server without the reference-backed tools |
| `--concurrency` | 8 | One WebSocket session per episode |
| `--max-cost` | none | Stops scheduling episodes once spend reaches this many USD |
| `--tool-choice` | `required` | Some HF providers only accept `auto` (novita, and cerebras for Qwen3.8) |

Keys come from the environment: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`, `OPENROUTER_API_KEY`, or `--api-key-env`.

## The protocol

Three loops implement the same protocol: `agent.py` for OpenAI-compatible chat, `agent_anthropic.py` for Claude and `agent_responses.py` for the OpenAI Responses API. All three live in `envs/retro_route/openenv/retroenv_openenv/`.

* The system prompt and task prompt are identical for every model. Each tool result reports the model turns left.
* The final turn exposes only `emit_routes`. Claude Opus 5.5 and Sonnet 5.5 reject a forced `tool_choice`, so that turn says so in text, as it does whenever `tool_choice` is `auto`.
* Two turns without a tool call trigger the final turn early. An episode that never calls `emit_routes` is closed with an empty route set, which scores the 0.05 floor.
* Claude runs at its default effort and thinking. GPT-5.6 runs at its default reasoning effort. Other models run at temperature 0.
* Some models send `submission` as a JSON string, sometimes with one stray closing bracket. The harness decodes that and records `submission_coerced`; the server's verifier stays strict, so training still sees the error.
* A refusal (Claude's `refusal` stop reason) counts as a failed episode. It is reported as `refusal_rate`, and requests are never re-routed to another model.

## Output

| File | Contents |
|---|---|
| `identity.json` | Model, provider, sampling, task IDs, toolset, and hashes of the tool schemas and agent code. A rerun must match. |
| `episodes/NNNN-<task>-aK.json` | Reward, components, failures, usage, cost, the submission and the full transcript |
| `progress.json`, `summary.json` | Coverage, pass@1 with a Wilson CI, exact route rate, reward and components, tool calls, no-emit and refusal rates, cost, and breakdowns by route count, step cap and heuristic tier |

Rerun the same command to resume. Graded episodes are kept, and those that hit a provider or transport error are retried. Transcripts contain solved routes, so `runs/` is gitignored and only `summarize.py` output is committed.

## The v1 board tools

These scripts reproduce the v1 board (20 tasks, run in-process over OpenRouter) and are unchanged. Predictions there are JSONL, one row per task with ranked attempts:

```json
{"task_id": "retro_...", "attempts": [{"submission": {"schema_version": "retro-route-graph-v1", "routes": []},
  "tool_calls": 6, "invalid_proposals": 1}]}
```

`retroenv-eval` recomputes every reward from the private tasks and the pinned stock, and never trusts a submitted reward. A missing task fails Pass@k instead of leaving the denominator; duplicate rows and unknown task IDs are errors.

* `run_baselines.py`: the oracle ceiling, one-route ablation and empty-graph floor.
* `run_model.py`: drives an OpenAI-compatible endpoint on the in-process session. It checkpoints each attempt, supports `--resume` and a provider-reported cost cap, and restricts the last turn to `emit_routes`.
* `estimate_board_cost.py`: projects cost from measured episode tokens. `benchmark_models.json` holds the model IDs and the dated price snapshot.
* `merge_model_shards.py`: merges `--start-index`/`--limit` shards and rejects duplicate, missing or mismatched ones.
* `summarize_board.py`: the cross-model JSON and Markdown table.
