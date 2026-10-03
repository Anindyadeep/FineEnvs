---
title: RetroEnv
emoji: ⚗️
colorFrom: gray
colorTo: green
sdk: docker
app_port: 8000
base_path: /web
pinned: false
tags: [openenv, chemistry, retrosynthesis, tool-use]
short_description: Plan retrosynthesis routes with verifier-backed rewards.
---

# RetroEnv: OpenEnv server

An agent gets a target molecule and works back to purchasable starting materials. It uses tools to inspect molecules, search a fixed stock and training precedents, and check disconnections. It ends the episode by submitting molecule/reaction trees with `emit_routes`, and the verifier scores them against hidden patent routes from PaRoutes.

The same server is used for training, evaluation and the browser playground at `/web`.

## Read the code

- [`retroenv_openenv/environment.py`](retroenv_openenv/environment.py): the MCP tools, the Task API, `reset`, and the terminal reward.
- [`retroenv_openenv/client.py`](retroenv_openenv/client.py): `RetroEnvClient` for one episode per WebSocket session, and `RemoteRetroRouteEnv` for TRL's `environment_factory`.
- [`retroenv_openenv/agent.py`](retroenv_openenv/agent.py), [`agent_anthropic.py`](retroenv_openenv/agent_anthropic.py), [`agent_responses.py`](retroenv_openenv/agent_responses.py): model loops for OpenAI-compatible chat, the Claude Messages API and the OpenAI Responses API.
- [`retroenv_openenv/ui.py`](retroenv_openenv/ui.py): the playground.
- [`../core/retroenv`](../core/retroenv): chemistry, stock search, tasks and the verifier. Every front-end shares it.

## The episode

| Step | What happens |
|---|---|
| `reset(split, index)` | Returns the prompt, target, step cap, route count and tool names. It never returns references, patents or the stock list. |
| tool calls | MCP `call_tool`. Up to 32 calls per episode (`RETROENV_MAX_TOOL_CALLS`); `emit_routes` always stays available. |
| `emit_routes(submission)` | Scores the trees. That step returns `done=True` with the dense reward, once. |

| Tool | Answers from |
|---|---|
| `inspect_molecule`, `pubchem_lookup` | RDKit on the given SMILES |
| `stock_retrieve` | The stock: exact, InChIKey, class, SMARTS or similarity search, capped at 20 results |
| `reaction_precedent_search`, `search_literature` | Training-split reactions only |
| `validate_disconnection`, `reaction_class_lookup` | **This task's hidden references.** Removed by `RETROENV_TOOLSET=unaided`. |
| `reaction_conditions_search` | The hidden record for a matching cut (full toolset), otherwise training analogues |
| `emit_routes` | The verifier |

The Task API lists splits and public task rows without references: `GET /retro_route/splits`, and `POST /retro_route/task`, `/tasks`, `/task_range` and `/num_tasks`. The API docs are at `/docs`.

## Run locally

From the project folder (`09-retroenv`):

```bash
uv sync --extra dev --extra eval
RETROENV_TASKS_REPO=AdithyaSK/RetroEnv-RL uv run bash envs/retro_route/openenv/start.sh
```

`start.sh` runs `prepare.py`, which downloads the v3 benchmark from
[AdithyaSK/RetroEnv-RL](https://huggingface.co/datasets/AdithyaSK/RetroEnv-RL) and verifies
its checksums, then starts the server on it. Add `RETROENV_TASKS_SUBDIR=retroeval-v2` to serve
v2. To serve a local directory instead:

```bash
RETROENV_BENCHMARK_DIR=benchmark/retroeval-v3 uv run uvicorn retroenv_openenv.server:app --port 8000
```

Open <http://localhost:8000/web/> for the playground. To run one model episode against the server:

```bash
uv run python envs/retro_route/openenv/rollout.py --server http://127.0.0.1:8000 \
  --split eval --index 0 --provider anthropic --model claude-opus-5-5
```

`tasks-private/` is not in git for v2 and v3; download it from the Hub or rebuild it (see the benchmark READMEs). The committed v1 benchmark (`benchmark/retroeval-v1`) works without either.

## Docker and Spaces

The image never contains the answer key. At startup `prepare.py` uses a mounted directory or downloads a task dataset; the image defaults to `RETROENV_TASKS_REPO=AdithyaSK/RetroEnv-RL`.

```bash
python deploy.py --stage-only --stage-dir /tmp/retroenv-space
docker build -t retroenv /tmp/retroenv-space
docker run --rm -p 8000:8000 retroenv                       # v3 from the Hub
docker run --rm -p 8000:8000 -v "$PWD/benchmark/retroeval-v3:/data:ro" \
  -e RETROENV_BENCHMARK_DIR=/data retroenv                  # a local directory
```

To deploy a Space that serves the published benchmark:

```bash
python deploy.py --repo YOUR_ORG/retroenv --tasks-repo AdithyaSK/RetroEnv-RL
```

A private task dataset also works; add `HF_TOKEN` as a Space secret so the Space can read it. The Space is private by default; `--public` makes it public.

| Variable | Default | Meaning |
|---|---|---|
| `RETROENV_BENCHMARK_DIR` | none | Local benchmark directory with `tasks-private/` and `stocks/` |
| `RETROENV_TASKS_REPO` | `AdithyaSK/RetroEnv-RL` in the image | Task dataset `org/name[@revision]`, used when no directory is set |
| `RETROENV_TASKS_SUBDIR` | none | Folder inside that dataset, e.g. `retroeval-v2` |
| `RETROENV_TOOLSET` | `full` | `unaided` removes the tools that answer from hidden references |
| `RETROENV_MAX_TOOL_CALLS` | `32` | Tool budget per episode |
| `RETROENV_DEFAULT_SPLIT` | `train` | Split used by a reset that names none |
| `MAX_CONCURRENT_ENVS` | `64` | WebSocket sessions per process |
| `ENABLE_WEB_INTERFACE` | `true` | Mount the playground at `/web` |
