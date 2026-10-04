# RetroEnv explorer

A local browser for everything RetroEnv has on disk: benchmark tasks, SFT datasets
and model runs.

```bash
hf download AdithyaSK/RetroEnv-RL --repo-type dataset --local-dir benchmark/retroeval-v3 \
  --exclude "retroeval-v2/*" "runs/*" "media/*"                     # the RL tasks
hf download AdithyaSK/RetroEnv-SFT --repo-type dataset --local-dir .local/sft  # the SFT exports
uv run --extra eval python explorer/server.py     # then open http://127.0.0.1:8050
```

Model runs are read from `runs/`; the v2 board's are in the RL dataset under `runs/v2-eval/`.

| View | What it shows |
|---|---|
| Environment | The RL side: tasks per split and what each split is for, what one episode looks like, the chosen split's step budget, first reaction, tier and target size (click a bar to list those tasks), the 9 tools and the reward components |
| RL tasks | Every task, filterable by split, steps, first reaction, tier, size, route count and SFT coverage, or searched by ID or SMILES; for benchmarks with model runs, how many models passed each one |
| Task | What the policy sees (target, budgets, exact prompt), the answer key the verifier grades against, drawn as a tree, a **Try it** panel, and every model-run episode and SFT row on the task, turn by turn |
| SFT data | The SFT exports side by side (recovery rate, exactness, tool use, tokens) and a browser over their rows |
| Model runs | The evaluation runs under `runs/` and their episodes |

**Try it** plays the task through the same core session the server runs. Call any tool
with JSON arguments, validate the candidate cuts the rule library proposes, submit the
answer key or an empty route, and see the reward split into its components. The first
episode on a benchmark builds its precedent index, which takes about 30 seconds for v3.

The server finds its data itself:

- benchmarks: every `benchmark/*/tasks-private`, with its difficulty sidecar and stock
- SFT datasets: every export under `.local/sft/`
- runs: everything under `runs/` except `runs/sft`, whose generation runs hold tens of thousands of files; their exports are the datasets

It indexes each SFT file once, by byte offset, and caches the index in
`.local/explorer-cache`; a row is read only when you open it. Molecules are drawn by
RDKit on the server and follow the page's light or dark theme.

Dev, eval and stress tasks hide their reference until you choose to show it, so the
default view is what a model sees. The server listens on localhost only.
