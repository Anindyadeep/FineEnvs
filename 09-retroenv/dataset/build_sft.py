#!/usr/bin/env python3
"""Turn RetroEnv episode runs into a chat-format SFT dataset.

Input is one or more ``eval/run_eval.py`` output directories on the train split,
from the scripted expert (``--provider reference``) or a teacher model. Output
is ``train.jsonl`` and ``validation.jsonl`` with ``{"messages", "tools", ...}``
rows in the OpenAI/TRL tool-calling layout, plus ``tools.json``,
``manifest.json`` and a dataset card.

Every row is an episode the server's verifier passed, cut after its
``emit_routes`` call. Three guards keep the benchmark clean:

1. Only train-split runs are read. A dev, eval or stress run is an error.
2. No task ID or target may appear in the guard benchmark's held-out splits.
3. The train tasks used, together with the guard's held-out tasks, must pass
   the strict split audit (``taskgen.audit_splits``): no shared target,
   scaffold, route product, reaction or patent.

Tool-call arguments and ``tools`` are JSON strings. Arrow merges the argument
objects of different tools into one struct and fills the gaps with nulls, which
a chat template would then render into the training text; a string survives
the round trip. ``decode`` below turns a row back into objects for
``apply_chat_template``.

    uv run python dataset/build_sft.py --run runs/sft/reference-train \\
        --train-tasks .local/retroeval-v2-sft/tasks-private/train.jsonl \\
        --output .local/sft/retroenv-sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from retroenv.chemistry import canonicalize_smiles
from retroenv.models import RetroTask
from retroenv.taskgen import SPLITS, audit_rules, audit_splits
from retroenv.tools import openai_tools
from retroenv_openenv.agent import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = tuple(split for split in SPLITS if split != "train")
# Transcript layout by run_eval provider; every other provider uses chat completions.
LAYOUTS = {"anthropic": "anthropic", "openai": "responses"}


def read_tasks(path: Path) -> list[RetroTask]:
    with path.open(encoding="utf-8") as handle:
        return [RetroTask.from_dict(json.loads(line)) for line in handle if line.strip()]


def _chat(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not row.get("prompt"):
        raise ValueError("chat-completions episode has no stored prompt; rerun it with the current run_eval.py")
    messages: list[dict[str, Any]] = [{"role": "user", "content": row["prompt"]}]
    for entry in row["transcript"]:
        if entry["role"] == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": entry.get("content") or ""}
            if entry.get("tool_calls"):
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]},
                    }
                    for call in entry["tool_calls"]
                ]
            messages.append(message)
        elif entry["role"] == "tool":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry["tool_call_id"],
                    "name": entry["name"],
                    "content": entry["content"],
                }
            )
        else:
            messages.append({"role": "user", "content": entry["content"]})
    return messages


def _anthropic(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    names: dict[str, str] = {}
    for entry in row["transcript"]:
        content = entry["content"]
        if entry["role"] == "assistant":
            # Thinking blocks are dropped: they are empty in stored Claude 5.x
            # transcripts, and the system prompt asks for no chain-of-thought.
            message: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(block["text"] for block in content if block.get("type") == "text"),
            }
            calls = [block for block in content if block.get("type") == "tool_use"]
            if calls:
                message["tool_calls"] = [
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {"name": block["name"], "arguments": json.dumps(block["input"])},
                    }
                    for block in calls
                ]
                names.update((block["id"], block["name"]) for block in calls)
            messages.append(message)
        elif isinstance(content, str):
            messages.append({"role": "user", "content": content})
        else:
            notes = []
            for block in content:
                if block.get("type") == "tool_result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "name": names[block["tool_use_id"]],
                            "content": block["content"],
                        }
                    )
                elif block.get("type") == "text":
                    notes.append(block["text"])
            if notes:  # the final-turn notice rides on the last tool-result turn
                messages.append({"role": "user", "content": "\n".join(notes)})
    return messages


def _responses(row: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending: list[str] = []
    for entry in row["transcript"]:
        if entry["role"] == "assistant":
            text, calls = "", []
            for item in entry["content"]:
                if item.get("type") == "message":
                    text += "".join(
                        part.get("text", "")
                        for part in item.get("content", [])
                        if part.get("type") in ("output_text", "text")
                    )
                elif item.get("type") == "function_call":
                    calls.append(item)
            message: dict[str, Any] = {"role": "assistant", "content": text}
            if calls:
                message["tool_calls"] = [
                    {
                        "id": item["call_id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": item["arguments"]},
                    }
                    for item in calls
                ]
                pending.extend(item["call_id"] for item in calls)
            messages.append(message)
        elif entry["role"] == "tool":
            # Results are stored in call order without IDs.
            messages.append(
                {"role": "tool", "tool_call_id": pending.pop(0), "name": entry["name"], "content": entry["content"]}
            )
        else:
            messages.append({"role": "user", "content": entry["content"]})
    return messages


CONVERTERS = {"chat": _chat, "anthropic": _anthropic, "responses": _responses}


def to_messages(row: dict[str, Any], layout: str, max_turns: int) -> list[dict[str, Any]]:
    """System prompt, then the episode up to and including the emit_routes call."""
    messages = CONVERTERS[layout](row)
    emits = [
        i
        for i, message in enumerate(messages)
        if any(call["function"]["name"] == "emit_routes" for call in message.get("tool_calls") or [])
    ]
    if not emits:
        raise ValueError("episode has no emit_routes call")
    system = {"role": "system", "content": SYSTEM_PROMPT.format(max_turns=max_turns)}
    return [system, *messages[: emits[-1] + 1]]


def drop_reason(row: dict[str, Any]) -> str | None:
    if not row.get("graded"):
        return "ungraded"
    if not row.get("valid"):
        return "not_passed"
    if row.get("auto_emitted"):
        return "no_emit"
    if row.get("submission_coerced"):
        return "coerced_submission"
    if row.get("errors"):
        return "errors"
    return None


def decode(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(messages, tools)`` with JSON objects in place of the stored strings."""
    messages = []
    for message in row["messages"]:
        message = dict(message)
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {**call, "function": {**call["function"], "arguments": json.loads(call["function"]["arguments"])}}
                for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages, json.loads(row["tools"])


def in_validation(task_id: str, fraction: float) -> bool:
    bucket = int(hashlib.sha256(task_id.encode()).hexdigest()[:12], 16) / 16**12
    return bucket < fraction


def _signature(messages: list[dict[str, Any]]) -> str:
    calls = [
        (call["function"]["name"], call["function"]["arguments"])
        for message in messages
        for call in message.get("tool_calls") or []
    ]
    return hashlib.sha256(json.dumps(calls).encode()).hexdigest()


def load_runs(
    runs: Iterable[Path], train: dict[str, RetroTask], held_ids: set[str], held_targets: set[str], drops: Counter[str]
) -> list[dict[str, Any]]:
    candidates = []
    for order, run in enumerate(runs):
        identity = json.loads((run / "identity.json").read_text())
        if identity["split"] != "train":
            raise SystemExit(f"{run} is a {identity['split']!r} run; SFT data comes from the train split only")
        toolset = identity["toolset"]
        tools = openai_tools(toolset)
        schema = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
        if identity["tool_schema_sha256"] != schema:
            raise SystemExit(f"{run} was served different tool schemas than retroenv.tools; rebuild from this checkout")
        layout = LAYOUTS.get(identity["provider"], "chat")
        tools_json = json.dumps(tools, sort_keys=True)
        for path in sorted((run / "episodes").glob("*.json")):
            row = json.loads(path.read_text())
            task_id = row["task_id"]
            target = canonicalize_smiles(row["target_smiles"])
            if row.get("split") != "train" or task_id in held_ids or target in held_targets:
                raise SystemExit(f"{path}: {task_id} belongs to a held-out split")
            if task_id not in train:
                raise SystemExit(f"{path}: {task_id} is not in --train-tasks; pass the file the run was built from")
            reason = drop_reason(row)
            if reason:
                drops[reason] += 1
                continue
            messages = to_messages(row, layout, identity["max_turns"])
            task = train[task_id]
            candidates.append(
                {
                    "order": order,
                    "row": {
                        "id": f"{task_id}:{identity['label']}:a{row['attempt']}",
                        "task_id": task_id,
                        "target_smiles": row["target_smiles"],
                        "source": identity["label"],
                        "provider": identity["provider"],
                        "toolset": toolset,
                        "max_steps": task.max_steps,
                        "kind": "two_route" if task.min_routes >= 2 else "single_route",
                        "reward": row["reward"],
                        "exact_match": row["exact_match"],
                        "tool_calls": row["tool_calls"],
                        "turns": row["turns"],
                        "messages": messages,
                        "tools": tools_json,
                    },
                }
            )
    return candidates


def select(
    candidates: list[dict[str, Any]], per_task: int, drops: Counter[str], per_steps: dict[int, int] | None = None
) -> list[dict[str, Any]]:
    """Up to ``per_task`` distinct trajectories per task (``per_steps`` overrides it by route
    length); earlier --run arguments win ties."""
    by_task: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_task.setdefault(candidate["row"]["task_id"], []).append(candidate)
    selected = []
    for task_id in sorted(by_task):
        # A stable hash breaks the remaining ties. Preferring fewer tool calls would
        # systematically drop the trajectories that recover from a rejected cut.
        ranked = sorted(
            by_task[task_id],
            key=lambda c: (
                c["order"],
                not c["row"]["exact_match"],
                -c["row"]["reward"],
                hashlib.sha256(c["row"]["id"].encode()).hexdigest(),
            ),
        )
        seen: set[str] = set()
        kept = 0
        cap = (per_steps or {}).get(ranked[0]["row"]["max_steps"], per_task)
        for candidate in ranked:
            signature = _signature(candidate["row"]["messages"])
            if signature in seen:
                drops["duplicate_trajectory"] += 1
                continue
            if kept == cap:
                drops["over_per_task_cap"] += 1
                continue
            seen.add(signature)
            kept += 1
            selected.append(candidate["row"])
    return selected


def behaviour(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the rows teach: recoveries from rejected cuts, tool use, and how varied the text is."""
    tools: Counter[str] = Counter()
    texts: set[str] = set()
    recovered = 0
    for row in rows:
        names = {}
        rejected = False
        for message in row["messages"]:
            for call in message.get("tool_calls") or []:
                tools[call["function"]["name"]] += 1
                names[call["id"]] = call["function"]["name"]
            if message["role"] == "assistant":
                texts.add(message.get("content") or "")
            elif message["role"] == "tool" and names.get(message.get("tool_call_id")) == "validate_disconnection":
                rejected = rejected or not json.loads(message["content"]).get("valid", True)
        recovered += rejected
    return {
        "rows_with_a_rejected_cut": round(recovered / len(rows), 4),
        "tool_calls": dict(tools.most_common()),
        "distinct_assistant_texts": len(texts),
    }


def token_stats(rows: list[dict[str, Any]], tokenizer_name: str) -> dict[str, Any]:
    """Render every row with the tokenizer's chat template; fails loudly if a row does not fit it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    lengths = []
    for row in rows:
        messages, tools = decode(row)
        # Render, then tokenize: tokenize=True returns a list or a BatchEncoding
        # depending on the transformers version.
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
    lengths.sort()

    def quantile(q: float) -> int:
        return lengths[min(len(lengths) - 1, int(q * len(lengths)))]

    return {
        "tokenizer": tokenizer_name,
        "rows": len(lengths),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": lengths[-1],
        "total": sum(lengths),
    }


def _relative(path: Path) -> str:
    """A path relative to the project, so a published manifest carries no local directories."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


CARD = """---
license: cc-by-4.0
task_categories:
- text-generation
tags:
- chemistry
- retrosynthesis
- tool-calling
- agent
- sft
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: validation.jsonl
---

# RetroEnv SFT trajectories

Multi-turn tool-calling episodes from RetroEnv (FineEnvs `09-retroenv`): plan a
retrosynthesis of a target molecule back to purchasable building blocks, using the
environment's tools, and finish with `emit_routes`. Every row is an episode that the
environment's verifier passed.

| | |
|---|---|
| Rows | {train} train, {validation} validation |
| Tasks | {tasks} distinct targets, train split only |
| Sources | {sources} |
| Toolset | {toolsets} |
| Route length | {depths} |
| Exact patent route | {exact} |
| Tokens per row | {tokens} |

## Row layout

`messages` is an OpenAI-style chat: a system prompt, the task prompt, then
alternating assistant tool calls and tool results, ending with the assistant's
`emit_routes` call. Tool results are verbatim server output, including
`model_turns_remaining`, so the format matches what a policy sees during
evaluation. `tools` holds the nine function schemas.

Tool-call `arguments` and `tools` are JSON strings, so that Arrow does not merge
different tools' arguments into one struct with null-filled gaps. Decode both before
`apply_chat_template`:

```python
import json
from datasets import load_dataset

ds = load_dataset("json", data_files={{"train": "train.jsonl"}})["train"]

def decode(row):
    messages = []
    for m in row["messages"]:
        m = {{k: v for k, v in m.items() if v is not None}}
        if m.get("tool_calls"):
            m["tool_calls"] = [{{**c, "function": {{**c["function"],
                "arguments": json.loads(c["function"]["arguments"])}}}} for c in m["tool_calls"]]
        messages.append(m)
    return messages, json.loads(row["tools"])
```

## How the episodes were made

`reference-expert` rows come from a scripted chemist that knows the task's patent route.
It acts through the same OpenEnv server and agent loop as any evaluated model, and plans
the way a chemist does:

1. Most episodes start by inspecting the target and searching the training precedents
   for close analogues.
2. It proposes disconnections in order. In about four episodes in ten the first is a
   plausible cut the patent did not make, such as an alkylation instead of the Suzuki
   coupling. `validate_disconnection` rejects it, and the expert moves on.
3. Each cut is validated and every new piece is looked up by exact `stock_retrieve`. A
   piece the stock lacks gets its own disconnection; one it holds stays a leaf, even
   where the patent made it, so some routes are shorter than the reference.
4. The submitted tree names each reaction, its reagent roles and its evidence.

Its text cites only tool results and chemistry read from the structures (reaction
families, reagent roles, strategic bonds; `retroenv.disconnections`). It cites a precedent
only when its reaction type matches, and it never writes the task's own patent number,
reaction IDs or reagents, because a student cannot observe them. Teacher-model rows, if
any, are that model's own episodes and were kept only if they passed.

Tasks come from the PaRoutes v2 archive, mined and filtered as described in the
RetroEnv README. None shares a target, scaffold, route product, reaction or patent with
the benchmark's dev, eval or stress tasks, and new tasks were admitted only if no route
molecule is a Morgan near-duplicate (Tanimoto at least 0.90) of a held-out one.
`manifest.json` records the audit, every filter count, and how often rows recover from a
rejected cut.

## Limitations

- Rejected cuts come from a rule library of strategic bonds, so the mistakes are
  textbook alternatives rather than the varied errors a model makes. Self-generated or
  teacher episodes add those.
- Text is drawn from short templates with variants. It is factual, not free-form
  reasoning.
- `validate_disconnection` answers from the task's hidden reference. Rows built with
  the `full` toolset teach a policy to lean on it: right for the benchmark's default,
  wrong for a deployment without that oracle. `run_eval.py --toolset unaided` builds
  rows without it.

## License and attribution

CC-BY-4.0. Routes derive from PaRoutes v2 (Genheden & Bjerrum, Zenodo 7341155,
CC-BY-4.0), which extracts reactions from USPTO patents.
"""


def write_card(path: Path, manifest: dict[str, Any]) -> None:
    def tally(counter: dict[str, int]) -> str:
        return ", ".join(f"{name} ({count})" for name, count in sorted(counter.items()))

    path.write_text(
        CARD.format(
            train=manifest["rows"]["train"],
            validation=manifest["rows"]["validation"],
            tasks=manifest["tasks"],
            sources=tally(manifest["by_source"]),
            toolsets=tally(manifest["by_toolset"]),
            depths=", ".join(f"{k}-step: {v}" for k, v in sorted(manifest["by_max_steps"].items())),
            exact=f"{manifest['exact_route_rate']:.1%} (the rest stop early at a purchasable intermediate)",
            tokens=(
                f"p50 {t['p50']:,}, p95 {t['p95']:,}, max {t['max']:,} ({t['tokenizer']} template)"
                if (t := manifest.get("tokens"))
                else "not measured"
            ),
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="run_eval.py output directory; repeatable, earlier runs win per-task ties",
    )
    parser.add_argument(
        "--train-tasks",
        type=Path,
        action="append",
        help="private train.jsonl the runs were built from (default: the v2 benchmark's)",
    )
    parser.add_argument(
        "--guard-dir",
        type=Path,
        default=ROOT / "benchmark" / "retroeval-v2",
        help="benchmark whose dev/eval/stress tasks must not leak",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-task", type=int, default=1, help="distinct trajectories kept per task")
    parser.add_argument(
        "--per-task-by-steps",
        default="",
        help="override --per-task by route length, e.g. '4=2,5=2' to upweight long routes",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--tokenizer", help="render every row with this tokenizer's chat template and report lengths")
    args = parser.parse_args(argv)

    train_files = args.train_tasks or [args.guard_dir / "tasks-private" / "train.jsonl"]
    train = {task.task_id: task for path in train_files for task in read_tasks(path)}
    held = [task for split in HELD_OUT for task in read_tasks(args.guard_dir / "tasks-private" / f"{split}.jsonl")]
    held_ids = {task.task_id for task in held}
    held_targets = {canonicalize_smiles(task.target_smiles) for task in held}

    drops: Counter[str] = Counter()
    candidates = load_runs(args.run, train, held_ids, held_targets, drops)
    per_steps = {int(k): int(v) for k, v in (item.split("=") for item in args.per_task_by_steps.split(",") if item)}
    rows = select(candidates, args.per_task, drops, per_steps)
    if not rows:
        raise SystemExit("no episode passed the filters")

    used = [train[task_id] for task_id in sorted({row["task_id"] for row in rows})]
    # The guard's own scaffold rules; without a manifest, v2's.
    manifest_path = args.guard_dir / "manifest.json"
    rules = (
        audit_rules(json.loads(manifest_path.read_text()))
        if manifest_path.exists()
        else {"exact_single_ring_scaffolds": True, "generic_scaffolds": frozenset()}
    )
    audit = audit_splits([*used, *held], **rules)
    if not audit["passed"]:
        raise SystemExit("leakage audit failed: " + json.dumps(audit["overlaps"][:3]))

    splits = {"train": [], "validation": []}
    for row in rows:
        splits["validation" if in_validation(row["task_id"], args.validation_fraction) else "train"].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        with (args.output / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in members:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    toolsets = sorted({row["toolset"] for row in rows})
    (args.output / "tools.json").write_text(
        json.dumps({name: openai_tools(name) for name in toolsets}, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": "retro-sft-v1",
        "rows": {name: len(members) for name, members in splits.items()},
        "tasks": len(used),
        "by_source": dict(Counter(row["source"] for row in rows)),
        "by_toolset": dict(Counter(row["toolset"] for row in rows)),
        "by_kind": dict(Counter(row["kind"] for row in rows)),
        "by_max_steps": {str(k): v for k, v in sorted(Counter(row["max_steps"] for row in rows).items())},
        "exact_route_rate": round(sum(row["exact_match"] for row in rows) / len(rows), 4),
        "mean_tool_calls": round(sum(row["tool_calls"] for row in rows) / len(rows), 2),
        "behaviour": behaviour(rows),
        "dropped": dict(sorted(drops.items())),
        "candidates": len(candidates),
        "runs": [_relative(run) for run in args.run],
        "train_tasks": [_relative(path) for path in train_files],
        "guard": {
            "dir": _relative(args.guard_dir),
            "held_out_tasks": len(held),
            "audit_passed": audit["passed"],
            "exact_single_ring_scaffolds": rules["exact_single_ring_scaffolds"],
            "generic_scaffolds": len(rules.get("generic_scaffolds", ())),
        },
        "per_task": args.per_task,
        "per_task_by_steps": per_steps,
        "validation_fraction": args.validation_fraction,
        "generator_commit": _git_head(),
        "license": "CC-BY-4.0",
        "attribution": "PaRoutes v2, Genheden & Bjerrum, https://zenodo.org/records/7341155 (CC-BY-4.0)",
    }
    if args.tokenizer:
        manifest["tokens"] = token_stats(rows, args.tokenizer)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_card(args.output / "README.md", manifest)
    print(
        json.dumps(
            {k: manifest[k] for k in ("rows", "tasks", "by_source", "dropped", "exact_route_rate", "behaviour")},
            indent=2,
        ),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
