from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from retroenv.chemistry import canonicalize_smiles
from retroenv.models import ReactionStep, ReferenceRoute, RetroTask
from retroenv.tools import openai_tools

_SPEC = importlib.util.spec_from_file_location(
    "retroenv_build_sft", Path(__file__).parents[1] / "dataset" / "build_sft.py"
)
assert _SPEC and _SPEC.loader
build_sft = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(build_sft)

SCHEMA = hashlib.sha256(json.dumps(openai_tools("full"), sort_keys=True).encode()).hexdigest()


def _task(task_id: str, target: str, reactants: tuple[str, ...], split: str) -> RetroTask:
    target = canonicalize_smiles(target)
    step = ReactionStep(
        product=target, reactants=tuple(canonicalize_smiles(r) for r in reactants), reaction_id=f"rxn_{task_id}"
    )
    route = ReferenceRoute(
        route_id=f"route_{task_id}",
        steps=(step,),
        source=({"name": "fixture", "group_id": f"patent_{task_id}", "license": "CC0"},),
    )
    return RetroTask(
        task_id=task_id,
        mode="route_planning",
        target_smiles=target,
        max_steps=1,
        stock_id="s",
        split=split,
        reference_routes=(route,),
    )


TRAIN = [
    _task("retro_a", "CCOC(C)=O", ("CCO", "CC(=O)O"), "train"),
    _task("retro_b", "CCNC(C)=O", ("CCN", "CC(=O)Cl"), "train"),
]
HELD = {
    "dev": _task("retro_dev", "c1ccc2ccccc2c1C(=O)O", ("c1ccc2ccccc2c1Br", "O=C=O"), "dev"),
    "eval": _task("retro_eval", "CC(C)(C)OC(=O)N1CCNCC1", ("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C", "C1CNCCN1"), "eval"),
    "stress": _task("retro_stress", "O=C(O)c1ccncc1F", ("Fc1cnccc1Br", "O=C=O"), "stress"),
}


def _write_tasks(path: Path, tasks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t.to_dict()) + "\n" for t in tasks))


@pytest.fixture
def layout(tmp_path):
    guard = tmp_path / "guard"
    for split, task in HELD.items():
        _write_tasks(guard / "tasks-private" / f"{split}.jsonl", [task])
    _write_tasks(guard / "tasks-private" / "train.jsonl", TRAIN)
    return guard


EMIT = {"submission": {"routes": [{"type": "mol", "smiles": "CCOC(C)=O", "in_stock": False, "children": []}]}}


def _chat_transcript():
    return [
        {
            "role": "assistant",
            "content": "Start.",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "stock_retrieve", "arguments": '{"query": "CCO"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "stock_retrieve", "content": '{"results": []}'},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c2", "type": "function", "function": {"name": "emit_routes", "arguments": json.dumps(EMIT)}}
            ],
        },
        {"role": "tool", "tool_call_id": "c2", "name": "emit_routes", "content": '{"done": true}'},
    ]


def _episode(task: RetroTask, attempt=0, transcript=None, **overrides):
    row = {
        "task_id": task.task_id,
        "target_smiles": task.target_smiles,
        "split": task.split,
        "attempt": attempt,
        "graded": True,
        "valid": True,
        "reward": 1.0,
        "exact_match": True,
        "auto_emitted": False,
        "submission_coerced": False,
        "errors": [],
        "tool_calls": 2,
        "turns": 2,
        "prompt": "Plan it.",
        "transcript": transcript or _chat_transcript(),
    }
    row.update(overrides)
    return row


def _run(path: Path, rows, split="train", provider="reference", label="reference-expert") -> Path:
    (path / "episodes").mkdir(parents=True)
    (path / "identity.json").write_text(
        json.dumps(
            {
                "split": split,
                "toolset": "full",
                "tool_schema_sha256": SCHEMA,
                "provider": provider,
                "label": label,
                "max_turns": 16,
            }
        )
    )
    for i, row in enumerate(rows):
        (path / "episodes" / f"{i:04d}.json").write_text(json.dumps(row))
    return path


def test_chat_episode_becomes_system_prompt_calls_and_ends_at_emit():
    messages = build_sft.to_messages(_episode(TRAIN[0]), "chat", 16)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert "16 model turns" in messages[0]["content"] and messages[1]["content"] == "Plan it."
    decoded, tools = build_sft.decode({"messages": messages, "tools": json.dumps(openai_tools("full"))})
    assert decoded[-1]["tool_calls"][0]["function"]["arguments"] == EMIT
    assert isinstance(messages[-1]["tool_calls"][0]["function"]["arguments"], str) and len(tools) == 9


def test_claude_transcript_keeps_text_drops_thinking_and_keeps_the_final_turn_notice():
    transcript = [
        {"role": "user", "content": "Plan it."},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Look up."},
                {"type": "tool_use", "id": "t1", "name": "stock_retrieve", "input": {"query": "CCO"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": '{"results": []}'},
                {"type": "text", "text": "This is your final turn."},
            ],
        },
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "emit_routes", "input": EMIT}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "{}"}]},
    ]
    messages = build_sft.to_messages({"transcript": transcript}, "anthropic", 16)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "user", "assistant"]
    assert messages[2]["content"] == "Look up." and messages[3]["name"] == "stock_retrieve"
    assert json.loads(messages[-1]["tool_calls"][0]["function"]["arguments"]) == EMIT


def test_responses_transcript_pairs_results_with_calls_in_order():
    transcript = [
        {"role": "user", "content": "Plan it."},
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "summary": []},
                {"type": "function_call", "call_id": "f1", "name": "stock_retrieve", "arguments": '{"query": "CCO"}'},
            ],
        },
        {"role": "tool", "name": "stock_retrieve", "content": '{"results": []}'},
        {
            "role": "assistant",
            "content": [
                {"type": "message", "content": [{"type": "output_text", "text": "Done."}]},
                {"type": "function_call", "call_id": "f2", "name": "emit_routes", "arguments": json.dumps(EMIT)},
            ],
        },
        {"role": "tool", "name": "emit_routes", "content": "{}"},
    ]
    messages = build_sft.to_messages({"transcript": transcript}, "responses", 16)
    assert messages[3]["tool_call_id"] == "f1" and messages[-1]["content"] == "Done."


def test_held_out_runs_and_tasks_are_errors(tmp_path, layout):
    eval_run = _run(tmp_path / "eval-run", [_episode(HELD["eval"])], split="eval")
    with pytest.raises(SystemExit, match="train split only"):
        build_sft.main(["--run", str(eval_run), "--guard-dir", str(layout), "--output", str(tmp_path / "o")])
    smuggled = _run(tmp_path / "smuggled", [_episode(HELD["eval"], split="train")])
    with pytest.raises(SystemExit, match="held-out split"):
        build_sft.main(["--run", str(smuggled), "--guard-dir", str(layout), "--output", str(tmp_path / "o")])


def test_filters_dedupe_and_run_order(tmp_path, layout):
    teacher = _run(
        tmp_path / "teacher",
        [
            _episode(TRAIN[0], transcript=[{**_chat_transcript()[0], "content": "Teacher."}, *_chat_transcript()[1:]]),
            _episode(TRAIN[1], valid=False),
        ],
        provider="hf",
        label="teacher",
    )
    expert = _run(
        tmp_path / "expert",
        [
            _episode(TRAIN[0]),  # same calls as the teacher's: a duplicate
            _episode(TRAIN[1]),
            _episode(
                TRAIN[1],
                attempt=1,
                transcript=[  # a different trajectory, over the cap of one
                    {
                        **_chat_transcript()[0],
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "stock_retrieve", "arguments": '{"query": "CCN"}'},
                            }
                        ],
                    },
                    *_chat_transcript()[1:],
                ],
            ),
            _episode(TRAIN[1], attempt=2, auto_emitted=True),
        ],
    )
    out = tmp_path / "out"
    assert (
        build_sft.main(
            [
                "--run",
                str(teacher),
                "--run",
                str(expert),
                "--guard-dir",
                str(layout),
                "--output",
                str(out),
                "--validation-fraction",
                "0",
            ]
        )
        == 0
    )
    rows = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert {row["task_id"]: row["source"] for row in rows} == {"retro_a": "teacher", "retro_b": "reference-expert"}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["guard"]["audit_passed"] and manifest["rows"] == {"train": 2, "validation": 0}
    assert manifest["dropped"] == {"duplicate_trajectory": 1, "no_emit": 1, "not_passed": 1, "over_per_task_cap": 1}
    assert (out / "README.md").read_text().startswith("---\nlicense: cc-by-4.0")


def test_validation_holdout_is_a_stable_function_of_the_task():
    picks = [build_sft.in_validation(f"retro_{i}", 0.1) for i in range(2000)]
    assert picks == [build_sft.in_validation(f"retro_{i}", 0.1) for i in range(2000)]
    assert 150 < sum(picks) < 250
