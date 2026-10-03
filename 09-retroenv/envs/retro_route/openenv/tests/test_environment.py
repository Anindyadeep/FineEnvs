"""The OpenEnv environment in-process: tools, terminal reward, budgets, toolsets, Task API."""

from __future__ import annotations

import json

import pytest
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from openenv_helpers import resources, settings  # noqa: F401  (resources is a fixture)
from retroenv.graph import routes_to_submission
from retroenv.tools import tool_names
from retroenv_openenv.config import Resources
from retroenv_openenv.environment import RetroRouteEnvironment


def _call(env, name, **arguments):
    return env.step(CallToolAction(tool_name=name, arguments=arguments))


def _oracle(env, split, index):
    task = env.resources.store.task(split, index)
    stock = env.resources.store.stock(task.stock_id)
    return routes_to_submission(task.target_smiles, task.reference_routes[: task.max_routes], stock, source="test")


def test_reset_reveals_no_references(resources):
    env = RetroRouteEnvironment(resources)
    opening = env.reset(split="eval", index=0)
    task = resources.store.task("eval", 0)
    text = json.dumps(opening.model_dump())
    assert opening.done is False and opening.reward is None
    assert opening.metadata["task_id"] == task.task_id
    for route in task.reference_routes:
        assert route.route_id not in text
        for source in route.source:
            assert source["group_id"] not in text
        for step in route.steps[1:]:
            assert step.product not in text


def test_tools_match_the_core_toolset_and_schemas(resources):
    env = RetroRouteEnvironment(resources)
    env.reset(split="eval", index=0)
    tools = env.step(ListToolsAction()).tools
    assert [tool.name for tool in tools] == list(tool_names("full"))
    emit = next(tool for tool in tools if tool.name == "emit_routes")
    assert "$defs" in emit.input_schema and "mol" in emit.input_schema["$defs"]


def test_emit_routes_ends_the_episode_and_reports_reward_once(resources):
    env = RetroRouteEnvironment(resources)
    env.reset(split="eval", index=2)
    first = _call(env, "emit_routes", submission=_oracle(env, "eval", 2))
    assert first.done is True and first.reward == 1.0
    assert env.state.done is True and env.state.reward == 1.0
    again = _call(env, "inspect_molecule", smiles="CCO")
    assert again.done is True and again.reward is None


def test_malformed_submission_is_scored_not_rejected(resources):
    env = RetroRouteEnvironment(resources)
    env.reset(split="eval", index=0)
    observation = _call(env, "emit_routes", submission={"routes": "not a list"})
    assert observation.error is None
    assert observation.done is True and observation.reward == 0.0


def test_budget_exhaustion_keeps_emit_available(resources):
    small = Resources(
        settings=settings(max_tool_calls=2), store=resources.store, precedent_index=resources.precedent_index
    )
    env = RetroRouteEnvironment(small)
    env.reset(split="eval", index=1)
    for _ in range(2):
        assert "error" not in _call(env, "inspect_molecule", smiles="CCO").result.data
    blocked = _call(env, "inspect_molecule", smiles="CCO").result.data
    assert "budget exhausted" in blocked["error"]
    assert _call(env, "emit_routes", submission=_oracle(env, "eval", 1)).reward == 1.0


def test_unaided_toolset_hides_the_reference_oracle(resources):
    unaided = Resources(
        settings=settings(toolset="unaided"), store=resources.store, precedent_index=resources.precedent_index
    )
    env = RetroRouteEnvironment(unaided)
    opening = env.reset(split="eval", index=0)
    names = [tool.name for tool in env.step(ListToolsAction()).tools]
    assert "validate_disconnection" not in names and "reaction_class_lookup" not in names
    assert "Validate proposed cuts" not in opening.metadata["prompt"]
    step = resources.store.task("eval", 0).reference_routes[0].steps[0]
    result = _call(
        env, "reaction_conditions_search", product_smiles=step.product, reactants=list(step.reactants)
    ).result.data
    assert result["supported"] is False and result["source"] == "training-visible analogues"


def test_task_api_serves_public_rows(resources):
    env = RetroRouteEnvironment(resources)
    splits = {row["name"]: row for row in env.list_splits()}
    assert splits["eval"]["num_tasks"] == env.num_tasks("eval") == 20
    row = env.get_task("eval", 3)
    assert row["index"] == 3 and "reference_routes" not in row
    assert len(env.get_task_range("eval", 0, 5)) == 5
    with pytest.raises(IndexError):
        env.get_task("eval", 999)


def test_reset_by_task_id_and_seed(resources):
    env = RetroRouteEnvironment(resources)
    task = resources.store.task("dev", 4)
    assert env.reset(split="dev", task_id=task.task_id).metadata["index"] == 4
    first = env.reset(split="dev", seed=7).metadata["task_id"]
    assert env.reset(split="dev", seed=7).metadata["task_id"] == first
    with pytest.raises(KeyError):
        env.reset(split="dev", task_id="retro_missing")
