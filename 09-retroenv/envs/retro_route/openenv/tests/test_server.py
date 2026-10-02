"""The running server: HTTP Task API, MCP over WebSocket, concurrent sessions, the UI mount."""

from __future__ import annotations

import threading

import httpx

from retroenv.graph import routes_to_submission
from retroenv.store import TaskStore
from retroenv_openenv.client import RetroEnvClient

from openenv_helpers import BENCHMARK, server_url  # noqa: F401  (server_url is a fixture)

STORE = TaskStore(BENCHMARK / "tasks-private", BENCHMARK / "stocks")


def _oracle(split, index):
    task = STORE.task(split, index)
    return routes_to_submission(task.target_smiles, task.reference_routes[: task.max_routes],
                                STORE.stock(task.stock_id), source="test")


def test_health_metadata_and_task_api(server_url):
    assert httpx.get(f"{server_url}/health").json() == {"status": "healthy"}
    assert httpx.get(f"{server_url}/metadata").json()["name"] == "retro_route"
    client = RetroEnvClient(server_url)
    assert {row["name"] for row in client.splits()} == {"train", "dev", "eval", "stress"}
    assert client.num_tasks("eval") == 20
    row = client.task("eval", 0)
    assert row["task_id"] == STORE.task("eval", 0).task_id and "reference_routes" not in row
    client.close()


def test_full_episode_over_mcp(server_url):
    with RetroEnvClient(server_url) as env:
        opening = env.reset("eval", index=5)
        assert opening["task_id"] == STORE.task("eval", 5).task_id
        assert {tool["function"]["name"] for tool in env.openai_tools()} >= {"stock_retrieve", "emit_routes"}
        lookup = env.call("stock_retrieve", {"query": opening["target_smiles"], "mode": "exact"})
        assert lookup.done is False and lookup.reward is None and "results" in lookup.result
        unknown = env.call("no_such_tool", {})
        assert unknown.error is not None
        final = env.call("emit_routes", {"submission": _oracle("eval", 5)})
        assert final.done is True and final.reward == 1.0 and final.result["score"]["valid"] is True


def test_concurrent_sessions_are_isolated(server_url):
    outcomes = {}

    def episode(index):
        with RetroEnvClient(server_url) as env:
            opening = env.reset("eval", index=index)
            reward = env.call("emit_routes", {"submission": _oracle("eval", index)}).reward
            outcomes[index] = (opening["task_id"] == STORE.task("eval", index).task_id, reward)

    threads = [threading.Thread(target=episode, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes == {i: (True, 1.0) for i in range(6)}


def test_playground_is_mounted(server_url):
    response = httpx.get(f"{server_url}/web/", follow_redirects=True, timeout=30)
    assert response.status_code == 200 and "RetroEnv" in response.text
