"""The HF Jobs launcher's plan and source upload, and the TRL environment's failure handling."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _launcher():
    spec = importlib.util.spec_from_file_location("hf_job", ROOT / "train/jobs/hf_job.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(monkeypatch, capsys, *argv):
    launcher = _launcher()
    monkeypatch.setattr(sys, "argv", ["hf_job.py", *argv])
    launcher.main()
    return json.loads(capsys.readouterr().out)


def test_the_plan_runs_one_entrypoint_action_on_two_h200s(monkeypatch, capsys):
    plan = _plan(monkeypatch, capsys, "train", "--mode", "async", "--name", "a", "--bucket", "o/b", "--steps", "100")
    assert plan["flavor"] == "h200x2" and plan["namespace"] == "o" and plan["timeout"] == "24h"
    command = plan["command"]
    assert command[:3] == ["bash", "/source/train/jobs/entrypoint.sh", "train"]
    assert command[command.index("--mode") + 1] == "async" and command[command.index("--output") + 1] == "/outputs/a"
    assert plan["data"] == "FineEnvs/retroenv-bucket"

    plan = _plan(
        monkeypatch, capsys, "eval", "--name", "e", "--bucket", "o/b", "--checkpoint", "/outputs/a/checkpoint-100"
    )
    assert "--mode" not in plan["command"] and plan["command"][-2:] == ["--checkpoint", "/outputs/a/checkpoint-100"]
    assert _plan(monkeypatch, capsys, "check", "--name", "c", "--bucket", "o/b")["flavor"] == "cpu-upgrade"


def test_the_plan_refuses_other_gpu_flavors_and_paths_as_names(monkeypatch, capsys):
    for argv in (["--flavor", "a100x4"], ["--name", "../x"]):
        with pytest.raises(SystemExit):
            _plan(monkeypatch, capsys, "smoke", "--name", "s", "--bucket", "o/b", *argv)


def test_the_source_upload_holds_the_project_and_nothing_local():
    files = {path.relative_to(ROOT).as_posix() for path in _launcher().source_files()}
    for needed in (
        "uv.lock",
        "pyproject.toml",
        "data/eval-core30.json",
        "envs/retro_route/openenv/corpus-manifest.json",
        "envs/retro_route/openenv/retroenv_openenv/server.py",
        "eval/evaluate.py",
        "train/sync_grpo.py",
        "train/jobs/entrypoint.sh",
    ):
        assert needed in files, needed
    assert any(name.startswith("envs/retro_route/openenv/retroenv_openenv/static/") for name in files)
    for name in files:
        assert not name.startswith(("runs/", "benchmark/", "data/release/", "data/build/")), name
        assert ".venv" not in name and "/prepared/" not in name and not name.endswith(".env"), name


class _DeadClient:
    """A session the server dropped: every call fails, as OpenEnv's cached socket does."""

    def __init__(self, server):
        self.server = server

    def reset(self, **_):
        raise ConnectionError("closed")

    def call(self, name, arguments):
        raise ConnectionError("closed")

    def close(self):
        pass


class _LiveClient(_DeadClient):
    """A server with a three-call budget: tools answer, emit_routes grades 0.6."""

    def reset(self, **kwargs):
        return {"prompt": "Plan a retrosynthesis for CCO.", "task_id": "retro_x", "max_tool_calls": 3, **kwargs}

    def call(self, name, arguments):
        from retroenv_openenv.client import ToolOutcome

        if name == "emit_routes":
            return ToolOutcome(name=name, result={"score": {"valid": True}}, done=True, reward=0.6, error=None)
        return ToolOutcome(name=name, result={"matches": []}, done=False, reward=None, error=None)


def test_a_broken_session_scores_nan_and_the_next_reset_reconnects(monkeypatch):
    from retroenv_openenv import client

    monkeypatch.setattr(client, "RetroEnvClient", _DeadClient)
    env = client.RemoteRetroRouteEnv("http://server")
    assert json.loads(env.stock_retrieve("CCO"))["error"].startswith("environment unavailable")
    assert math.isnan(env.get_reward())  # no grade, not a wrong answer

    monkeypatch.setattr(client, "RetroEnvClient", _LiveClient)
    assert env.reset(index=3) == "Plan a retrosynthesis for CCO."  # a fresh session after the dead one
    assert env.get_reward() == 0.0


def test_tool_results_count_down_the_budget_and_the_trace_records_each_episode(monkeypatch, tmp_path):
    from retroenv_openenv import client

    trace = tmp_path / "episodes.jsonl"
    monkeypatch.setenv("RETROENV_TRACE_PATH", str(trace))
    monkeypatch.setattr(client, "RetroEnvClient", _LiveClient)
    env = client.RemoteRetroRouteEnv("http://server")
    env.reset(index=0)
    first = json.loads(env.stock_retrieve("CCO"))
    assert first["tool_calls_remaining"] == 2 and "reminder" in first  # 2 left of 3, under REMIND_AT
    assert env.get_reward() == 0.0  # no submission: 0, as in evaluation
    env.reset(index=1)
    env.inspect_molecule("CCO")
    json.loads(env.emit_routes({"routes": []}))
    assert env.get_reward() == 0.6
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [(r["index"], r["submitted"], r["passed"], r["tool_calls"], r["reward"]) for r in records] == [
        (0, False, False, 1, 0.0),
        (1, True, True, 1, 0.6),
    ]
    assert env.submitted and env.passed and not env.failed
