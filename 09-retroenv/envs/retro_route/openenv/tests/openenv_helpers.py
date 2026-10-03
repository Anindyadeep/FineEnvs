from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from retroenv_openenv.config import Resources, Settings

# The committed v1 benchmark has private references, so tests can score oracles.
BENCHMARK = Path(__file__).resolve().parents[4] / "benchmark" / "retroeval-v1"


def settings(**overrides) -> Settings:
    values = {
        "tasks_dir": BENCHMARK / "tasks-private",
        "stocks_dir": BENCHMARK / "stocks",
        "default_split": "train",
        "max_tool_calls": 32,
        "toolset": "full",
    }
    values.update(overrides)
    return Settings(**values)


@pytest.fixture(scope="session")
def resources() -> Resources:
    if not BENCHMARK.exists():
        pytest.skip("benchmark/retroeval-v1 is not available")
    return Resources.load(settings())


def _serve(extra_env: dict[str, str]):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {**os.environ, "RETROENV_BENCHMARK_DIR": str(BENCHMARK), "MAX_CONCURRENT_ENVS": "16", **extra_env}
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "retroenv_openenv.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=1).status_code == 200:
                return process, url
        except httpx.HTTPError:
            time.sleep(0.3)
    process.kill()
    raise RuntimeError("server did not start")


@pytest.fixture(scope="session")
def server_url():
    if not BENCHMARK.exists():
        pytest.skip("benchmark/retroeval-v1 is not available")
    process, url = _serve({"ENABLE_WEB_INTERFACE": "true"})
    yield url
    process.terminate()
    process.wait(timeout=20)
