from __future__ import annotations

import fnmatch
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "mini-release"
HELD_OUT = ("tasks-private/test_id.jsonl", "tasks-private/test_hard.jsonl")
_SPEC = importlib.util.spec_from_file_location("retroenv_prepare", ROOT / "envs/retro_route/openenv/prepare.py")
assert _SPEC and _SPEC.loader
prepare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare)


def _repo(tmp_path: Path, name: str, files: list[str]) -> Path:
    root = tmp_path / "hub" / name
    for rel in files:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIXTURE / rel, root / rel)
    return root


def _fixture_files(prefix: str = "") -> list[str]:
    return [
        str(p.relative_to(FIXTURE))
        for p in FIXTURE.rglob("*")
        if p.is_file() and str(p.relative_to(FIXTURE)).startswith(prefix)
    ]


def _fake_hub(monkeypatch, repos: dict[str, Path]) -> list[dict]:
    """A snapshot_download that copies the files of a local repo matching the patterns, and records each call."""
    calls: list[dict] = []

    def snapshot_download(repo_id, *, repo_type, revision, local_dir, allow_patterns, token):
        calls.append({"repo": repo_id, "revision": revision, "patterns": allow_patterns, "token": token})
        root = repos[repo_id]
        for path in root.rglob("*"):
            rel = str(path.relative_to(root))
            if path.is_file() and any(fnmatch.fnmatch(rel, pattern) for pattern in allow_patterns):
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, Path(local_dir) / rel)
        return str(local_dir)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    return calls


@pytest.fixture
def hub_env(monkeypatch, tmp_path):
    for name in ("RETROENV_BENCHMARK_DIR", "RETROENV_HELDOUT_REPO", "RETROENV_TASKS_SUBDIR", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("RETROENV_PREPARED_DIR", str(tmp_path / "prepared"))
    return tmp_path / "prepared"


def test_a_public_repo_downloads_without_a_token_and_prints_its_directory(tmp_path, monkeypatch, capsys, hub_env):
    calls = _fake_hub(monkeypatch, {"LiteFold/RetroEnv": _repo(tmp_path, "public", _fixture_files())})
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv@main")
    assert prepare.main() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(hub_env)
    assert calls == [
        {
            "repo": "LiteFold/RetroEnv",
            "revision": "main",
            "token": None,
            "patterns": ["tasks-private/*", "stocks/*", "library/*", "checksums.json", "manifest.json"],
        }
    ]


def test_a_release_without_held_out_routes_serves_the_other_splits(tmp_path, monkeypatch, capsys, hub_env):
    public = [f for f in _fixture_files() if f not in HELD_OUT]
    _fake_hub(monkeypatch, {"LiteFold/RetroEnv": _repo(tmp_path, "public", public)})
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv")
    assert prepare.main() == 0
    assert "not serving test_hard, test_id (known routes held out)" in capsys.readouterr().err
    assert sorted(p.name for p in (hub_env / "tasks-private").iterdir()) == ["dev.jsonl", "train.jsonl"]


def test_the_held_out_repo_adds_its_splits_with_the_token(tmp_path, monkeypatch, capsys, hub_env):
    public = [f for f in _fixture_files() if f not in HELD_OUT]
    calls = _fake_hub(
        monkeypatch,
        {
            "LiteFold/RetroEnv": _repo(tmp_path, "public", public),
            "LiteFold/RetroEnv-heldout": _repo(tmp_path, "heldout", list(HELD_OUT)),
        },
    )
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv")
    monkeypatch.setenv("RETROENV_HELDOUT_REPO", "LiteFold/RetroEnv-heldout")
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    assert prepare.main() == 0
    assert "not serving" not in capsys.readouterr().err
    assert calls[1] == {
        "repo": "LiteFold/RetroEnv-heldout",
        "revision": None,
        "patterns": ["tasks-private/*"],
        "token": "hf_test",
    }
    assert all((hub_env / name).exists() for name in HELD_OUT)


def test_an_unreadable_held_out_repo_is_a_startup_failure(tmp_path, monkeypatch, hub_env):
    public = [f for f in _fixture_files() if f not in HELD_OUT]
    _fake_hub(
        monkeypatch,
        {
            "LiteFold/RetroEnv": _repo(tmp_path, "public", public),
            "LiteFold/RetroEnv-heldout": tmp_path / "hub" / "empty",
        },
    )
    (tmp_path / "hub" / "empty").mkdir(parents=True)
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv")
    monkeypatch.setenv("RETROENV_HELDOUT_REPO", "LiteFold/RetroEnv-heldout")
    with pytest.raises(SystemExit, match="did not provide test_hard, test_id"):
        prepare.main()


def test_an_altered_task_file_still_fails(tmp_path, monkeypatch, hub_env):
    root = _repo(tmp_path, "public", _fixture_files())
    with (root / "tasks-private" / "dev.jsonl").open("a") as handle:
        handle.write("\n")
    _fake_hub(monkeypatch, {"LiteFold/RetroEnv": root})
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        prepare.main()


def test_a_subdirectory_serves_another_benchmark_from_the_same_repo(tmp_path, monkeypatch, capsys, hub_env):
    root = tmp_path / "hub" / "nested"
    shutil.copytree(FIXTURE, root / "previous-release")
    calls = _fake_hub(monkeypatch, {"LiteFold/RetroEnv": root})
    monkeypatch.setenv("RETROENV_TASKS_REPO", "LiteFold/RetroEnv")
    monkeypatch.setenv("RETROENV_TASKS_SUBDIR", "previous-release")
    assert prepare.main() == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == str(hub_env / "previous-release")
    assert all(pattern.startswith("previous-release/") for pattern in calls[0]["patterns"])


def test_without_a_source_it_says_what_to_set(monkeypatch, hub_env):
    monkeypatch.delenv("RETROENV_TASKS_REPO", raising=False)
    with pytest.raises(SystemExit, match="RETROENV_TASKS_REPO"):
        prepare.main()
