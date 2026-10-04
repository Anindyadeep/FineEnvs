#!/usr/bin/env python3
"""Make the task bundle available before the server starts.

The bundle is a benchmark directory (``tasks-private/``, ``stocks/`` and,
optionally, ``checksums.json``) as written by the dataset scripts. It holds
the reference routes, so it is never baked into the image:

* ``RETROENV_BENCHMARK_DIR`` set: use that directory as is (local runs).
* ``RETROENV_TASKS_REPO=org/name[@revision]``: download that HF dataset into
  ``RETROENV_PREPARED_DIR`` (default ``prepared``). The published benchmark is
  ``AdithyaSK/RetroEnv-RL``: v3 at its root, v2 under ``retroeval-v2/``
  (``RETROENV_TASKS_SUBDIR=retroeval-v2``). ``HF_TOKEN`` is needed only for a
  private repo.

Either way, files listed in ``checksums.json`` are verified.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path


def verify(root: Path) -> None:
    if not (root / "tasks-private").is_dir() or not (root / "stocks").is_dir():
        raise SystemExit(f"{root} needs tasks-private/ and stocks/")
    manifest = root / "checksums.json"
    if not manifest.exists():
        print(f"prepare: {root} has no checksums.json; skipping verification", file=sys.stderr)
        return
    expected = json.loads(manifest.read_text())["sha256"]
    # The tasks and the stock both decide rewards, so a missing or altered file
    # is a startup failure, not something to skip.
    required = {name for name in expected if name.startswith(("tasks-private/", "stocks/"))}
    for name in sorted(required):
        path = root / name
        if not path.exists():
            raise SystemExit(f"prepare: {path} is listed in checksums.json but missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise SystemExit(f"prepare: checksum mismatch for {path}")
    print(f"prepare: verified {len(required)} file(s) in {root}", file=sys.stderr)


def main() -> int:
    local = os.getenv("RETROENV_BENCHMARK_DIR")
    if local:
        verify(Path(local))
        print(local)
        return 0
    repo = os.getenv("RETROENV_TASKS_REPO")
    if not repo:
        raise SystemExit(
            "Set RETROENV_BENCHMARK_DIR, or RETROENV_TASKS_REPO to a task dataset (for example AdithyaSK/RetroEnv-RL)."
        )
    from huggingface_hub import snapshot_download

    repo_id, _, revision = repo.partition("@")
    subdir = os.getenv("RETROENV_TASKS_SUBDIR", "").strip("/")
    prefix = f"{subdir}/" if subdir else ""
    download = Path(os.getenv("RETROENV_PREPARED_DIR", "prepared"))
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision or None,
        local_dir=download,
        allow_patterns=[
            f"{prefix}{name}" for name in ("tasks-private/*", "stocks/*", "checksums.json", "manifest.json")
        ],
        token=os.getenv("HF_TOKEN") or None,
    )
    target = download / subdir if subdir else download
    verify(target)
    # start.sh serves whatever this prints last.
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
