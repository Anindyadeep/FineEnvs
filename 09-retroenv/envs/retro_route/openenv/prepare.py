#!/usr/bin/env python3
"""Make the private task bundle available before the server starts.

The bundle is a benchmark directory (``tasks-private/``, ``stocks/`` and,
optionally, ``checksums.json``) as written by the dataset scripts. It holds
the reference routes, so it is never baked into the image:

* ``RETROENV_BENCHMARK_DIR`` set: use that directory as is (local runs).
* ``RETROENV_TASKS_REPO=org/name[@revision]``: download that private HF
  dataset into ``RETROENV_PREPARED_DIR`` (default ``prepared``) with HF_TOKEN.

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
    for name, digest in expected.items():
        path = root / name
        if not name.startswith("tasks-private/") or not path.exists():
            continue  # only the files the server reads are required
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SystemExit(f"prepare: checksum mismatch for {path}")
    print(f"prepare: verified {root}", file=sys.stderr)


def main() -> int:
    local = os.getenv("RETROENV_BENCHMARK_DIR")
    if local:
        verify(Path(local))
        return 0
    repo = os.getenv("RETROENV_TASKS_REPO")
    if not repo:
        raise SystemExit("Set RETROENV_BENCHMARK_DIR, or RETROENV_TASKS_REPO to a private task dataset.")
    from huggingface_hub import snapshot_download

    repo_id, _, revision = repo.partition("@")
    target = Path(os.getenv("RETROENV_PREPARED_DIR", "prepared"))
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision or None,
        local_dir=target,
        allow_patterns=["tasks-private/*", "stocks/*", "checksums.json", "manifest.json"],
        token=os.getenv("HF_TOKEN"),
    )
    verify(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
