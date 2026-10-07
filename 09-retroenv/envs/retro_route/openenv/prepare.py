#!/usr/bin/env python3
"""Make the task bundle available before the server starts.

The bundle is a release directory (``tasks-private/``, ``stocks/``, ``library/``,
``manifest.json`` and ``checksums.json``) as written by ``dataset/build_release.py``.
It holds the hidden routes, so it is never baked into the image:

* ``RETROENV_BENCHMARK_DIR`` set: use that directory as is (local runs).
* ``RETROENV_TASKS_REPO=org/name[@revision]``: download that HF dataset into
  ``RETROENV_PREPARED_DIR`` (default ``prepared``); ``RETROENV_TASKS_SUBDIR`` selects a
  subdirectory of the repo. The public release carries the known routes of every split
  except the held-out test splits.
* ``RETROENV_HELDOUT_REPO=org/name[@revision]``: also download the held-out splits'
  ``tasks-private/`` rows from that (private) dataset into the same directory.
  ``HF_TOKEN`` is needed only for a private repo.

Either way, files listed in ``checksums.json`` are verified. A split whose known routes
are absent is simply not served.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

BUNDLE = ("tasks-private/*", "stocks/*", "library/*", "checksums.json", "manifest.json")


def verify(root: Path) -> list[str]:
    if not all((root / name).is_dir() for name in ("tasks-private", "stocks", "library")):
        raise SystemExit(f"{root} needs tasks-private/, stocks/ and library/")
    if not any((root / "tasks-private").glob("*.jsonl")):
        raise SystemExit(f"{root}/tasks-private holds no split")
    manifest = root / "checksums.json"
    if not manifest.exists():
        print(f"prepare: {root} has no checksums.json; skipping verification", file=sys.stderr)
        return []
    expected = json.loads(manifest.read_text())["sha256"]
    # The stock and the reaction library decide every reward, so a missing or altered file
    # is a startup failure. A split's known routes may be held out, but never altered.
    required = {name for name in expected if name.startswith(("stocks/", "library/"))}
    tasks = {name for name in expected if name.startswith("tasks-private/")}
    present = required | {name for name in tasks if (root / name).exists()}
    for name in sorted(present):
        path = root / name
        if not path.exists():
            raise SystemExit(f"prepare: {path} is listed in checksums.json but missing")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[name]:
            raise SystemExit(f"prepare: checksum mismatch for {path}")
    held_out = sorted(Path(name).stem for name in tasks - present)
    print(f"prepare: verified {len(present)} file(s) in {root}", file=sys.stderr)
    if held_out:
        print(f"prepare: not serving {', '.join(held_out)} (known routes held out)", file=sys.stderr)
    return held_out


def download(repo: str, local_dir: Path, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    repo_id, _, revision = repo.partition("@")
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision or None,
        local_dir=local_dir,
        allow_patterns=patterns,
        token=os.getenv("HF_TOKEN") or None,
    )


def main() -> int:
    local = os.getenv("RETROENV_BENCHMARK_DIR")
    if local:
        verify(Path(local))
        print(local)
        return 0
    repo = os.getenv("RETROENV_TASKS_REPO")
    if not repo:
        raise SystemExit("Set RETROENV_BENCHMARK_DIR, or RETROENV_TASKS_REPO to a released task dataset on the Hub.")
    subdir = os.getenv("RETROENV_TASKS_SUBDIR", "").strip("/")
    prefix = f"{subdir}/" if subdir else ""
    prepared = Path(os.getenv("RETROENV_PREPARED_DIR", "prepared"))
    download(repo, prepared, [f"{prefix}{name}" for name in BUNDLE])
    target = prepared / subdir if subdir else prepared
    heldout = os.getenv("RETROENV_HELDOUT_REPO")
    if heldout:
        download(heldout, target, ["tasks-private/*"])
    missing = verify(target)
    # huggingface_hub falls back to the local directory when a repo is unreadable.
    if heldout and missing:
        raise SystemExit(f"prepare: {heldout} did not provide {', '.join(missing)}; can HF_TOKEN read it?")
    # start.sh serves whatever this prints last.
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
