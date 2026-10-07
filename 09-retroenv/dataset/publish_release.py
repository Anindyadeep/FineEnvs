"""Publish a built release to the Hub: everything public except the held-out splits' known routes.

    uv run python -m dataset.publish_release                  # LiteFold/RetroEnv + LiteFold/RetroEnv-heldout
    uv run python -m dataset.publish_release --runs runs      # also the eval runs, into the held-out repo

The public repo gets every split's public tasks, the known routes of every split but the
held-out ones, the stock, the reaction library and the manifests. The private held-out repo
gets only tasks-private/<held-out split>.jsonl and, optionally, eval runs, whose exact
solutions would otherwise reveal held-out answers. HF_TOKEN must be able to write both.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from huggingface_hub import HfApi

HELD_OUT = ("test_id", "test_hard")
PRIVATE_ROW = "| `tasks-private/` | the same tasks with hidden known routes and difficulty labels |"


def public_card(card: str, held_out: list[str], heldout_repo: str) -> str:
    for split in held_out:
        card = re.sub(rf"  - split: {split}\n    path: tasks-private/{split}\.jsonl\n", "", card)
    if PRIVATE_ROW not in card:
        raise SystemExit("release README has no tasks-private/ row to rewrite")
    return card.replace(
        PRIVATE_ROW,
        f"| `tasks-private/` | the same tasks with known routes and difficulty labels, for every split except "
        f"{' and '.join(held_out)}, whose known routes are in the private `{heldout_repo}` |",
    )


def heldout_card(held_out: list[str], public_repo: str) -> str:
    files = "\n".join(f"- `tasks-private/{split}.jsonl`" for split in held_out)
    return f"""---
license: cc-by-4.0
---

# RetroEnv held-out known routes

The known routes and difficulty labels of the held-out splits of [{public_repo}](https://huggingface.co/datasets/{public_repo}),
kept out of the public release so they stay out of training data:

{files}

Download them into the public release to evaluate on these splits; `runs/`, when present, holds
model evaluation episodes on them.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", type=Path, default=Path("data/release/RetroEnv-RL"))
    parser.add_argument("--public-repo", default="LiteFold/RetroEnv")
    parser.add_argument("--heldout-repo", default="LiteFold/RetroEnv-heldout")
    parser.add_argument("--held-out", nargs="+", default=list(HELD_OUT))
    parser.add_argument("--runs", type=Path, help="eval run directory to publish into the held-out repo")
    parser.add_argument(
        "--squash",
        action="store_true",
        help="squash the public repo's history so earlier commits cannot leak held-out files",
    )
    args = parser.parse_args(argv)
    held_files = [f"tasks-private/{split}.jsonl" for split in args.held_out]
    missing = [name for name in held_files if not (args.release / name).exists()]
    if missing:
        raise SystemExit(f"missing from {args.release}: {missing}")

    api = HfApi()
    api.create_repo(args.public_repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=args.release,
        repo_id=args.public_repo,
        repo_type="dataset",
        ignore_patterns=["README.md", *held_files],
        delete_patterns=[*held_files, "runs/*"],
        commit_message="Release without the held-out known routes",
    )
    card = public_card((args.release / "README.md").read_text(), args.held_out, args.heldout_repo)
    api.upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.public_repo, repo_type="dataset"
    )
    if args.squash:
        api.super_squash_history(repo_id=args.public_repo, repo_type="dataset")

    api.create_repo(args.heldout_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=args.release,
        repo_id=args.heldout_repo,
        repo_type="dataset",
        allow_patterns=held_files,
        commit_message="Held-out known routes",
    )
    card = heldout_card(args.held_out, args.public_repo)
    api.upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=args.heldout_repo, repo_type="dataset"
    )
    if args.runs:
        api.upload_folder(
            folder_path=args.runs,
            path_in_repo="runs",
            repo_id=args.heldout_repo,
            repo_type="dataset",
            ignore_patterns=["*.log"],
            commit_message="Eval runs",
        )
    print(
        f"https://huggingface.co/datasets/{args.public_repo}\nhttps://huggingface.co/datasets/{args.heldout_repo} (private)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
