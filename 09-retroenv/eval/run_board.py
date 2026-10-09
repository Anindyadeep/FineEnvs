#!/usr/bin/env python3
"""Run every model of a board through run_eval.py against one server, then summarize.

    uv run --extra eval python eval/run_board.py eval/boards/core30-nothink.json \\
        --server http://127.0.0.1:8000 --output runs/core30-nothink

A board file names an evaluation set, the split that serves it, the run_eval flags every model
shares, and per-model providers and flags (see eval/boards/). Each model runs in its own
run_eval process, writing runs/<board>/<model>/ and a log beside it; a rerun resumes every
model where it stopped. --tasks N runs only the first N tasks, as a smoke test.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", model).strip("_")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("board", type=Path)
    parser.add_argument("--server", required=True, help="URL of a running RetroEnv server")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, default=ROOT / "data" / "release" / "RetroEnv-RL")
    parser.add_argument("--parallel", type=int, default=6, help="models run at once")
    parser.add_argument("--concurrency", type=int, default=4, help="episodes at once per model")
    parser.add_argument("--tasks", type=int, help="only the first N tasks of the evaluation set (a smoke test)")
    parser.add_argument("--only", nargs="+", help="run only these models (by model id)")
    args = parser.parse_args(argv)

    board = json.loads(args.board.read_text())
    evalset = json.loads((ROOT / board["evalset"]).read_text())
    task_ids = [task["task_id"] for task in evalset["tasks"]][: args.tasks]
    args.output.mkdir(parents=True, exist_ok=True)
    ids_file = args.output / "task_ids.txt"
    ids_file.write_text("\n".join(task_ids) + "\n")
    models = [m for m in board["models"] if not args.only or m["model"] in args.only]

    def run(spec: dict) -> tuple[str, int]:
        out = args.output / slug(spec["model"])
        command = [
            sys.executable,
            str(ROOT / "eval" / "run_eval.py"),
            "--provider", spec["provider"],
            "--model", spec["model"],
            "--split", board["split"],
            "--task-ids", str(ids_file),
            "--server", args.server,
            "--benchmark-dir", str(args.benchmark_dir),
            "--concurrency", str(spec.get("concurrency", args.concurrency)),
            "--max-cost", str(spec.get("max_cost", board.get("max_cost", 25))),
            "--output", str(out),
            *board.get("common", []),
            *spec.get("args", []),
        ]  # fmt: skip
        with out.parent.joinpath(f"{out.name}.log").open("w") as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        print(f"{'done' if code == 0 else f'exit {code}'}: {spec['model']}", flush=True)
        return spec["model"], code

    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(run, models))
    failed = [model for model, code in results if code]
    finished = [
        args.output / slug(m["model"]) for m in models if (args.output / slug(m["model"]) / "summary.json").exists()
    ]
    if finished:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "eval" / "summarize.py"),
                *map(str, finished),
                "--output",
                str(args.output / "results"),
            ],
            cwd=ROOT,
        )
    print(json.dumps({"models": len(models), "failed": failed, "tasks": len(task_ids)}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
