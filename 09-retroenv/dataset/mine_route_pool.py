#!/usr/bin/env python3
"""Mine every verified route-planning candidate from PaRoutes v2, before any sampling.

Stage 1 of the v2 benchmark build. It applies the same raw filters as
``build_training_sample.py`` (tree shape, at most ``max_steps`` reactions, every
leaf in the stock) and keeps up to five routes per target with distinct first
cuts. Each kept route is replayed through the serving verifier. Unlike the v1
sampler it keeps single-route targets too, so the pool can be sampled for
larger benchmarks. Stage 2 (``build_benchmark_v2.py``) samples and splits it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_training_sample as v1  # noqa: E402  (reuse the v1 extraction exactly)
from retroenv.chemistry import canonicalize_smiles  # noqa: E402
from retroenv.models import RetroTask  # noqa: E402
from retroenv.store import load_stock  # noqa: E402
from retroenv.verifier import RouteVerifier  # noqa: E402


def _leaves(root: dict) -> list[str]:
    out: list[str] = []

    def visit(mol: dict) -> None:
        reactions = [c for c in mol.get("children") or [] if c.get("type") == "reaction"]
        if not reactions:
            out.append(str(mol.get("smiles", "")))
            return
        for child in reactions[0].get("children") or []:
            if child.get("type") == "mol":
                visit(child)

    visit(root)
    return out


def mine(archive: Path, stock_path: Path, output: Path, max_steps: int) -> dict:
    stock = load_stock(stock_path)
    raw_stock = {
        line.split(maxsplit=1)[0]
        for line in stock_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    with gzip.open(archive, "rt", encoding="utf-8") as handle:
        roots = json.load(handle)

    # Popularity signals over the whole archive, independent of the filters.
    routes_per_target: Counter[str] = Counter()
    leaf_frequency: Counter[str] = Counter()
    for root in roots:
        routes_per_target[str(root.get("smiles"))] += 1
        leaf_frequency.update(set(_leaves(root)))

    connection, counters = v1._candidate_rows(roots, raw_stock, max_steps)
    by_target: dict[str, list[int]] = defaultdict(list)
    for target, route_index in connection.execute("SELECT target, route_index FROM routes ORDER BY target, first_cut"):
        if len(by_target[target]) < 5:  # same LIMIT 5 as the v1 sampler
            by_target[target].append(int(route_index))
    connection.close()
    counters["candidate_targets"] = len(by_target)

    verifier = RouteVerifier()
    output.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with output.open("w", encoding="utf-8") as handle:
        for raw_target in sorted(by_target):
            target = canonicalize_smiles(raw_target)
            if target in stock:
                counters["excluded_target_in_stock"] += 1
                continue
            references = []
            for route_index in by_target[raw_target]:
                try:
                    reference = v1._extract_reference(roots[route_index], route_index)
                except Exception:
                    counters["extraction_rejected"] += 1
                    continue
                replay = RetroTask(
                    task_id="generation-replay",
                    mode="route_planning",
                    target_smiles=target,
                    max_steps=max_steps,
                    stock_id="replay",
                    split="unassigned",
                    reference_routes=(reference,),
                )
                flat = {"route": [s.to_dict(include_evidence=False) for s in reference.steps]}
                if verifier.score_route(replay, flat, stock).valid:
                    references.append(reference)
                else:
                    counters["verifier_rejected_routes"] += 1
            first_cuts = {tuple(sorted(r.steps[0].reactants)) for r in references if r.steps}
            if not references:
                counters["targets_without_verified_route"] += 1
                continue
            leaves = sorted(
                {x for r in references for s in r.steps for x in s.reactants if x not in {q.product for q in r.steps}}
            )
            row = {
                "target_smiles": target,
                "reference_routes": [r.to_dict() for r in references],
                "distinct_first_cuts": len(first_cuts),
                "steps": [len(r.steps) for r in references],
                "patents": sorted({s["group_id"] for r in references for s in r.source}),
                "archive_routes_for_target": routes_per_target[raw_target],
                "leaf_archive_frequency": {x: leaf_frequency.get(x, 0) for x in leaves},
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            kept += 1
            counters["targets_kept"] += 1
            counters[f"targets_with_{min(len(first_cuts), 3)}{'+' if len(first_cuts) >= 3 else ''}_first_cuts"] += 1
    report = {
        "archive": str(archive),
        "stock": str(stock_path),
        "stock_molecules": len(stock),
        "max_steps": max_steps,
        "counters": dict(sorted(counters.items())),
    }
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/paroutes-v2-benchmark/all_loaded_routes.json.gz")
    )
    parser.add_argument("--stock-file", type=Path, default=Path("data/raw/paroutes-v2-benchmark/stock_n1.txt"))
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(".local/pool/pool-n1-s3.jsonl"))
    args = parser.parse_args(argv)
    print(json.dumps(mine(args.archive, args.stock_file, args.output, args.max_steps), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
