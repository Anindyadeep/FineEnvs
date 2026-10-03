#!/usr/bin/env python3
"""Design a stratified route-planning benchmark from the stage-1 pool (v3).

v2 sampled 1,000 tasks and then split them, so its eval set inherited the pool's
skew: 2- and 3-step routes only, and mostly two reaction families. v3 designs the
held-out splits first and fills train with what is left:

1. eval, dev and stress are drawn in that order, in salted stable-hash order,
   against quotas on the shortest route length (2-5 steps), the first-step
   reaction family (``disconnections.step_family``), target heavy atoms, and
   two-route tasks. Family quotas sit halfway between the pool's mix and uniform,
   so the rarest family still gets enough tasks to report on.
2. A held-out task must touch no leakage group already taken by any split. Eval
   tasks are therefore independent of each other (no shared patent, scaffold,
   route product, reaction or near-duplicate), not only of train.
3. Every other pool target joins train unless it touches a held-out group.

Leakage groups are v2's (``build_benchmark_v2.group_keys``, Morgan Tanimoto
>= 0.90 between route molecules), with one-ring and acyclic molecules grouped by
exact structure. The output has the v2 layout, so ``audit_benchmark.py``,
``label_difficulty.py``, ``run_baselines.py`` and the server read it.

    uv run python dataset/build_benchmark_v3.py --output-dir benchmark/retroeval-v3
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_benchmark_v2 import STOCK_ID, Groups, _with_split, make_task, write  # noqa: E402
from rdkit import Chem  # noqa: E402
from retroenv.chemistry import canonicalize_smiles, scaffold_smiles, stable_hash  # noqa: E402
from retroenv.disconnections import step_family  # noqa: E402
from retroenv.models import RetroTask  # noqa: E402
from retroenv.store import load_stock  # noqa: E402
from retroenv.taskgen import audit_splits, is_single_ring_scaffold  # noqa: E402

HELD_OUT = {"eval": 250, "dev": 150, "stress": 150}
DEPTHS = {2: 0.28, 3: 0.28, 4: 0.24, 5: 0.20}
SIZES = {"<=20": 0.28, "21-30": 0.34, "31-40": 0.30, "41+": 0.08}
TWO_ROUTE = 0.10
# A multi-ring scaffold in at least this many pool tasks (biphenyl, indole,
# quinoline, ...) groups by exact structure, as one-ring scaffolds already do.
# Grouping on it would let one eval task bar hundreds of unrelated train tasks.
GENERIC_SCAFFOLD_TASKS = 100
SALT = "retroeval-v3"


def size_bin(heavy_atoms: int) -> str:
    return "<=20" if heavy_atoms <= 20 else "21-30" if heavy_atoms <= 30 else "31-40" if heavy_atoms <= 40 else "41+"


def quotas(shares: dict, total: int) -> dict:
    """Largest-remainder rounding, so the counts sum to ``total``."""
    raw = {key: share * total for key, share in shares.items()}
    counts = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)[: total - sum(counts.values())]:
        counts[key] += 1
    return counts


def describe(row: dict) -> dict:
    first = row["reference_routes"][0]["steps"][0]
    return {
        "depth": min(row["steps"]),
        "family": step_family(first["reactants"], first["product"]),
        "two_route": row["distinct_first_cuts"] >= 2,
        "scaffold": scaffold_smiles(row["target_smiles"]),
        "size": size_bin(Chem.MolFromSmiles(row["target_smiles"]).GetNumHeavyAtoms()),
    }


def generic_scaffolds(pool: list[dict], minimum: int) -> frozenset[str]:
    """Multi-ring scaffolds on a target or route product of at least ``minimum`` pool tasks."""
    counts: Counter[str] = Counter()
    for row in pool:
        molecules = {canonicalize_smiles(row["target_smiles"])} | {
            canonicalize_smiles(step["product"]) for route in row["reference_routes"] for step in route["steps"]
        }
        counts.update({scaffold_smiles(m) for m in molecules} - {""})
    return frozenset(s for s, n in counts.items() if n >= minimum and not is_single_ring_scaffold(s))


def select(
    name: str,
    total: int,
    pool: list[dict],
    features: dict,
    family_shares: dict,
    groups: Groups,
    taken: set[str],
    tasks: dict[str, RetroTask],
) -> tuple[list[str], Counter]:
    """Fill one held-out split; later passes relax the size, then the family quota."""
    want = {
        "depth": quotas(DEPTHS, total),
        "family": quotas(family_shares, total),
        "size": quotas(SIZES, total),
        "two_route": round(TWO_ROUTE * total),
    }
    have = {"depth": Counter(), "family": Counter(), "size": Counter(), "two_route": 0}
    scaffolds: Counter[str] = Counter()
    chosen: list[str] = []
    skips: Counter[str] = Counter()

    def offer(row: dict, check: tuple[str, ...]) -> None:
        target = row["target_smiles"]
        f = features[target]
        size = f["size"]
        if have["depth"][f["depth"]] >= want["depth"].get(f["depth"], 0):
            return
        if "family" in check and have["family"][f["family"]] >= want["family"][f["family"]]:
            return
        if "size" in check and have["size"][size] >= want["size"][size]:
            return
        cap = 1 if not is_single_ring_scaffold(f["scaffold"]) else 2
        if scaffolds[f["scaffold"]] >= cap:
            skips["scaffold_cap"] += 1
            return
        task = tasks.get(target) or make_task(row)
        tasks[target] = task
        contacts = groups.contacts(task)
        if contacts[2] or contacts[3]:  # shares a group with a task some split already holds
            skips["touches_held_out_group"] += 1
            return
        groups.admit(task, name, contacts)
        taken.add(target)
        chosen.append(target)
        have["depth"][f["depth"]] += 1
        have["family"][f["family"]] += 1
        have["size"][size] += 1
        have["two_route"] += f["two_route"]
        scaffolds[f["scaffold"]] += 1

    # Rare strata choose first, so common families cannot use up the depth
    # slots before a rare one is reached; hash order decides within a stratum.
    ordered = [row for row in pool if row["target_smiles"] not in taken]
    count = Counter(v for t in features for v in (features[t]["family"], features[t]["size"]))
    ordered.sort(
        key=lambda row: min(
            count[features[row["target_smiles"]]["family"]], count[features[row["target_smiles"]]["size"]]
        )
    )
    two_route = [row for row in ordered if features[row["target_smiles"]]["two_route"]]
    for row in two_route:
        if have["two_route"] >= want["two_route"]:
            break
        if row["target_smiles"] not in taken:
            offer(row, ())
    for check in (("family", "size"), ("family",), ()):
        for row in ordered:
            if len(chosen) >= total:
                break
            if row["target_smiles"] not in taken and not features[row["target_smiles"]]["two_route"]:
                offer(row, check)
        skips[f"filled_after_{'+'.join(check) or 'depth_only'}"] = len(chosen)
    if len(chosen) < total:
        raise SystemExit(f"{name}: only {len(chosen)} of {total} tasks fit the quotas")
    return chosen, skips


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, default=Path(".local/pool/pool-n1-s5.jsonl"))
    parser.add_argument("--stock-file", type=Path, default=Path("data/raw/paroutes-v2-benchmark/stock_n1.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark/retroeval-v3"))
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument(
        "--generic-scaffold-tasks",
        type=int,
        default=GENERIC_SCAFFOLD_TASKS,
        help="multi-ring scaffolds in at least this many pool tasks group by exact structure",
    )
    parser.add_argument("--dry-run", action="store_true", help="report the design only")
    args = parser.parse_args(argv)

    pool = [json.loads(line) for line in args.pool.open(encoding="utf-8")]
    pool.sort(key=lambda row: stable_hash(f"{SALT}:{row['target_smiles']}", length=32))
    features = {row["target_smiles"]: describe(row) for row in pool}
    natural = Counter(f["family"] for f in features.values())
    family_shares = {k: 0.5 * v / len(pool) + 0.5 / len(natural) for k, v in natural.items()}

    generic = generic_scaffolds(pool, args.generic_scaffold_tasks)
    groups = Groups(
        exact_single_ring=True, threshold=args.near_duplicate_threshold, fill_split="train", generic=generic
    )
    tasks: dict[str, RetroTask] = {}
    taken: set[str] = set()
    splits: dict[str, list[str]] = {}
    report: dict = {"held_out": {}}
    for name, total in HELD_OUT.items():
        splits[name], skips = select(name, total, pool, features, family_shares, groups, taken, tasks)
        report["held_out"][name] = dict(skips)
        print(f"{name}: {len(splits[name])} tasks", file=sys.stderr)

    splits["train"] = []
    skipped = 0
    for row in pool:
        target = row["target_smiles"]
        if target in taken:
            continue
        task = tasks.get(target) or make_task(row)
        if groups.admit(task, "train"):
            tasks[target] = task
            splits["train"].append(target)
        else:
            skipped += 1
    print(f"train: {len(splits['train'])} tasks, {skipped} skipped for touching a held-out group", file=sys.stderr)

    final = [_with_split(tasks[target], name) for name, targets in splits.items() for target in targets]
    audit = audit_splits(final, exact_single_ring_scaffolds=True, generic_scaffolds=generic)
    composition = {}
    for name, targets in splits.items():
        f = [features[t] for t in targets]
        composition[name] = {
            "tasks": len(targets),
            "depth": dict(sorted(Counter(x["depth"] for x in f).items())),
            "family": dict(Counter(x["family"] for x in f).most_common()),
            "size": dict(sorted(Counter(x["size"] for x in f).items())),
            "two_route": sum(x["two_route"] for x in f),
            "stereocentres": sum(bool(tasks[t].difficulty["stereochemistry"]) for t in targets),
        }
    report.update(
        {
            "train_skipped_touching_held_out": skipped,
            "composition": composition,
            "strict_audit_passed": audit["passed"],
            "audit_overlap_kinds": sorted({o["kind"] for o in audit["overlaps"]}),
        }
    )
    print(json.dumps(report, indent=2))
    if not audit["passed"]:
        raise SystemExit("split audit failed")
    if args.dry_run:
        return 0
    stock = load_stock(args.stock_file)
    manifest = {
        "schema_version": "retro-benchmark-v3",
        "source": {
            "name": "paroutes-v2-benchmark",
            "license": "CC-BY-4.0",
            "url": "https://zenodo.org/records/7341155",
            "archive": "data/raw/paroutes-v2-benchmark/all_loaded_routes.json.gz",
            "archive_sha256": "d504e5964af1b09632d4048f6ab81d626babc5d344dff1be00ece312716f2405",
        },
        "stock": {
            "id": STOCK_ID,
            "molecules": len(stock),
            "source_path": str(args.stock_file),
            "sha256": "b8641e2028846d995953b30f7acd36e7521d0301159518d172497340a0bfb115",
        },
        "design": {
            "pool": str(args.pool),
            "salt": SALT,
            "held_out": HELD_OUT,
            "depth_shares": DEPTHS,
            "size_shares": SIZES,
            "two_route_share": TWO_ROUTE,
            "family_shares": {k: round(v, 4) for k, v in sorted(family_shares.items())},
            "held_out_independent": True,
            "exact_single_ring_scaffolds": True,
            "generic_scaffold_tasks": args.generic_scaffold_tasks,
            "generic_scaffolds": sorted(generic),
            "near_duplicate_threshold": args.near_duplicate_threshold,
            "scaffold_cap_per_split": {"multi_ring": 1, "single_ring_or_acyclic": 2},
        },
        **report,
    }
    write(args.output_dir, final, stock, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
