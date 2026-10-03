#!/usr/bin/env python3
"""Sample and split a larger route-planning benchmark from the stage-1 pool.

Stage 2 of the v2 build (stage 1 is ``mine_route_pool.py``). Selection order:

1. Every v1 task whose target survives stage 1, pinned to its v1 split.
2. Every other target with >= 2 verified distinct first cuts.
3. Single-route targets in stable-hash order until ``--tasks`` is reached, with
   at most one task per patent, at most one per multi-ring scaffold, a
   proportional cap for single-ring and acyclic scaffolds, and equal shares of
   2- and 3-step routes.

Leakage groups use the keys of ``taskgen.assign_strict_splits`` (target, target
scaffold, route, route products and their scaffolds, reaction, patent, Morgan
Tanimoto >= 0.90 between route products). One change by default: one-ring and
acyclic molecules group by exact structure, not by scaffold
(``taskgen.scaffold_group_key``). On 1,000 tasks the v1 rule puts every
benzene-only intermediate in one 254-task group, so train gets 35% of them and
eval none; ``--strict-scaffolds`` restores the v1 rule.

Groups are tracked while tasks are admitted: a candidate that would merge two
groups already pinned to different v1 splits is skipped. Unpinned groups are
then assigned with the same largest-first, largest-deficit rule as v1. Output
has the v1 layout, so ``audit_benchmark.py`` (with
``--exact-single-ring-scaffolds``), ``run_baselines.py`` and the server read it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from retroenv.chemistry import canonicalize_smiles, inspect_molecule, scaffold_smiles, stable_hash
from retroenv.models import ReferenceRoute, RetroTask
from retroenv.store import load_stock
from retroenv.taskgen import SPLITS, audit_splits, is_single_ring_scaffold, scaffold_group_key, write_tasks

STOCK_ID = "paroutes-v2-n1"
FINGERPRINTS = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def make_task(row: dict, split: str = "unassigned") -> RetroTask:
    target = row["target_smiles"]
    routes = tuple(ReferenceRoute.from_dict(r) for r in row["reference_routes"])
    first_cuts = {tuple(sorted(r.steps[0].reactants)) for r in routes}
    details = inspect_molecule(target)
    return RetroTask(
        # Same ID formula as build_training_sample.py, so surviving v1 tasks keep their IDs.
        task_id=stable_hash(f"sample-route:{target}", prefix="retro_", length=20),
        mode="route_planning",
        target_smiles=target,
        max_steps=max(len(r.steps) for r in routes),
        stock_id=STOCK_ID,
        split=split,
        reference_routes=routes,
        min_routes=min(2, len(first_cuts)),
        max_routes=min(5, len(routes)),
        difficulty={
            "depth": min(len(r.steps) for r in routes),
            "max_reference_depth": max(len(r.steps) for r in routes),
            "heavy_atoms": details["heavy_atoms"],
            "stereochemistry": details["chiral_centres"] > 0,
            "scaffold": details["murcko_scaffold"],
            "reference_alternatives": len(routes),
            "distinct_first_cuts": len(first_cuts),
        },
    )


def group_keys(
    task: RetroTask, exact_single_ring: bool, generic: frozenset[str] = frozenset()
) -> list[tuple[str, str]]:
    """The keys of taskgen.assign_strict_splits, with the optional exact-structure scaffold rules."""

    def scaffold_key(kind: str, smiles: str) -> tuple[str, str]:
        return (kind, scaffold_group_key(smiles, exact_single_ring=exact_single_ring, generic=generic))

    keys = [("target", canonicalize_smiles(task.target_smiles)), scaffold_key("scaffold", task.target_smiles)]
    for route in task.reference_routes:
        keys.append(("route", route.route_id))
        for step in route.steps:
            product = canonicalize_smiles(step.product)
            keys.append(("route_product", product))
            keys.append(scaffold_key("route_product_scaffold", product))
            if step.reaction_id:
                keys.append(("reaction", step.reaction_id))
        for source in route.source:
            if source.get("group_id"):
                keys.append((f"source:{source.get('name', '')}", str(source["group_id"])))
    return keys


class Groups:
    """Incremental union-find over admitted tasks, with optional split pins per group."""

    def __init__(
        self,
        exact_single_ring: bool,
        threshold: float,
        fill_split: str | None = None,
        generic: frozenset[str] = frozenset(),
    ):
        self.exact_single_ring, self.threshold, self.generic = exact_single_ring, threshold, generic
        # In fill mode every new task joins ``fill_split``, so near-duplicates only
        # matter against tasks pinned elsewhere; skipping the rest keeps 50k builds fast.
        self.fill_split = fill_split
        self.parent: list[int] = []
        self.pin: dict[int, str] = {}
        self.keyed: dict[tuple[str, str], int] = {}
        self.fps: list = []
        self.owners: list[int] = []
        self.near_duplicate_edges = 0

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def contacts(self, task: RetroTask) -> tuple[list, list, set[int], set[int]]:
        """The task's keys and fingerprints, and the groups it touches by key and by near-duplicate."""
        keys = group_keys(task, self.exact_single_ring, self.generic)
        products = sorted(
            {canonicalize_smiles(s.product) for r in task.reference_routes for s in r.steps}
            | {canonicalize_smiles(task.target_smiles)}
        )
        fps = [FINGERPRINTS.GetFingerprint(Chem.MolFromSmiles(p)) for p in products]
        roots = {self.find(self.keyed[k]) for k in keys if k in self.keyed}
        near = set()
        if self.threshold > 0 and self.fps:
            for fp in fps:
                sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self.fps))
                for j in np.nonzero(sims >= self.threshold)[0]:
                    near.add(self.find(self.owners[j]))
        return keys, fps, roots, near

    def admit(self, task: RetroTask, pin: str | None = None, contacts: tuple | None = None) -> bool:
        keys, fps, roots, near = contacts or self.contacts(task)
        touched = roots | near
        pins = {self.pin[r] for r in touched if r in self.pin} | ({pin} if pin else set())
        if len(pins) > 1:
            return False
        index = len(self.parent)
        self.parent.append(index)
        self.near_duplicate_edges += len(near - roots)
        for root in touched:
            self.parent[self.find(root)] = index
            self.pin.pop(root, None)
        if pins:
            self.pin[index] = pins.pop()
        for key in keys:
            self.keyed.setdefault(key, index)
        if self.fill_split is None or pin != self.fill_split:
            self.fps.extend(fps)
            self.owners.extend([index] * len(fps))
        return True

    def components(self) -> list[list[int]]:
        members: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            members[self.find(i)].append(i)
        return list(members.values())


def build(
    pool: list[dict],
    v1_splits: dict[str, str],
    n_total: int,
    ratios,
    exact_single_ring: bool,
    threshold: float,
    *,
    max_per_patent: int = 1,
    max_per_scaffold: int = 1,
    step_quota: bool = True,
    fill_split: str | None = None,
):
    """Caps of 0 mean unlimited. Without the step quota, routes keep the pool's natural length mix.

    ``v1_splits`` maps pinned targets to their split (the v1 benchmark by default).
    With ``fill_split``, every new task is pinned to that split, so a candidate
    that touches a group of another split is skipped instead of moving the group.
    """
    counters: Counter[str] = Counter()
    groups = Groups(exact_single_ring, threshold, fill_split)
    tasks: list[RetroTask] = []
    tags: list[str] = []
    by_target = {row["target_smiles"]: row for row in pool}
    scaffold_of = {row["target_smiles"]: scaffold_smiles(row["target_smiles"]) for row in pool}
    pool_scaffold = Counter(scaffold_of.values())
    patent_use: Counter[str] = Counter()
    scaffold_use: Counter[str] = Counter()

    def offer(row: dict, tag: str, pin: str | None = None) -> bool:
        task = make_task(row)
        if not groups.admit(task, pin):
            counters[f"skip_{tag}_pin_conflict"] += 1
            return False
        tasks.append(task)
        tags.append(tag)
        patent_use.update(row["patents"])
        scaffold_use[scaffold_of[row["target_smiles"]]] += 1
        counters[f"selected_{tag}"] += 1
        return True

    for target, split in sorted(v1_splits.items()):
        if target in by_target:
            offer(by_target[target], "v1_pinned", split)
        else:
            counters["v1_dropped_target_in_stock"] += 1
    ordered = sorted(pool, key=lambda r: stable_hash(r["target_smiles"], length=32))
    for row in ordered:
        if row["distinct_first_cuts"] >= 2 and row["target_smiles"] not in v1_splits:
            offer(row, "two_route", fill_split)

    remaining = n_total - len(tasks)
    quota = {2: remaining // 2, 3: remaining - remaining // 2}
    filled: Counter[int] = Counter()
    for row in ordered:
        if len(tasks) >= n_total:
            break
        if row["distinct_first_cuts"] >= 2 or row["target_smiles"] in v1_splits:
            continue
        steps = min(row["steps"])
        if step_quota and filled[steps] >= quota.get(steps, 0):
            continue
        if max_per_patent and any(patent_use[p] >= max_per_patent for p in row["patents"]):
            counters["skip_patent_already_used"] += 1
            continue
        scaffold = scaffold_of[row["target_smiles"]]
        if is_single_ring_scaffold(scaffold):
            cap = max(1, round(pool_scaffold[scaffold] / len(pool) * n_total))
        else:
            cap = max_per_scaffold
        if cap and scaffold_use[scaffold] >= cap:
            counters["skip_scaffold_cap"] += 1
            continue
        if offer(row, "single_route", fill_split):
            filled[steps] += 1

    desired = {name: ratios[k] * len(tasks) for k, name in enumerate(SPLITS)}
    counts = {name: 0 for name in SPLITS}
    assignment: dict[int, str] = {}
    comps = groups.components()
    pinned = [c for c in comps if groups.find(c[0]) in groups.pin]
    free = [c for c in comps if groups.find(c[0]) not in groups.pin]
    for comp in pinned:
        split = groups.pin[groups.find(comp[0])]
        for i in comp:
            assignment[i] = split
        counts[split] += len(comp)
    free.sort(key=lambda c: (-len(c), stable_hash("|".join(sorted(tasks[i].task_id for i in c)), length=32)))
    for comp in free:
        split = max(
            SPLITS,
            key=lambda name: (
                (desired[name] - counts[name]) / max(desired[name], 1.0),
                desired[name] - counts[name],
                stable_hash(name + tasks[comp[0]].task_id, length=12),
            ),
        )
        for i in comp:
            assignment[i] = split
        counts[split] += len(comp)
    final = [_with_split(task, assignment[i]) for i, task in enumerate(tasks)]
    sizes = sorted((len(c) for c in comps), reverse=True)
    by_split_tag = Counter((assignment[i], tags[i]) for i in range(len(tasks)))
    info = {
        "components": len(comps),
        "component_sizes_top": sizes[:12],
        "largest_component": sizes[0],
        "near_duplicate_edges": groups.near_duplicate_edges,
        "counts": counts,
        "by_split_and_kind": {f"{s}/{t}": n for (s, t), n in sorted(by_split_tag.items())},
    }
    return final, counters, info


def _with_split(task: RetroTask, split: str) -> RetroTask:
    return RetroTask(
        task_id=task.task_id,
        mode=task.mode,
        target_smiles=task.target_smiles,
        max_steps=task.max_steps,
        stock_id=task.stock_id,
        split=split,
        reference_routes=task.reference_routes,
        min_routes=task.min_routes,
        max_routes=task.max_routes,
        difficulty=task.difficulty,
        schema_version=task.schema_version,
    )


def write_checksums(out: Path) -> None:
    """Hash the files a server must not run with silently altered: tasks and stock."""
    import hashlib

    files = sorted(
        [
            *out.glob("tasks-private/*.jsonl"),
            out / "tasks-private/manifest.json",
            out / "normalized-routes.jsonl",
            *out.glob("stocks/*.smi"),
        ]
    )
    digests = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.exists()
    }
    (out / "checksums.json").write_text(
        json.dumps({"schema_version": "retro-private-checksums-v1", "sha256": digests}, indent=2) + "\n",
        encoding="utf-8",
    )


def write(out: Path, tasks: list[RetroTask], stock: frozenset[str], manifest: dict) -> None:
    write_tasks(tasks, out / "tasks-private", manifest)
    (out / "tasks-public").mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with (out / "tasks-public" / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for task in sorted((t for t in tasks if t.split == split), key=lambda t: t.task_id):
                handle.write(
                    json.dumps(task.to_dict(include_references=False), sort_keys=True, separators=(",", ":")) + "\n"
                )
    (out / "stocks").mkdir(parents=True, exist_ok=True)
    (out / "stocks" / f"{STOCK_ID}.smi").write_text("".join(f"{s}\n" for s in sorted(stock)), encoding="utf-8")
    with (out / "normalized-routes.jsonl").open("w", encoding="utf-8") as handle:
        for task in sorted(tasks, key=lambda t: t.task_id):
            for route in task.reference_routes:
                row = {"task_id": task.task_id, "target_smiles": task.target_smiles, **route.to_dict()}
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_checksums(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, default=Path(".local/pool/pool-n1-s3.jsonl"))
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=Path("benchmark/retroeval-v1"),
        help="benchmark whose tasks keep their split (pinned)",
    )
    parser.add_argument("--fill-split", choices=SPLITS, help="pin every new task to this split")
    parser.add_argument("--stock-file", type=Path, default=Path("data/raw/paroutes-v2-benchmark/stock_n1.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path(".local/retroeval-v2"))
    parser.add_argument("--tasks", type=int, default=1000)
    parser.add_argument("--ratios", nargs=4, type=float, default=(0.6, 0.1, 0.15, 0.15))
    parser.add_argument(
        "--strict-scaffolds", action="store_true", help="v1 rule: group on every Murcko scaffold, including benzene"
    )
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.90)
    parser.add_argument("--max-per-patent", type=int, default=1, help="0 = unlimited")
    parser.add_argument("--max-per-scaffold", type=int, default=1, help="multi-ring scaffolds; 0 = unlimited")
    parser.add_argument("--no-step-quota", action="store_true", help="keep the pool's natural route-length mix")
    parser.add_argument("--dry-run", action="store_true", help="report selection and groups only")
    args = parser.parse_args(argv)

    pool = [json.loads(line) for line in args.pool.open(encoding="utf-8")]
    v1_splits = {}
    for split in SPLITS:
        for line in (args.v1_dir / "tasks-private" / f"{split}.jsonl").open(encoding="utf-8"):
            v1_splits[json.loads(line)["target_smiles"]] = split
    tasks, counters, info = build(
        pool,
        v1_splits,
        args.tasks,
        tuple(args.ratios),
        not args.strict_scaffolds,
        args.near_duplicate_threshold,
        max_per_patent=args.max_per_patent,
        max_per_scaffold=args.max_per_scaffold,
        step_quota=not args.no_step_quota,
        fill_split=args.fill_split,
    )
    audit = audit_splits(tasks, exact_single_ring_scaffolds=not args.strict_scaffolds)
    report = {
        "selection": dict(sorted(counters.items())),
        "split": info,
        "strict_audit_passed": audit["passed"],
        "audit_overlap_kinds": sorted({o["kind"] for o in audit["overlaps"]}),
    }
    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0
    stock = load_stock(args.stock_file)
    manifest = {
        "schema_version": "retro-benchmark-v2-draft",
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
        "selection_rules": {
            "pool": str(args.pool),
            "exclude_target_in_stock": True,
            "max_per_patent": args.max_per_patent,
            "multi_ring_scaffold_cap": args.max_per_scaffold,
            "single_ring_or_acyclic_scaffold_cap": "pool share x tasks",
            "single_route_step_shares": None if args.no_step_quota else {"2": 0.5, "3": 0.5},
            "v1_tasks_pinned_to_v1_split": True,
            "exact_single_ring_scaffolds": not args.strict_scaffolds,
            "near_duplicate_threshold": args.near_duplicate_threshold,
        },
        "requested_tasks": args.tasks,
        "ratios": dict(zip(SPLITS, args.ratios)),
        **report,
    }
    write(args.output_dir, tasks, stock, manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
