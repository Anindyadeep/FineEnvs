#!/usr/bin/env python3
"""Attach heuristic difficulty labels to a built benchmark (private sidecar).

Labels go to ``<benchmark>/difficulty.jsonl``, never into public task rows, because
features such as the first-step reaction family are hints. The tier is a
transparent point score meant for stratifying and curriculum ordering until
model pass rates replace it:

  +1  the shortest reference route has 3 or more steps
  +1  it has 4 or more steps (v3 and later; v2 routes stop at 3, so v2 tiers are unchanged)
  +1  no train-split target with Morgan Tanimoto >= 0.30 (weak precedent search)
  +1  first-step family outside the common set (Boc in/out, amide/ester, reductive
      amination, one-reactant FGI, alkylation/SNAr)
  +1  some leaf appears in <= 5 routes of the whole PaRoutes archive

  tier: 0-1 easy, 2 medium, 3-5 hard

On the 20 v1 eval tasks with 8 models, only train similarity was significant
(Spearman 0.50 with mean reward); treat the tier as a prior, not a measurement.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_route_pool import _leaves  # noqa: E402

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402
from retroenv.disconnections import step_family  # noqa: E402  (one copy, shared with the expert)

SPLITS = ("train", "dev", "eval", "stress")
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
COMMON = {
    "Boc protection",
    "Boc deprotection",
    "amide/ester coupling",
    "reductive amination",
    "one-reactant FGI",
    "alkylation / SNAr",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, default=Path(".local/retroeval-v2"))
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/paroutes-v2-benchmark/all_loaded_routes.json.gz")
    )
    args = parser.parse_args(argv)

    with gzip.open(args.archive, "rt", encoding="utf-8") as handle:
        roots = json.load(handle)
    leaf_frequency: Counter[str] = Counter()
    for root in roots:
        leaf_frequency.update(set(_leaves(root)))
    del roots

    tasks = []
    for split in SPLITS:
        for line in (args.benchmark_dir / "tasks-private" / f"{split}.jsonl").open(encoding="utf-8"):
            tasks.append(json.loads(line))
    fps = {t["task_id"]: FPG.GetFingerprint(Chem.MolFromSmiles(t["target_smiles"])) for t in tasks}
    train_ids = [t["task_id"] for t in tasks if t["split"] == "train"]
    train_fps = [fps[i] for i in train_ids]

    rows = []
    for task in tasks:
        mol = Chem.MolFromSmiles(task["target_smiles"])
        sims = DataStructs.BulkTanimotoSimilarity(fps[task["task_id"]], train_fps)
        if task["split"] == "train":
            sims = [s for i, s in zip(train_ids, sims) if i != task["task_id"]]
        refs = task["reference_routes"]
        leaves = sorted(
            {x for r in refs for s in r["steps"] for x in s["reactants"] if x not in {q["product"] for q in r["steps"]}}
        )
        families = sorted({step_family(r["steps"][0]["reactants"], r["steps"][0]["product"]) for r in refs})
        min_depth = min(len(r["steps"]) for r in refs)
        nn = max(sims) if sims else 0.0
        min_leaf = min(leaf_frequency.get(x, 0) for x in leaves)
        points = {
            "three_steps": int(min_depth >= 3),
            "four_plus_steps": int(min_depth >= 4),
            "no_close_train_target": int(nn < 0.30),
            "uncommon_first_step": int(not set(families) & COMMON),
            "rare_leaf": int(min_leaf <= 5),
        }
        score = sum(points.values())
        rows.append(
            {
                "task_id": task["task_id"],
                "split": task["split"],
                "kind": "two_route" if task["min_routes"] >= 2 else "single_route",
                "min_depth": min_depth,
                "heavy_atoms": mol.GetNumHeavyAtoms(),
                "rings": rdMolDescriptors.CalcNumRings(mol),
                "sa_score": round(sascorer.calculateScore(mol), 2),
                "first_step_families": families,
                "nn_train_similarity": round(nn, 3),
                "min_leaf_archive_routes": min_leaf,
                "points": points,
                "score": score,
                "tier": "easy" if score <= 1 else ("medium" if score == 2 else "hard"),
            }
        )
    out = args.benchmark_dir / "difficulty.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: (SPLITS.index(r["split"]), r["task_id"])):
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "tiers_by_split": {s: dict(Counter(r["tier"] for r in rows if r["split"] == s)) for s in SPLITS},
        "first_step_families": Counter(f for r in rows for f in r["first_step_families"]).most_common(),
        "points_rate": {k: round(sum(r["points"][k] for r in rows) / len(rows), 3) for k in rows[0]["points"]},
    }
    (args.benchmark_dir / "difficulty.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
