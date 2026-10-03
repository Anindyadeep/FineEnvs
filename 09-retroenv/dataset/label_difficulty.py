#!/usr/bin/env python3
"""Attach heuristic difficulty labels to a built benchmark (private sidecar).

Labels go to ``<benchmark>/difficulty.jsonl``, never into public task rows, because
features such as the first-step reaction family are hints. The tier is a
transparent point score meant for stratifying and curriculum ordering until
model pass rates replace it:

  +1  the shortest reference route has 3 steps
  +1  no train-split target with Morgan Tanimoto >= 0.30 (weak precedent search)
  +1  first-step family outside the common set (Boc in/out, amide/ester, reductive
      amination, one-reactant FGI, alkylation/SNAr)
  +1  some leaf appears in <= 5 routes of the whole PaRoutes archive

  tier: 0-1 easy, 2 medium, 3-4 hard

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

SPLITS = ("train", "dev", "eval", "stress")
FPG = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
PATTERNS = {
    "boronic": Chem.MolFromSmarts("[#6]B([OX2])[OX2]"),
    "acyl_halide": Chem.MolFromSmarts("C(=O)[Cl,Br]"),
    "acid": Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    "amine": Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]"),
    "alcohol": Chem.MolFromSmarts("[OX2H][CX4,c]"),
    "carbonyl": Chem.MolFromSmarts("[CX3H1,CX3H0;!$(C(=O)[O,N])](=O)[#6,#1]"),
    "halide": Chem.MolFromSmarts("[#6][Cl,Br,I]"),
    "sulfonyl_halide": Chem.MolFromSmarts("S(=O)(=O)[Cl,F]"),
    "isocyanate": Chem.MolFromSmarts("N=C=O"),
}
BOC = Chem.MolFromSmarts("CC(C)(C)OC(=O)[N,O,n]")
BOC2O = Chem.MolFromSmiles("CC(C)(C)OC(=O)OC(=O)OC(C)(C)C")
COMMON = {
    "Boc protection",
    "Boc deprotection",
    "amide/ester coupling",
    "reductive amination",
    "one-reactant FGI",
    "alkylation / SNAr",
}


def _has(smiles: str, key: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.HasSubstructMatch(PATTERNS[key])


def step_family(reactants: list[str], product: str) -> str:
    """Coarse rule-based reaction family; 'other' when no rule fires."""
    mols = [Chem.MolFromSmiles(r) for r in reactants]
    if any(m is not None and m.HasSubstructMatch(BOC2O) for m in mols):
        return "Boc protection"
    if len(reactants) == 1:
        p = Chem.MolFromSmiles(product)
        if mols[0].HasSubstructMatch(BOC) and len(p.GetSubstructMatches(BOC)) < len(mols[0].GetSubstructMatches(BOC)):
            return "Boc deprotection"
        return "one-reactant FGI"
    if any(_has(r, "boronic") for r in reactants):
        return "Suzuki-type coupling"
    if any(_has(r, "sulfonyl_halide") for r in reactants):
        return "sulfonylation"
    if any(_has(r, "isocyanate") for r in reactants):
        return "urea / carbamate"
    if any(_has(r, "acyl_halide") or _has(r, "acid") for r in reactants) and any(
        _has(r, "amine") or _has(r, "alcohol") for r in reactants
    ):
        return "amide/ester coupling"
    if any(_has(r, "carbonyl") for r in reactants) and any(_has(r, "amine") for r in reactants):
        return "reductive amination"
    if any(_has(r, "halide") for r in reactants):
        return "alkylation / SNAr"
    return "other"


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
