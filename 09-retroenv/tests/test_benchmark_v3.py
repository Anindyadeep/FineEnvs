from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from retroenv.taskgen import audit_rules, split_rules

DATASET = Path(__file__).parents[1] / "dataset"
sys.path.insert(0, str(DATASET))
_SPEC = importlib.util.spec_from_file_location("retroenv_build_v3", DATASET / "build_benchmark_v3.py")
assert _SPEC and _SPEC.loader
v3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v3)


def test_quotas_sum_to_the_split_size():
    for total in (150, 250, 251):
        for shares in (v3.DEPTHS, v3.SIZES):
            counts = v3.quotas(shares, total)
            assert sum(counts.values()) == total and set(counts) == set(shares)
    assert v3.quotas(v3.DEPTHS, 250) == {2: 70, 3: 70, 4: 60, 5: 50}


def test_split_rules_come_from_the_manifest(tmp_path):
    assert split_rules(tmp_path) == {"exact_single_ring_scaffolds": False, "generic_scaffolds": frozenset()}
    (tmp_path / "manifest.json").write_text(
        json.dumps({"design": {"exact_single_ring_scaffolds": True, "generic_scaffolds": ["c1ccc2ccccc2c1"]}})
    )
    assert split_rules(tmp_path)["generic_scaffolds"] == frozenset({"c1ccc2ccccc2c1"})
    assert audit_rules({"selection_rules": {"exact_single_ring_scaffolds": True}})["exact_single_ring_scaffolds"]


def test_committed_v3_design_meets_its_quotas():
    manifest_path = Path(__file__).parents[1] / "benchmark" / "retroeval-v3" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["strict_audit_passed"]
    for split, total in v3.HELD_OUT.items():
        composition = manifest["composition"][split]
        assert composition["tasks"] == total
        assert composition["depth"] == {str(k): v for k, v in v3.quotas(v3.DEPTHS, total).items()}
        assert composition["two_route"] == round(v3.TWO_ROUTE * total)
    assert set(manifest["composition"]["eval"]["family"]) == set(manifest["design"]["family_shares"])
