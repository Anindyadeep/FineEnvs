"""Non-thinking and no-repair settings the evaluation harness sends to providers."""

from __future__ import annotations

import json

from retroenv_openenv.agent import normalize_arguments
from retroenv_openenv.agent_anthropic import thinking_off


def test_claude_models_turn_thinking_off_the_way_each_accepts():
    assert thinking_off("claude-haiku-5-5") == ({"type": "disabled"}, None, True)
    assert thinking_off("claude-sonnet-5-5") == ({"type": "between_tools"}, None, True)
    # Opus 5.5 and Fable 5.1 reject disabled thinking; low effort is their floor and is labelled so.
    assert thinking_off("claude-opus-5-5") == (None, "low", False)
    assert thinking_off("claude-fable-5-1") == (None, "low", False)


def test_without_repair_a_broken_submission_reaches_the_verifier_unchanged():
    broken = '{"routes": [{"type": "mol", "smiles": "CCO", "children": []}]]'  # one stray bracket
    repaired, coerced = normalize_arguments("emit_routes", {"submission": broken})
    assert coerced and isinstance(repaired["submission"], dict)
    untouched, coerced = normalize_arguments("emit_routes", {"submission": broken}, repair=False)
    assert not coerced and untouched["submission"] == broken
    valid = json.dumps({"routes": []})
    assert normalize_arguments("emit_routes", {"submission": valid}, repair=False)[0]["submission"] == valid
