from __future__ import annotations

from dataset.publish_release import PRIVATE_ROW, heldout_card, public_card

CARD = f"""---
configs:
- config_name: tasks_with_known_routes
  data_files:
  - split: train
    path: tasks-private/train.jsonl
  - split: dev
    path: tasks-private/dev.jsonl
  - split: test_id
    path: tasks-private/test_id.jsonl
  - split: test_hard
    path: tasks-private/test_hard.jsonl
---

| Path | Contents |
|---|---|
{PRIVATE_ROW}
"""


def test_the_public_card_offers_known_routes_only_for_published_splits():
    card = public_card(CARD, ["test_id", "test_hard"], "LiteFold/RetroEnv-heldout")
    assert "tasks-private/train.jsonl" in card and "tasks-private/dev.jsonl" in card
    assert "tasks-private/test_id.jsonl" not in card and "tasks-private/test_hard.jsonl" not in card
    assert "LiteFold/RetroEnv-heldout" in card and PRIVATE_ROW not in card


def test_the_held_out_card_lists_its_files_and_links_the_public_release():
    card = heldout_card(["test_id", "test_hard"], "LiteFold/RetroEnv")
    assert "`tasks-private/test_id.jsonl`" in card and "`tasks-private/test_hard.jsonl`" in card
    assert "https://huggingface.co/datasets/LiteFold/RetroEnv" in card
