"""Server settings from environment variables, and the read-only data every session shares.

Tasks are read from a benchmark directory with the layout the dataset scripts
write (``tasks-private/<split>.jsonl`` and ``stocks/<stock_id>.smi``). The
private references stay inside the server process; nothing here is bundled
into the Docker image (``prepare.py`` fetches them at startup).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from retroenv.retrieval import PrecedentIndex, cached_stock_index
from retroenv.store import TaskStore
from retroenv.tools import tool_names

ENV_NAME = "retro_route"


def _resolve_dirs() -> tuple[Path, Path]:
    benchmark = os.getenv("RETROENV_BENCHMARK_DIR")
    if benchmark:
        root = Path(benchmark)
        return root / "tasks-private", root / "stocks"
    tasks = os.getenv("RETROENV_TASKS_DIR")
    stocks = os.getenv("RETROENV_STOCKS_DIR")
    if tasks and stocks:
        return Path(tasks), Path(stocks)
    prepared = Path(os.getenv("RETROENV_PREPARED_DIR", "prepared"))
    if (prepared / "tasks-private").is_dir():
        return prepared / "tasks-private", prepared / "stocks"
    raise RuntimeError(
        "No tasks configured. Set RETROENV_BENCHMARK_DIR to a benchmark directory "
        "(tasks-private/ and stocks/), or run prepare.py first."
    )


@dataclass(frozen=True)
class Settings:
    tasks_dir: Path
    stocks_dir: Path
    default_split: str = "train"
    max_tool_calls: int = 32
    toolset: str = "full"
    pubchem_cache: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        tasks_dir, stocks_dir = _resolve_dirs()
        cache = os.getenv("RETROENV_PUBCHEM_CACHE") or None
        settings = cls(
            tasks_dir=tasks_dir,
            stocks_dir=stocks_dir,
            default_split=os.getenv("RETROENV_DEFAULT_SPLIT", "train"),
            max_tool_calls=int(os.getenv("RETROENV_MAX_TOOL_CALLS", "32")),
            toolset=os.getenv("RETROENV_TOOLSET", "full"),
            pubchem_cache=Path(cache) if cache else None,
        )
        tool_names(settings.toolset)  # fail fast on an unknown toolset
        return settings


@dataclass
class Resources:
    settings: Settings
    store: TaskStore
    precedent_index: PrecedentIndex
    pubchem_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, settings: Settings) -> "Resources":
        store = TaskStore(settings.tasks_dir, settings.stocks_dir)
        splits = store.splits()
        if not splits:
            raise RuntimeError(f"no task splits found in {settings.tasks_dir}")
        if settings.default_split not in splits:
            raise RuntimeError(
                f"default split {settings.default_split!r} is not among {splits}"
            )
        # Eval references never enter retrieval: precedents come from train only.
        training = store.tasks("train") if "train" in splits else ()
        # Canonicalize every stock once, so the first reset is not slow.
        for stock_id in sorted({task.stock_id for task in store.iter_all()}):
            cached_stock_index(store.stock(stock_id))
        return cls(
            settings=settings,
            store=store,
            precedent_index=PrecedentIndex(training),
            pubchem_cache=_load_pubchem_cache(settings.pubchem_cache),
        )


def _load_pubchem_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"PubChem cache must be a JSON object: {path}")
    return {str(key).casefold(): row for key, row in value.items() if isinstance(row, dict)}


@lru_cache(maxsize=1)
def shared_resources() -> Resources:
    """Loaded once per process, on the first environment instance."""
    return Resources.load(Settings.from_env())
