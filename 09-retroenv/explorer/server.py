#!/usr/bin/env python3
"""Local explorer for RetroEnv: benchmark tasks, SFT datasets and model runs.

    uv run --extra eval python explorer/server.py            # http://127.0.0.1:8050

It reads the files in place: every ``benchmark/*/tasks-private`` set (with its
difficulty sidecar and stock), every SFT export under ``.local/sft/``, and the
model runs under ``runs/`` (the multi-gigabyte SFT generation runs under
``runs/sft`` are skipped; their exports are the datasets). SFT files are indexed
once by byte offset and the index is cached in ``.local/explorer-cache``, so later
starts are fast and a row is read only when it is opened.

References of dev, eval and stress tasks are sent only when the page asks with
``reveal=1``, so the default view shows what a model sees. It binds to localhost.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
import sys
import threading
import time
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from retroenv.chemistry import canonicalize_smiles
from retroenv.disconnections import describe, reaction_phrase, step_family, strategic_disconnections
from retroenv.environment import PROMPT, RetroRouteSession
from retroenv.graph import routes_to_submission
from retroenv.models import RetroTask
from retroenv.retrieval import PrecedentIndex
from retroenv.store import load_stock
from retroenv.taskgen import split_rules
from retroenv.tools import ORACLE_TOOLS, TOOLS, openai_tools
from retroenv.verifier import GRAPH_WEIGHTS

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
CACHE = ROOT / ".local" / "explorer-cache"
HELD_OUT = ("dev", "eval", "stress")
SPLITS = ("train", "dev", "eval", "stress")

_spec = importlib.util.spec_from_file_location("build_sft", ROOT / "dataset" / "build_sft.py")
build_sft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_sft)


def size_bin(heavy: int) -> str:
    return "<=20" if heavy <= 20 else "21-30" if heavy <= 30 else "31-40" if heavy <= 40 else "41+"


# --- benchmarks -------------------------------------------------------------------


class Benchmark:
    def __init__(self, root: Path):
        self.name = root.name
        self.root = root
        self.tasks: dict[str, dict[str, Any]] = {}
        for split in SPLITS:
            path = root / "tasks-private" / f"{split}.jsonl"
            if path.exists():
                for line in path.open(encoding="utf-8"):
                    task = json.loads(line)
                    self.tasks[task["task_id"]] = task
        sidecar = root / "difficulty.jsonl"
        self.difficulty = {}
        if sidecar.exists():
            self.difficulty = {row["task_id"]: row for row in map(json.loads, sidecar.open(encoding="utf-8"))}
        self.stock_id = next(iter(self.tasks.values()))["stock_id"] if self.tasks else None
        self._stock: frozenset[str] | None = None
        self.rows = [self._row(task) for task in self.tasks.values()]
        self.rows.sort(key=lambda r: (SPLITS.index(r["split"]), r["id"]))

    @property
    def stock(self) -> frozenset[str]:
        if self._stock is None:
            self._stock = frozenset(
                canonicalize_smiles(s) for s in load_stock(self.root / "stocks" / f"{self.stock_id}.smi")
            )
        return self._stock

    def _row(self, task: dict[str, Any]) -> dict[str, Any]:
        diff = self.difficulty.get(task["task_id"], {})
        first = task["reference_routes"][0]["steps"][0]
        heavy = task["difficulty"].get("heavy_atoms") or diff.get("heavy_atoms") or 0
        return {
            "id": task["task_id"],
            "split": task["split"],
            "smiles": task["target_smiles"],
            "steps": task["max_steps"],
            "routes": task["min_routes"],
            "heavy": heavy,
            "size": size_bin(heavy),
            "family": (diff.get("first_step_families") or [step_family(first["reactants"], first["product"])])[0],
            "tier": diff.get("tier"),
            "nn": diff.get("nn_train_similarity"),
            "stereo": bool(task["difficulty"].get("stereochemistry")),
        }


def discover_benchmarks() -> dict[str, Benchmark]:
    found = {}
    for path in sorted((ROOT / "benchmark").glob("*/tasks-private")):
        print(f"loading {path.parent.name} ...", file=sys.stderr)
        found[path.parent.name] = Benchmark(path.parent)
    return found


# --- SFT datasets ---------------------------------------------------------------------


class Dataset:
    """An SFT export, indexed by byte offset so rows are read only when opened."""

    def __init__(self, root: Path):
        self.name = root.name
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {}
        self.files = [root / f"{part}.jsonl" for part in ("train", "validation") if (root / f"{part}.jsonl").exists()]
        stamp = hashlib.sha256(
            json.dumps([(str(f), f.stat().st_size, f.stat().st_mtime) for f in self.files]).encode()
        ).hexdigest()[:16]
        cache = CACHE / f"{self.name}.{stamp}.json"
        if cache.exists():
            self.rows = json.loads(cache.read_text())
        else:
            print(f"indexing {self.name} (once) ...", file=sys.stderr)
            self.rows = self._index()
            CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(self.rows))
        self.by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.by_id = {}
        for row in self.rows:
            self.by_task[row["task_id"]].append(row)
            self.by_id[row["id"]] = row

    def _index(self) -> list[dict[str, Any]]:
        rows = []
        for part, path in zip(("train", "validation"), self.files):
            with path.open("rb") as handle:
                offset = 0
                for line in handle:
                    row = json.loads(line)
                    names, rejected, turns = {}, False, 0
                    for message in row["messages"]:
                        for call in message.get("tool_calls") or []:
                            names[call["id"]] = call["function"]["name"]
                        if message["role"] == "assistant":
                            turns += 1
                        elif (
                            message["role"] == "tool"
                            and names.get(message.get("tool_call_id")) == "validate_disconnection"
                        ):
                            rejected = rejected or not json.loads(message["content"]).get("valid", True)
                    rows.append(
                        {
                            "id": row["id"],
                            "task_id": row["task_id"],
                            "part": part,
                            "file": path.name,
                            "offset": offset,
                            "length": len(line),
                            "source": row["source"],
                            "steps": row["max_steps"],
                            "reward": row["reward"],
                            "exact": row["exact_match"],
                            "calls": row["tool_calls"],
                            "turns": turns,
                            "recovered": rejected,
                            "tools": sorted(set(names.values())),
                        }
                    )
                    offset += len(line)
        return rows

    def read(self, row_id: str) -> dict[str, Any]:
        meta = self.by_id[row_id]
        with (self.root / meta["file"]).open("rb") as handle:
            handle.seek(meta["offset"])
            return json.loads(handle.read(meta["length"]))


def discover_datasets() -> dict[str, Dataset]:
    found = {}
    for path in sorted((ROOT / ".local" / "sft").glob("*/manifest.json")):
        found[path.parent.name] = Dataset(path.parent)
    return found


# --- model runs ---------------------------------------------------------------------------


class Run:
    def __init__(self, root: Path, benchmarks: dict[str, Benchmark]):
        self.root = root
        self.name = str(root.relative_to(ROOT / "runs"))
        self.identity = json.loads((root / "identity.json").read_text())
        ids = set(self.identity.get("task_ids", []))
        self.benchmark = next((name for name, b in benchmarks.items() if ids and ids <= b.tasks.keys()), None)
        self._episodes: list[dict[str, Any]] | None = None

    @property
    def episodes(self) -> list[dict[str, Any]]:
        if self._episodes is None:
            rows = []
            for path in sorted((self.root / "episodes").glob("*.json")):
                row = json.loads(path.read_text())
                rows.append(
                    {
                        "file": path.name,
                        "task_id": row["task_id"],
                        "attempt": row.get("attempt", 0),
                        "reward": row.get("reward"),
                        "valid": row.get("valid"),
                        "exact": row.get("exact_match"),
                        "calls": row.get("tool_calls"),
                        "turns": row.get("turns"),
                        "refused": any(str(e).startswith("refusal") for e in row.get("errors", [])),
                    }
                )
            self._episodes = rows
        return self._episodes

    def summary(self) -> dict[str, Any]:
        summary = json.loads((self.root / "summary.json").read_text()) if (self.root / "summary.json").exists() else {}
        return {
            "name": self.name,
            "label": self.identity.get("label"),
            "provider": self.identity.get("provider"),
            "split": self.identity.get("split"),
            "benchmark": self.benchmark,
            "toolset": self.identity.get("toolset"),
            "tasks": len(self.identity.get("task_ids", [])),
            "pass": summary.get("pass_at_1"),
            "exact": summary.get("exact_route_rate"),
            "reward": summary.get("mean_reward"),
            "cost": summary.get("cost_usd"),
        }


def model_passes(benchmark: str) -> dict[str, list[int]]:
    """task ID -> [episodes that passed, episodes] across the model runs on this benchmark."""
    cache = STATE.setdefault("passes", {})
    if benchmark not in cache:
        tally: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for run in STATE["runs"].values():
            if run.benchmark == benchmark and run.identity.get("provider") != "reference":
                for episode in run.episodes:
                    tally[episode["task_id"]][0] += bool(episode["valid"])
                    tally[episode["task_id"]][1] += 1
        cache[benchmark] = dict(tally)
    return cache[benchmark]


def discover_runs(benchmarks: dict[str, Benchmark]) -> dict[str, Run]:
    found = {}
    for identity in sorted((ROOT / "runs").glob("**/identity.json")):
        if "runs/sft" in str(identity):
            continue  # generation runs: tens of thousands of files; their exports are the datasets
        run = Run(identity.parent, benchmarks)
        found[run.name] = run
    return found


# --- the app ---------------------------------------------------------------------------------

app = FastAPI(title="RetroEnv explorer")
STATE: dict[str, Any] = {}


def bench(name: str) -> Benchmark:
    try:
        return STATE["benchmarks"][name]
    except KeyError:
        raise HTTPException(404, f"unknown benchmark {name!r}") from None


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "benchmarks": [
            {"name": n, "splits": dict(Counter(r["split"] for r in b.rows))} for n, b in STATE["benchmarks"].items()
        ],
        "datasets": [{"name": n, "rows": len(d.rows), "tasks": len(d.by_task)} for n, d in STATE["datasets"].items()],
        "runs": [r.summary() for r in STATE["runs"].values()],
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
                "oracle": t["function"]["name"] in ORACLE_TOOLS,
            }
            for t in TOOLS
        ],
        "weights": GRAPH_WEIGHTS,
    }


@app.get("/api/overview")
def overview(benchmark: str, dataset: str | None = None) -> dict[str, Any]:
    b = bench(benchmark)
    out: dict[str, Any] = {"splits": {}}
    for split in SPLITS:
        rows = [r for r in b.rows if r["split"] == split]
        if not rows:
            continue
        out["splits"][split] = {
            "tasks": len(rows),
            "two_route": sum(r["routes"] >= 2 for r in rows),
            "stereo": sum(r["stereo"] for r in rows),
            **{key: dict(Counter(str(r[key]) for r in rows)) for key in ("steps", "family", "tier", "size")},
        }
    d = STATE["datasets"].get(dataset or "")
    if d:
        rows = d.rows
        out["dataset"] = {
            "name": d.name,
            "rows": len(rows),
            "tasks": len(d.by_task),
            "manifest": d.manifest,
            "recovered": sum(r["recovered"] for r in rows) / len(rows),
            "steps": dict(Counter(str(r["steps"]) for r in rows)),
            "turns": dict(Counter(str(min(r["turns"], 12)) for r in rows)),
            "calls": dict(Counter(str(min(r["calls"] // 2 * 2, 30)) for r in rows)),
            "recovered_by_steps": {
                s: sum(r["recovered"] for r in rows if str(r["steps"]) == s)
                / max(1, sum(str(r["steps"]) == s for r in rows))
                for s in sorted({str(r["steps"]) for r in rows})
            },
            "tools": dict(Counter(t for r in rows for t in r["tools"])),
        }
    out["datasets"] = {
        name: {
            "rows": len(x.rows),
            "recovered": sum(r["recovered"] for r in x.rows) / max(1, len(x.rows)),
            "exact": x.manifest.get("exact_route_rate"),
            "calls": x.manifest.get("mean_tool_calls"),
            "tokens": x.manifest.get("tokens"),
            "behaviour": x.manifest.get("behaviour"),
            "steps": x.manifest.get("by_max_steps"),
        }
        for name, x in STATE["datasets"].items()
    }
    return out


@app.get("/api/tasks")
def tasks(
    benchmark: str,
    dataset: str | None = None,
    split: str = "",
    steps: str = "",
    family: str = "",
    tier: str = "",
    size: str = "",
    routes: str = "",
    sft: str = "",
    q: str = "",
    offset: int = 0,
    limit: int = Query(50, le=200),
) -> dict[str, Any]:
    b = bench(benchmark)
    d = STATE["datasets"].get(dataset or "")
    needle = q.strip()
    canonical = None
    if needle and not needle.startswith("retro_"):
        try:
            canonical = canonicalize_smiles(needle)
        except Exception:
            canonical = None
    out = []
    for row in b.rows:
        if (
            split
            and row["split"] != split
            or steps
            and str(row["steps"]) != steps
            or family
            and row["family"] != family
        ):
            continue
        if tier and row["tier"] != tier or size and row["size"] != size or routes and str(row["routes"]) != routes:
            continue
        if needle and needle not in row["id"] and needle not in row["smiles"] and row["smiles"] != canonical:
            continue
        sft_rows = d.by_task.get(row["id"], []) if d else []
        if (
            sft == "has"
            and not sft_rows
            or sft == "recovered"
            and not any(r["recovered"] for r in sft_rows)
            or sft == "none"
            and sft_rows
        ):
            continue
        out.append({**row, "sft": len(sft_rows), "recovered": any(r["recovered"] for r in sft_rows)})
    passes = model_passes(benchmark)
    page = [{**row, "models": passes.get(row["id"])} for row in out[offset : offset + limit]]
    return {"total": len(out), "rows": page, "has_models": bool(passes)}


@app.get("/api/sft_rows")
def sft_rows(
    dataset: str,
    steps: str = "",
    recovered: str = "",
    tool: str = "",
    q: str = "",
    offset: int = 0,
    limit: int = Query(50, le=200),
) -> dict[str, Any]:
    d = STATE["datasets"].get(dataset)
    if not d:
        raise HTTPException(404, f"no dataset {dataset}")
    rows = [
        r
        for r in d.rows
        if (not steps or str(r["steps"]) == steps)
        and (not recovered or r["recovered"] == (recovered == "yes"))
        and (not tool or tool in r["tools"])
        and (not q or q in r["id"])
    ]
    return {"total": len(rows), "rows": rows[offset : offset + limit]}


@app.get("/api/disconnections")
def disconnections(smi: str) -> list[dict[str, Any]]:
    """Rule-based candidate cuts, the same library the scripted chemist proposes from."""
    return [
        {
            "reactants": list(d.reactants),
            "bond": d.bond,
            "family": d.family,
            "score": d.score,
            "text": describe(d.family, d.reactants, smi),
        }
        for d in strategic_disconnections(smi)[:8]
    ]


# --- live sessions: play a task through the same core session the server uses -------------

SESSIONS: dict[str, dict[str, Any]] = {}
_INDEXES: dict[str, PrecedentIndex] = {}
_INDEX_LOCK = threading.Lock()


def _precedents(b: Benchmark) -> PrecedentIndex:
    with _INDEX_LOCK:
        if b.name not in _INDEXES:
            train = [RetroTask.from_dict(t) for t in b.tasks.values() if t["split"] == "train"]
            _INDEXES[b.name] = PrecedentIndex(train, **split_rules(b.root / "tasks-private"))
        return _INDEXES[b.name]


class NewSession(BaseModel):
    benchmark: str
    task: str
    toolset: str = "full"


class ToolCall(BaseModel):
    tool: str
    arguments: dict[str, Any] = {}


@app.post("/api/session")
def new_session(request: NewSession) -> dict[str, Any]:
    b = bench(request.benchmark)
    if request.task not in b.tasks:
        raise HTTPException(404, f"no task {request.task}")
    session = RetroRouteSession(precedent_index=_precedents(b), toolset=request.toolset)
    observation = session.reset(RetroTask.from_dict(b.tasks[request.task]), b.stock)
    for key in sorted(SESSIONS, key=lambda k: SESSIONS[k]["at"])[:-40]:
        del SESSIONS[key]  # keep the 40 most recent
    sid = uuid.uuid4().hex
    SESSIONS[sid] = {"session": session, "benchmark": b, "task": request.task, "at": time.time()}
    return {"session": sid, "observation": observation, "tools": openai_tools(request.toolset)}


def _session(sid: str) -> dict[str, Any]:
    if sid not in SESSIONS:
        raise HTTPException(404, "session expired; start a new one")
    SESSIONS[sid]["at"] = time.time()
    return SESSIONS[sid]


def _outcome(session: RetroRouteSession, tool: str, result: Any) -> dict[str, Any]:
    reward = result.get("score", {}).get("reward") if tool == "emit_routes" and isinstance(result, dict) else None
    return {
        "tool": tool,
        "result": result,
        "done": session.done,
        "reward": reward,
        "tool_calls_used": session.tool_calls,
        "tool_calls_remaining": max(0, session.max_tool_calls - session.tool_calls),
    }


@app.post("/api/session/{sid}/call")
def session_call(sid: str, request: ToolCall) -> dict[str, Any]:
    session = _session(sid)["session"]
    if request.tool not in session.tool_names:
        return {"tool": request.tool, "result": {"error": f"{request.tool} is not in the {session.toolset!r} toolset"}}
    try:
        result = getattr(session, request.tool)(**request.arguments)
    except TypeError as exc:
        result = {"error": f"bad arguments: {exc}"}
    return _outcome(session, request.tool, result)


@app.post("/api/session/{sid}/reference")
def session_reference(sid: str, reveal: int = 0) -> dict[str, Any]:
    """Submit the answer key, to see what a perfect episode scores (held-out tasks only after reveal)."""
    entry = _session(sid)
    task = entry["benchmark"].tasks[entry["task"]]
    if task["split"] in HELD_OUT and not reveal:
        raise HTTPException(403, "reveal the reference first")
    retro = RetroTask.from_dict(task)
    submission = routes_to_submission(
        retro.target_smiles, retro.reference_routes[: retro.max_routes], entry["session"].stock, source="answer key"
    )
    out = _outcome(entry["session"], "emit_routes", entry["session"].emit_routes(submission))
    return {**out, "arguments": {"submission": submission}}


def _route(b: Benchmark, route: dict[str, Any]) -> dict[str, Any]:
    steps = []
    for step in route["steps"]:
        family = step_family(step["reactants"], step["product"])
        steps.append(
            {
                "product": step["product"],
                "reactants": step["reactants"],
                "family": family,
                "phrase": reaction_phrase(family, step["reactants"], step["product"]),
                "text": describe(family, step["reactants"], step["product"]),
            }
        )
    molecules = {m for s in route["steps"] for m in [s["product"], *s["reactants"]]}
    return {
        "steps": steps,
        "source": [{k: v for k, v in s.items() if k in ("group_id", "name")} for s in route.get("source", [])],
        "in_stock": {m: canonicalize_smiles(m) in b.stock for m in molecules},
    }


@app.get("/api/task")
def task(benchmark: str, id: str, reveal: int = 0, dataset: str | None = None) -> dict[str, Any]:
    b = bench(benchmark)
    if id not in b.tasks:
        raise HTTPException(404, f"no task {id}")
    t = b.tasks[id]
    row = next(r for r in b.rows if r["id"] == id)
    hidden = t["split"] in HELD_OUT and not reveal
    prompt = PROMPT.format(
        target=t["target_smiles"], max_steps=t["max_steps"], min_routes=t["min_routes"], max_routes=t["max_routes"]
    )
    d = STATE["datasets"].get(dataset or "")
    runs = []
    for run in STATE["runs"].values():
        if run.benchmark != benchmark or id not in run.identity.get("task_ids", []):
            continue
        for episode in run.episodes:
            if episode["task_id"] == id:
                runs.append({"run": run.name, "label": run.identity.get("label"), **episode})
    return {
        "row": row,
        "task": {k: v for k, v in t.items() if k != "reference_routes"},
        "prompt": prompt,
        "difficulty": b.difficulty.get(id),
        "hidden": hidden,
        "target_in_stock": canonicalize_smiles(t["target_smiles"]) in b.stock,
        "routes": None if hidden else [_route(b, r) for r in t["reference_routes"]],
        "sft": d.by_task.get(id, []) if d else [],
        "runs": runs,
    }


@app.get("/api/sft_row")
def sft_row(dataset: str, id: str) -> dict[str, Any]:
    d = STATE["datasets"].get(dataset)
    if not d or id not in d.by_id:
        raise HTTPException(404, "no such row")
    row = d.read(id)
    return {"meta": d.by_id[id], "messages": row["messages"], "tools": json.loads(row["tools"])}


@app.get("/api/episode")
def episode(run: str, file: str) -> dict[str, Any]:
    r = STATE["runs"].get(run)
    if not r or "/" in file or not (r.root / "episodes" / file).exists():
        raise HTTPException(404, "no such episode")
    row = json.loads((r.root / "episodes" / file).read_text())
    layout = build_sft.LAYOUTS.get(r.identity.get("provider"), "chat")
    if layout == "chat" and not row.get("prompt"):
        row = {**row, "prompt": "(this run did not store its prompt)"}
    try:
        messages = build_sft.CONVERTERS[layout](row)
    except Exception as exc:  # an unusual transcript: show it raw rather than fail
        messages = [{"role": "user", "content": f"(could not normalise this transcript: {exc})"}]
    return {
        "messages": messages,
        "score": {
            k: row.get(k)
            for k in (
                "reward",
                "valid",
                "exact_match",
                "components",
                "hard_failures",
                "errors",
                "submission_coerced",
                "auto_emitted",
                "usage",
            )
        },
    }


@app.get("/api/run")
def run_detail(run: str) -> dict[str, Any]:
    r = STATE["runs"].get(run)
    if not r:
        raise HTTPException(404, f"no run {run}")
    return {"summary": r.summary(), "episodes": r.episodes}


@lru_cache(maxsize=20000)
def _svg(smiles: str, width: int, height: int) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'></svg>"
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.clearBackground = False
    options.bondLineWidth = 1.3
    options.padding = 0.05
    options.minFontSize = 10
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    text = drawer.GetDrawingText().replace("#000000", "currentColor")
    return text[text.index("<svg") :]


@app.get("/api/mol.svg")
def mol_svg(smi: str, w: int = Query(200, le=800), h: int = Query(110, le=600)) -> Response:
    return Response(_svg(smi, w, h), media_type="image/svg+xml", headers={"Cache-Control": "max-age=86400"})


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    # Fail before loading anything if another explorer already holds the port.
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise SystemExit(
                f"port {args.port} is in use (another explorer?); open http://127.0.0.1:{args.port} "
                f"or start this one with --port {args.port + 1}"
            )
    STATE["benchmarks"] = discover_benchmarks()
    STATE["datasets"] = discover_datasets()
    STATE["runs"] = discover_runs(STATE["benchmarks"])
    print(
        f"ready: {len(STATE['benchmarks'])} benchmarks, {len(STATE['datasets'])} SFT datasets, {len(STATE['runs'])} runs "
        f"on http://127.0.0.1:{args.port}",
        file=sys.stderr,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
