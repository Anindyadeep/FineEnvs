# retroeval-v3

Published at [AdithyaSK/RetroEnv-RL](https://huggingface.co/datasets/AdithyaSK/RetroEnv-RL)
(this directory is the repo root). Download it here with
`hf download AdithyaSK/RetroEnv-RL --repo-type dataset --local-dir benchmark/retroeval-v3 --exclude "retroeval-v2/*" "runs/*" "media/*"`.

28,039 route-planning tasks mined from PaRoutes v2 (Zenodo 7341155, CC-BY-4.0). Every
leaf is in the n1 stock (13,432 molecules); routes have 2 to 5 steps. v2 sampled its tasks and
then split them, so its eval set kept the pool's skew: 2- and 3-step routes only, half
of them from two reaction families. v3 designs the held-out splits first and fills train
with every task that shares nothing with them.

| Split | Tasks | 2 / 3 / 4 / 5 steps | Two-route | Heavy atoms ≤20 / 21–30 / 31–40 / 41+ | Easy / medium / hard |
|---|---:|---:|---:|---:|---:|
| train | 27,489 | 13,376 / 8,878 / 3,724 / 1,511 | 47 | 11,134 / 11,386 / 4,705 / 264 | 17,568 / 7,132 / 2,789 |
| dev | 150 | 42 / 42 / 36 / 30 | 15 | 42 / 51 / 45 / 12 | 60 / 56 / 34 |
| eval | 250 | 70 / 70 / 60 / 50 | 25 | 70 / 85 / 75 / 20 | 89 / 106 / 55 |
| stress | 150 | 42 / 42 / 36 / 30 | 15 | 35 / 51 / 47 / 17 | 58 / 56 / 36 |

The first reaction of the reference route (`label_difficulty.step_family`):

| First-step family | eval | dev | stress | train |
|---|---:|---:|---:|---:|
| alkylation / SNAr | 48 | 29 | 29 | 8,017 |
| one-reactant FGI | 41 | 25 | 25 | 6,658 |
| amide/ester coupling | 31 | 19 | 19 | 3,873 |
| other | 31 | 18 | 18 | 4,079 |
| Suzuki-type coupling | 20 | 12 | 12 | 1,426 |
| reductive amination | 20 | 12 | 12 | 1,529 |
| sulfonylation | 17 | 10 | 10 | 913 |
| Boc deprotection | 15 | 9 | 9 | 591 |
| urea / carbamate | 14 | 8 | 8 | 214 |
| Boc protection | 13 | 8 | 8 | 189 |

Use eval for the board, dev for model selection and RL validation curves, and keep
stress sealed as a second test. Tiers are the heuristic of `label_difficulty.py`; treat
them as a prior, not a measurement.

## How the splits were designed

1. **Mine.** `mine_route_pool.py --max-steps 5` keeps 37,135 targets whose routes have 2
   to 5 reactions, one reaction per molecule, every leaf in stock, and replay through the
   serving verifier.
2. **Held-out first.** eval, then dev, then stress are drawn in salted stable-hash order
   against quotas:
   - shortest route length: 28% / 28% / 24% / 20% for 2 / 3 / 4 / 5 steps
   - first-step family: halfway between the pool's mix and uniform
   - target heavy atoms: four bins
   - two-route tasks: 10%

   Rare strata choose first, so the common families cannot use up the depth slots. Each
   split takes at most one task per multi-ring scaffold and two per one-ring or acyclic
   one.
3. **Independent held-out tasks.** A held-out task must share no leakage group with any task
   already held out, so no two eval tasks share a patent, scaffold, intermediate, reaction
   or near-duplicate.
4. **Train is the rest.** Every other pool target joins train unless it touches a held-out
   group; 9,096 do.

The leakage groups are v2's: target, scaffold, route, intermediates and their scaffolds,
reaction, patent, and Morgan Tanimoto ≥ 0.90 between route molecules. One-ring and acyclic
molecules group by exact structure, as in v2.

One rule is new. The 33 multi-ring scaffolds found in at least 100 pool tasks, such as
biphenyl, indole, quinoline and phenylpiperazine, also group by exact structure
(`manifest.json` lists them). Under the plain scaffold rule, one eval biphenyl would bar 719
unrelated train tasks, and the held-out splits would cut train from 27,489 to 20,979.
Patent, intermediate, reaction and near-duplicate grouping still apply to these molecules.

## Checks

- The split audit passes under the recorded rules.
- The scripted reference expert passes every eval, dev and stress task within the
  16-turn, 32-call budget. It needs 7.4 / 9.4 / 11.6 / 14.2 calls for 2 / 3 / 4 / 5-step
  eval tasks, so a 5-step task leaves an agent 18 calls for search.
- Every one of the 28,141 reference routes and oracle graphs replays through the serving
  verifier (`audit.json`). The oracle scores 1.000 and an empty submission 0.050.
- No held-out task loses any precedent to the train-time precedent filter (see below), so
  held-out episodes see exactly what they would without it.

The precedent index is built from all 27,489 train tasks. A train task's own reactions, its
patent, its scaffolds and its near-duplicates are hidden from its precedent search
(`PrecedentIndex.hidden`). This mirrors what the split withholds from a held-out task, so a
policy trained on train cannot learn to look up answers that evaluation never offers.

## Rebuild the private files

From `09-retroenv`, after downloading `all_loaded_routes.json.gz` and `stock_n1.txt` into
`data/raw/paroutes-v2-benchmark/` (see `dataset/README.md`):

```bash
uv run python dataset/mine_route_pool.py --max-steps 5 --output .local/pool/pool-n1-s5.jsonl
uv run python dataset/build_benchmark_v3.py --output-dir benchmark/retroeval-v3
uv run python dataset/label_difficulty.py --benchmark-dir benchmark/retroeval-v3
uv run python dataset/audit_benchmark.py --benchmark-dir benchmark/retroeval-v3 \
  --expected-eval-tasks 250 --manifest-rules
```

The build is deterministic; compare the result with `checksums.json`. Private references,
the difficulty sidecar and the 27k public train rows stay out of git (`.gitignore`).
