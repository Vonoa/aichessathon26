# Handover — next session

Full history is in [PROGRESS.md](PROGRESS.md). This is the state + the immediate task.

## Where the engine is

`main` @ `3940d42`. One file ships: `agent.py` + `search.py` + `evaluate.py` (all root `*.py`
go in the zip). Current engine:

- **Search**: negamax + alpha-beta, iterative deepening, per-move transposition table
  (dict, cleared each move), quiescence search, killer + history move ordering, MVV-LVA,
  check extension, late-move reductions, contempt (draws score -25 cp from the root side).
- **Eval** (`evaluate.py`): tapered midgame/endgame. Material + PeSTO-style piece-square
  tables for all 6 pieces + pawn structure (doubled/isolated/passed) + mobility + phase-
  faded king safety. Side-to-move relative. **Weights are untuned placeholders.**
- **Time**: `agent.py` infers the increment from clock deltas; `_budget_s` paces for long
  games with a 500 ms reserve.
- **Strength**: 100% vs every starter baseline (incl. `baselines/numba`). ~+200 Elo over
  `versions/phase5eval` (the pre-fix version) on the seeded-opening arena, wide CI.
- **Submitted**: live zip has the PST-orientation fix + contempt. Check-extension and LMR
  are merged but NOT yet uploaded — re-`make zip` and upload after the jit lands.

## The problem the jit solves

The search is **eval-cost-bound**, not node-bound. ~10-17k nodes/sec, depth stuck at 2-3
in middlegames. `evaluate()` calls `board.attacks_mask()` ~24x per leaf (slider rays) for
mobility + king safety. Null-move (tried, reverted), check extension, and LMR all measured
**flat** because of this ceiling. Nothing in the search moves the needle until the eval is
fast. Target after jit: 80-150k nps, depth 5-7 in middlegames — then null-move/PVS/LMR all
start earning Elo.

## The task: jit the eval  (branch `jit-eval`, already created off main, empty)

Build in verifiable increments, one commit each. The golden-value test in
`tests/test_engine.py` (`_EVAL_GOLDEN`, `test_eval_golden_values`) is the guardrail — jitted
output must match the current eval, or a small delta is consciously documented.

1. **`_encode(board)` + jitted classical ray attacks + import-time warm-up.** Encode to
   fixed arrays once per call (12 piece `uint64` bitboards, occupancy, turn). `@njit`
   `_ray_attacks(occ, sq, deltas)` loops each direction from `sq` until off-board/blocked.
   Knight/king are precomputed `uint64[64]` tables. Call the jitted eval once at module
   import with the real dtypes so numba compiles inside the 90 s budget.
2. **Jitted material + tapered PST.** Port `_game_phase` + the MG/EG PST sum. Golden test.
3. **Jitted pawn structure** (file + front-span masks passed in as constants).
4. **Jitted mobility + king safety** via `_ray_attacks` — this is where the speed lands.
5. **Swap `evaluate()` to the jitted path**; keep the old body as `_evaluate_reference`
   for the golden test. Bench + arena vs `versions/phase5eval`.

## Gotchas

- **PeSTO tables in `evaluate.py` are `_flip_ranks`'d at import** — they're written rank-8-
  first, we index a1 = 0. Don't double-flip in the jit; either flip the raw lists once and
  bake the flipped arrays, or apply `sq ^ 56` consistently.
- **`_game_phase`** = sum of non-pawn material (24 = full midgame, 0 = bare kings). It was
  inverted once; keep it this way. Blend is `mg*phase + eg*(24-phase)`, `int(.../24)` for
  colour-symmetric truncation (not `//`).
- Eval is **side-to-move relative** (`return tapered if board.turn == WHITE else -tapered`).
- **numba `cache=True` is useless** — `/tmp` is wiped per game (AGENTS.md). Warm at import.
- **mypy `files`** in pyproject = only `agent.py`, `search.py`, `evaluate.py`. `harness/`
  was dropped after the upstream merge (it now has OS-specific `SIGKILL`/`killpg`).
- **Windows / PowerShell 5.1**: no `&&` (one command per line). If `uv` isn't found, open a
  new terminal or use `.venv\Scripts\python.exe -m ...`. Repo is under OneDrive, so `.git`
  lock errors happen intermittently — just retry.
- **Upstream was merged**: `make arena` is now 16 seeded-opening games with a 95%
  confidence interval — this is the real measurement tool (startpos mirror matches drew
  everything and were useless). `versions/phase{1,2,3b,3c,4a,4b,5,5eval}` still work as
  opponents. `make zip` now smoke-tests the built zip.

## Backlog after the jit (re-measure each on the seeded arena)

- Re-test **null-move** and add **PVS** (null-window re-search) — both need the depth the
  jit unlocks.
- **Texel tuning** the eval weights — overlaps the teammate's Phase 7 data pipeline;
  their original eval prototype is in `versions/phase5/`, they reference a `texel_tune.py`
  that doesn't exist yet.
- **KX-vs-K mate driver** (Phase 5b) — drive the lone king to a corner + king proximity;
  fixes 80-move won-endgame conversions. Also fixes qsearch stalemate blindness.
- **Syzygy 3-4-man tablebases** (`chess.syzygy` is in the base image; 5-man too big).
- **qsearch**: `generate_legal_captures()` + delta pruning.
- **Persistent fixed-size TT** (currently per-move dict).

## Rules / constraints (verify at aichessathon.com/docs before relying)

120 s + 0.5 s/move, 2 GB, one core, no network, 90 s import. Preinstalled: torch 2.13 CPU,
numpy 2.5, python-chess 1.11, onnxruntime 1.29, numba 0.67. No third-party engines or
published nets; PeSTO PST *values* are fine (common positional heuristic, not an engine).
10 uploads/team/day, latest valid plays, rated rounds hourly 08:00-22:00.

## Git / team

Repo `github.com/Vonoa/aichessathon` (private). Flow: branch -> push -> PR -> CI green ->
merge. `upstream` remote = the starter, pull its harness fixes with
`git fetch upstream && git merge upstream/main`. Teammate is on the Phase 7 data/training
pipeline.
