# Progress log

Running record of what's been done. Newest entry at the bottom. The plan and phase
definitions live in [docs/PLAN.md](docs/PLAN.md).

## Current status (2026-09-06)

- **Engine:** Phase 3b merged to `main`. Negamax + alpha-beta + iterative deepening,
  MVV-LVA ordering, transposition table (cleared per move), previous-iteration move first,
  material + pawn-PST eval, mate-distance scoring, history-driven draw awareness,
  increment-aware time budget.
- **Strength:** Phase 3b scored 83.8% vs frozen Phase 2 at 10 s + 0.1 s (+33 =1 -6);
  60-65% at 3 s + 0.05 s after the leaf-repetition fix. Phase 2 itself: 100% vs
  `baselines/greedy`, 55.5% vs frozen Phase 1. Zero self-inflicted losses throughout.
- **Tests:** 13 passing, run by `make gate` (ruff + mypy strict + pytest + 2 games).
- **In flight:** branch `time-management` — increment-aware `_budget_s` + determinism note.
- **Next:** 3c numba-jitted eval, 3d incremental eval, 3e jitted move generator.

## Branches / versions

- `main` — Phase 3b engine.
- `versions/phase1/`, `versions/phase2/`, `versions/phase3b/` — frozen arena opponents.
- Baselines: `random` < `greedy` (1-ply material) < `minimax` (2-ply) < `numba`.

## Log

### 2026-09-06 — Repo setup

- Split the single `agent.py` into `agent.py` (entrypoint + non-raising fallback),
  `search.py`, `evaluate.py`. Packager zips all root `*.py`, so this still submits fine.
- Team repo created at `github.com/Vonoa/aichessathon` (private). Starter kept as the
  `upstream` remote for pulling harness fixes. `main` protected — changes go via PR + CI.
- Wrote [docs/PLAN.md](docs/PLAN.md): phased plan, priority order (robustness -> speed ->
  pruning -> evaluation), competitive read, leakage checklist, performance rules,
  working agreement.

### 2026-09-06 — Phase 1: minimal working engine (merged, PR #1)

- `search.py`: negamax with fail-soft alpha-beta, iterative deepening keeping the last
  completed depth, MVV-LVA capture ordering, time budget = `time_left_ms / max(20,
  50 - fullmove)` capped at `clock - 300 ms`, node-counted clock checks.
- `evaluate.py`: material (P/N/B/R/Q = 100/320/330/500/900) + a pawn piece-square table,
  returned from the side-to-move's point of view for negamax.
- Result: **+31 =9 -0, 88.8% vs greedy.** No crashes or flags. The 9 draws were won
  games shuffled into a threefold repetition — the Phase 2 target.
- Frozen to `versions/phase1/`.

### 2026-09-06 — Audit

- [audit.md](audit.md): full read of the Phase 1 code. Confirmed the structure is sound
  and found the real gaps — no mate-distance scoring, no repetition/draw awareness,
  terminal-root re-raises through the fallback, MVV-LVA breaks for a king attacker, no
  tests. All folded into the Phase 2 work below.

### 2026-09-06 — Phase 2: robustness + draws (merged, PR #2)

- **Mate distance:** `_negamax` threads a ply counter and returns `-MATE + ply`, so the
  engine goes for the fastest mate and the longest defence; the search stops deepening
  once a forced mate is found.
- **Draw awareness:** `agent.py` accumulates `_history` (transposition keys of every
  position we've moved in this game). `_negamax` returns 0 for the 50-move rule,
  insufficient material, an in-search repetition, or reaching a position already seen in
  the game. This is what stops won games leaking to a draw.
- **Edge cases:** safe terminal root (no `StopIteration`), single-legal-move
  short-circuit, king-attacker MVV-LVA fix (piece-type ordinals, not centipawns),
  queen promotions ordered ahead of quiets, tighter clock check (every 255 nodes),
  `AGENT_DEBUG=1` prints depth/score/nodes/ms.
- Result: **100% vs greedy** (all 40 by checkmate — repetition draws gone), **55.5% vs
  Phase 1.** Zero self-inflicted losses.
- Frozen to `versions/phase2/`.

### 2026-09-06 — Test suite (PR `tests-and-gate`)

- `tests/test_engine.py`: mate-in-1 found, single-legal-move short-circuit, promotion
  UCI, six tricky FENs each return a legal move, determinism, and a move-1
  within-budget / did-search assertion (guards Phase 3 against numba compiling on the
  clock).
- `pytest` added to the dev group; `make gate` now runs it. 11 tests, ~0.8 s.

### 2026-09-06 — Phase 3 started: profiling (branch `phase-3-speed`)

- `tools/bench.py` (`make bench`): runs the search on three fixed positions (open
  middlegame, sharp middlegame, rook endgame) and prints depth / nodes / nodes-per-second.
  `search._last_depth` added as an observability global.
- **Baseline (pure Python, one core):**

  | position | depth | nodes | nodes/sec |
  |---|---|---|---|
  | open middlegame | 4 | 34,425 | 23,432 |
  | sharp middlegame | 4 | 36,720 | 23,247 |
  | rook endgame | 6 | 44,625 | 36,341 |
  | overall | | 115,770 | 27,070 |

  Depth 4 in a middlegame is the ceiling right now — the target for Phase 3 is depth 6-7.
- Plan for the rest of Phase 3: 3b transposition table, 3c numba-jitted evaluation
  (warmed at import), 3d incremental eval on push/pop, 3e jitted bitboard move generator
  (its own multi-PR sub-project, done last).

### 2026-09-06 — Phase 3b: transposition table (branch `phase-3-tt`)

- `_negamax` computes the position key once (leaves skip it entirely), probes `_tt` for a
  cached result usable at the current depth/window, and stores `(depth, score, bound,
  best move)` after searching. Bound kind (`_EXACT` / `_LOWER` / `_UPPER`) records whether
  the score is exact, a floor (beta cutoff), or a ceiling (failed low). Mate scores are
  not cached — ours are root-relative, so path-dependent. Table cleared each move; Phase 4
  makes it persistent + fixed-size.
- `_search_root` now takes the previous iteration's best move and searches it first — the
  "iterative deepening feeds move ordering" win the audit flagged as missing.
- Bench barely moved (middlegame still depth 4, ~-5% nodes/sec; endgame depth 6 -> 7).
  But the arena jumped: **83.8% vs frozen Phase 2 (+33 =1 -6) at 10 s + 0.1 s**, draws
  down from 73/100 to 1/40. The gain is move-ordering quality and plan stability, not raw
  depth.
- **Regression found in review and fixed:** moving the transposition-key computation
  below the `depth <= 0` leaf return also skipped the `is_repetition(2) / key in _seen`
  draw check at the horizon, so a repetition landing exactly on the last ply scored by
  material instead of 0. Invisible at 10 s (search is deep enough to catch it a ply up),
  but at 3 s + 0.05 s it leaked 9/20 games into threefolds vs a 3/20 control. Fix: run
  the draw check before the leaf return, computing the key only when `halfmove_clock >= 4`
  makes a repetition possible. Regression test added.

### 2026-09-06 — Time management (branch `time-management`)

- `_budget_s` adds `0.8 * increment` to each move's share of the clock: every move we make
  refills the clock by the increment, so it is time to spend, not hoard. Capped at
  `clock - 300 ms`, floored at 10 ms.
- First cut hardcoded the increment at 500 ms (the competition value). It scored **32.5%
  vs frozen Phase 3b at 3 s + 0.05 s** — the arena's real increment is 50 ms, so the
  budget assumed a 500 ms refill it never got and drained the clock in ~7 moves. No flags,
  just collapse. Lesson: never hardcode a clock parameter the harness can vary.
- Fix: `agent.get_move` infers the increment from the clock delta since our last move
  (`new_clock == old_clock - our_spend + increment`), measuring our own spend with
  `time.monotonic()`, and passes it through. Defaults to 0 (the old safe formula) until
  the second move. Our spend measures a hair short of the referee's, biasing the estimate
  low - the safe direction.
- Determinism stated in the module docstring (no RNG, stable sort over fixed move order,
  first-seen tie-break). Tests added for `_budget_s` sanity and `_infer_increment`.
- Carry-overs still open: Phase 2's 300-game clean bar, `_TT_MAX` clear-vs-evict, and the
  persistent-TT path-dependence (deferred to Phase 4 / a longer run).
