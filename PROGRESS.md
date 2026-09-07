# Progress log

Running record of what's been done. Newest entry at the bottom. The plan and phase
definitions live in [docs/PLAN.md](docs/PLAN.md).

## Current status (2026-09-06)

- **Engine:** Phase 4b merged to `main`. Negamax + alpha-beta + iterative deepening,
  transposition table (per move) + previous-iteration move first, MVV-LVA + killers +
  history ordering, quiescence search at the horizon, bitboard material + pawn-PST eval,
  mate-distance scoring, history-driven draw awareness, increment-aware time budget.
- **Strength:** each frozen step beats the last decisively — 3b 83.8% vs 2, 3c 67.5% vs
  3b, 4a 100% vs 3c. Beats every baseline (97.5% vs minimax, 87.5% vs numba). Won its
  first rated game. Zero self-inflicted losses throughout.
- **Tests:** run by `make gate` (ruff + mypy strict + pytest + 2 games).
- **Submitted:** first upload passed validation (ready in 0.5 s, won its smoke game). Live
  in the hourly rated rounds.
- **Reordered:** the engine already beats every baseline (97.5% vs minimax, 87.5% vs
  numba), so Phase 4 (pruning stack) and Phase 5 (real eval) come before the big, risky
  3e jitted move generator. "Get it right, then get it fast."
- **In flight:** branch `budget-longgame` — `_budget_s` reworked to survive long games.
- **Next:** Phase 5 (real evaluation). Pruning-stack items (null-move, LMR, PVS, persistent
  TT) come *after* Phase 5, when a real eval makes games decisive enough to measure and
  4b's gain can be validated properly.

## Branches / versions

- `main` — Phase 4a engine.
- `versions/phase{1,2,3b,3c,4a}/` — frozen arena opponents.
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

### 2026-09-06 — Phase 3c: bitboard evaluation (branch `phase-3-eval`)

- `evaluate()` rewritten: material via `int.bit_count()` on masked piece bitboards, pawn
  table walked by bit iteration (`sq ^ 56` mirrors for Black). No more `board.pieces()`
  SquareSet allocation - a dozen objects per call gone.
- Same values as before, so it is a pure speed change. `test_bitboard_eval_matches_reference`
  pins the new output against the old SquareSet formula across six positions.
- Decision: the standalone "numba the eval" step is dropped. `baselines/numba` in this
  repo already shows jitting a small eval is "barely stronger", and the bench says the
  move generator, not the eval, is the per-node cost. numba goes into 3e.

### 2026-09-06 — First submission

- `submission.zip` (agent.py, evaluate.py, search.py at the zip root, 12 KB unzipped)
  passed platform validation: `ready in 0.5 s of the 90 s init budget`, won its first
  smoke game by checkmate, slowest move 3.2 s. Live in the hourly rated rounds.
- Smoke game 2 ended `draw by ply_cap` - more evidence the thin eval has no plan in
  balanced positions and needs Phase 5.

### 2026-09-06 — Phase 4a: quiescence search (branch `phase-4-quiescence`)

- `_negamax` at `depth <= 0` now calls `_qsearch` instead of `evaluate()` directly.
  `_qsearch` stands pat on the static eval, then searches only captures and promotions
  (all legal evasions when in check) with alpha-beta until the position is quiet. Ply is
  hard-capped at `_MAX_DEPTH + 32` as a safety net; `_tick` runs so it respects the clock.
- Fixes the horizon effect: the eval was being read one ply before a recapture.
- Tests: `_qsearch` resolves a hanging rook (>= +450, and more than the static eval), and
  leaves a quiet position exactly at the static eval.

### 2026-09-06 — First rated game

- Won a rated game by checkmate against another team's agent. Stockfish-16 depth-16
  review: 86.8% accuracy, 51 ACPL, 45 best / 17 excellent / 10 good, but **7 blunders**
  and it took ~80 moves to convert a totally won position (king shuffling until pawn
  promotions forced a mate). Finished with 4.5 s on the clock to the opponent's 21 s.
- Read: the blunders are depth/tactics; the 80-move conversion is missing mating
  technique = the Phase 5 signal. The tight clock is `moves_left` pinned at 20 being too
  aggressive in a 91-move game - a minor later tweak.

### 2026-09-06 — Phase 4b: killers + history (branch `phase-4-ordering`)

- `_killers` (two flat slots per ply) and `_hist` (4096 from/to counts), both reset each
  move. A quiet move that beta-cuts is stored as a killer for its ply and adds `depth*depth`
  to its history score.
- `_ordered` now takes `ply` and ranks: captures/promotions (MVV-LVA) > killer 0 > killer
  1 > quiet moves by history. The TT move is still forced to the front by the caller.
  `_qsearch` and `_search_root` pass no ply, so they keep pure capture ordering.
- Test: `_record_cutoff` updates the killer slot and adds `depth*depth` to history.

### 2026-09-06 — Budget: survive long games (branch `budget-longgame`)

- `_budget_s`: `moves_left` floor raised 20 -> 30 and the horizon to `56 - fullmove`, the
  increment share cut 0.8 -> 0.5, plus two hard caps: never more than a third of the clock
  on one move, and always keep `_RESERVE_MS` (300 -> 500 ms) on the clock. The old formula
  converged to a ~2 s clock in a long game; this converges near 7-8 s at the real 500 ms
  increment. First rated game ended with 4.5 s to the opponent's 21 s - this is the fix.
- Also captured a quick audit's deferred findings in `audit.md`: qsearch stalemate
  blindness (folds into 5b), qsearch has no delta pruning (with the pruning stack), dirty
  board after `_Timeout` (latent), TT path-dependent draw (persistent-TT step), history
  cap (after 3e), and that Phase 4b's gain is unproven on 20-game samples.

### 2026-09-07 — Jit the eval, step 1: encode + ray attacks + warm-up (branch `jit-eval`)

- The search is eval-cost-bound: the Phase 5 tapered eval is ~28x slower than the old
  material eval (~114 us/call), middlegame depth stuck at 2-3. The jit is the unlock.
  Building it in five verifiable increments (one commit each) against the golden-value
  test in `tests/test_engine.py`; see `HANDOVER.md` for the plan.
- Step 1 adds the scaffold to `evaluate.py`, `evaluate()` itself untouched:
  - `_encode(board)` -> `(pieces[2,6] uint64, occ[3] uint64, turn)`, built once per call.
  - `@njit _ray_attacks(occ, sq, dirs)` -- classical ray loop, indexes a `_BB_SQUARES`
    uint64 table instead of numba-fragile `1 << sq` shifts; includes the blocker square
    like `board.attacks_mask()`. `_BISHOP/_ROOK/_QUEEN_DIRS` as (file, rank) step pairs.
  - `_KNIGHT_ATTACKS` / `_KING_ATTACKS` uint64[64], copied from `chess.BB_*_ATTACKS`.
  - `_warm_up()` runs at import (all three dir sets, real dtypes) so numba compiles in the
    90 s budget, not on the clock. `import evaluate` is ~2.5 s; move-1 budget test still green.
- `tests/test_evaljit.py`: `_ray_attacks` matches python-chess's `BB_DIAG/RANK/FILE`
  lookup for every square across 5 occupancies; leaper + `_BB_SQUARES` tables match;
  `_encode` round-trips. 41 tests pass, ruff + mypy strict clean (numba added to the
  mypy `ignore_missing_imports` override).
- PROGRESS.md gap noted in `audit.md` (no entries for Phase 5 eval / contempt / check-ext /
  LMR / harness merge) is still open -- backfill on a separate pass.
