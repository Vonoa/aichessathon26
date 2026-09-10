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

### 2026-09-07 — KX-vs-K mate driver (branch `kx-mate-driver`)

- `_mopup(board)` / jitted `_mopup_jit`: white-relative, 0 unless exactly one side is a
  bare king. Adds, for the winning side, `10 * centre-manhattan-distance(lone king)` +
  `4 * (7 - Chebyshev(kings))` -- push the lone king to a corner, march the winning king
  up. Added after the tapered blend, before the side-to-move flip, in both
  `_evaluate_reference` and `_evaluate_jit`; the equivalence test pins them equal.
- Targets the recurring fifty-move / threefold draws vs `phase5eval`/`phase5jit`: the eval
  was flat in KQ/KR-vs-K (no signal to close the net), so the search shuffled to the
  50-move rule. Small vs the material lead -- a gradient to convert by.
- Golden `"8/2k5/8/8/8/8/5K2/6R1 w"` moved 540 -> 578 (deliberate; only the bare-king
  case changes). Colour symmetry preserved.
- Gate + arena vs `versions/phase5jit` pending.

### 2026-09-07 — Non-linear king safety (branch `king-safety-nonlinear`)

- The attacker-zone term in `_king_safety_mg` / `_king_safety_side` was linear (`-weight`
  per enemy piece touching the king's 3x3 zone). Real king danger is roughly quadratic in
  the number of attackers -- one piece is nothing, three is often decisive. Rated 56 was
  lost to a Greek-gift sac the eval never flagged (up ~3 pieces of material, eval said
  -33; a proper king-safety term scores that near -400).
- New: sum `unit * (zone squares the piece attacks)` over every attacker, count the
  attackers, then `1 -> -danger`, `2+ -> -min(450, danger^2 * 65 // 100)`. Units
  P2 N3 B3 R5 Q7. All first-guess constants -- arena calibrates. `ATTACKER_ZONE_WEIGHT`
  removed, replaced by `KING_ATTACK_UNIT` + `KING_DANGER_SCALE` + `KING_DANGER_MAX`.
- Reference and jit changed identically; equivalence test holds. Several middlegame golden
  values re-captured (a deliberate eval rework); the symmetric golden stays 0, the
  phase-faded endgame goldens are unchanged.
- Gate + arena vs `versions/phase5jit` pending.

### 2026-09-07 — Endgame king activity (branch `endgame-king-activity`)

- New white-relative eval term `_king_activity` / jitted `_king_activity_jit`, added after
  the tapered blend alongside the mate driver. Fires only when game phase < 8, one side
  leads by >= 200 cp (pawn+piece value), and the trailing side is not a bare king (that's
  the mate driver's job). Rewards the leading side `6 * (7 - Chebyshev(kings))`, faded
  linearly to 0 at the phase ceiling -- a small tie-breaker that turns aimless shuffling
  (Rated 57: promoted a queen, drew by threefold with the king wandering) into "march the
  king toward the enemy to finish".
- `_material_lead` helper added. Reference and jit changed identically; equivalence test
  covers new endgame FENs. Zero golden-value changes (the bare-king guard + phase gate +
  lead gate keep every golden position out of scope).
- Gate + arena vs `versions/phase5jit` pending.

### 2026-09-08 — Tweaks from the 18-game review (branch `game-review-tweaks`)

Reviewed rounds 43-60 (rounds <= 50 ran older engines; 55-60 are the current one).
Two failure modes dominate: king-hunt losses (king left in the centre / unsound sac
accepted) and conversion stalls (won position shuffled to a draw, slow KQ/KR-vs-K mates).

- **Mop-up weights 10/4 -> 16/8.** Rated 58 took ~30 moves to mate KQ-vs-K; the driver's
  gradient was too shallow at platform depth. Golden `"8/2k5/.../6R1 w"` 578 -> 604.
- **Flat-position budget cut -- TRIED AND REVERTED.** Cut `_budget_s` by 4x when the last
  8 completed-search scores were all within +-35 cp. Meant for dead K+R-vs-K+R shuffles
  (Rated 43 / 60 ran the clock down to 4 s / 10 s) but +-35 for 8 moves also matches a
  normal balanced middlegame -- in the 10 s + 0.1 s arena it dropped the search to depth
  1-2 and lost 9 of 16 (Elo -89), with a death-spiral (a timed-out depth-1 move logs
  score 0, which keeps the cut engaged). A real clock fix needs `fullmove >= ~40`, a
  tighter band, and a floor -- deferred.

King-in-centre eval term and a position test set from the losses are noted but not done
here. Next eval work: king defenders + escape squares, rook-on-open-file, outposts.

### 2026-09-08 -- Checks in quiescence + ordered fallback move (branch `qsearch-checks`)

- **Quiet checks in qsearch.** `_qsearch` gains a `qply` counter; for the first
  `_QS_CHECK_PLIES` (1) plies past the horizon, when not in check, it also generates up
  to `_QS_CHECK_CAP` (6) non-capturing checking moves alongside the captures/promotions.
  Purely a chance to raise the score -- a bad check just scores low and is ignored -- so
  stand-pat stays sound. Catches the forcing shot (knight fork with check, back-rank
  skewer) a captures-only qsearch walked past. `board.gives_check` is only called on
  quiet non-promo moves while under the cap and inside the check window, so deeper
  qnodes pay nothing. New unit test: Nf4+ fork, stand-pat -332 -> qsearch +210.
- **Ordered fallback move.** `search_move` seeded `best` with `legal[0]` (raw
  python-chess order); an interrupted first ID pass under severe time pressure then
  returned a near-random move. Now `best = _ordered(board, legal)[0]` -- MVV-LVA/history
  order, deterministic, strictly better. Matters in the endgame clock scrambles
  (Rated 60/61 finished under 13 s).
- Not expected to fix the Greek-gift losses (R56): the mate there needs a quiet
  non-check follow-up (Qh5) two plies past the sac -- outside a one-ply check window.
- Gate + bench (watch the nps hit from `gives_check`) + arena vs `versions/phase5jit`
  pending.

### 2026-09-08 -- Persistent transposition table (branch `persistent-tt`)

The per-move dict TT is replaced by a fixed-size table kept across moves within a game
(docs/PLAN.md Phase 4: "fixed-size, replace-by-depth, kept across moves ... not an
unbounded dict"). A fresh process per game resets it; tests call `search._reset_tt()`.

- Two flat `np.uint64` arrays, `_TT_BITS = 22` -> 4.2M slots, 64 MB, allocated at import,
  no per-entry Python objects and no GC churn. Open-addressed, one probe at
  `slot = key64 & mask`; `key64 = hash(_transposition_key()) & 2**64-1` (a tuple of
  ints/bool/None, so `hash` is stable across runs), 0 reserved for the empty slot.
- One entry packed per uint64: value (16-bit, offset-encoded), depth (8), flag (2),
  best-move code (16), generation (16). `_move_code` / `_code_move` round-trip a
  `chess.Move` through 16 bits (from | to<<6 | promo<<12).
- Replace-by-depth *within* a generation; a slot from an older search or a different
  position is always taken. `search_move` bumps `_tt_gen` instead of clearing.
- `_TT_VALUE_MAX = 30_000`: scores outside +-this are not stored -- they can't fit the
  field, and this subsumes the "don't cache mate scores" rule (measured from the root,
  wrong down another path) without a separate check. The in-search-repetition draw is
  still returned before the store, so it is never cached; a value *derived* from one
  deeper down is mildly path-dependent but the generation stamp refreshes it within a
  move or two -- the standard trade-off for a persistent table.
- Local check: a second search of the same position at fixed depth 4 visits 27 nodes
  vs 8,594 cold, same score. KR-vs-K depth 6 fills 3,399 slots, none oversized.
- Tests: entry round-trip, persist-and-cut-nodes, no-oversized-value, `_reset_tt` wipe;
  `_reset_tt()` added to the autouse fixtures in test_engine.py / test_positions.py and
  to `test_is_deterministic` (the carried-over table is otherwise a hidden input).
- Gate + bench (watch for an nps change from the packed probe) + arena vs
  `versions/phase5jit` pending.

### 2026-09-08 -- Aspiration windows + principal variation search (branch `aspiration`)

Next two items of the Phase 4 pruning stack, on top of the persistent TT.

- **PVS.** In `_search_root` and `_negamax`, the first (PV) move is searched with the
  full (alpha, beta) window; every later move is scouted with a null window
  (-alpha-1, -alpha) and only re-searched in full if the scout beats alpha (for a
  reduced LMR scout, any beat triggers the re-search; for an unreduced one, only a
  beat strictly below beta). Folds the old separate LMR re-search into the same path.
- **Aspiration.** `search_move` now drives the root through `_aspiration_search`: for
  depth > 3 the window is `last_score +- _ASPIRATION` (40 cp). A result outside it
  widens that side to infinity and re-searches once -- cheap now that the persistent
  TT carries the tree between the narrow and wide passes.
- Local checks: mate-in-1 still solved, `get_move` still deterministic, and a
  fixed-depth root search with the narrow window stays within ~10% of the
  full-window node count (no blow-up from re-searches).
- Gate + bench (depth reached in fixed time is the signal) + arena vs
  `versions/phase5jit` pending. If the bench shows aspiration costing depth on
  volatile scores, widen `_ASPIRATION` or gate it to deeper plies; PVS stays either way.

### 2026-09-08 -- Syzygy tablebases at the root (branch `syzygy`)

3-man WDL + DTZ Syzygy files ship in `syzygy/` (~26 KB: KP/KQ/KR/KB/KN vs K). When the
board is down to `_TB_MAX_PIECES` (5) men or fewer and the files are present,
`search_move` picks the move straight from the tables and skips the search:

- `_tb_root_move`: for each legal move, probe WDL (outcome) and DTZ (plies-to-zero) from
  our point of view. Rank by best WDL first; among those, when winning: mate-in-1, then a
  fifty-move-counter-resetting move (capture / pawn push), then smallest DTZ; when losing:
  drag it out (largest |DTZ|, keep the counter running); a draw just holds.
- `chess.syzygy.open_tablebase("syzygy")` at import, wrapped so a missing/empty dir or a
  bad file leaves `_tablebase = None` and nothing changes. Any probe gap (piece count not
  covered, missing file) makes `_tb_root_move` return None and the search runs as normal.
- A **first attempt** put a WDL probe inside `_negamax`/`_qsearch`, but that suppressed
  the search and the 1-ply mop-up gradient just orbited the lone king without mating
  (KRvK not mated in 40+). Reverted; root-only DTZ selection mates KRvK in 27 plies,
  KQvK in 11, converts won K+P-vs-K, and correctly holds drawn K+P-vs-K (the Rated
  61/62 failure mode).
- **Packaging:** the folder is not auto-detected. Build with
  `uv run python -m harness.package --include syzygy` or the tables do not ship and the
  engine silently falls back to search.
- Gate + arena vs `versions/phase5jit` pending. `make gate` skips the syzygy tests if
  `search._tablebase is None`.

### 2026-09-08 -- Jitted move generator, Phase A (branch `jit-movegen`)

`movegen.py`: a numba-jitted bitboard generator, built and validated on its own before
it touches `search.py` (docs/PLAN.md Phase 3 -- "the hardest phase"; a movegen bug is a
lost game).

- Representation: `bb` uint64[2,6] (same layout as `evaluate._encode`) + `state` int64[4]
  `[turn, castling, ep, halfmove]`. Moves packed into int32 (`from | to<<6 | promo<<12 |
  flag<<15`, flag = normal / double-push / en-passant / castle).
- Jitted: `_gen` (pseudo-legal), `_make` / `_unmake` (in place, undo word), `_attacked_by`
  (occupancy-parametrised), `_gen_legal` (gen + king-safety filter), `_perft`.
- **Validated:** perft matches the published numbers for all six standard positions
  (initial d5 4,865,609; Kiwipete d4 4,085,603; positions 3-6) *and* python-chess's own
  counts. The legal-move SET matches python-chess move-for-move over 21,307 positions
  from 600 pseudo-random games -- zero mismatches (castling, en passant, promotions
  included).
- **Speed:** ~4.4-5.2M nps for jitted perft vs python-chess's ~0.26M -- ~17x at the raw
  gen + make + legality + unmake work.
- Cost: the numba compile of `_gen` / `_gen_legal` is ~22 s at import (`_perft` is not
  warmed -- validation only). Fits the 90 s init budget with room; `cache=False` because
  `/tmp` is wiped per game.
- Tests: `tests/test_movegen.py` (perft to d3-d4, legal-set divergence, encode round
  trip). `movegen.py` added to the mypy files list.
- **Not integrated.** Phase B decides how deep to wire it into the search -- the real
  win needs the search to carry a lightweight board the whole way down, not push/pop a
  chess.Board per node.

### 2026-09-08 -- Jitted move generator wired into the search, Phase B (branch `jit-movegen`)

The search now runs on `movegen.py`'s bitboard board: `(bb, state)` numpy arrays,
`_gen_legal` / `_make` / `_unmake` in place of `board.legal_moves` / `push` / `pop`,
`_zobrist` for the TT and repetition keys, `_attacked_by` for check detection, and
`evaluate._evaluate_jit` read straight off `bb`. A `chess.Board` is touched only at the
root -- parse the FEN, probe Syzygy, format the UCI reply. `agent.py` keys `_history` by
`movegen.zobrist`.

- **Sub-steps, each validated:** eval bridge == `evaluate.evaluate` over 8,432 positions
  (0 mismatch); Zobrist transposition-consistent + make/unmake round-trips + deterministic;
  search score == old python-chess search on **561/562 positions at fixed depth 3**
  (worst gap 31 cp, the known LMR-fail-soft ordering effect). Move matches 69% -- the
  rest are equal-value alternatives (the generator's move order differs from
  python-chess's, so the stable-sort tie-break picks differently).
- **Speed: ~2.4x nps, +1-2 plies.** Bench: open middlegame d3->d4 (~22k->49k nps),
  sharp middlegame d3->d5 (~25k->60k), rook endgame d7->d8 (~30k->74k). Base node cost
  ~40 us -> ~15 us.
- `search.warm_up()` runs one tiny search at import so numba compiles the whole path in
  the ~14 s import, not on move one. First real move: d3 in 188 ms, no compile stall.
- `_insufficient` is a coarse jitted check (KvK, K+minor vs K); same-colour KBvKB and
  KNNvK fall through to the eval / repetition -- rare, never a blunder.
- Tests: the six search-internal tests rewritten to the `(bb, state)` interface; gate
  green (245 pass, 1 xfail).
- **Still open:** SEE / NMP un-parked on the fast substrate (both should flip positive
  now); a full arena vs `versions/phase5jit` + `make zip` smoke before any upload;
  `docs/ENGINE.md` / `PLAN.md` describe the old python-chess search and need a rewrite.

### 2026-09-08 -- SEE + null-move pruning on the jitted substrate (branch `jit-movegen`)

Both were parked at ~break-even on the old python-chess search (a node was ~40 us so the
per-node cost swallowed the pruning). Re-added now that a node is ~15 us.

- **NMP.** In `_negamax` after the TT probe: not in check, `depth >= _NMP_MIN_DEPTH` (3),
  beta not a mate score, side to move has a piece (`_has_non_pawn_material` -- the
  zugzwang guard), and static eval already `>= beta`. Flip `state[0]`/ep, search
  `depth - 1 - r` (r = 3 at depth >= 6 else 2) zero-window at beta; a fail-high prunes.
- **SEE.** `movegen._see(bb, turn, code)` -- a jitted static exchange evaluation with
  x-ray, `_attackers_to` off the ray tables. Quiescence drops a non-promo capture whose
  SEE is worse than `-_SEE_QS_MARGIN` (90), unless it captures equal-or-up (structurally
  safe, skip the call) or gives check.
- **Bench, isolated:** NMP alone takes the rook endgame d8 -> d10 at no nps cost (it
  finally pays -- reaches the depth where R=3 compounds). SEE alone is ~3% nps, no bench
  depth change (its value is tactical -- not misevaluating a losing-capture line). Both
  together: open middlegame d4 -> d5, rook d8 -> d10, sharp unchanged; overall nps
  64k -> 60k.
- SEE unit-tested (undefended / defended / x-ray / en passant); NMP tested to cut nodes
  without changing the score; `_has_non_pawn_material` tested. Gate green (249 pass).
- Arena vs `versions/phase5jit` pending, then the rated ladder.
