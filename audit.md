# Engine audit

**History (all 2026-09-06 unless noted):** Phase 1 → Phase 2 (mate distance, draws) → 3a bench →
3b transposition table → 3b–4b full read → Phase 5 eval / contempt / check-ext → **this pass
(2026-09-07): re-run after the upstream-harness merge + LMR.**

**Branch:** `lmr`. `search.py` has **uncommitted** LMR. HEAD carries the committed check
extension. The upstream harness was merged (PRs #16–#18): seeded openings, suspend/resume, a big
`sandbox.py` / `arena.py` / `referee.py` rewrite. Two contributors: `Aled von Oppell` (20
commits) and `Rezqy <rez-qy@hotmail.com>` (5 — the "Negamax search algorithm" + eval-merge
commits). Frozen opponents through `versions/phase5eval/`.

Findings verified by running the code. `ruff` + `mypy --strict` clean, **37 tests pass**.

---

## 0. The two headline corrections vs the last pass

### The 13/20 "crash" run was a harness / machine-load artifact — resolved

Last pass, `current vs phase4b` reported `crash 13`. That run was on the **old local harness**
(pre-merge). Re-run on the **new** harness, sequentially, nothing else on the machine:

| current (`lmr`) vs | TC | result | terminations |
|---|---|---|---|
| `versions/phase4b` | 10 s + 0.1 s ×20 | **+12 =2 −6, 65 %** (Elo +108) | checkmate 18, threefold 2 — **0 crash** |
| `baselines/minimax` | 10 s + 0.1 s ×12 | **+12 =0 −0, 100 %** | checkmate 12 |
| `versions/phase2` | 3 s + 0.05 s ×16 | **+15 =1 −0, 96.9 %** | checkmate 15, fifty-moves 1 — **0 crash** |

48 clean games, zero crashes. Direct stress (`get_move` on Fool's-mate, near-stalemate, and
sharp tactical FENs) also never crashed — terminal positions correctly return `"0000"`. The
upstream merge rewrote process handling in `sandbox.py`; the old harness evidently mishandled a
slow agent under load. **One** crash did appear in a single `harness.play` game while three
other jobs ran concurrently, attributed to the *frozen phase4b* agent — Windows subprocess
flakiness under contention, not an engine defect. Worth a watch (the platform runs two of your
games at once, though in separate containers).

### Playing strength did NOT regress — it improved

Last pass concluded "strength regressed" from the bench (middlegame depth 4 → 2) and the
artifact crash run. **The clean arena overrules that:** +65 % / Elo +108 vs phase4b, 100 % vs
minimax, 96.9 % vs phase2, and the fifty-move draw floods are gone (seeded openings + a real
eval that actually has a plan). A smarter eval at depth 2–3 beats a material-only eval at depth
4. The slowness is still worth fixing — more depth on top of this eval is strictly better — but
it is **not a blocker and not a regression.**

---

## 1. New finding — determinism is broken under time pressure *(Important)*

The `search.py` docstring claims: *"The same position and clock always produce the same move."*
**This is false.** Verified: `get_move(fen, 1500)` on one FEN, 8 runs →
`['a1c1','f4b8','a1c1','f4b8','f4b8','a1c1','f4b8','f4b8']`.

The mechanism: the search is fully deterministic **at a fixed depth** (`_search_root` at a fixed
depth returns `f3e5 +59` every time; no RNG, stable sort, insertion-ordered dict). What jitters
is **which iterative-deepening depth completes before the deadline** — pure wall-clock timing.
The Phase 5 eval (§2) makes this far worse: searches are shallow and land on a depth boundary
almost every move, so machine jitter flips the result between depth N and N+1.

`test_is_deterministic` passes only because its position + 2000 ms happens to land the same both
times. PLAN wants "reproduce a lost game and explain any move at the finals" — that is not
currently possible from FEN + clock alone.

Fix options: correct the docstring claim; log the completed depth beside each move for replay;
add a fixed-depth / fixed-nodes repro mode; consider a small hysteresis so a marginal deeper
depth doesn't change the move.

---

## 2. `evaluate.py` — Phase 5 tapered eval: correct, ~28× too slow

**Cost (clean, separate processes):** phase4b eval **4.1 µs/call** → phase5 eval **113.9 µs/call**.
Bench: open middlegame **depth 3**, sharp middlegame **depth 2**, rook endgame depth 6, overall
~8 k nps (was ~40 k).

### Structure (unchanged from last pass, re-confirmed)

Tapered MG/EG: `phase = _game_phase()` (Σ weight·count over N1 B1 R2 Q4, `min(24, ·)`), then
`int((mg·phase + eg·(24−phase)) / 24)`. `int()` (truncate toward zero) not `//` keeps the score
colour-symmetric. Material 100/320/330/500/900; PeSTO-style MG/EG PST per piece,
`_flip_ranks`ed once at import to a1 = 0; pawn structure (doubled −12, isolated −10, passed by
rank); mobility (`attacks_mask` count × weight, MG/EG); king safety (pawn shield, open/semi-open
files, attacker-zone pressure) added to `mg` only and faded by the phase blend.

### Verified

- **Colour-symmetric** on every FEN tried (`test_eval_is_colour_symmetric` + 9 pinned golden
  values + start = 0 + material/knight/pawn-advance direction tests, all pass).
- **`_game_phase` is correct now** — the currently checked-out file has `phase = 0; phase += …`.
  The **`versions/phase5/` freeze has it inverted** (`phase = 24; phase -= …` with `//`) — a
  known-bad freeze; use `versions/phase5eval/` as the reference.
- **Stalemate / insufficient material → 0** inside `evaluate()`. This incidentally closes most
  of the earlier "qsearch is stalemate-blind" gap — a stalemate reached by a capture in
  `_qsearch` now scores 0. Cost: `board.is_stalemate()` is a full movegen (**~6 µs**) at every
  leaf, and `_negamax` already tested `not moves` while `_qsearch`'s quiet branch builds the
  move list itself — so `evaluate()` generates moves a **third** time. Remove it: have the
  callers, which already know, pass the fact down.

### The cost, itemised

`mobility` + `king_safety` `attacks_mask` loops over every piece; the O(pawns²) passed-pawn
scan with a per-call dict rebuild (run twice per `evaluate()`); the redundant `is_stalemate()`.
The docstring says the plan is to tune this version, then port the tuned tables into a jitted
bitboard eval — that port has to happen before real depth comes back.

---

## 3. `search.py`

Negamax / alpha-beta / ID / TT / quiescence / killers+history — **unchanged and still verified
sound** (TT returns identical best move and score neutered; mate distance correct;
leaf-repetition draw check runs before the `depth ≤ 0` return). Three newer pieces:

### Contempt — `_draw_score(ply)` (committed)

`_CONTEMPT = 25`. Every draw path returns `−25` at even ply, `+25` at odd ply. A position always
recurs at the *same ply parity* in one search (its side-to-move and the root's are both fixed),
so after `k` negamax negations the value is `−25` regardless of `k` — a draw is worth `−25` cp
from the moving side at the root. **Verified** (parity table; regression test updated to assert
`∓_CONTEMPT`). Sane magnitude; consistent with "don't leak a won game, grab a draw when lost by
> 25 cp". TT-safe because of the parity invariant.

### Check extension (committed)

```python
in_check = board.is_check()
if in_check and ply < _MAX_DEPTH: depth += 1
```

- **Verified:** mate distance preserved (`Rb8#` = `MATE − 1`; `in_check` captured before the
  bump and reused for the `-MATE + ply` return); TT-consistent (extension before the probe, so
  a checked position is always probed/stored at the same effective depth).
- **Recursion is bounded in practice** — worst case ≈ `64 + nominal_depth` `_negamax` frames,
  but real checking lines end in mate / repetition / `_Timeout` long before: observed max
  nesting ~14 on a queen-checks position. No `RecursionError` reproduced.
- **No damping** (no cap on consecutive extensions) and **no dedicated test**. Low risk given
  the bound, but add a test and consider capping total extension.

### LMR — late-move reductions (UNCOMMITTED)

```python
reduce = quiet and not in_check and depth >= 3 and move_index >= 3 and not board.is_check()
if reduce:
    r = 2 if move_index >= 6 else 1
    score = -_negamax(board, depth-1-r, ply+1, -beta, -alpha, deadline)   # full window, reduced depth
    if score > alpha:
        score = -_negamax(board, depth-1, ply+1, -beta, -alpha, deadline) # re-search full depth
```

- `quiet` computed before `push`; skips captures, promotions, checking moves, and nodes that
  are themselves in check. `depth-1-r ≥ 0` always (`depth ≥ 3`, `r ≤ 2`).
- **Not applied at the root** (`_search_root` calls `_negamax` at full depth for every root
  move) — so no LMR-induced root blunder.
- Uses the **full** `(-beta, -alpha)` window on the reduced search rather than a null window —
  unusual but not unsound; the `score > alpha` re-search bounds the error. A reduced-depth score
  that beats `value` but not `alpha` can still be stored in the TT at full `depth` — a minor,
  contained imprecision (table cleared each move).
- Deterministic at fixed depth. **No dedicated test.** Bench: bought back ~1 ply in the open
  middlegame (depth 2 → 3). The arena results in §0 include LMR.

### `_budget_s` (committed, reworked for long games)

`moves_left = max(30, 56 − fullmove)`, `+ 0.5·increment`, capped at both `clock/3` and
`clock − 500 ms`, floored at 10 ms. `test_budget_leaves_a_reserve_and_caps_at_a_third` covers
the caps. Addresses the "4.5 s left in a 91-move game" note.

---

## 4. `agent.py` — unchanged

Entrypoint, `_history` → `_seen`, `_infer_increment` (backs the increment out of the clock
delta, biased low = safe), non-raising fallback. Verified previously; no changes.

---

## 5. Findings summary

| # | Finding | Severity |
|---|---|---|
| 1 | Determinism broken under time pressure; docstring claim is false (§1) | **Important** |
| 2 | Phase 5 eval ~28× slower; middlegame depth 2–3 — real depth left on the table (§2) | **Important** (not a regression — arena is +Elo) |
| 3 | `evaluate()` runs a redundant `is_stalemate()` movegen at every leaf (§2) | Important (folds into 2) |
| 4 | Check extension has no damping / no test (§3) | Minor |
| 5 | LMR uncommitted, no test, full-window reduced search (§3) | Minor |
| 6 | One crash under concurrent load (frozen phase4b, `play` game); 48 clean games had none | Minor — watch |
| 7 | `versions/phase5/` is a known-bad freeze (inverted `_game_phase`) | Minor |
| 8 | PROGRESS.md stale — no entry for Phase 5 eval, contempt, check-ext, LMR, harness merge | Minor |
| 9 | Provenance of the second contributor's "Negamax search algorithm" commit — confirm it is the team's own work and both can explain it at finals | Important (competition risk, not a code bug) |

### Carried from earlier audits (still open)

- **Phase 2's clean bar** — 300+ games vs `random` and `greedy`, zero crash / flag / illegal —
  still never run as a gate. Engine is live in rated rounds.
- TT can cache a path-dependent draw score (now `±25`) — handle before a **persistent** TT.
- `_qsearch` has no width limit (delta pruning / SEE) — worse now each node costs ~115 µs.
- `_hist` has no cap — re-check band separation once nps recovers.
- On `_Timeout` the board is left with unpopped moves — latent, harmless (fresh board per call).
- `board._transposition_key()` private API; `int.bit_count()` ⇒ Python ≥ 3.10 (platform 3.12).

---

## 6. Verdict

**Correctness:** sound. Contempt is parity-consistent and TT-safe; the check extension preserves
mate scores; LMR is gated correctly and never negative-depth; the Phase 5 eval is
colour-symmetric with good test coverage; the inverted-phase bug is fixed in the working tree.
**No crashes** in 48 clean games — the earlier crash run was an old-harness / load artifact.

**Strength:** improved — Elo +108 vs phase4b, 100 % vs minimax, 96.9 % vs phase2, draw floods
gone. The real eval earns its keep even at depth 2–3.

**Weak points:** (1) not reproducible under time pressure — same FEN + clock gives different
moves; (2) the eval is ~28× too slow, so the engine is searching 1–2 plies shallower than it
should.

### Fix list

**Important**

1. **Determinism / reproducibility** (§1). At minimum correct the docstring; better, log the
   completed depth per move and add a fixed-depth repro path so a lost game can be replayed.
2. **Speed up `evaluate()`** toward the Phase-4b ~5 µs — port the (tuned) tables into a jitted
   bitboard eval, or cut the mobility / king-safety / passed-pawn / `is_stalemate` costs. Re-bench
   and confirm the middlegame is back to depth ≥ 4; the eval + the extra depth together should
   be a large gain.
3. **Confirm the imported search commit's provenance** and that both contributors can walk a
   panel through their code (§5.9).
4. **Finally run Phase 2's 300-game clean bar** and wire it into `make gate` / a pre-upload step.

**Minor**

5. Remove the redundant `is_stalemate()` movegen from `evaluate()` (§2).
6. Commit LMR with a test; add a check-extension test; consider damping the extension.
7. Update PROGRESS.md (§5.8); mark `versions/phase5/` known-bad (§5.7).
8. Watch for crashes under the platform's two-games-at-once (§5.6).

**Carry-overs:** TT path-dependence before a persistent TT; qsearch delta pruning; `_hist` cap;
`try/finally` push/pop.

No redesign. The search spine and the Phase 5 eval design are right. The work is: make it fast
enough to search deep, and make it reproducible.
