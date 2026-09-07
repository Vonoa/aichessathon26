# Engine audit

**Audit history (all 2026-09-06 unless noted):** Phase 1 skeleton → Phase 2 (mate distance,
draw awareness) → Phase 3a (bench) → Phase 3b (transposition table) → Phase 3b–4b full read →
**this pass (2026-09-07): Phase 5 evaluation, contempt, check extension.**

**Branch:** `search-extensions`. `search.py` has **uncommitted** changes (the check extension).
`evaluate.py` at HEAD = the Phase 5 tapered eval. **Two contributors** now: `Aled von Oppell`
(20 commits) and `Rezqy <rez-qy@hotmail.com>` (5 commits — the "Negamax search algorithm" and
some eval-merge commits). Frozen opponents: `versions/phase{1,2,3b,3c,4a,4b,5,5eval}/`.

Scope: [`agent.py`](agent.py), [`search.py`](search.py), [`evaluate.py`](evaluate.py),
[`tests/test_engine.py`](tests/test_engine.py), [`tools/bench.py`](tools/bench.py). Findings
verified by running the code.

---

## 0. Status and the headline

`ruff` + `mypy --strict` clean. **37 tests pass** (~0.8 s). No crash / flag / illegal in any
run this pass.

**Headline: the Phase 5 evaluation is ~19× slower than the Phase 3c eval it replaces, and it
has dropped the middlegame search from depth 4 to depth 2.**

| | Phase 4b (3c eval) | current (Phase 5 eval + contempt + check-ext) |
|---|---|---|
| `evaluate()` cost | **6.1 µs/call** (~164 k/s) | **117.7 µs/call** (~8.5 k/s) |
| bench — open middlegame | depth 4, ~32 k nps | **depth 2**, ~7.4 k nps |
| bench — sharp middlegame | depth 4, ~40 k nps | **depth 2**, ~6.3 k nps |
| bench — rook endgame | depth 7, ~49 k nps | depth 5, ~13.9 k nps |
| bench — overall nps | ~40 k | **~8.9 k** |

Depth 2 in a middlegame means the engine cannot see a two-move tactic — it will hang pieces.
The Phase 5 eval is *smarter per position* (tapered PST, pawn structure, mobility, king safety)
but the engine that uses it is *tactically much weaker* because it can no longer search deep
enough. A nominal depth-4 search of a lightly tactical position now takes **~3.1 s** (23 983
nodes) — more than a whole move's budget — partly because check extensions widen the tree on top
of the slow eval.

This eval is fine as a **tuning target** (its own docstring says "NOT YET JITTED … tune this
version first, then port the tuned tables into Phase 3's jitted bitboard eval"). It is **not
shippable as-is** — if `make zip` runs against this tree, the submission plays at depth 2.

---

## 1. `evaluate.py` — Phase 5 tapered evaluation

### Structure

```
phase   = _game_phase(board)                       # 24 = opening, 0 = bare kings
mg, eg  = 0, 0
for each piece:  mg += value + PST_MG[sq];  eg += value + PST_EG[sq]   (Black mirrored, sq^56, subtracted)
for each colour: add pawn-structure + mobility (mob_mg into mg, mob_eg into eg)
mg     += king-safety (White) − king-safety (Black)     # MG-scale only
blended = mg·phase + eg·(24 − phase)
tapered = int(blended / 24)                         # int() not // — see below
return tapered if White to move else −tapered
```

### The pieces

- **Material** — P/N/B/R/Q = 100/320/330/500/900, K = 0.
- **PST** — MG and EG tables per piece, PeSTO-style public seed values. Written rank-8-first
  and flipped once at import by `_flip_ranks` to this repo's a1 = 0 indexing (`test_pst_tables_are_oriented_a1_first`
  guards the orientation). All weights are placeholders for `texel_tune.py`.
- **`_game_phase`** — `Σ weight·count` over N (1), B (1), R (2), Q (4), capped `min(24, …)`.
  24 at the start, 0 at bare kings. The `min` cap is essential (promoted-queen piles push the
  raw sum > 24); the sum can't go negative so no `max(0, …)` is needed. **The currently
  checked-out file has this correct** — an earlier version (frozen as `versions/phase5/`) had
  the phase *inverted* (`phase = 24; phase -= …`), which commit `d5a618c` fixed. Treat the
  `versions/phase5/` freeze as known-bad; use `versions/phase5eval/` as the reference.
- **Pawn structure** (`_pawn_structure`) — doubled (−12 each extra), isolated (−10 each),
  passed-pawn bonus by rank `[0,5,10,20,35,60,100,0]`. The passed-pawn test is an O(pawns²)
  scan with a per-call dict rebuild, run once per colour per `evaluate()`.
- **Mobility** (`_mobility`) — `attacks_mask(sq).bit_count()` per N/B/R/Q, weighted MG/EG. Raw
  attack count — does not exclude own-piece squares or account for pins. Placeholder.
- **King safety** (`_king_safety_mg`) — pawn-shield bonus (+12/pawn), open/semi-open file
  penalties on the king's three files (−22 / −12), and −weight for every enemy N/B/R/Q whose
  `attacks_mask` intersects the king zone. Added to `mg` only, so the `·phase / 24` blend fades
  it to 0 at bare kings — deliberate (an exposed king is a middlegame liability, not an
  endgame one).

### Verified correct

- **Colour symmetry** — `evaluate(board) == evaluate(board.mirror())` on every position tried,
  because `tapered = int(blended / 24)` truncates toward zero (`//` would floor and make a
  position and its mirror differ by 1). `test_eval_is_colour_symmetric` (5 FENs) +
  `test_eval_golden_values` (9 exact pinned outputs) cover this.
- **Direction** — startpos = 0; +queen > +700; centralised knight > rim knight; advanced pawn
  > home pawn; MG king wants the back rank; EG pawn near promotion scores ~+150 over home.
  All are tests and all pass.
- **Stalemate / insufficient material** — `evaluate()` now returns `0` for both (line 342).
  This incidentally closes most of the "qsearch is stalemate-blind" gap the previous audit
  flagged: a stalemate reached by a capture inside `_qsearch` now scores 0, verified. Cost:
  `board.is_stalemate()` is a full movegen (**5.8 µs**) run at *every* leaf.

### The cost problem (see §0)

`evaluate()` is 117.7 µs/call. The expensive parts, in order: the mobility and king-safety
`attacks_mask` loops over every piece; the O(pawns²) passed-pawn scan; and the redundant
`is_stalemate()` movegen (`_negamax` already tested `not moves`, and `_qsearch`'s non-check
branch builds the full legal-move list anyway — so `evaluate()` generates moves a third time).

---

## 2. `search.py` — unchanged core + contempt (5f) + check extension (uncommitted)

The negamax / alpha-beta / iterative-deepening / TT / quiescence / killers+history machinery is
**unchanged from the Phase 4b audit and still verified sound** (TT returns identical best move
and score with the table neutered; mate distance correct; leaf-repetition draw check runs
before the `depth <= 0` return; determinism holds). Two additions:

### Contempt — `_draw_score(ply)` (committed, PR #15)

```python
_CONTEMPT = 25
def _draw_score(ply):  return -_CONTEMPT if ply % 2 == 0 else _CONTEMPT
```

Every draw path (`is_fifty_moves`, insufficient material, stalemate, `is_repetition(2)`,
`key in _seen`) now returns `_draw_score(ply)` instead of `0`.

**Verified correct.** A given position always recurs at the *same ply parity* within one search
(its side-to-move is fixed, the root's is fixed, so the parity is determined). So:

- `_draw_score` returns `−25` at even ply, `+25` at odd ply (checked, plies 0-5).
- After `k` negamax negations up to the root, the value is `−25` regardless of `k` — a draw is
  worth `−25` cp **from the moving side's point of view at the root**, so the engine plays to
  avoid draws when it is not worse, and still takes a draw when it is losing by more than
  25 cp.
- Because parity is fixed per position, a contempt-adjusted score cached in the TT is
  self-consistent — the code comment's claim holds up.

`_CONTEMPT = 25` (a quarter-pawn) is a sane magnitude. The regression test
`test_seen_position_is_a_draw_at_the_horizon` was correctly updated to assert `∓_CONTEMPT`
instead of `0`.

### Check extension (UNCOMMITTED, branch `search-extensions`)

```python
in_check = board.is_check()
if in_check and ply < _MAX_DEPTH:
    depth += 1        # search a checked node one ply deeper
```

- **Verified:** mate distance is preserved (`Rb8#` still scores exactly `MATE − 1`, because
  `in_check` is captured *before* the `depth += 1` and the `-MATE + ply` uses it); the
  extension happens before the TT probe so the same checked position is always probed/stored at
  the same effective depth — TT-consistent.
- **Concern — no damping.** There is no cap on *consecutive* extensions or on total extension
  relative to the nominal depth. A long forcing checking line keeps `depth` from decreasing;
  the only bounds are `ply < _MAX_DEPTH` (64), `_QS_MAX_PLY` (96) and `_tick`. In a
  check-heavy position this multiplies node count — and it already shows: a *nominal* depth-4
  search took 3.1 s / 24 k nodes. On top of the slow eval this is why the middlegame collapsed
  to depth 2.
- **Uncommitted and untested** — no test exercises the extension; it is exploratory work on a
  side branch.

---

## 3. `agent.py` — unchanged

Entrypoint, `_history` (position keys → `_seen`), `_infer_increment` (backs the increment out
of the clock delta, biased low = safe), non-raising fallback. All verified in the previous
audit; no changes this pass.

`_budget_s` was reworked (PR #11 / `3b6489e`) for long games and is now in the tree:
`moves_left = max(30, 56 − fullmove)`, `+ 0.5·increment`, capped at both `clock/3` and
`clock − 500 ms`, floored at 10 ms. `test_budget_leaves_a_reserve_and_caps_at_a_third`
verifies the caps. This addresses the "4.5 s left in a 91-move game" note from the first rated
game.

---

## 4. Findings

### Finding 1 — Phase 5 eval is ~19× too slow; middlegame search fell to depth 2 *(Critical for any upload; not a correctness bug)*

Covered in §0. `evaluate()` 6.1 µs → 117.7 µs; overall nps 40 k → 8.9 k; middlegame depth 4 → 2;
nominal depth-4 search ≈ 3.1 s. **Do not `make zip` / upload against this tree.** Options:

- Port the (once-tuned) tables into the jitted bitboard eval before this ships — the eval
  docstring already says this is the plan; it means 3e (or at least a jitted `evaluate`) has to
  land first, contradicting the "pruning + eval before 3e" reorder in PROGRESS.
- Or cut the per-leaf cost hard now: drop the redundant `is_stalemate()` movegen (get
  stalemate=0 from `_qsearch`/`_negamax` which already generate moves); cache/skip the
  passed-pawn O(n²) scan; gate king-safety and mobility behind cheaper masks.
- Keep this version strictly as the Texel tuning target and keep shipping the Phase 4b engine
  until the fast port exists.

### Finding 2 — check extension has no damping, compounds the slowdown *(Important)*

§2. Bound the extension (e.g. only extend if not already extended past `depth + N`, or halve
the extension after the first). Commit it with a test, or shelve it until the eval speed is
fixed — measuring it now, on top of finding 1, tells you nothing.

### Finding 3 — `evaluate()` runs a redundant full movegen at every leaf *(Important, folds into finding 1)*

`board.is_stalemate()` (5.8 µs) inside `evaluate()`, while `_negamax` already returned on
`not moves` and `_qsearch`'s quiet branch builds the legal-move list itself. Have `_qsearch`
pass "is this stalemate" down, or return 0 from the branch that already knows.

### Finding 4 — verify the provenance of the second contributor's search commit *(Important — competition risk, not a code bug)*

`Rezqy`'s `0ed83cb "Add Negamax search algorithm with optimizations"` and the
`"Add files via upload"` / `"Merge evaluations for midgame/endgame"` commits have terse,
generic messages unlike the rest of the history. The PeSTO tables are explicitly public seed
values (PLAN allows a hand-tuned PST from public starting points), so the *tables* are fine.
But the competition DQs retroactively for shipping a third-party engine, and finalists walk a
panel through how the engine was built — so the search-algorithm contribution needs a human
confirmation that it is the team's own work, and both contributors need to be able to explain
their parts.

### Finding 5 — PROGRESS.md is stale *(Minor)*

Its "Current status" still says "Phase 4b merged to main"; it has no entry for the Phase 5
eval, the `_game_phase` inversion fix, contempt, or the `search-extensions` branch. The last
logged entry is `budget-longgame`. Bring it up to date — the log's value is that it is trusted.

### Finding 6 — `versions/phase5/` is a known-bad freeze *(Minor)*

It carries the inverted `_game_phase` (`phase = 24; phase -= …` with `//` truncation). Fine as
a historical arena opponent, but do not use it as an eval reference or a base for a revert —
use `versions/phase5eval/`.

### Carried from earlier audits (still open)

- **Phase 2's clean bar** — 300+ games vs `random` and vs `greedy`, zero crash / flag /
  illegal — has still never been run as a gate. The engine is live in rated rounds.
- TT can cache a path-dependent draw score (now `±25`, same reasoning) — must be handled
  before a **persistent** TT.
- `_qsearch` has no width limit (no delta pruning / SEE) — worse now that each node costs
  ~120 µs.
- `_hist` has no cap — re-check band separation once nps recovers.
- On `_Timeout` the board is left with unpopped moves — latent, harmless while `agent.py`
  builds a fresh board per call.
- `board._transposition_key()` is a private API; `int.bit_count()` implies Python ≥ 3.10
  (platform is 3.12 — fine).
- No explicit RNG seed (deterministic by construction, documented).

### Arena (this pass)

*Running — 48 games, current vs `versions/phase4b` and vs `minimax` / `numba` at 10 s + 0.1 s.
Table to be filled in; the bench already establishes the depth-2 regression regardless of the
game scores.*

---

## 5. Verdict

**Correctness:** sound. Contempt is wired correctly (parity-consistent, TT-safe). The check
extension preserves mate scores. The Phase 5 eval is colour-symmetric and directionally
sensible, with good test coverage. The inverted-phase bug is already fixed in the working tree.
No crash / flag / illegal.

**Playing strength:** **regressed.** The Phase 5 eval as written costs ~19× the Phase 3c eval
and drops middlegame search to depth 2; check extensions make it worse. A smarter static score
does not compensate for not seeing a two-move tactic.

### Fix list

**Critical**

1. **Do not upload this tree.** Keep shipping the Phase 4b engine. Treat the current
   `evaluate.py` as the Texel tuning target only, exactly as its docstring says.
2. **Get the eval back under ~10 µs/call before it ships** — jitted port of the tuned tables,
   or aggressive pruning of the mobility / king-safety / passed-pawn / `is_stalemate` costs.
   Re-bench and confirm the middlegame is back to depth ≥ 4.

**Important**

3. Bound the check extension (damping) and commit it with a test — or shelve it until the eval
   speed is fixed.
4. Remove the redundant `is_stalemate()` movegen from `evaluate()` (finding 3).
5. Confirm the provenance of the imported search commit and that both contributors can explain
   their code (finding 4).
6. Finally run Phase 2's 300-game clean bar and wire it into `make gate` or a pre-upload step.

**Optional**

7. Update PROGRESS.md (finding 5); note `versions/phase5/` as known-bad (finding 6).
8. Carry-overs: TT path-dependence before a persistent TT; qsearch delta pruning; `_hist` cap;
   `try/finally` push/pop; explicit seed.

No redesign — the search spine is right and the eval design (tapered PST + structure + mobility
+ king safety, Texel-tuned) is the correct Phase 5. The problem is purely that it must be fast
enough to run at the leaves, and right now it is not.
