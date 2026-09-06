# Engine audit

**Audit history (all 2026-09-06):** Phase 1 skeleton → Phase 2 (mate distance, draw awareness)
→ Phase 3a (bench) → Phase 3b (transposition table) → **this pass: the full engine through
Phase 4b.**

**Branch:** `phase-4-ordering` (Phase 4b committed, frozen to `versions/phase4b/`, tree clean).
**Next:** null-move / LMR / PVS / persistent TT, then Phase 5 (real eval). 3e (jitted move
generator) is deliberately deferred — see §6.

Scope: [`agent.py`](agent.py), [`search.py`](search.py), [`evaluate.py`](evaluate.py),
[`tools/bench.py`](tools/bench.py), [`tests/test_engine.py`](tests/test_engine.py); harness
context from [`harness/referee.py`](harness/referee.py), [`harness/sandbox.py`](harness/sandbox.py).
Every finding was verified by running the code.

---

## 0. Status

| Component | State |
|---|---|
| Phase 1 — minimal engine | **Complete** |
| Phase 2 — robustness + draws | **Complete** (the 3b leaf-repetition regression was found in review, fixed, and regression-tested) |
| Phase 3a — bench | **Complete** (instrumentation) |
| Phase 3b — transposition table | **Complete** — verified sound with qsearch + killers in the mix |
| Phase 3c — bitboard eval | **Complete** — output pinned against the old formula |
| Time management — increment-aware budget | **Complete** — with a known long-game weakness (finding 5) |
| Phase 4a — quiescence search | **Complete** — with stalemate / width caveats (findings 1, 2) |
| Phase 4b — killers + history | **Complete and clean** |
| Phase 3e — jitted movegen | **Deferred on purpose** (§6) |

`ruff` + `mypy --strict` clean. **23 tests pass** (~0.8 s). Bench: overall ~40k nps (up from the
27k Phase 3a baseline), middlegame still **depth 4**, endgame depth 7. `make zip` ships
`agent.py` + `evaluate.py` + `search.py` only (16 KB unzipped); `tools/` and `tests/` excluded.
Engine is submitted and live in rated rounds.

Arenas this pass (54 games, **zero crash / flag / illegal**):

| phase4b vs | TC | result | draws |
|---|---|---|---|
| `baselines/minimax` | 10 s + 0.1 s | **+14 =0 −0, 100 %** | 0 |
| `versions/phase2` | 3 s + 0.05 s | **+17 =1 −2, 87.5 %** | 1 threefold — the low-clock draw guard is healthy again (was 47.5 % / 9 draws before the 3b fix) |
| `versions/phase4a` | 10 s + 0.1 s | **+2 =12 −6, 40 %** | 10 fifty-move, 2 threefold — see finding 7 |

**No critical issues. No correctness bugs in the main search.** The findings are horizon-effect
gaps in quiescence, a near-mirror draw flood against the previous version, and a few latent
traps — details in §4.

---

## 1. `evaluate.py` — bitboard material + pawn PST (Phase 3c)

### What static evaluation is

A heuristic score for a position with **no search** — the leaves of the tree call it to guess
who is better. All tactics come from the search; the evaluator only needs to be roughly right
about quiet positions.

### The 3c rewrite

`evaluate()` used to build a `SquareSet` per piece type via `board.pieces(...)` — ~a dozen
object allocations per call. It now works straight on the raw bitboards:

```python
white = board.occupied_co[chess.WHITE]; black = board.occupied_co[chess.BLACK]
score = _PAWN * ((board.pawns & white).bit_count() - (board.pawns & black).bit_count()) + ...
```

`int.bit_count()` (Python 3.10+; platform is 3.12) is popcount. Pawn PST is walked by bit
iteration: `wp & -wp` isolates the low bit, `.bit_length() - 1` is its square index,
`wp &= wp - 1` clears it. Black mirrors with `sq ^ 56` (flip the rank, same as
`square_mirror`) and is **subtracted**. Same piece values (100/320/330/500/900), same table, so
**it is a pure speed change** — `test_bitboard_eval_matches_reference` pins the output against
the old `board.pieces()` formula across six positions; this audit re-checked it on a
promoted-queen pile (+2700), bare kings, an all-pawns wall, black-to-move, and a black pawn one
square from promotion — **all match.**

### The pawn table (unchanged)

Rank 2 d2/e2 = −20 (push the centre pawns → a 40 cp swing for `d4`/`e4`); rank 4 d4/e4 = +20;
rank 5 = +25; rank 7 = +50 across (near promotion); ranks 1 & 8 = 0. Classic CPW table.

### Why centipawns / the side-to-move flip

Integers scaled ×100 avoid float rounding in a hot function and match every published table.
`return score if board.turn == chess.WHITE else -score` — everything above is White-relative;
negamax needs every node scored **relative to the side to move**, so Black-to-move negates.

### Understands / cannot understand

**Understands:** material; central vs wing pawns; whether the d/e pawns moved; pawn
advancement, especially near promotion.

**Cannot:** king safety, piece activity/mobility/outposts, non-pawn placement (no PST for
N/B/R/Q), pawn structure beyond raw rank, open files, the bishop pair as a term, space, tempo;
any tactic beyond search depth; fortresses and wrong-bishop draws; **mating technique** (the
rated game took ~80 moves to convert a won position — this is the Phase 5 signal).

---

## 2. `search.py` — negamax + AB + ID + TT + quiescence + killers/history

### Module state (lines 30-54)

`MATE = 1_000_000`; `_MATE_THRESHOLD = MATE - 1_000` (any `|score| ≥` this is a forced mate;
the smallest real mate magnitude is `MATE - 64`, safely above). `_CHECK_INTERVAL = 255` (clock
tested every 255 nodes). `_QS_MAX_PLY = 96` (quiescence recursion cap). TT bound kinds
`_EXACT/_LOWER/_UPPER`; `_TT_MAX = 1_000_000`. Move-ordering bands: `_CAPTURE_BASE = 10_000_000`,
`_KILLER_0 = 9_000_000`, `_KILLER_1 = 8_000_000`, history below. `_killers` (2 slots × 65 plies,
flat) and `_hist` (4096 from/to counts) — both reset every move.

### `search_move(board, time_left_ms, history=None, increment_ms=0.0) -> str` (lines 61-104)

Resets `_nodes` / `_last_depth`; `_seen = frozenset(history)`; **clears `_tt`, `_killers`,
`_hist`**. No legal moves → `"0000"`; one → return it. Then iterative deepening `depth = 1..64`:
`move, score = _search_root(board, depth, deadline, best)` with `best` (the previous iteration's
move) passed in as the first move to try; `_Timeout` → `break` (keep the previous `best`). After
a completed depth: `best = move`, `_last_depth = depth`, optional `AGENT_DEBUG` print, `break`
if `|score| ≥ _MATE_THRESHOLD` (proven mate) or past the deadline.

### `_budget_s(board, time_left_ms, increment_ms) -> float` (lines 107-117)

```python
moves_left = max(20, 50 - board.fullmove_number)
budget_ms  = time_left_ms / moves_left + 0.8 * increment_ms
budget_ms  = min(budget_ms, time_left_ms - 300)      # watchdog margin
return       max(budget_ms, 10.0) / 1000.0           # 10 ms floor
```

Every move we make refills the clock by the increment, so `0.8 ×` it is time to spend, not
hoard. `agent.py` **infers** `increment_ms` from the clock delta (see §3) rather than
hardcoding it — an earlier cut hardcoded 500 ms, assumed a refill the 50 ms arena never gave,
and drained the clock in ~7 moves (PROGRESS 2026-09-06 "Time management"). Verified: budget is
always in `[10 ms, clock]` across clocks 1 ms … 120 s × increments 0/50/500.

### `_search_root(board, depth, deadline, first) -> (Move, int)` (lines 120-139)

`_ordered(...)` the legal moves, move `first` to the front. Full-window alpha (`-MATE-1`) on
each root move, narrowing beta via `alpha = max(alpha, score)`. Keeps the max with strict `>`
(ties keep the earlier / better-ordered move → deterministic). The returned `best_score` is
**exact for the chosen move**: move 1 is searched with an open lower bound (→ exact), and any
later move only replaces the best when its child did *not* fail high (→ also exact); rejected
moves get a fail-soft upper bound that can't affect the max. No TT probe/store at the root
itself (it is re-searched every iteration; `first` supplies the ordering).

### `_negamax(board, depth, ply, alpha, beta, deadline) -> int` (lines 142-212)

1. `_tick(deadline)`.
2. `is_fifty_moves()` → 0.
3. `popcount(occupied) ≤ 4 and is_insufficient_material()` → 0 (the popcount guard is a safe
   necessary condition — every insufficient-material config has ≤ 4 pieces).
4. `moves = list(board.legal_moves)`; empty → `-MATE + ply` (check) or 0 (stalemate).
5. **Repetition / seen check, *before* the leaf return** (this is the fix for the 3b
   regression): `key = board._transposition_key() if halfmove_clock >= 4 else None`; if
   `key is not None and (is_repetition(2) or key in _seen)` → 0. Quiet leaves (`halfmove_clock
   < 4`, common) still skip the key.
6. `if depth <= 0: return _qsearch(board, ply, alpha, beta, deadline)`.
7. Compute `key` if not already; **TT probe:** extract `tt_move` always; if `e_depth >= depth`
   return `e_value` for `_EXACT`, or `_LOWER and e_value >= beta`, or `_UPPER and e_value <=
   alpha`.
8. `_ordered(board, moves, ply)`, then move `tt_move` to the front.
9. Search loop (fail-soft, `alpha >= beta` cutoff). On a cutoff by a **quiet** move (not a
   capture, no promotion), `_record_cutoff(move, ply, depth)`.
10. **TT store**, only if `|value| < _MATE_THRESHOLD`: flag from the fail-soft result
    (`value <= alpha_orig` → `_UPPER`; `value >= beta` → `_LOWER`; else `_EXACT`);
    `if len(_tt) >= _TT_MAX: _tt.clear()` first.

**Negamax / negation, alpha / beta, why alpha-beta is fast:** covered in earlier audits and
unchanged — every node maximises its own score and negates the child's; `alpha` is the best the
mover has already secured, `beta` the best the opponent can hold them to (`= -alpha` of the
parent); `value >= beta` → opponent avoids the line → stop; with good ordering the tree shrinks
from `b^d` to ≈ `b^(d/2)`. The TT + `first`/`tt_move` + killers/history all exist to make the
ordering better.

**Mate distance (verified):** checkmate is `-MATE + ply`, `ply` from the root. Negated up the
tree an immediate mate is `MATE - 1`, a mate two of our moves off `MATE - 3`, etc. — larger =
faster, so the engine takes the fastest mate and the longest defence. Mate scores are **never
cached** (root-relative → path-dependent).

**TT soundness (verified this pass):** identical best move *and* score with the TT live vs
neutered, at depth 5 across four middlegame/endgame positions, with quiescence and killers
active; node counts 25-30 % lower with the TT.

### `_qsearch(board, ply, alpha, beta, deadline) -> int` (lines 215-249)

At `depth <= 0`, instead of calling `evaluate()` mid-exchange, keep searching:

- `ply >= _QS_MAX_PLY` → `evaluate(board)` (safety net).
- **In check:** search *all* legal evasions; no moves → `-MATE + ply` (mate found in
  quiescence, distance-correct); no stand-pat.
- **Not in check:** `best = evaluate(board)` (stand pat); `best >= beta` → return; else
  `alpha = max(alpha, best)` and search only captures + promotions.
- Recurse with `-_qsearch(..., ply+1, -beta, -alpha)`, fail-soft, `alpha >= beta` cutoff.

`_ordered` is called without `ply` here → pure MVV-LVA capture ordering (no killers). Stand-pat
assumes the mover can hold at least the static eval — fails only in zugzwang, a standard
trade-off, correctly documented. Verified: resolves a hanging rook to ≥ +450 (> the static
eval); leaves a quiet position exactly at the static eval; a capture-dense position terminates
via `_Timeout` (no `RecursionError` — max ~160 stack frames vs the 1000 limit).

### `_ordered(board, moves, ply=-1) -> list[Move]` (lines 252-280)

`sorted(moves, key=score, reverse=True)` (stable → deterministic). `score`:
- capture → `_CAPTURE_BASE + 8*victim - attacker + promo` (victim/attacker are piece-type
  ordinals 1..6; `8 > 6` guarantees a bigger victim always sorts first, king attacker included;
  `promo` = 100 for =Q, 10 otherwise).
- non-capture promotion → `_CAPTURE_BASE + (100 | 10)`.
- `== killer0` → `_KILLER_0`; `== killer1` → `_KILLER_1` (this ply's slots, or `None` if
  `ply < 0` / out of range).
- else → `_hist[from*64 + to]`.

Killers are looked up only against moves actually in the list, so an illegal killer from a
sibling branch simply never matches — safe.

### `_record_cutoff(move, ply, depth)` (lines 283-290)

Shift the ply's killer slots down and store `move` in slot 0 (guarded against duplicating it);
`_hist[from*64 + to] += depth * depth` (deep cutoffs weigh more). Bounds verified: `ply` in a
real `_negamax` node is ≤ 63 when `_ordered`/`_record_cutoff` run, so `base = ply*2 ≤ 126 <
129` — always in range; the `0 <= base < _KILLER_SLOTS - 1` guard is belt-and-braces.

### `_tick(deadline)` (lines 293-297)

`_nodes += 1`; every 255 nodes, `time.monotonic() >= deadline` → raise `_Timeout`. `deadline`
is `_budget_s` (capped at `clock − 300 ms`); the 0.5 s increment tops the clock back up, so a
one-slice overrun does not flag. Returning the last completed depth is safe — each depth is an
independent search; a partial depth is discarded when `_Timeout` unwinds.

---

## 3. `agent.py` — entrypoint, history, increment inference

```python
board = chess.Board(fen)
key = board._transposition_key()
_history[key] = _history.get(key, 0) + 1          # count every position we move in
increment_ms = _infer_increment(time_left_ms)
started = time.monotonic()
try:    return search_move(board, time_left_ms, _history, increment_ms)
except Exception:                                  # a search bug must never forfeit
    legal = list(board.legal_moves); return legal[0].uci() if legal else "0000"
finally:
    _clock["prev"]  = float(time_left_ms)
    _clock["spent"] = (time.monotonic() - started) * 1000.0
```

- **`_history`** feeds `_seen` in the search. Counts are kept but the search only uses the key
  set — the deliberate "first repetition = draw" heuristic (safe: never misses a draw; can
  under-rate a won line that transposes through an earlier position).
- **`_infer_increment`**: `increment = time_left_ms - (prev - spent)`, clamped `[0, 2000]`.
  Derives the per-move increment (`get_move` is not told it) from the clock delta since our last
  move. `spent` is measured from *after* `_infer_increment` to the `finally`, so it runs a hair
  short of the referee's measurement → the estimate is biased **low** → a smaller budget → the
  safe direction. The process is suspended between our moves, so `spent` measures only our
  compute — **no clock leakage.** Verified: a simulated true 50 ms increment is inferred as
  49 ms.
- The fallback can no longer raise (`legal[0]` / `"0000"`, never `next(iter(...))`).
- `chess.Board(fen)` and `_transposition_key()` are outside the `try` — a malformed FEN would
  propagate, but that is a platform-contract violation and cannot occur in a real game.

---

## 4. Findings

### Verified correct

- **TT is sound** with quiescence and killers in the mix (identical best move + score, TT on
  vs off).
- **Leaf-level repetition / `_seen`** now returns 0 at depth 0 *and* depth 1 — the 3b
  regression this audit series flagged is fixed and has a regression test
  (`test_seen_position_is_a_draw_at_the_horizon`).
- **Mate distance** — `MATE - 1` for an immediate mate, distance-correct when being mated; mate
  scores excluded from the TT.
- **Bitboard `evaluate()`** matches the reference formula on every case tried, promoted pieces
  and empty boards included.
- **Determinism** — no RNG; stable sort over python-chess's fixed move order; first-seen
  tie-break; killers/history populate deterministically and reset per move. Two full
  `search_move` calls on one FEN return the same move.
- **Time safety** — `_budget_s` always in `[10 ms, clock]`; `time_left_ms` of 5 / 1 / 0 / **−50**
  all return a legal move in ~15 ms with no exception.
- **`_qsearch`** finds mates (distance-correct), resolves hanging material, leaves quiet
  positions at the static eval, and cannot `RecursionError`.
- **Bounds** — `_killers` / `_hist` indexing is always in range for real search plies.
- **Packaging** — the zip contains only the three root modules; `tools/` and `tests/` are not
  shipped.

### Finding 1 — `_qsearch` does not detect stalemate (or 50-move / repetition / insufficient) *(Minor→Moderate)*

The not-in-check branch does `best = evaluate(board)` and then searches only captures/promotions.
If the side to move has **no legal moves at all**, that is stalemate (a draw, 0) — but qsearch
returns the material eval instead. Same for a capture line that trips the 50-move rule or
repeats inside quiescence.

`_negamax` catches stalemate for any move at `depth >= 1` (step 4), and iterative deepening
corrects the score at the next depth — so this only bites when the search finishes shallow, i.e.
**at low clock**, and only for a stalemate reached *by a capture or promotion* (quiet stalemates
never arise in qsearch). It is uncommon, but it lines up with the observed weakness: the rated
game took ~80 moves to convert a won position and a smoke game drew by `ply_cap`. Given PLAN
priority #3 ("don't lose games to ourselves"), a cheap guard is worth it:

```python
else:  # not in check
    if not any(board.generate_legal_moves()):   # stalemate
        return 0
    best = evaluate(board)
    ...
```

(one extra movegen per quiet qsearch node; measure the nps cost, or gate it on a low piece
count).

### Finding 2 — `_qsearch` has no width limit *(Minor)*

No delta pruning, no SEE. A capture-dense position (e.g. many major pieces mutually en prise)
can spend the **entire move budget** inside one `_qsearch` call — verified: a queen-pile
position ran until the deadline. It is *safe* (`_Timeout` propagates to `search_move`, which
returns the last completed depth), but the engine can then complete only depth 1-2 in that
position. Delta pruning is already on the Phase 4 list — keep it there.

### Finding 3 — the board is left dirty after a `_Timeout` *(Minor, latent)*

`board.push(move)` on line 187/132, then the recursive call raises `_Timeout`, so
`board.pop()` is skipped — at every frame up the stack. Verified: after a forced timeout the
caller's board has 7-8 unpopped moves and a changed FEN.

**Currently harmless** — `agent.get_move` builds a fresh `chess.Board(fen)` every call and never
touches it again once `search_move` returns. But it is a trap for any future change that reuses
the board (a persistent board object, calling `search_move` twice, a test that inspects the
board afterward). A `try: ... finally: board.pop()` around each push/recurse, or catching
`_Timeout` one level higher and resetting, removes the trap.

### Finding 4 — TT can cache a path-dependent draw score *(Minor, carried; Phase 4 blocker)*

An interior node whose subtree returned 0 via the in-search `is_repetition(2)` folds that 0
into its stored `value`; probed later on a path where the repetition would not occur, the cached
value is wrong. `key in _seen` is path-independent (fixed game history), so only in-search
repetitions contribute, and the damage is bounded to one move because `_tt` is cleared each
move. The code comment already flags this. **It must be handled before the Phase 4 persistent
TT lands** (don't store a node whose subtree hit a repetition, or tag such entries).

### Finding 5 — `_budget_s` is too aggressive in long games *(Minor, carried)*

`moves_left = max(20, 50 - fullmove_number)` pins at 20 from move 30 on, so every later move
budgets `clock/20 + 0.8·inc` regardless of how many moves actually remain. The first rated game
finished with 4.5 s on the clock (opponent had 21 s) in a 91-move game. Not a flag risk (the
`0.8·inc` term sustains ~400 ms/move), but it leaves the engine thin on time exactly when long
technical endgames need it. PROGRESS already notes this as "a minor later tweak."

### Finding 6 — move-ordering bands can theoretically collide *(Minor)*

`_hist` has no cap and only grows within a move. The bands are `_CAPTURE_BASE = 10 M`, killers
9 M / 8 M, history from 0. In the current ~50-200 k node/search regime a hot `_hist` slot
reaches ~10-50 k — nowhere near 8 M — so ordering is well-separated. Once 3e multiplies nps,
re-check that a hot slot can't pass `_KILLER_1`; clamp the per-slot history contribution if
needed.

### Finding 7 — Phase 4b does not visibly beat Phase 4a in the arena *(Important to resolve, not a code bug)*

PROGRESS says "each frozen step beats the last decisively." This audit's run of **phase4b vs
`versions/phase4a`, 20 games @ 10 s + 0.1 s, scored 40 % (+2 =12 −6)** — 10 of the draws by the
fifty-move rule. Meanwhile phase4b is 100 % vs `minimax` and 87.5 % vs Phase 2, so the engine is
strong and not broken; this is specifically the near-mirror match against the previous version.

Two readings, and they need separating:

- **Benign:** 12 of 20 games are draws, so the result rests on 8 decisive games — pure variance
  on top of the thin-eval "two engines with no plan shuffle to a fifty-move draw" problem
  PROGRESS already owns. 40 % is ~1.3 σ from 50 %.
- **Not benign:** killer/history ordering does not change the move at a fixed depth (both sides
  are exact alpha-beta), only which depth the budget buys. A mis-tuned history heuristic can
  *cost* depth in some positions and make 4b a net-neutral-or-negative change despite the
  bench looking fine.

Either way, **4b's gain is unproven.** Before stacking more ordering heuristics (null-move, LMR,
PVS) on top, run a few-hundred-game arena vs 4a at a recorded time control and confirm the
direction — the same lesson the 3b audit already flagged about recording the TC and running
enough games. If 4b turns out neutral, that is worth knowing now.

### Nits

- `board._transposition_key()` is a private python-chess API, used in `agent.py` and
  `search.py`. Fine while python-chess is pinned at 1.11 (platform + local); a one-line comment
  would pin the assumption.
- `evaluate()` depends on `int.bit_count()` (Python 3.10+). Platform is 3.12, so fine — but it
  is an implicit version floor worth a note.
- No explicit RNG seed. Deterministic by construction and documented in the `search.py`
  docstring; PLAN wants a seed stated for finals reproducibility. The nuance: determinism holds
  *at a fixed depth*, and wall-clock timing decides the depth reached, so the same position can
  yield different moves on a slower machine.

### Docstring / PROGRESS accuracy

`search.py`'s docstring accurately describes 3b + 4a + 4b and the determinism guarantee.
PROGRESS is accurate and unusually candid — it records the hardcoded-increment mistake, the 3b
leaf-repetition regression and its fix, and the 80-move conversion as the Phase 5 signal. Good
discipline; keep it.

Still open from earlier audits (not this phase's job): **Phase 2's formal clean bar — "300+
games vs `random` and vs `greedy`, zero crash / flag / illegal" — has never been run as a
gate**, even though the engine is now submitted and playing rated games.

---

## 5. Speed reality check

| position | Phase 3a baseline | now (4b) |
|---|---|---|
| open middlegame | depth 4, 23 k nps | depth 4, ~32 k nps |
| sharp middlegame | depth 4, 23 k nps | depth 4, ~40 k nps |
| rook endgame | depth 6, 36 k nps | depth 7, ~49 k nps |
| overall | 27 k nps | ~40 k nps |

3c (bitboard eval) + the search work bought ~1.5× nps, and the endgame gained a ply, but **the
middlegame is still stuck at depth 4** — the Phase 3 target was 6-7. `board.legal_moves` (full
legality in Python) is the per-node cost and only 3e (the jitted move generator) removes it.
PROGRESS's reorder — pruning + real eval before 3e — is defensible ("get it right, then get it
fast", and the engine already beats every baseline), but depth 4 in sharp middlegames is the
ceiling on tactical strength until 3e happens, and it is visible in the rated game's 7 blunders.

---

## 6. Verdict

**Phase 1-3c, time management, Phase 4a-4b: Complete.** No critical issues, no correctness bugs
in the main search. The engine is sound, deterministic, submitted, and winning rated games.

### Fix list

**Critical**

- None.

**Important**

1. **Confirm Phase 4b is actually a gain** (finding 7). A few hundred games vs `versions/phase4a`
   at a recorded time control before adding null-move / LMR / PVS. This run scored 40 %; that
   is probably mirror-match + thin-eval variance, but "probably" is not the bar, and a
   mis-tuned history heuristic costing depth is a real alternative.
2. **`_qsearch` stalemate guard** (finding 1). Small change, and it targets the exact weakness
   already costing games — slow won-position conversion and a `ply_cap` draw. Measure the nps
   cost of the extra movegen; gate it on low piece count if needed.
3. **Run Phase 2's clean bar now** — 300+ games vs `random` and vs `greedy`, assert zero
   crash / flag / illegal, and wire it into `make gate` or a documented pre-upload step. The
   engine is live in rated rounds without this having ever been run formally.

**Optional**

3. `try/finally` (or a helper) around push/pop so a `_Timeout` leaves the board clean
   (finding 3) — cheap insurance against a future refactor.
4. Handle the TT path-dependent-draw case **before** the persistent TT (finding 4).
5. `_budget_s`: use real moves-remaining, not `max(20, …)`, so long endgames aren't starved
   (finding 5).
6. Clamp the per-slot history contribution before 3e multiplies node counts (finding 6).
7. qsearch delta pruning (finding 2) — already on the Phase 4 list.
8. State an explicit seed / tie-break for finals reproducibility; note the `int.bit_count()`
   Python floor and the private `_transposition_key()` dependency.
9. The KX-vs-K mate driver PROGRESS already plans for Phase 5 — it directly addresses the
   80-move conversion.

No redesign — negamax + alpha-beta + iterative deepening + TT + quiescence + killers/history is
the right spine, cleanly implemented. The open work is quiescence robustness, the deferred
speed (3e), and real evaluation (Phase 5).
