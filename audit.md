# Engine audit

**Audits so far (all 2026-09-06):**
1. Phase 1 skeleton.
2. Phase 2 (mate distance, draw awareness, edge cases).
3. Phase 3a (profiling bench — instrumentation only).
4. **Phase 3b (transposition table + iterative-deepening move ordering) — this document.**

**Branch:** `phase-3-tt` (3b committed and frozen to `versions/phase3b/`). **Next:** 3c
numba-jitted eval. Scope: [`evaluate.py`](evaluate.py), [`search.py`](search.py),
[`agent.py`](agent.py), [`tools/bench.py`](tools/bench.py); harness context from
[`harness/referee.py`](harness/referee.py). Findings verified by running the code.

---

## 0. Status

| Phase | State |
|---|---|
| 1 — minimal engine | **Complete** |
| 2 — robustness + draws | **Complete** — but 3b partially regressed the leaf-level repetition guard (finding 1) |
| 3a — profiling bench | **Complete** (instrumentation only) |
| 3b — transposition table | **Mostly complete** — TT is sound, ordering win is real and measured; fix finding 1 before 3c |

`ruff` + `mypy --strict` clean. 11 tests pass. Arenas (this audit):

| phase-3-tt vs `versions/phase2` | result | draws |
|---|---|---|
| 10 s + 0.1 s (arena default) | **+15 =1 -4, 77.5%** | 0 threefold, 1 fifty-move |
| 3 s + 0.05 s (stress) | +5 =9 -6, 47.5% | **9 threefold / 20** |
| `phase2` vs `phase2` @ 3 s + 0.05 s (control) | +9 =5 -6, 57.5% | 3 threefold / 20 |

The 10 s result corroborates PROGRESS's "83.8% vs Phase 2". The 3 s result is finding 1
biting — see section 4.

---

## 1. `evaluate.py` (unchanged since Phase 1)

### What static evaluation is

A heuristic score for a position computed **without searching**. The search bottoms out at leaf
nodes and calls `evaluate()` to guess who is better and by how much. All tactical understanding
comes from the search; the evaluator only needs to be roughly right about quiet positions.

### Material values (lines 9-15)

`PAWN 100, KNIGHT 320, BISHOP 330, ROOK 500, QUEEN 900`. Lines 33-35: `(white − black) count ×
value`. Standard Kaufman-ish values; knight ≈ bishop with a slight bishop edge; king has no
material value (checkmate is handled in the search).

### Why centipawns

1 cp = 1/100 pawn. Integer math scaled by 100 avoids float rounding in a function called
millions of times, leaves room for sub-pawn positional terms, and matches every published table.
`MATE = 1_000_000` sits far above any realistic material sum, so mate can't be confused with
material.

### Pawn piece-square table (lines 18-27)

`_PAWN_PST[square]` is a centipawn bonus for a white pawn, indexed `a1..h8`. By rank: rank 2 has
d2/e2 = −20 (push the centre pawns); rank 4 d4/e4 = +20; rank 5 d5/e5 = +25; rank 7 = +50 across
(near promotion); ranks 1 & 8 = 0. Classic CPW pawn table.

### Why Black's squares are mirrored (line 39)

`chess.square_mirror` flips the rank (a7↔a2, e5↔e4) so a black pawn is scored against the
positionally-equivalent white square; the result is **subtracted**.

### Why the final flip (line 40)

`return score if board.turn == chess.WHITE else -score`. Everything above is White-relative;
negamax needs every node scored **relative to the side to move**, so if Black is to move, negate.

### Why side-to-move perspective fits negamax

Negamax has every node maximise its own score and negate the child's. That only works if "score"
always means "good for whoever is on move." Line 40 makes `evaluate()` speak that convention, so
the search never needs to know its colour. A leaf returns "good for the player to move at the
leaf"; each `-_negamax(...)` one ply up flips it; each `board.push` swaps the side — signs stay
consistent from leaf to root.

### Understands / cannot understand

**Understands:** material; central vs wing pawns; whether the d/e pawns moved; pawn advancement,
especially near promotion; symmetrically for both sides.

**Cannot understand:** king safety, piece activity/mobility/outposts, non-pawn placement (no PST
for N/B/R/Q), pawn structure (doubled/isolated/backward/passed beyond raw rank), open files,
bishop pair as a term, space, tempo, initiative; any tactic beyond search depth; fortresses and
wrong-bishop draws. `evaluate()` itself ignores the halfmove clock and repetition (the search
handles those).

### Sensible / fails

**Sensible:** up a rook → +500; 1.e4 (−20→+20) preferred over 1.a3; winning a pawn → +100-ish;
KQ vs KR → +400.

**Fails:** positional sacrifice one ply past the horizon (declines it); opposite-side-castling
attack at equal material → 0.00; rim knight vs outpost knight → 0.00; healthy vs doubled
isolated pawns at equal count → 0.00; grabbing a pawn that wrecks its own king; fortress →
"+330 winning".

---

## 2. `search.py`, function by function (Phase 3b)

### Module state (lines 18-32)

- `MATE = 1_000_000`; `_MATE_THRESHOLD = MATE - 1_000` — any `|score| ≥` this is a forced mate
  (smallest possible mate magnitude is `MATE - 64`, safely above).
- `_CHECK_INTERVAL = 255` — wall clock tested every 255 nodes.
- `_EXACT, _LOWER, _UPPER = 0, 1, 2` — TT bound kinds.
- `_TT_MAX = 1_000_000` — clear the table rather than grow past this.
- `_nodes`, `_last_depth` (last fully completed ID depth; read by `tools/bench.py`),
  `_seen: frozenset` (transposition keys the real game has visited), and
  `_tt: dict[key -> (depth, value, flag, best_move)]`.

### `search_move(board, time_left_ms, history=None) -> str` (lines 39-77)

Resets `_nodes` / `_last_depth`; `_seen = frozenset(history)`; **`_tt.clear()`** (the table is
per-move for now). Then:

- no legal moves → `"0000"`; exactly one → return it immediately.
- iterative deepening `depth = 1..64`: `move, score = _search_root(board, depth, deadline, best)`
  — **`best` (the previous iteration's move) is passed in as the first move to try.** `_Timeout`
  → `break` (keep the previous `best`). After a completed depth: `best = move`,
  `_last_depth = depth`, optional `AGENT_DEBUG` print, `break` if `|score| ≥ _MATE_THRESHOLD`
  (forced mate — a deeper search can't improve a proven mate) or past the deadline.

### `_budget_s(board, time_left_ms) -> float` (lines 80-85) — unchanged since Phase 1

`moves_left = max(20, 50 - fullmove_number)`; `share = time_left_ms / moves_left`; capped at
`time_left_ms - 300`; floored at 10 ms. **Still ignores the 0.5 s/move increment.**

### `_search_root(board, depth, deadline, first) -> tuple[Move, int]` (lines 88-107)

`_ordered(...)` the legal moves, then move `first` to the front if present. Full-window
(`-MATE-1`) alpha on every root move, narrowing beta via `alpha = max(alpha, score)`. Keeps the
max with strict `>` (ties keep the earlier / better-ordered move → deterministic). Returns
`(best_move, best_score)`. Does **not** probe or store the TT for the root itself (fine — the
root is re-searched each iteration and `first` supplies the ordering).

### `_negamax(board, depth, ply, alpha, beta, deadline) -> int` (lines 110-170)

Order of operations:

1. `_tick(deadline)`.
2. `is_fifty_moves()` → 0.
3. `popcount(occupied) ≤ 4 and is_insufficient_material()` → 0.
4. `moves = list(board.legal_moves)`; if empty → `-MATE + ply` (check) or 0 (stalemate).
5. **`if depth <= 0: return evaluate(board)`** — leaf; the comment says "skip the transposition
   key and table entirely". **This is now above the repetition check** — see finding 1.
6. `key = board._transposition_key()` (interior nodes only).
7. `if halfmove_clock >= 4 and (is_repetition(2) or key in _seen): return 0`.
8. **TT probe:** `entry = _tt.get(key)`. Always extract `tt_move` for ordering. If
   `e_depth >= depth`: return `e_value` when `_EXACT`, or `_LOWER and e_value >= beta`, or
   `_UPPER and e_value <= alpha`.
9. `_ordered(...)`, then move `tt_move` to the front if present.
10. Search loop (fail-soft negamax + `alpha >= beta` cutoff), tracking `best_move`.
11. **TT store**, only if `|value| < _MATE_THRESHOLD` (mate scores are root-relative → path-
    dependent → not cacheable by position). Flag from the fail-soft result:
    `value <= alpha_orig` → `_UPPER`; `value >= beta` → `_LOWER`; else `_EXACT`.
    `if len(_tt) >= _TT_MAX: _tt.clear()` before insert.

**Negamax / negation:** after `board.push(move)` it is the opponent's turn and `_negamax`
returns a score relative to *that* side; `-` converts it back to this node's perspective.

**alpha / beta:** `alpha` = the best the side to move has already secured elsewhere (a lower
bound worth beating); `beta` = the best the opponent can hold them to (an upper bound),
`= -alpha` of the parent. `value >= beta` → the opponent avoids this whole line → stop (beta
cutoff / fail-high).

**Why alpha-beta is faster:** minimax visits `b^d` nodes (b ≈ 35); with good ordering
alpha-beta visits ≈ `b^(d/2)` — the square root — because one refutation discards a line and
ordering finds it first. Roughly doubles reachable depth. The TT and the `first` / `tt_move`
hints exist to make the ordering better still.

**Mate distance (verified):** checkmate is `-MATE + ply`, `ply` counted from the root. Negated
up the tree, an immediate mate is `MATE - 1`, a mate two of our moves off is `MATE - 3`, etc.
Larger = faster, so the engine **prefers the fastest mate and the longest defence**. `search_move`
stops deepening once `|score| ≥ _MATE_THRESHOLD`.

**Draw detection:** 50-move, insufficient material, in-search 2-fold repetition
(`is_repetition(2)`), and reaching a position in the real game's history (`key in _seen`) all
score 0. `_seen` is a set of keys with counts discarded (`agent.py` counts them, `search_move`
drops the counts) — the deliberate **"first repetition = draw"** heuristic: safe (never misses a
draw), at the cost of occasionally under-rating a won line that transposes through an earlier
position.

### `_ordered(board, moves) -> list[Move]` (lines 173-186) — unchanged since Phase 2

MVV-LVA on **piece-type ordinals** (1..6): `8 * victim - attacker`. `8 > 6` (king) guarantees a
bigger victim always sorts first, including king captures (Phase 1's centipawn version scored
`Kxp` at −19000). Queen promotions `+100`, other promotions `+10`. En passant:
`piece_type_at(to_square)` is `None`, `or chess.PAWN` → victim = pawn (correct). `sorted` is
stable → deterministic.

### `_tick(deadline)` (lines 189-193)

`_nodes += 1`; every 255 nodes check `time.monotonic() >= deadline` and raise `_Timeout`.
`deadline` is `_budget_s` (capped at `clock − 300 ms`), and the 0.5 s increment tops the clock
back up, so a one-slice overrun does not flag. Returning the last completed depth is safe: each
depth is an independent search; a partial depth is discarded when `_Timeout` unwinds
`_search_root`; tree cost is dominated by the deepest ply so the aborted depth wasted little.

---

## 3. How the files interact

```
get_move(fen, time_left_ms)                              agent.py
  board = chess.Board(fen)
  _history[board._transposition_key()] += 1              count this position for the game
  search_move(board, time_left_ms, _history)             search.py
    _seen = frozenset(_history);  _tt.clear()
    legal = board.legal_moves
      none -> "0000"   |   one -> return it
    deadline = now + _budget_s(...)
    best = legal[0]
    for depth in 1, 2, 3, ...:
      _search_root(board, depth, deadline, first=best)
        moves = _ordered(legal); move `first` to the front
        for each move: push; score = -_negamax(board, depth-1, ply=1, -MATE-1, -alpha, dl); pop
                       keep max; alpha = max(alpha, score)
           _negamax:
             _tick(); 50-move / insufficient -> 0
             moves = board.legal_moves; none -> -MATE+ply / 0
             depth<=0 -> return evaluate(board)                 <-- leaf (finding 1: no rep check)
             key = board._transposition_key()
             halfmove_clock>=4 and (is_repetition(2) or key in _seen) -> 0
             TT probe: e_depth>=depth and bound usable -> return e_value
             _ordered(moves); move tt_move to the front
             loop: push; -_negamax(... ply+1 ...); pop; alpha/beta cutoff; track best_move
             |value| < MATE-1000 -> _tt[key] = (depth, value, flag, best_move)
      best = move;  |score| >= MATE-1000 or past deadline -> break
    return best.uci()
  except Exception: return legal[0].uci() if legal else "0000"      fallback that can't raise
```

**Why side-to-move eval fits negamax:** covered in section 1 — the signs stay consistent from
leaf to root because `evaluate()` committed to the side-to-move convention on line 40, and every
`-_negamax(...)` plus every `board.push` flips together.

---

## 4. Audit of the Phase 3b code

### Verified correct (by running it)

- **TT soundness.** Identical best move *and* score with the TT live vs neutered, at depth 4-7
  across the three bench positions; node count ~5-6% lower with the TT. Alpha-beta + this TT is
  still exact for the returned value.
- **Fail-soft bound flags.** `alpha_orig` captured before the loop; `_UPPER` / `_LOWER` /
  `_EXACT` assigned correctly; probe only returns on `e_depth >= depth` with a window-compatible
  bound. `tt_move` is used for ordering even when the entry is too shallow to return — correct
  and desirable.
- **Mate scores are not cached** (`|value| < _MATE_THRESHOLD` guard) — right call, since
  `MATE - ply` is root-relative. Verified: `Rb8#` still scores `MATE - 1`; mate found and ID
  stops.
- **`first` (previous-iteration best) tried first at the root** — the "iterative deepening feeds
  move ordering" win the Phase 1 audit flagged as missing. This is where the arena gain comes
  from (see below), not raw depth.
- **`_transposition_key()` computed for interior nodes only** — leaves skip it (the intended
  speed trade), which is exactly what causes finding 1.
- **Determinism** preserved — `_tt` cleared per move and populated in a fixed node order;
  `search_move` returns the same move on repeat runs.
- **`best_move` is never stored as `None`** — the move loop always runs at least once and the
  first iteration always sets `value`/`best_move` (the `-MATE-1` sentinel can't survive).
- **Edge cases** from Phase 2 still hold: `"0000"` on no legal moves, single-move short-circuit,
  non-raising `get_move` fallback, `AGENT_DEBUG` gating the only print.
- `ruff` + `mypy --strict` clean; 11 tests pass; bench: endgame depth 6 → **7**, middlegame
  still depth 4, overall nps ≈ −5% (TT overhead > nodes saved at these depths — expected).

### Findings

**1. [Important — fix before 3c] Repetition / `_seen` draw detection regressed at leaf nodes.**

Phase 2 checked `is_repetition(2) or key in _seen` at the *top* of `_negamax`, before the
`depth <= 0` leaf return. Phase 3b moved that check *below* the leaf return (to piggyback on the
`key` it now computes only for interior nodes). So a position that is an in-search 2-fold
repetition, or one the real game has already visited, **now returns `evaluate(board)` instead of
0 when it lands exactly on the search horizon.**

Verified directly: a KR-vs-K position back at a `_seen` key returns **+500 at depth 0**, **0 at
depth 1**. Same for an in-search repetition.

This partially re-opens the "leak a won game into a threefold" hole that Phase 2 closed. It is
bounded to the horizon and to repetition-type draws (50-move and insufficient-material are still
checked before the leaf return). Iterative deepening self-corrects at the next depth — *when
there is time for it*.

**It is not just theoretical.** At 3 s + 0.05 s the search often finishes only depth 3-4, and
the arena shows the effect: phase-3-tt vs `versions/phase2` scored **47.5% with 9 threefold
draws / 20**, versus `phase2`-vs-`phase2` at the same control (**57.5%, 3 threefold / 20**) and
versus phase-3-tt's own **77.5%, 0 threefold** at the normal 10 s control. The competition is
120 s + 0.5 s, so a long game routinely reaches low clock — this will cost games there, against
PLAN priority #3 ("don't lose games to ourselves").

*Minimal fix:* in the `depth <= 0` branch, still run the repetition/`_seen` test before
returning `evaluate` — pay `_transposition_key()` there only when `halfmove_clock >= 4` (rare in
sharp middlegames, so the leaf fast-path is mostly preserved):

```python
if depth <= 0:
    if board.halfmove_clock >= 4 and (board.is_repetition(2)
            or board._transposition_key() in _seen):
        return 0
    return evaluate(board)
```

**2. [Minor — handle in Phase 4] TT can cache a path-dependent draw score.**

An interior node whose subtree returned 0 via the in-search `is_repetition(2)` folds that 0 into
its stored `value`. If the same `key` is later probed on a path where that repetition would not
occur, the cached value is wrong. `key in _seen` is path-independent (fixed game history), so
only in-search repetitions contribute, and the contamination is bounded to one move because
`_tt` is cleared each move. **This must be addressed when the Phase 4 persistent TT lands** —
e.g. don't store a node whose subtree hit a repetition, or tag such entries.

**3. [Minor] The TT is currently a small net speed loss** (~5% nodes/sec) — at depth 4-7 the
key/probe/store overhead exceeds the nodes saved. The payoff is entirely move-ordering
stability from seeding `first`/`tt_move` (draws 73/100 → ~1/20 at normal TC, +15 vs Phase 2).
Fine — it's Phase 4 groundwork — but don't expect the bench to move until 3c/3d.

**4. [Minor] `_TT_MAX` clears the whole table on overflow** rather than evicting, losing all
ordering info mid-search. Cannot trigger in the pure-Python regime (~50 k nodes/search ≪ 1 M
entries), so moot now; revisit with the fixed-size table in Phase 4.

**5. [Nit, carried] `board._transposition_key()` is a private API**, used in `agent.py` and
`search.py`. Fine while python-chess is pinned at 1.11 (platform + local); worth a comment.

**6. [Nit] Redundant move generation** — `search_move` builds `board.legal_moves`, then
`_search_root` builds it again. Trivial; Phase 3e movegen work subsumes it.

### No bugs found in

TT flag/probe logic, mate handling with the TT, the fail-soft window, determinism, the
`get_move` fallback, `_last_depth` on timeout (correctly holds the last *completed* depth — the
`break` precedes the assignment).

### Docstring match

The new `search.py` docstring accurately describes 3b: "a search result is cached by position…
the best move from the last iteration is tried first… cleared each move for now; Phase 4 makes
it persistent and fixed-size." True and verified.

Against PLAN §Phase 3: 3a (profile) and 3b (TT) done; "arena vs Phase 2 shows the extra ply
paying off" is **partially** met — the extra ply shows only in the endgame (6→7); the
middlegame is still depth 4 and the arena gain is ordering, not depth. That is expected before
3c/3d/3e and PROGRESS says as much.

Still open from earlier audits (unchanged, not 3b's job): `_budget_s` ignores the 0.5 s
increment; no explicit seed / documented tie-break (deterministic in practice); PLAN's "300+
games vs random and vs greedy, zero crash/flag/illegal" not run since Phase 2.

---

## 5. Verdict

**Phase 1: Complete. Phase 2: Complete** (with the leaf-repetition caveat below).
**Phase 3a: Complete. Phase 3b: Mostly complete** — the TT is sound and the move-ordering win is
real and measured, but one Phase 2 guarantee regressed.

### Fix before 3c

**Critical**

- None.

**Important**

1. **Restore leaf-level repetition / `_seen` detection** (finding 1). Two lines. Without it the
   engine leaks won/equal games into threefold draws whenever it is low on clock — measured, not
   hypothetical.
2. **Record the arena time control in PROGRESS.** "83.8% vs Phase 2" holds at 10 s + 0.1 s and
   collapses at 3 s + 0.05 s; a strength claim without its TC is not reproducible. Re-run the
   comparison after fixing finding 1.

**Optional**

3. Note finding 2 (TT + path-dependent draw scores) in the Phase 4 plan so the persistent table
   handles it from the start.
4. Reconsider `_TT_MAX` behaviour (evict vs clear) when the fixed-size Phase 4 table lands.
5. Carry-overs: increment-aware `_budget_s`; explicit seed + documented tie-break; run the
   Phase 2 "done" bar (300+ games vs random and greedy).

No Phase 3c+ work and no redesign — negamax + alpha-beta + iterative deepening + TT is the right
spine. Finding 1 is the one thing standing between "Phase 2 is genuinely intact" and not.
