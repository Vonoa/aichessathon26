# Phase 1 / Phase 2 engine audit

**First audit:** 2026-09-06 (against the Phase 1 skeleton)
**Re-audit:** 2026-09-06 (against the Phase 2 implementation now in the working tree)
**Branch:** `phase-2-robustness`
**Scope:** [`evaluate.py`](evaluate.py), [`search.py`](search.py), [`agent.py`](agent.py), with context from
[`versions/phase1/`](versions/phase1/) and the harness ([`harness/referee.py`](harness/referee.py),
[`harness/sandbox.py`](harness/sandbox.py), [`harness/rules.py`](harness/rules.py)).
Findings below were verified by running the code, not only by reading it.

---

## 0. Status

- The first audit found `search.py`/`agent.py` byte-identical to `versions/phase1/` — no Phase 2
  code at all.
- Since then, Phase 2 has been implemented in `search.py` and `agent.py` (uncommitted at
  re-audit time). `evaluate.py` is unchanged from Phase 1.
- `ruff` and `mypy --strict` are clean. A 4-game arena vs `baselines/greedy` at 4 s base:
  **+4 =0 -0, all won by checkmate**, no crash / flag / illegal.

**Phase 1: Complete.** **Phase 2: Mostly complete** — the core is implemented and verified; the
gaps are in time management and test/repro scaffolding. Details in section 5.

---

## 1. `evaluate.py` (unchanged since Phase 1)

### What static evaluation is

A heuristic score for a position computed **without searching** — no lookahead, no move
generation. The search bottoms out at leaf nodes and calls `evaluate()` to guess who is better
and by how much. All tactical understanding comes from the search; the evaluator only needs to
be roughly right about quiet positions.

### Material values (lines 9-15)

`PAWN 100, KNIGHT 320, BISHOP 330, ROOK 500, QUEEN 900`. Lines 33-35 compute
`(white count - black count) * value` per piece type. Standard Kaufman-ish values: knight approx
bishop with a slight bishop edge (330 vs 320) as a crude bishop-pair proxy; rook = 5 pawns;
queen = 9. King has no material value here — its "value" is handled by checkmate detection in
the search.

### Why centipawns

1 centipawn = 1/100 of a pawn. Integer math scaled by 100 avoids floating-point rounding in a
function called millions of times, leaves room for sub-pawn positional terms (a 25 cp PST bonus)
without fractions, and matches the units every other engine and every published table uses.
`MATE = 1_000_000` in the search sits far above any realistic material sum (~4000 cp), so a mate
score can never be confused with a material score.

### The pawn piece-square table (lines 18-27)

`_PAWN_PST[square]` is a centipawn bonus for a white pawn on that square, indexed `a1..h8`
(square 0 = a1, 8 = a2, ..., 56 = a8). By rank:

- Rank 1 & 8: all 0 — pawns cannot be there.
- Rank 2 (`5,10,10,-20,-20,10,10,5`): d2/e2 get **-20** — penalise an unmoved centre pawn, so
  `d4`/`e4` is a 40 cp swing. Flank pawns +5/+10.
- Rank 3 (`5,-5,-10,0,0,-10,-5,5`): mild penalty for `c3`/`f3`-type blocking moves.
- Rank 4 (`0,0,0,20,20,0,0,0`): d4/e4 +20 — occupy the centre.
- Rank 5 (`5,5,10,25,25,10,5,5`): d5/e5 +25.
- Rank 6 (`10,10,20,30,30,20,10,10`): further advancement.
- Rank 7 (all `50`): a pawn one step from promotion is +50 regardless of file.

Classic Chess Programming Wiki pawn table: control the centre, push the d/e pawns, value
advanced/near-promotion pawns.

### Why Black's squares are mirrored (line 39)

The table is written from White's viewpoint. A black pawn on a7 is in the same strategic
situation as a white pawn on a2. `chess.square_mirror(square)` flips the rank (a7<->a2, e5<->e4)
so the lookup uses the positionally-equivalent white square. The result is **subtracted** because
Black's advantages are negative in a White-relative score.

### Why the final flip (line 40)

```python
return score if board.turn == chess.WHITE else -score
```

Everything above is **White-relative** (positive = good for White). Negamax requires every node
to return a score **relative to the side to move at that node**. If it is Black to move, negate.

### Why side-to-move perspective fits negamax

Negamax uses `max(a, b) = -min(-a, -b)`: every node maximises its own score and negates the
child's, instead of separate maximise-for-White / minimise-for-Black code. That only works if
"score" always means "good for whoever is on move." Line 40 makes `evaluate()` speak that
convention directly, so the search never needs to know its colour. Concretely: a leaf returns
"good for the player to move at the leaf"; each `-_negamax(...)` one ply up flips it to that
node's perspective, and each `board.push` swaps the side to move, so the signs stay consistent
from leaf to root.

### What the evaluator understands

Material balance; pawns in the centre vs on the wings; whether the d/e pawns have been pushed (a
development proxy); how far advanced each pawn is, especially near promotion; all of it
symmetrically for both sides.

### What it cannot understand

- King safety — exposed king scores the same as a castled one.
- Piece activity / mobility / outposts — knight on a3 == knight on e5.
- Non-pawn placement — knight/bishop/rook/queen have no PST; only count matters.
- Pawn structure — doubled, isolated, backward, passed (beyond raw rank), islands, majorities.
- Rooks on open files, bishop pair as a real term, space, tempo, initiative.
- Any tactic not resolved by search depth — pins, forks, discovered attacks, trapped pieces
  (a doomed bishop counts full value until actually captured within the horizon).
- Fortresses, wrong-bishop endings — reports "+330, winning" in a dead draw.
- Draw proximity — `evaluate()` itself ignores the halfmove clock and repetition (the *search*
  now handles those; see section 2).

### Where it is sensible

- Up a clean rook, quiet position: approx +500, correct; the search trades down and wins.
- 1.e4 vs 1.a3: e-pawn -20 -> +20 (40 cp gain), a3 stays 5 -> 5. Prefers a real opening move.
- Winning a pawn in a symmetric structure: +100-ish, right direction and magnitude.
- KQ vs KR, nothing loose: +400, knows which side to be.

### Where it fails

- Positional sacrifice: give up a knight for a mate one ply past the horizon -> eval says -320,
  engine declines the winning sac.
- Opposite-side-castling attack, material equal: three pawns storming a bare king -> eval 0.00.
- Knight on the rim vs knight on a central outpost, equal material -> 0.00.
- Healthy majority vs doubled isolated pawns, equal count -> 0.00.
- Grab a pawn that shatters your own king cover, refutation just past the horizon -> takes it.
- Fortress / wrong rook-pawn bishop ending -> "+330, winning", misplays a draw.

---

## 2. `search.py`, function by function (Phase 2 code)

### Module state (lines 17-26)

- `MATE = 1_000_000`; `_MATE_THRESHOLD = MATE - 1_000` — any score with `abs >= _MATE_THRESHOLD`
  is a forced mate. Mate scores are `MATE - ply` with `ply <= _MAX_DEPTH = 64`, so the smallest
  possible mate magnitude is `MATE - 64 = 999_936`, comfortably above the threshold and above
  any eval. Verified.
- `_CHECK_INTERVAL = 255` (was 1023 in Phase 1) — the wall clock is now tested 4x more often.
- `_DEBUG = os.environ.get("AGENT_DEBUG") == "1"` — gates the per-depth `print`.
- `_nodes` — node counter for the clock check. `_seen: frozenset[Hashable]` — transposition keys
  of every position the game has actually visited, refreshed each call from `agent.py`'s history.

### `search_move(board, time_left_ms, history=None) -> str` (lines 33-67)

Public entry. Resets `_nodes`; sets `_seen = frozenset(history)` (keys only — see the note on
counts below). Then:

- `legal = list(board.legal_moves)`; **no legal moves -> return `"0000"`** (the null-move UCI —
  a "can't move" sentinel; in a real game this call never happens because the referee ends a
  mated/stalemated game first).
- **exactly one legal move -> return it immediately**, no search, no clock spent.
- Otherwise iterative deepening `for depth in 1..64`:
  - `move, score = _search_root(board, depth, deadline)`; `_Timeout` -> `break` (keep the
    previous depth's `best`).
  - `best = move`.
  - if `_DEBUG`: print `depth / score / nodes / elapsed ms`.
  - **if `abs(score) >= _MATE_THRESHOLD`: `break`** — a forced mate was found (or proven against
    us); a deeper search of the same tree cannot change a proven mate.
  - if past the deadline: `break`.
- return `best.uci()`.

Why iterative deepening: always have a finished answer when time runs out; re-searching shallow
depths is cheap because the tree grows geometrically; it also sets up move ordering (not yet
exploited — no PV carry-over).

### `_budget_s(board, time_left_ms) -> float` (lines 70-75) — unchanged from Phase 1

```python
moves_left = max(20, 50 - board.fullmove_number)
share      = time_left_ms / moves_left
capped     = min(share, time_left_ms - _SAFETY_MS)   # _SAFETY_MS = 300
return       max(capped, 10.0) / 1000.0              # 10 ms floor
```

Move 1 / 120 s -> ~2.45 s. Move 20 / 60 s -> 2.0 s. Move 40 / 5 s -> 0.25 s. Any clock
`<= ~300 ms` -> the 10 ms floor. **The 0.5 s/move increment is not used at all**, and the
position term is only `50 - fullmove_number`. See section 5.

### `_search_root(board, depth, deadline) -> tuple[chess.Move, int]` (lines 78-90)

The top ply, separate because it returns a **move** and has no beta above it. Seeds
`best_move` / `best_score` / `alpha` at `-MATE-1`. For each ordered legal move: `push`,
`score = -_negamax(board, depth-1, ply=1, -MATE-1, -alpha, deadline)`, `pop`, keep the max
(`>`, so ties keep the earlier / better-ordered move -> deterministic), `alpha = max(alpha, score)`
so later siblings get a narrowing window. Returns `(best_move, best_score)`. A `_Timeout` from
inside `_negamax` propagates straight out to `search_move`, discarding the whole depth.

### `_negamax(board, depth, ply, alpha, beta, deadline) -> int` (lines 93-118)

```python
_tick(deadline)                                              # may raise _Timeout
if board.is_fifty_moves():                    return 0       # 50-move rule
if popcount(occupied) <= 4 and board.is_insufficient_material(): return 0
if board.halfmove_clock >= 4 and (board.is_repetition(2)
        or board._transposition_key() in _seen):            return 0   # repetition
moves = list(board.legal_moves)
if not moves:   return -MATE + ply if board.is_check() else 0          # mate / stalemate
if depth <= 0:  return evaluate(board)                                 # leaf
value = -MATE - 1
for move in _ordered(board, moves):
    board.push(move)
    value = max(value, -_negamax(board, depth-1, ply+1, -beta, -alpha, deadline))
    board.pop()
    alpha = max(alpha, value)
    if alpha >= beta:  break                                          # beta cutoff
return value
```

**Negamax + why the negation:** after `board.push(move)` it is the opponent's turn and
`_negamax` returns a score relative to *that* side; `-` converts it back to this node's
perspective.

**alpha / beta:** `alpha` = the best the side to move has already secured elsewhere (a lower
bound worth beating). `beta` = the best the opponent can already hold them to (an upper bound);
`beta = -alpha` of the parent, because the opponent's floor is this node's ceiling. If a move
yields `value >= beta` the opponent will avoid this whole line, so stop — a **beta cutoff /
fail-high**.

**Why alpha-beta is much faster:** plain minimax visits `b^d` nodes (b ~ 35). With good move
ordering alpha-beta visits about `b^(d/2)` — the square root — because one refutation is enough
to discard a line, and ordering finds it first. Roughly doubles the depth reachable in the same
time; hence `_ordered`.

**Root-relative mate scores (verified):**

- Checkmate is scored `-MATE + ply` where `ply` counts half-moves from the root. Negated up the
  tree, an immediate mate becomes `MATE - 1`, a mate two of our moves away becomes `MATE - 3`,
  etc. Larger score = faster mate, so `_search_root`'s max-pick **prefers the faster mate**.
  Verified: back-rank `Rb8#` scores `999999`; a forced mate-in-N scores `MATE - (2N-1)`.
- Being mated is `-MATE + ply` from our side: mated-in-1 = `-MATE + 2 = -999998`, mated-in-2 =
  `-MATE + 4`. `-999998 < -999996`, so the max-pick **prefers the line that is mated later** —
  it delays the mate. Verified.
- `search_move` stops iterative deepening as soon as `|score| >= _MATE_THRESHOLD`.

**Draw detection inside the search (verified):**

- `is_fifty_moves()` -> 0. Verified (and python-chess already returns False here if the position
  is actually checkmate, so mate keeps precedence — checked).
- insufficient material (with a `popcount <= 4` fast guard) -> 0.
- **Repetition** -> 0, when `halfmove_clock >= 4` (below that no repetition is reachable) and
  either:
  - `board.is_repetition(2)` — the current position already occurred once earlier *in this
    search line* (in-search repetition), or
  - `board._transposition_key() in _seen` — the current position is one the **actual game** has
    already visited.
  Both verified to return 0.

This is the **"first repetition = draw" heuristic**: a 2nd occurrence (not yet a legal
threefold) is already scored as a draw. It is the conventional, deliberately-safe choice — it
errs toward *seeing* draws, never toward missing one, which is the right direction for "don't
lose a won game to a repetition." The cost: `_seen` is a set of keys with the counts discarded
(`agent.py` counts them, `search_move` throws the counts away), so a position the game visited
just once, if the PV transposes through it, is scored 0 — the engine can **under-rate a
genuinely winning line that passes back through an earlier position**. Acceptable trade-off,
worth knowing.

Ordering note: the draw checks run before the "no legal moves" (mate/stalemate) check. python-chess
guards `is_fifty_moves()` / `is_insufficient_material()` against a checkmate position, and a
checkmate position cannot also be a repetition (the game would have ended at the first
occurrence), so no real precedence bug — confirmed by test.

### `_ordered(board, moves) -> list[chess.Move]` (lines 121-134) — rewritten in Phase 2

```python
value = 0
if board.is_capture(move):
    victim   = board.piece_type_at(move.to_square)   or chess.PAWN   # ordinals 1..6
    attacker = board.piece_type_at(move.from_square) or chess.PAWN
    value = 8 * victim - attacker
if move.promotion is not None:
    value += 100 if move.promotion == chess.QUEEN else 10
return value                                          # sorted(..., reverse=True), stable
```

MVV-LVA now uses **piece-type ordinals (1..6)**, not centipawn values. Because `8 > 6` (the
largest ordinal, the king), `8 * victim` always dominates the attacker term, so a bigger victim
always sorts first — **including king captures**, which in the Phase 1 centipawn version scored
`10*100 - 20000 = -19000` and sorted below quiet moves. Verified: `Kxe3` now scores `+2`.
Queen promotions get `+100` (above any capture), other promotions `+10`. En passant:
`piece_type_at(to_square)` is `None`, `or chess.PAWN` makes the victim a pawn — correct.
Non-capture non-promotion -> 0. `sorted` is stable, so equal scores keep python-chess's fixed
order -> deterministic.

### `_tick(deadline) -> None` (lines 137-141)

`_nodes += 1`; every `_CHECK_INTERVAL = 255` nodes, check `time.monotonic() >= deadline` and
raise `_Timeout`. Checking every node would be measurable overhead; every ~255 nodes amortises
it while bounding overrun to one 255-node slice. `deadline` comes from `_budget_s`, already
capped at `time_left_ms - 300 ms`, and the 0.5 s increment tops the clock back up, so an overrun
of one slice does not flag.

**Why returning the last completed depth is safe:** each depth is a complete, independent search
to that depth; a partially-searched depth is discarded entirely when `_Timeout` unwinds
`_search_root`. The depth-(N-1) move is fully valid, just shallower. And tree cost is dominated
by the deepest ply, so the aborted depth wasted little.

### The `8 * victim - attacker` formula (ordinals)

| Capture | victim,attacker ordinals | score |
|---|---|---|
| Pawn takes Queen | 5, 1 | 8*5 - 1 = 39 |
| Knight takes Queen | 5, 3 | 37 |
| Queen takes Queen | 5, 5 | 35 |
| Pawn takes Rook | 4, 1 | 31 |
| Pawn takes Pawn | 1, 1 | 7 |
| Queen takes Pawn | 1, 5 | 3 |
| King takes Pawn | 1, 6 | 2 (Phase 1: -19000) |
| quiet move | - | 0 |
| queen promotion | - | +100 (+capture term if also a capture) |

x8 guarantees "larger victim always wins" for every attacker including the king, because the
smallest victim gap (1) times 8 exceeds the largest attacker ordinal (6).

---

## 3. How the two files interact

```
get_move(fen, time_left_ms)                              agent.py
  board = chess.Board(fen)
  key = board._transposition_key()
  _history[key] += 1                                     count this position for the game
  search_move(board, time_left_ms, _history)             search.py
    _seen = frozenset(_history)                          keys of every visited position
    legal = board.legal_moves
      none  -> return "0000"
      one   -> return it
    deadline = now + _budget_s(...)
    for depth in 1, 2, 3, ...:
      _search_root(board, depth, deadline)
        for each ordered legal move m:                   _ordered: captures MVV-LVA, then promos
          board.push(m)
          score = -_negamax(board, depth-1, ply=1, -MATE-1, -alpha, deadline)
             _negamax:
               _tick(deadline)                                 may raise _Timeout
               fifty-move / insufficient / repetition -> 0     draw detection
               moves = board.legal_moves
               if none: return -MATE+ply / 0                   mate (distance-encoded) / stalemate
               if depth<=0: return evaluate(board)             LEAF: evaluate.py, side-to-move int
               else: for each child: push; v=max(v,-_negamax(...ply+1...)); pop
                     alpha=max(alpha,v); if alpha>=beta: break prune
          track best (score, move); alpha=max(alpha,score)
      best = move
      if |score| >= MATE-1000: break                     forced mate found
      if past deadline: break
    return best.uci()                                     "e2e4" / "e7e8q"

  except Exception:                                       fallback that cannot raise
    legal = list(board.legal_moves)
    return legal[0].uci() if legal else "0000"
```

**Why side-to-move eval fits negamax:** the leaf returns "good for the player to move at the
leaf"; every `-_negamax(...)` one ply up flips it to that node's perspective, and every
`board.push` swaps the side to move, so the signs stay consistent from leaf to root purely
because `evaluate()` committed to that convention on line 40. At the root, `_search_root`
maximises `-_negamax(child)` = "best for us." A White-relative eval would invert the sign at
every Black node and the engine would help its opponent on alternate plies.

---

## 4. Audit of the current code

### Implemented correctly (verified by running it)

- **Fail-soft negamax + alpha-beta.** Window passing, the `alpha >= beta` cutoff, the `-MATE-1`
  sentinel that never leaks.
- **Root-relative mate scores.** Immediate mate `MATE-1`; forced mate-in-N `MATE-(2N-1)`; being
  mated `-MATE+ply`. The engine **prefers faster mates and delays being mated** (both verified),
  and stops iterative deepening once a mate is proven.
- **Draw detection in the search:** 50-move, insufficient material, in-search repetition
  (`is_repetition(2)`), and game-history repetition (`_seen`). All verified to score 0. This
  closes the Phase 1 "wins material straight into a threefold" hole.
- **Edge cases:** no legal moves -> `"0000"`; one legal move -> instant; the `get_move`
  `except` fallback can no longer raise (Phase 1's double `StopIteration` is gone).
- **MVV-LVA** fixed for king attackers via ordinals; queen-promotion ordering bonus added.
- **En passant / promotions / castling** come out of `board.legal_moves`, push/pop and evaluate
  correctly; `search_move` emits `e7e8q` for promotions.
- **Determinism.** No RNG anywhere; `sorted` stable; `_search_root` keeps the first move on ties.
- **`AGENT_DEBUG=1`** gates the only `print` (PLAN's "DEBUG flag gates every print").
- **Tooling:** `ruff` and `mypy --strict` clean. 4/4 vs `baselines/greedy` by checkmate, no
  crash / flag / illegal.

### Partially implemented / weak

- **Time management vs PLAN.** PLAN Phase 2: "Time budget adapts to the position and the
  increment." `_budget_s` is unchanged from Phase 1 — it never uses the 0.5 s/move increment,
  and its position term is just `max(20, 50 - fullmove_number)`. It is safe (10 ms floor,
  300 ms watchdog margin, 255-node clock checks) but not adaptive. This is the main Phase 2
  shortfall.
- **Reproducibility scaffolding.** PLAN Phase 2 wants "a deterministic tie-break plus a global
  seed" and "a move-1 nodes/sec assertion in the tests." The engine is deterministic in
  practice, but there is no explicit seed, no documented tie-break rationale, and **no tests in
  the repo at all** (no `tests/`, `make gate` only plays two games).
- **`board._transposition_key()` is a private API**, used in both `agent.py` and `search.py`.
  Fine while python-chess is pinned at 1.11 (platform and local), but note it.

### Bugs

None found in the re-audit. Three hypotheses were checked and dismissed:

- *"Draw-by-rule is checked before checkmate -> a mate on the 100th half-move is scored 0."*
  False: python-chess `is_fifty_moves()` / `is_insufficient_material()` return False on a
  checkmate position, and a checkmate position cannot repeat. Verified: `_negamax` returns
  `-999999` (mate), not 0, for a checkmate at `halfmove_clock = 101`.
- *"`is_repetition(2)` at every node tanks nodes/sec."* Measured negligible — the
  `halfmove_clock >= 4` guard skips it in most of the tree, and where it runs the difference is
  within run-to-run noise (~22k nps midgame, ~19k nps in a rook ending either way).
- *"No-legal-moves at the root crashes."* Fixed — `search_move` returns `"0000"` and the
  fallback in `get_move` mirrors it.

### Conceptual / minor observations (not bugs)

1. **"First repetition = draw" heuristic.** `_seen` keeps keys, not counts, and
   `is_repetition(2)` fires on the first repeat. The engine treats a 2nd occurrence as a draw
   even though a legal threefold needs a 3rd. Deliberately safe (never misses a draw), but it
   can make the engine **under-convert a won position whose best line transposes back through an
   earlier position** — it sees 0 there and may steer away. Conventional trade-off; keep it in
   mind when a won game peters out.
2. **Early break when being mated.** `search_move` breaks out of iterative deepening as soon as
   `|score| >= _MATE_THRESHOLD`, including when the score is `-MATE + ply` (we are lost). A
   proven forced mate cannot be un-proven by a deeper search, so this is sound, but it does stop
   the engine from looking for an even longer defence than the one found at that depth. Impact
   is negligible (the game is lost either way) and it saves clock.
3. **`"0000"` on a no-move root** would be scored "illegal" if it ever reached the referee, but
   it cannot — the referee ends a mated/stalemated game before calling the agent. There is no
   better value to return (there is no legal move).
4. **`_budget_s` floor + overrun at a tiny clock.** At `time_left_ms <= ~300`, budget is the
   10 ms floor; a real search returns in ~15-20 ms (verified), overrunning by a few ms. Only
   relevant below ~20 ms on the clock, where the game is already lost, and the increment
   recovers it.
5. **Redundant move generation.** `search_move` generates `board.legal_moves`, then
   `_search_root` calls `next(iter(board.legal_moves))` and `list(board.legal_moves)` again.
   Trivial; Phase 3 movegen work subsumes it.

### Does it match the docstrings?

Yes, now. `search.py`'s Phase 2 docstring claims "mate scores are relative to the root ply,
draws (repetition, 50-move, insufficient material) are detected inside the search using the game
history passed in from agent.py, and AGENT_DEBUG=1 prints a line per completed depth." All three
are present and verified. `agent.py`'s comment about `_history` being read by the search to spot
a repeated position is accurate.

Against PLAN section "Phase 2":

| PLAN Phase 2 item | Status |
|---|---|
| Mate scores relative to the search root ply | DONE, verified |
| Prefers mate sooner / delays being mated | DONE, verified |
| Repetition awareness (keep the positions we are asked about) | DONE (`_history` -> `_seen`) |
| Edge-case sweep: no legal moves, stalemate, checkmate, promotion, en passant | DONE, verified |
| Hard clock checks inside the search + watchdog margin | DONE (`_tick` @255, `_SAFETY_MS`) |
| Time budget adapts to the position and the increment | PARTIAL — no increment term, weak position term |
| Deterministic tie-break plus a global seed | PARTIAL — deterministic, but no explicit seed/doc |
| A `DEBUG` flag gates every `print` | DONE (`AGENT_DEBUG`) |
| A move-1 nodes/sec assertion in the tests | MISSING — no tests exist |
| Done when: 300+ games vs random and vs greedy, zero crash/flag/illegal | NOT YET RUN (4/4 clean so far) |

---

## 5. Verdict

**Phase 1: Complete.** The two Phase 1 blemishes from the first audit — king-attacker move
ordering and the `StopIteration` in the `get_move` fallback — are both fixed.

**Phase 2: Mostly complete.** The hard parts (root-relative mate distance, prefer/delay mate,
repetition + 50-move + insufficient-material draw detection using real game history, the
edge-case sweep, tighter clock checks, `AGENT_DEBUG`) are implemented and verified working.
`ruff`/`mypy` clean; early arena results clean. What remains is time-management polish and the
repro/test scaffolding PLAN asks for.

### Fix before moving to Phase 3

**Critical**

- None. No correctness bugs were found in the re-audit.

**Important**

1. **Run the Phase 2 "done" bar:** 300+ games vs `random` and vs `greedy`, assert zero
   crash / flag / illegal. This is the gate PLAN sets and it has not been run.
2. **Make the time budget use the increment** (0.5 s/move) — e.g. allow roughly
   `increment + remaining_share`, still under the `time_left_ms - _SAFETY_MS` cap. Right now a
   long game leaves time on the table because only the base clock is divided up.
3. **Add tests** (there are none): a move-1 nodes/sec assertion (guards against numba compiling
   on the clock in Phase 3), plus fixed-position checks for mate-in-1 score, being-mated score,
   stalemate = 0, repetition = 0, 50-move = 0, one-legal-move fast path, en passant, promotion,
   castling.
4. **Document the deterministic tie-break and set an explicit seed** even if nothing is random
   yet — PLAN wants a lost game to be reproducible and explainable at the finals.

**Optional**

5. Decide deliberately whether the "first repetition = draw" eagerness (keys-only `_seen`,
   `is_repetition(2)`) is what you want, or whether to pass counts through and require a true
   3rd occurrence. Current behaviour is the safe default; just make it a choice on the record.
6. Replace the private `board._transposition_key()` with a small helper, or add a comment
   pinning the assumption to python-chess 1.11.
7. Drop the redundant `board.legal_moves` regeneration in `_search_root`.
8. Consider not breaking out of iterative deepening when the mate score is against us, so a
   longer defence can still be found (very low value).

No Phase 3 work and no redesign — negamax + alpha-beta + iterative deepening is the right spine,
and Phase 2 is now genuinely on it.
