# How the engine works — the maths and the ideas

Reference for the team and the finals walkthrough. Covers every technique in `search.py`
and `evaluate.py` and why it is there. Companion to [PROGRESS.md](../PROGRESS.md) (history)
and [HANDOVER.md](../HANDOVER.md) (current task).

---

## 1. The setting

`get_move(fen, time_left_ms) -> str`. One core, pure Python, ~120 s for the whole game.
Node counts are in the **thousands per move**, not the millions a C engine gets. So the
trade is different: evaluation quality and move ordering buy more than raw depth, and every
constant is chosen against a tiny time budget.

A "node" = one position the search looks at. "Ply" = one half-move (one side moving).
"Depth" = how many plies ahead the main line looks before the evaluation is called.

---

## 2. Negamax

Minimax says White maximises the score, Black minimises it — two mirror-image code paths.
**Negamax** collapses them into one using the identity

```
min(a, b)  =  -max(-a, -b)
```

Score every position **from the point of view of the side to move**. Then whatever is good
for me is exactly `-1 x` as good for my opponent, so one function handles both sides:

```
value(node) = max over legal moves m of  -value(child after m)
```

The `-` in front of the recursive call is the whole trick: after you play `m` it is the
opponent's turn, the child returns a score in *their* frame, and negating it puts it back
in yours. `evaluate()` commits to the same convention (`return score if white_to_move else
-score`), so signs stay consistent from leaf to root automatically.

---

## 3. Alpha-beta pruning

Plain minimax visits `b^d` nodes, where `b` is the branching factor (~35 legal moves in a
middlegame) and `d` is the depth. Alpha-beta carries two bounds down the tree:

- **alpha** — the best score the side to move has already guaranteed somewhere else. A
  floor worth beating.
- **beta** — the best the opponent will allow (it is `-alpha` of the parent, because the
  opponent's floor is this node's ceiling).

If a move returns `value >= beta`, the opponent has a cheaper alternative one ply up and
will never enter this line, so you stop searching the siblings — a **beta cutoff**.

With **perfectly ordered** moves alpha-beta visits about `b^(d/2)` nodes — the square root.
Roughly: you need one good reply to refute a line, and good ordering finds it first, so
most subtrees collapse to a single child. In practice this doubles the reachable depth for
the same node budget. Ordering quality is therefore everything — hence section 4.

Our version is **fail-soft**: it returns the true best value found even when it is outside
the `[alpha, beta]` window, which gives the transposition table tighter bounds to store.

---

## 4. Move ordering

Alpha-beta only pays off if the best move is tried first. We score each move and sort
descending. The bands (from `search.py`):

**Captures / promotions** — score `_CAPTURE_BASE + 8*victim - attacker (+ promo bonus)`.
This is **MVV-LVA** (Most Valuable Victim, Least Valuable Attacker): `victim` and `attacker`
are piece-type ordinals 1..6 (pawn..king). Multiplying the victim by 8 guarantees a bigger
victim always outranks a smaller one regardless of attacker (max attacker 6 < 8), and
subtracting the attacker prefers taking with the cheapest piece (less at risk on the
recapture). "Pawn takes queen" beats "queen takes queen" beats "pawn takes knight".

**Killer moves** — two slots per ply. A *quiet* move that caused a beta cutoff at one node
is very often good at a sibling node too (same tactical motif), so it is tried right after
the captures.

**History heuristic** — a `64 x 64` table indexed `from_square * 64 + to_square`. Every
time a quiet move causes a cutoff we add `depth * depth` to its cell. The `depth^2` weight
means a cutoff found deep in the tree (where searching is expensive) counts far more than a
shallow one. Remaining quiet moves are ordered by this score.

The transposition-table move (section 5), if any, is forced ahead of all of these.

Killers and history reset every move (`search_move` clears them). They compound *with
depth* — at depth 2-3 they measure flat, which is why the engine is currently search-
feature-saturated and eval-speed-bound.

---

## 5. Iterative deepening

Instead of picking a depth and searching once, search depth 1, then throw it away and
search depth 2, then 3, ... until the time budget for this move runs out.

**Why the re-search is nearly free.** Tree size grows geometrically. If depth `d` costs
`c * r^d` nodes for some ratio `r > 1`, then all the shallower passes together cost

```
c(r + r^2 + ... + r^(d-1))  =  c * r^d * (1/r + 1/r^2 + ...)  ~  c * r^d / (r - 1)
```

so the overhead over just searching depth `d` once is a *constant fraction*, about
`1/(r-1)` — typically 30-60%, not a multiple.

**What you buy for that fraction:**
1. Always a finished answer. If time runs out mid-pass you return the best move from the
   last *completed* depth — you never flag with nothing.
2. Free move ordering. Depth `d`'s best move is almost always still best at depth `d+1`, so
   you try it first and alpha-beta prunes `d+1` far harder. Iterative deepening usually
   pays for its own overhead this way.

---

## 6. Transposition table

Different move orders reach the same position ("transpose"): `1.e4 e5 2.Nf3` and
`1.Nf3 e5 2.e4` are identical. Without a cache the search explores that subtree twice.

We key a dict on `board._transposition_key()` — a hashable tuple of piece placement, side
to move, castling rights and en-passant square (the same thing python-chess uses for
repetition detection). Each entry stores `(depth, value, bound, best_move)`.

**Bound kinds** — a fail-soft alpha-beta value is not always exact:
- `_EXACT` — the search completed inside the window; `value` is the true score.
- `_LOWER` — a beta cutoff happened; the true score is `>= value` (a floor).
- `_UPPER` — no move beat alpha; the true score is `<= value` (a ceiling).

On entering a node, if a stored entry was searched at least as deep as we need now
(`e_depth >= depth`):
- `_EXACT` -> return it.
- `_LOWER` and `value >= beta` -> return it (already good enough to cut).
- `_UPPER` and `value <= alpha` -> return it (already too weak to matter).

Otherwise the stored `best_move` is still used as the first move to try.

**Not cached:** mate scores (they are root-relative, section 7) and the draw scores from
repetition detection (path-dependent). The table is cleared every move for now; a
persistent fixed-size table is on the backlog.

---

## 7. Quiescence search and the horizon effect

If you stop the search at a fixed depth and evaluate, you often stop **in the middle of an
exchange** — "I'm up a queen!" one ply before the recapture. The evaluation is measured in
a position it was never designed for and is wrong. This is the *horizon effect*.

**Quiescence search** (`_qsearch`): at the horizon, instead of evaluating immediately, keep
searching **captures only** (plus all legal moves when in check, since being in check is
not "quiet") until no forcing move remains, then evaluate.

- **Stand-pat**: the side to move can usually do at least as well as the static score by
  *not* capturing, so `stand_pat = evaluate(board)` is a lower bound. If `stand_pat >= beta`
  we cut immediately; if `stand_pat > alpha` it raises alpha.
- It **terminates** because material runs out — capture chains are short.
- A `_QS_MAX_PLY` cap is a safety net against pathological check/capture loops.

---

## 8. Mate scores

A checkmate for the side to move (no legal moves, in check) is scored `-MATE + ply`, where
`ply` is the distance from the search root and `MATE = 1,000,000`.

- Subtracting `ply` makes a **faster mate score higher** (mate in 1 beats mate in 5), and
  makes the losing side **prefer the longest defence**.
- Scores past `_MATE_THRESHOLD = MATE - 1000` are recognised as forced mates: iterative
  deepening stops (a deeper search cannot beat a forced mate) and they are never stored in
  the transposition table (the `ply` offset is only correct relative to *this* root).
- `MATE` sits far above any material sum (~4000 cp), so a mate score can never be confused
  with a material score.

---

## 9. Contempt

A draw is scored `_CONTEMPT = 25` centipawns **below equal, from the root side's point of
view**, so the engine only accepts a draw when it genuinely believes it is worse by more
than a quarter of a pawn — it plays for the win.

The sign has to survive negamax's per-ply negation. `_draw_score(ply)` returns
`-_CONTEMPT` when `ply` is even (root side to move) and `+_CONTEMPT` when odd. Because a
given position always recurs at the **same ply parity** (side to move alternates with ply),
this is path-consistent: propagate a draw score up the tree and every node sees "a draw is
`_CONTEMPT` worse for the root side", which is what we want.

---

## 10. Search extensions and reductions

**Check extension.** When a node is in check, `depth += 1` before recursing — a forcing
line gets a full extra ply to resolve before evaluation, so short tactical shots are not
cut off by the horizon.

**Late-move reductions (LMR).** Moves ordered late are probably bad (that is what the
ordering is for). For a *quiet* move past `_LMR_MIN_MOVE`, at `depth >= _LMR_MIN_DEPTH`, not
in or giving check, search it first at `depth - 1 - r` (`r` = 1, or 2 for very late moves).
If that reduced search still beats alpha it "surprised" us, so re-search it at full depth.
Net effect: the search spends its nodes on the moves that matter. (Currently measures flat
because the search is eval-bound, not node-bound — it will pay off after the eval jit.)

---

## 11. Time management

**Per-move budget** (`_budget_s`):

```
moves_left = max(30, 56 - fullmove_number)
budget     = time_left_ms / moves_left + 0.5 * increment_ms
budget     = min(budget, time_left_ms / 3, time_left_ms - 500)     # cap + reserve
budget     = max(budget, 10 ms)                                     # floor
```

`time_left_ms / moves_left` is the sustainable share of the clock. Adding half the
increment reflects that every move refills the clock, so the increment is time to spend
rather than hoard. Assume at least 30 moves left so we do not drain the clock in a long
game. Never spend more than a third of the clock on one move, and always keep 500 ms —
the referee's watchdog does not forgive.

**Why it survives a long game.** If we always spend `T/m + i/2` and get `i` back per move,
then `T_next = T - (T/m + i/2) + i = T(1 - 1/m) + i/2`. The fixed point is
`T* = (i/2) * m = m*i/2`; with `m = 30`, `i = 500 ms` that is a stable ~7.5 s reserve, not
the ~2 s the earlier `m = 20`, no-cap formula converged to.

**Inferring the increment.** `get_move` is not told the increment. But after our move the
clock goes `new = old - our_spend + increment`, so

```
increment ~= new_clock - (old_clock - our_spend)
```

`agent.py` stores `old_clock` and measures `our_spend` with `time.monotonic()` around the
search, then backs the increment out, clamped to `[0, 2000] ms`, default 0 until the second
move. Our own timer runs a hair short of the referee's, which biases the estimate low — the
safe direction (smaller increment -> smaller budget).

**Clock checks inside the search.** `_tick` bumps a node counter and, once every
`_CHECK_INTERVAL` nodes, checks `time.monotonic()` against the deadline and raises
`_Timeout`, which unwinds to `search_move` and returns the last completed depth's move.

---

## 12. Evaluation — `evaluate.py`

Returns a centipawn score (1 pawn = 100) from the **side-to-move's** point of view.

### 12.1 Material

`P=100  N=320  B=330  R=500  Q=900`, king 0 (its value is handled by mate detection). Bishop
slightly above knight is a crude bishop-pair proxy. Counted with `int.bit_count()` on the
piece bitboards masked by colour — no per-square Python loop.

### 12.2 Piece-square tables

A `[64]` table per piece giving a centipawn bonus for that piece on that square (knights
toward the centre, rooks toward open files and the 7th, a midgame king in the corner, an
endgame king central). Values are the public **PeSTO** tables — a standard positional
heuristic, not a third-party engine.

Squares are indexed `a1 = 0 ... h8 = 63` (python-chess). A white piece on `sq` reads
`table[sq]`; a black piece reads `table[sq ^ 56]` — XOR by 56 flips the rank (`a1 <-> a8`),
mirroring the board so the same table serves both colours. Black's contribution is
subtracted. (The PeSTO arrays are written rank-8-first, so they are flipped once at import
by `_flip_ranks` to match `a1 = 0` — getting this wrong made the midgame king want to march
up the board.)

### 12.3 Tapered evaluation

A queen matters differently in a queenless endgame; the king wants opposite things in the
middlegame and the endgame. So every square-dependent term has a **midgame** and an
**endgame** value, and the final score interpolates between them by *how much material is
left*.

**Game phase**: `phase = 1*N + 1*B + 2*R + 4*Q` summed over both sides, clamped to `[0, 24]`.
24 = full non-pawn material (pure midgame), 0 = bare kings (pure endgame).

**Blend**:

```
score = ( mg_score * phase  +  eg_score * (24 - phase) ) / 24
```

with `int(.../24)` (truncate toward zero) rather than `//` (which rounds toward negative
infinity and would make a position and its colour-mirror differ by 1 cp).

### 12.4 Pawn structure

Per side, in centipawns:
- **Doubled** — `-12` per extra pawn on a file (two pawns on the e-file = one penalty).
- **Isolated** — `-10` per pawn on a file with no friendly pawn on either adjacent file
  (no pawn can ever defend it).
- **Passed** — a bonus by rank (`0, 5, 10, 20, 35, 60, 100, 0` from the pawn's own side) if
  no enemy pawn stands on the same or an adjacent file ahead of it — nothing can stop it
  promoting except pieces.

### 12.5 Mobility

For each knight, bishop, rook, queen: count the squares it attacks
(`board.attacks_mask(sq).bit_count()`) and multiply by a small weight (a few cp per square,
slightly higher in the endgame). A crude "active pieces are worth more" term. The
`attacks_mask` calls for the sliding pieces are the current per-node bottleneck and the
reason for the eval jit.

### 12.6 King safety (midgame only, phase-faded)

Computed on the `mg_score` side only, because an exposed king is a middlegame liability but
in the endgame the king is *supposed* to walk into the open. The phase blend then fades it
out: at `phase = 0` it contributes exactly nothing.

- **Pawn shield** — `+12` per friendly pawn on the three files in front of the king.
- **Open / half-open file** near the king — `-22` if neither side has a pawn on a file
  beside the king, `-12` if only the enemy does.
- **Attacker zone** — for each enemy piece whose attacks reach the king's 3x3 zone,
  subtract a weight by piece type (`P 2, N/B 6, R 9, Q 14`). Linear, which is crude — real
  king safety is non-linear in the number of attackers — but a reasonable placeholder.

### 12.7 The weights are placeholders

Every number above is a hand-picked or public starting value, not tuned for *our* search.
The real gain is **Texel tuning**: play a few hundred thousand self-play positions, label
each with the eventual game result (1 / 0.5 / 0), and fit the weights by logistic
regression so `sigmoid(k * eval)` best predicts the result. That is a separate data project
(overlaps the teammate's work).

---

## 13. Bit tricks used throughout

A **bitboard** is a 64-bit integer, one bit per square (bit `i` = square `i`, `a1 = 0`).

- `bb & occupied_co[WHITE]` — restrict a piece bitboard to one colour.
- `bb.bit_count()` — how many pieces (population count, a CPU instruction in Python 3.10+).
- `bb & -bb` — isolates the lowest set bit. `(bb & -bb).bit_length() - 1` is that square's
  index. `bb &= bb - 1` clears it. Together they iterate the set squares.
- `sq ^ 56` — flip the rank (vertical mirror), used to read a white-oriented table for a
  black piece.
- `0x0101010101010101 << f` — the bitboard of file `f` (a1, a2, ... a8 then shifted).

python-chess exposes `board.pawns`, `board.knights`, ... (all pieces of that type, both
colours), `board.occupied_co[colour]`, and `chess.scan_forward(bb)` to iterate set bits.
