# Our plan

Working doc for the two of us. `docs/IDEAS.md` is upstream's and we don't edit it;
this file is ours. Canonical rules live at
[aichessathon.com/docs](https://aichessathon.com/docs) and change — read them before any upload.

## The deliverable

`agent.py` at the root of a zip, exposing `get_move(fen, time_left_ms) -> str` (UCI). The
platform does `import agent`, runs it once per game, keeps the process alive between our
moves and **suspends it while the opponent thinks**. Everything at repo root that matches
`*.py` gets zipped (`harness/package.py`), so splitting into modules is fine.

Key numbers (verify against the site before relying on them):

- 120 s + 0.5 s/move per side, wall clock, one core of an EPYC 9V74, 2 GB RAM.
- 90 s import budget before the clock starts. Load weights and warm numba here.
- Flag = loss (draw only if the other side cannot mate). Illegal / crash / OOM = loss.
  Move reply > 4 KB = illegal. 600 plies = draw.
- Read-only filesystem except 256 MB at `/tmp`, wiped per game. No network, no GPU.
- Preinstalled stack only: `torch` 2.13 CPU, `numpy` 2.5, `python-chess` 1.11,
  `onnxruntime` 1.29, `numba` 0.67. A `requirements.txt` in the zip is ignored.
- 10 uploads per team per day; the latest one that passed validation plays.

## Rules on AI (from aichessathon.com/docs/rules.md)

Allowed: using AI tools to write the code (as long as the result follows the rules and we
can explain it at the finals); a classical engine we wrote; training our own network on
data labelled by any engine; opening books and Syzygy tablebases as shipped data in the
50 MB cap.

Banned: third-party engines (Stockfish, Lc0, Maia, any wrapper/port/translation); starting
from a published network (fine-tuning or re-exporting one counts as shipping it); a database
of another engine's moves or evals shipped for runtime lookup; obfuscation (instant DQ).

Verification is retroactive and finalists walk a panel through how the agent was built. If
we ship a network we must show how it was trained — so we keep every training script,
dataset manifest, and run log.

## Priority order

We picked, in order: **(3) don't lose games to ourselves → (1) raw speed → (2) the
pruning stack past the basics**, then evaluation. Phase 1 is unavoidable groundwork; the
priorities start at Phase 2.

## Phases

Each phase is a branch and a PR. After it merges, freeze the working tree to
`versions/phaseN/` and commit it — that frozen copy is the arena opponent for the next
phase. "Better than our last version" over a few hundred fast games is the only signal
that counts; two games are noise.

### Phase 1 — minimal working engine  *(in progress)*

Negamax + alpha-beta, iterative deepening, MVV-LVA capture ordering, material + one pawn
PST, a per-move time budget from `time_left_ms`, and a fallback in `get_move` that can
never raise. Split into `agent.py` (entrypoint + fallback), `search.py`, `evaluate.py`.

- **Done when:** beats `baselines/greedy` over 100 games; `make gate` green.

### Phase 2 — robustness and time management  *(priority #3)*

- Mate scores stored **relative to the search root ply** (a persistent TT otherwise
  misplays won positions — "mate in 3" becomes "mate in 1" two moves later).
- Time budget adapts to the position and the increment; hard clock checks inside the
  search return the best move from the last finished depth; keep a watchdog margin.
- Edge-case sweep: no legal moves, stalemate, checkmate, promotion, en passant.
- Repetition awareness: keep the positions we have been asked about, so we can claim a
  draw when lost or avoid one when winning (the referee auto-claims threefold).
- Deterministic tie-break plus a global seed — we must be able to reproduce a lost game
  and explain any move at the finals. A `DEBUG` flag gates every `print`.
- A move-1 nodes/sec assertion in the tests, to catch numba compiling on the clock later.
- **Done when:** 300+ games vs `random` and vs `greedy` with zero crash / flag / illegal.

### Phase 3 — speed  *(priority #1)*

- Profile first. The cost is `board.legal_moves` (full legality in Python) and eval.
- numba-jit the evaluation; warm it at import with the exact dtype it will see.
- The big one: a **jitted bitboard move generator**, so the search stops paying
  python-chess per node. This is the hardest phase and the one that separates the field.
- Incremental material/PST updates on push/pop instead of rescanning 64 squares.
- **Done when:** nodes/sec up several-fold; arena vs Phase 2 shows the extra ply paying off.

### Phase 4 — the pruning stack  *(priority #2)*

Added and arena-measured one at a time, keeping only what pays, recording each one's gain:

- Transposition table — fixed-size, numpy-backed, replace-by-depth, kept across moves in
  a game. Not an unbounded dict (millions of entries churn GC and eat the 2 GB).
- Quiescence search (captures only) at the leaves, with delta pruning.
- Principal variation search.
- Killer-move and history heuristics for quiet-move ordering.
- Null-move pruning.
- Late-move reductions.
- Aspiration windows around the previous iteration's score.
- Check extensions.
- **Done when:** arena vs Phase 3 is a decisive gain.

### Phase 5 — evaluation

Tapered midgame/endgame PST for every piece, pawn structure, king safety, mobility.
**Texel-tune** the weights against self-play results instead of eyeballing them.

### Phase 6 — endgame tablebases

Ship Syzygy 3–4-piece WDL (fits the cap); probe when few pieces remain to convert won
endings cleanly and hold drawn ones.

### Phase 7 — NNUE evaluation  *(optional, separate offline track — start early, it's the long pole)*

Offline: collect positions, label with Stockfish, train a small quantised net **from random
init**, export ONNX, run via onnxruntime, batch a search pass's leaf evals into one call.
Ships only if it beats the Phase 5 hand-crafted eval in the arena. Keep full training
provenance for the panel.

## Competitive read

Every entrant got the same starter, and `docs/IDEAS.md` dictates the standard build, so the
median entry looks the same: `board.legal_moves` in the loop, a dict TT on
`_transposition_key()`, recursion, a hand-tuned PST, maybe a jitted eval. Where we pull ahead:

| Crowd does | Why it's weak | Our move |
|---|---|---|
| python-chess board as the search node | it's a correctness library, not a speed one | jitted bitboard movegen (Phase 3) |
| unbounded dict TT | GC churn, eats the 2 GB | fixed numpy table, replace-by-depth (Phase 4) |
| `random.choice` tie-break (from the baselines) | non-deterministic → can't reproduce or explain a game | seed + deterministic tie-break (Phase 2) |
| ordering stops at MVV-LVA | alpha-beta underperforms on quiet moves | killers + history + SEE (Phase 4) |
| re-scan `board.pieces()` every node | O(64) + dict alloc per node | incremental eval (Phase 3) |
| test with 2–20 games | pure noise | few-hundred-game arena vs the previous version |
| numba warm-up with the wrong dtype | recompiles on move 1 → flag | warm with the exact signature; assert move-1 nodes/sec |

What does **not** help: a big opening book (rated games start from unpublished curated
positions), or hunting for an exotic search paradigm — negamax + alpha-beta is the game.

## Leakage checklist

### Training data (Phase 7)

- **Opening distribution shift.** Rated games start from curated positions we never see.
  Generate training positions from randomized / curated-style openings and perturbations,
  not only from the standard start.
- **Transposition contamination.** The same position reaches you via many move orders; a
  random split drops near-identical positions on both sides and inflates the val metric.
  De-duplicate by `_transposition_key` **and its horizontal mirror** before splitting.
- **Target leakage → mimicry.** Training on deep-engine centipawns and then scoring the net
  by how closely it matches that engine builds an impersonator — the move-match signature
  that triggers retroactive DQ. Train on a blend of game outcome (WDL) and shallow eval.
- **Chasing our own tail.** Iterating "train on games from version N, test against version
  N" overfits to our current weaknesses. Keep one frozen external yardstick all project.
- **Tablebase distortion.** If we ship Syzygy and train on TB-perfect endgame labels, don't
  let them dominate the loss and warp the middlegame.

### Runtime (matters now)

- **Clock leakage.** The process is suspended during the opponent's move, so any wall-clock
  timer carried across turns over-counts. Budget only from the `time_left_ms` argument.
- **Mate scores in a persistent TT** — see Phase 2. Store relative to root ply.
- **Draw-state blindness.** The search must see real game history or it will "win" material
  straight into a threefold draw. Keep the move stack; check repetition near the root.
- **Debug on the clock.** `print` is safe but not free; 8 KB/move is CPU and I/O we pay for.
  Gate it behind `DEBUG`.

## Performance rules for the hot path

For 64-element board work called millions of times, the ranking is
**numba-jitted scalar loop > Python `for` loop > numpy**. numpy has ~1–5 µs of dispatch +
allocation per call; at n=64 you never amortize it. `@njit` compiles the loop to machine
code over a preallocated fixed-dtype array — nanoseconds.

1. Don't vectorize small fixed-size work with numpy — jit a scalar loop instead.
2. numpy / onnx calls only at batch boundaries (≥ ~100 items), e.g. NNUE leaf batches.
3. No allocation in the node loop: no `list(...)`, no `board.copy()`; reuse buffers; push/pop.
4. Hoist attribute lookups (`push = board.push`) before the loop.
5. Integer eval only; `int16` / `int32` if jitted.
6. `board.piece_map()` / `board.pieces()` allocate — track material incrementally.
7. `board.is_check()` / `is_checkmate()` generate moves internally — derive from our movegen.
8. `move.uci()` once at the very end, never inside search.
9. Explicit move stack instead of recursion only if a profile says frame setup is hot.

## Working agreement

- **Repo:** our private team repo is `origin`; the starter stays as `upstream` for pulling
  harness fixes (`git fetch upstream && git merge upstream/main`).
- **Branches:** one per phase/feature. Push, open a PR, wait for CI green, the other reviews,
  merge. CI runs `make gate` (ruff + mypy strict + two must-finish games) on Linux/mac/Windows.
- **Never edit `harness/`** — it mirrors the platform; changing it makes local results lies.
- **Testing:** commit frozen `versions/phaseN/` copies. Agree a fixed arena config
  (opponent, games, time control) so scores are comparable between our machines.
- **Uploads:** 10/team/day. Agree who uploads and when, or we burn the quota.
- **Division of labour:** Phases 1–4 (engine) are one track; the Phase 7 offline data +
  training pipeline is independent and starts now. Natural split — one drives the engine,
  the other stands up data/training; converge at Phase 5–6.
- **Style:** Python 3.12, type-annotated, ruff + mypy strict clean. Keep `agent.py`
  readable — it's what a judge reads if our games get flagged.
