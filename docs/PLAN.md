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

## Status (2026-09-08, build `diag-8`)

Phases 1-4 and 6 are shipped. The engine is a jitted-eval negamax with the full
classical pruning stack and 3-man Syzygy:

- **Search:** negamax + fail-soft alpha-beta, iterative deepening, **principal variation
  search**, **aspiration windows** (±40 cp around the last score), quiescence with
  captures + promotions + one ply of quiet checks, MVV-LVA + killers + history ordering,
  check extension, late-move reductions, contempt (25 cp), mate-distance scoring.
- **Transposition table:** persistent, fixed-size (2 x 4.2 M `uint64`, ~64 MB), packed
  one entry per word, replace-by-depth with a per-search generation stamp. Kept across
  moves within a game. This is the single biggest gain so far (~2.4x nps, +1 ply).
- **Evaluation:** numba-jitted tapered PST + material + pawn structure + mobility +
  non-linear (quadratic) king safety + KX-vs-K mop-up driver + endgame king activity.
- **Endgame:** 3-man Syzygy WDL+DTZ (`syzygy/`, ~26 KB). At <=5 men and the files
  present, `search_move` returns the tablebase move (WDL then DTZ) and skips the search.
  Package with `--include syzygy` or it does not ship.
- **Reached on the match machine:** d5-d7 middlegame, ~20-28k nps.

Two open fronts:

1. **Depth in sharp middlegames.** Rated losses (R56, R59, R63, R67) share a shape:
   equal opening -> one tactical/positional sequence the d5-d6 search does not see ->
   lost -> mated. The jitted move generator is the headline fix; the section below is
   the alternative path.
2. **NNUE** (`docs/phase7-nnue.md`) -- teammate's offline track, not competitive yet.

Not a current problem: endgame conversion at <=3 men (Syzygy fixed it); the clock
(`diag-8` finishes rated games with 60 s+).

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

### Phase 1 — minimal working engine  *(done)*

Negamax + alpha-beta, iterative deepening, MVV-LVA capture ordering, material + one pawn
PST, a per-move time budget from `time_left_ms`, and a fallback in `get_move` that can
never raise. Split into `agent.py` (entrypoint + fallback), `search.py`, `evaluate.py`.

### Phase 2 — robustness and time management  *(done)*

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

### Phase 3 — speed  *(eval done; movegen open)*

- **Done:** numba-jitted evaluation, warmed at import (~3x nps, +1 ply). Golden +
  equivalence tests pin the jitted path against a pure-Python reference.
- **Open:** the **jitted bitboard move generator**. `board.legal_moves` + `board.push` /
  `pop` + `board.is_check` are the per-node cost now that eval is jitted. This is the
  hardest single item and the one that separates the field. Deferred while the pruning
  stack and Syzygy were landed; see "Depth without the jitted movegen" below for the
  cheaper alternatives to try first.

### Phase 4 — the pruning stack  *(done)*

All shipped and arena/ladder-checked one at a time:

- Persistent fixed-size TT (packed `uint64` arrays, replace-by-depth + generation, kept
  across moves). The big win: ~2.4x nps, +1 ply.
- Quiescence: captures + promotions + one ply of quiet checks. (Delta pruning: not yet.)
- Principal variation search + aspiration windows.
- Killer-move and history heuristics.
- Late-move reductions.
- Check extensions.
- **Null-move pruning: tried at d3-d4, lost 83 Elo, reverted** — needs depth to pay.
  Worth re-trying now that the engine reaches d5-d7 (see below).
- SEE: not yet — see below.

### Phase 5 — evaluation  *(partly done; Texel tuning open)*

Shipped: tapered PST + material + pawn structure + mobility + **non-linear (quadratic)**
king safety + KX-vs-K mop-up driver + endgame king activity. Every weight is still a
hand-picked / public starting value. **Texel-tune** them against self-play WDL labels
(the pipeline the teammate built for NNUE is 90% reusable). A `king-safety-v2` branch
(flight squares + off-back-rank exposure) is parked -- it did not fix its target
(Greek-gift losses are a horizon problem, not a static-eval one).

### Phase 6 — endgame tablebases  *(done, 3-man)*

3-man Syzygy WDL+DTZ ships in `syzygy/`. `search_move` returns the tablebase move (WDL
for the outcome, DTZ for the fastest conversion) at <=5 men and skips the search. A
first attempt probed WDL inside the search and just orbited the lone king without mating
-- reverted; root-only DTZ mates KRvK in 27 plies. 4-man (~21 MB, still under the cap)
is a drop-in -- add the files, no code change -- but held: KRvKR is usually drawn,
KPvKP is niche, and 21 MB of binary bloats the repo. Add it if rated 4-man endings leak.

### Phase 7 — NNUE evaluation  *(offline track, not competitive yet)*

Pipeline works end to end (`versions/phase7/`); the net is a proof of concept — trained
on 3,875 positions, per-node ONNX inference, search forked from an old build. The full
gap analysis and the plan to close it are in **`docs/phase7-nnue.md`**. Ships only if it
clearly beats the classical `main` in a 100+ game arena.

## Depth without the jitted movegen

The movegen rewrite is the biggest single depth lever but it is also the riskiest,
multi-day item. These add roughly the same effective depth (~1.5-2.5 ply combined) for a
fraction of the risk, and each is independent. Ship one per branch as its own `diag-N`
so the rated ladder can attribute it — the mirror arena (`versions/phase5jit`) cannot
measure changes this size.

Ranked by value / effort:

1. **SEE (static exchange evaluation).** ~40-60 lines of pure Python off
   `board.attackers_mask`. Two uses: (a) in quiescence, skip captures with `SEE < 0` --
   qsearch is a large fraction of all nodes and it currently searches every capture
   including `QxP`-defended; (b) order captures by SEE instead of MVV-LVA. Also lets us
   reduce / skip bad captures in the main search. The single biggest non-movegen lever.

2. **Null-move pruning, retry.** Not in check, opponent gets a free move, search
   `depth - 1 - R` (R ~= 2-3); if it still fails high, prune the node. Reverted at d3-d4
   (-83 Elo, "needs depth to pay"); the engine is at d5-d7 now. Guard with the
   `_has_non_pawn_material` helper (already in the file) against zugzwang, and never in
   the PV or in check. Standard +50-80 Elo when it works.

3. **Reverse futility / static null-move pruning.** At shallow depth, if
   `static_eval - margin(depth) >= beta`, return `static_eval` without searching. Very
   cheap, ~+20-30 Elo.

4. **Futility pruning.** At frontier nodes (depth 1-2), skip quiet moves when
   `static_eval + margin < alpha` -- they cannot raise alpha. Conservative margins keep
   it safe. ~+20-40 Elo.

5. **Late-move pruning (move-count).** Near the leaves (depth <= 3-4), once the first
   `3 + depth*depth` quiet moves have not improved alpha, skip the rest outright (not
   just reduce). Not in check, not the PV. ~+20-40 Elo.

6. **LMR formula.** Replace the `r = 1 or 2` step with `r ~= 0.8 + ln(depth)*ln(move)/2.3`,
   less on PV / killers / when improving, more for very late moves and bad captures.
   ~+20-50 Elo, medium effort.

7. **Adaptive time.** `_budget_s` is flat per move. Spend more when the best move changed
   between iterations or the position is sharp (many captures / checks / a big eval
   swing), less on forced recaptures and quiet positions -- depth where it decides games.
   ~+20-40 Elo.

8. **TT polish.** Store the static eval in the entry (skip re-evaluating on revisit);
   bucketed slots (2-4 entries per index, depth-preferred + always-replace) to cut
   collisions; internal iterative reduction (`depth -= 1` at a high-depth node with no TT
   move). ~+15-30 Elo combined, low effort.

**Risk note.** 2-5 are forward pruning -- they can drop a move that was a winning quiet
sacrifice. Mitigations are standard (never in check, never the PV, conservative
depth-scaled margins, NMP verification in the endgame). Net positive at d5+, but each
needs a rated-ladder read, not just the gate.

## Competitive read

Every entrant got the same starter, and `docs/IDEAS.md` dictates the standard build, so the
median entry looks the same: `board.legal_moves` in the loop, a dict TT on
`_transposition_key()`, recursion, a hand-tuned PST, maybe a jitted eval. Where we pull ahead:

| Crowd does | Why it's weak | Our move | Status |
|---|---|---|---|
| python-chess board as the search node | it's a correctness library, not a speed one | jitted bitboard movegen | **open** — the last big lever |
| unbounded dict TT | GC churn, eats the 2 GB | fixed `uint64`-array table, replace-by-depth + generation | done (`diag-8`) |
| `random.choice` tie-break | non-deterministic → can't reproduce or explain a game | seed + deterministic tie-break | done |
| ordering stops at MVV-LVA | alpha-beta underperforms on quiet moves | killers + history (done); SEE | SEE open |
| re-scan `board.pieces()` every node | O(64) + dict alloc per node | jitted bitboard eval reading colour-masked piece bitboards | done |
| test with 2–20 games | pure noise | rated ladder as the real signal; 16-game arena only as a "did it crater" check | done |
| numba warm-up with the wrong dtype | recompiles on move 1 → flag | warm with the exact signature at import | done |

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
