# Phase 7 (NNUE) handoff — status as of 2026-09-08

For: Aled (author of `docs/phase7-nnue.md`, the original roadmap this follows)
Branch: `phase7-real-data` (pushed to `origin`, up to date, working tree clean)
Latest commits: `ffe9946` "updates", `4520702` "carlsen", `53f5a3e` "phase7: retrain on 34,617-position dataset"

This is a status update against the roadmap in `docs/phase7-nnue.md`, not a replacement for
it — read that first if you haven't. Everything below assumes it.

---

## TL;DR

- All three "critical path" items from the roadmap are done: search rebased onto current
  `main` (diag-8), data pipeline scaled and fixed, retrain done properly (BCE + LR schedule +
  early stopping). Speed work (incremental accumulator) is also done — that was listed as a
  *later* priority in the roadmap but got pulled forward.
- Dataset grew from 3,875 → **44,529** positions (Lichess bulk archive + self-play + Magnus
  Carlsen's full 2001–2022 tournament career, 4,314 games). Still far short of the 5–10M
  target — see "What's still the real gap" below.
- Two arena results exist against `versions/phase5`, on two different checkpoints (see
  "Results" — read the caveat, they don't agree as cleanly as I'd like).
- **The vs-`main` comparison the roadmap's decision point calls for has not been rerun on the
  current checkpoint.** The only vs-`main` data point on file is stale (pre-dates the labelling
  bug fix and the full retrain). This needs to happen before anyone decides whether to ship
  NNUE over classical.

---

## What's done, mapped to the roadmap's table

| Thing | Roadmap's "Now" | Current state |
|---|---|---|
| Training positions | 3,875 | **44,529** (still need 5–50M for the stated params:data target) |
| Params : data ratio | ~53 params/sample | **~4.6 params/sample** (205,377 params ÷ 44,529 rows) — target is ≤0.05 (20+ samples/param), so still ~92× more data needed to hit that bar |
| Leaf inference | one `session.run` per node | **incremental accumulator** (numpy push/pop deltas, no ONNX per-leaf calls at all — see below) |
| Search around the net | forked from `diag-2` | **current `main` (diag-8)**: persistent TT, PVS, aspiration windows, Syzygy 3-man, mop-up/king-safety terms |
| Features | 768 absolute + STM bit | **unchanged** — still absolute, not king-relative. This is the biggest strength lever still untouched (see below) |
| Node rate with the net | ~hundreds/sec (unmeasured) | **8,430 nps** vs classical's **9,724 nps** on the same 3 fixed bench positions (`nnue_agent/bench.py`, mirrors `tools/bench.py`) — ~13% slower, comparable depth reached |

---

## Data pipeline (all scripts in `versions/phase7/`)

The pipeline is: **source → raw `(game_id, fen, result)` → dedupe → label → train → export**.
Three independent sources feed the same format so they compose:

1. **Lichess bulk archive** (`pgn_to_positions.py`): one small 2013 monthly archive from
   `database.lichess.org` (2,038,084 raw positions from that file alone), quiet-filtered and
   subsampled down to 30,000 (`filter_and_subsample.py`, reuses `generate_positions._is_quiet`:
   drops any position in check or with a capture/promotion available).
   - The unauthenticated/authenticated Lichess *user* API (`/api/games/user/DrNykterstein`,
     i.e. Magnus's Lichess account) never worked — persistent 429s regardless of auth, almost
     certainly deliberate rate-limiting on that specific high-profile account. Abandoned in
     favour of the bulk database dumps, which have no such issue.
2. **Self-play**: `generate_positions.py`, `diag-8` vs itself with randomised openings, run in
   5 parallel seeded batches. Only ~5,900 raw positions — quiet-filtering yields very few
   rows per game (~1.36/game) since self-play games tend to be quieter/shorter than human
   games, so this alone can't scale fast. It's a minority slice of the current dataset by
   design, not a bug.
3. **Magnus Carlsen's career, 2001–2022** (`csv_to_positions.py`, new this session): a Kaggle
   CSV export (`Carlsen_game_info.csv` + `Carlsen_moves.csv`, one row per game / one row per
   ply respectively — **not** PGN). The CSV's own `fen` column is board-placement only (no
   side-to-move/castling/en-passant/clocks), so it's unusable directly; the script instead
   replays each game from the start position via `board.push_san()` on the CSV's `notation`
   column, which gives an exact full FEN at every ply. All 4,314 games replayed cleanly (zero
   SAN failures). Quiet-filtered down to 11,630 positions.
   - **These raw CSVs are NOT in the repo** (176 MB — same "too large, reconstructable"
     treatment as the other raw intermediates). They're on this machine at
     `Downloads/Carlsen_moves.csv` and `Downloads/Carlsen_game_info.csv`.  If you need to
     rerun this step on your own machine, you'll need your own copy of that Kaggle dataset:
     https://www.kaggle.com/datasets/zq1200/magnus-carlsen-complete-chess-games-20012022

All three raw sources get concatenated, deduped (`dedupe.py` — transposition-key + horizontal
mirror, so the same position reached by a different move order or as a mirror image only
counts once), then labelled (`label.py`).

**Labelling**: every target is `λ·outcome + (1−λ)·sigmoid(shallow_eval/400)`, λ=0.3 (i.e.
leaning 0.7 toward the shallow eval, per the roadmap's spec). The shallow eval is the actual
`main/search.py` engine (not a placeholder), run under a **0.3s wall-clock budget** via its own
iterative-deepening loop — not a fixed depth, since a fixed-depth call on this engine can run
away in sharp positions. This is genuinely the safe "engine labels, doesn't get imitated"
recipe the roadmap and `docs/PLAN.md` call for, not deep-engine centipawns.

Labelling ~44.5k positions single-threaded would take ~3.75 hours at 0.3s/position. It's
parallelized 16-way across OS processes (one per CPU core) — this is the dev pipeline, not the
shipped agent, so the "one core" rule doesn't apply here. Full run takes ~20-25 minutes this
way. **Caveat learned the hard way this session: a backgrounded shell command tied to a Claude
Code session can get torn down if that session restarts mid-run, silently losing partial
progress** — if you rerun this, check each chunk's log actually ends with "wrote ..." before
trusting its output file's row count.

**Bug fixed this session, worth knowing about if you touch `label.py` again**: the shallow-eval
function originally read `board.turn` *after* running the search to decide whose perspective
the score was in. `search.py`'s `_Timeout` unwinds through `_negamax`'s recursion without
popping the board back (documented behavior — `search_move()` itself works around this the same
way), so post-search `board.turn` can reflect some mid-tree position, not the original FEN. This
silently flipped the sign of the label for some positions. Confirmed via direct trace (a
rook-up endgame scored +644 internally, came out as −644). Fixed by capturing
`root_is_white = board.turn == chess.WHITE` before the search loop starts. If this hadn't been
caught, it would have corrupted training data at scale without any obvious symptom.

---

## Training

`train.py`, BCE loss (not MSE — target is a probability), `OneCycleLR` schedule, early stopping
on validation loss (patience 20), batch size 2048 (clamped down for small datasets), up to 300
epochs. Current run log: `versions/phase7/runs_v2/model.pt/run_1788890751.json` — best val_loss
0.581, early-stopped at epoch 28.

Every run writes a JSON log with git commit, dataset hash-equivalent (row count), hyperparams,
and epochs run — the provenance trail `docs/PLAN.md` asks for.

Weights are exported two ways: `export_onnx.py` (ONNX, used by the old per-leaf
`nnue_eval.py` path) and `export_weights.py` (raw numpy arrays into `.npz`, used by the
incremental accumulator below). `nnue_agent/weights.npz` is the one actually wired into the
test agent right now.

---

## Speed: incremental accumulator (pulled forward from the roadmap's "later" section)

`versions/phase7/nnue_agent/nnue_incremental.py` replaces the old "call ONNX once per leaf"
approach entirely. The feature-transformer's hidden state for a position differs from its
parent's by only the moved piece (plus captured piece, plus rook shift on castling, plus the
en-passant pawn) — so `Accumulator.push(board, move)` computes just those delta feature
columns and adds/subtracts them from a running stack, and `pop()` just decrements a stack
pointer. `set_root()` does one full recompute when a game starts.

`nnue_agent/search.py` is `main/search.py` verbatim with `_acc.push()`/`_acc.pop()` wired into
the three real push/pop sites (`_search_root`, `_negamax`, `_qsearch`) and
`_acc.evaluate_cp(board)` replacing the old `evaluate(board)` calls.

Correctness: `nnue_agent/test_incremental.py` compares the incremental path against a
from-scratch recompute at every ply across 100 random games plus 8 hand-built edge cases
(castling ×4, en passant ×2, promotion ×2). All pass, including after the retrain (weights
changing doesn't change correctness, but it was re-checked anyway).

Result: **8,430 nps vs 9,724 nps classical (~13% slower)**, comparable search depth — this
closes what the roadmap flagged as the single scariest number ("~hundreds/sec, unmeasured").
Batched inference and int8 quantization (the roadmap's other two speed levers) were
deliberately **not** pursued — speed is no longer the bottleneck at this ratio, data and
features are.

---

## Results

Two checkpoints have been arena-tested against `versions/phase5` (this repo's name for what
the roadmap calls `phase5jit`):

| Dataset | Positions | Games | Score | Elo | 95% CI |
|---|---|---|---|---|---|
| Lichess + self-play only | 34,617 | 16 | 59.4% | +66 | −78 to +239 |
| Lichess + self-play only | 34,617 | 64 | 65.6% | +112 | **+41 to +194** |
| + Carlsen career (current) | 44,529 | 64 | 60.2% | +72 | −4 to +155 |

**Read this honestly, not optimistically**: adding the Carlsen data moved the point estimate
down (+112 → +72) and the interval now just barely touches zero, so it's no longer an
unambiguous win at 95% confidence the way the pre-Carlsen 64-game result was. The two
intervals overlap heavily though (both centered somewhere around +70–110 Elo), so this is
plausibly sample noise from a 64-game batch rather than the Carlsen data actually hurting —
elite 2001–2022 OTB games are a genuinely different distribution (different time controls,
very early games from when Carlsen was ~2000-rated) than Lichess blitz/rapid, so some dilution
wouldn't be shocking either. **This is not resolved. A larger batch (128+ games) on the
current checkpoint would settle it** — I didn't run one yet given the Thursday deadline
pressure, and made the call to keep the Carlsen-augmented checkpoint as current since it's
still positive on point estimate and adds real distributional diversity. Overridable — the
data and both checkpoints (`runs_bigdata/model.pt` pre-Carlsen, `runs_v2/model.pt` current)
are both still in the repo if you want to compare them yourself or revert.

**Not yet done, and it's the roadmap's actual decision point**: `nnue_agent` vs classical
`main`, 100+ games. The only number on file for that comparison (main +2=2−12 vs an early NNUE
build, Elo −255, 16 games) predates both the labelling sign-flip fix and every retrain since —
**treat it as meaningless for deciding anything now.** This needs a fresh run before anyone
decides whether NNUE is worth shipping over the classical engine.

---

## What's still the real gap (in priority order, if you're picking this up)

1. **Data volume — this is the one to scale up next, concretely.** 44,529 positions is still
   tiny for a net this size — the params:data ratio is ~92× off the roadmap's own target. The
   pipeline is fully built and tested end to end, so this is "run the same scripts on
   more/bigger inputs," not new engineering.

   **Important if you're doing this on a different machine**: the raw 2,038,084-position
   Lichess file (`versions/phase7/raw_positions_lichess.tsv` and the `.pgn` it came from) is
   **not in the repo** — gitignored as a large reconstructable intermediate, same as the
   Carlsen CSVs. It only exists on the machine that generated it. To regenerate it:
   ```
   curl -o lichess_2013-01.pgn.zst https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst
   # decompress with zstandard (added as a dev dep this session: uv add zstandard)
   python -c "import zstandard, pathlib; zstandard.ZstdDecompressor().copy_stream(open('lichess_2013-01.pgn.zst','rb'), open('lichess_2013-01.pgn','wb'))"
   uv run python pgn_to_positions.py lichess_2013-01.pgn --out raw_positions_lichess.tsv
   ```
   **Quality caveat worth knowing**: 2013 is very early Lichess — `WhiteElo`/`BlackElo` in that
   archive skew low (e.g. 1639/1403 in the first game) and the player pool isn't representative
   of current play. A more recent archive (2020+) would carry more reliable rating fields and
   let you actually rating-filter toward stronger games, not just quiet-filter. Worth pulling a
   newer archive instead of / in addition to the 2013 one for the scale-up, not just resampling
   the same file harder.

   Once you have a raw file (existing or freshly pulled), scaling to a specific target is just:
   `filter_and_subsample.py <raw>.tsv <out>.tsv --target <N>`, then the usual
   `cat` → `dedupe.py` → `label.py` (parallelize across cores same as this session did — see
   "Data pipeline" above for the caveat about backgrounded jobs surviving session restarts) →
   `train.py`.

   **Sizing/cost guide**: going to **~1M positions** would take the params:data ratio from the
   current ~4.6 params/sample down to ~0.2 (still ~4× short of the roadmap's ≤0.05 target, but
   a big practical step, not just cosmetic). Labelling cost scales linearly with position count
   at ~0.3s each — ~300k is a 2–2.5 hour labelling run across 16 cores, ~1M is 5+ hours. If
   time is tight, lowering `label.py --budget-s` (e.g. 0.3 → 0.15) roughly halves labelling
   time at the cost of a noisier (but still real, not fake) shallow-eval signal.
2. **King-relative features (HalfKP/HalfKA).** Still absolute 768 one-hot, per the roadmap's
   own "this is what NNUE actually is" note. Bigger ceiling than more data alone, but it's a
   real architecture change — new feature encoding, new incremental-accumulator delta logic,
   more implementation risk this close to the deadline. Not started.
3. **The vs-`main` arena run** described above — needed regardless of what else happens next,
   since it's the actual bar the roadmap sets for shipping NNUE at all.
4. Texel-tuning the classical eval (the roadmap's parallel hedge) — not started. Still a live
   option if NNUE data volume can't scale far enough by Thursday; reuses the same labelled
   data this pipeline already produces.

---

## Files added/changed this session

New: `versions/phase7/csv_to_positions.py`, `filter_and_subsample.py`, `export_weights.py`,
`nnue_agent/nnue_incremental.py`, `nnue_agent/test_incremental.py`, `nnue_agent/bench.py`.
Fixed: `label.py` (sign-flip bug, real engine shallow-eval, quiet-only sampling),
`train.py` (BCE loss, LR schedule, early stopping), `generate_positions.py` (crash fix,
cross-game state reset). Full detail in `git log` on this branch — commit messages are
descriptive.

Deleted (superseded by the incremental path): the old per-leaf ONNX `evaluate.py` and
`model.onnx`/`model.onnx.data` inside `nnue_agent/`.

Committed dataset/model artifacts: `versions/phase7/labelled_combined.tsv` (34,617 rows, prior
checkpoint's data), `labelled_combined_v2.tsv` (44,529 rows, current), `runs_bigdata/model.pt`,
`runs_v2/model.pt/model.pt`, `nnue_agent/weights.npz` (current). All raw/intermediate files
upstream of dedupe are gitignored — reconstructable by rerunning the pipeline scripts, not
worth the repo bloat.
