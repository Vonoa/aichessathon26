# Phase 7 (NNUE): where it stands and what it needs

Notes for the NNUE track. The pipeline in `versions/phase7/` works end to end
(generate → label → dedupe → train → export → run), which is the hard infrastructure
part and it's done. What's there now is a **proof of concept, not a competitive agent**.
This is the gap and how to close it, roughly in priority order.

The classical engine (`main`, currently `diag-8`: persistent TT + PVS/aspiration +
Syzygy 3-man) is beating `versions/phase5jit`. The bar for shipping NNUE instead is
"clearly stronger than that," not "beats the greedy baseline."

---

## Where it stands

| Thing | Now | Needs to be |
|---|---|---|
| Training positions | 3,875 | 5–50 million |
| Params : data ratio | ~53 params per sample | at least 20 samples per param |
| Leaf inference | one `session.run` per node | batched, and/or incremental |
| Search around the net | forked from `diag-2` | current `main` |
| Features | 768 absolute + STM bit | dual-perspective king-relative |
| Node rate with the net | ~hundreds/sec (unmeasured) | within 2–3× of classical (~22k) |

Beating `baselines/greedy` 100% only means "better than a depth-1 material grabber."
It says nothing about strength at 120 s + 0.5 s on one core.

---

## Critical path — makes it a testable agent

Do these three first. Nothing else matters until the net can be measured fairly.

### 1. Rebase the search onto `main` (hours)

`versions/phase7/nnue_agent/search.py` is a `diag-2` fork. It's missing the persistent
TT, PVS, aspiration windows, quiescence checks, LMR tuning, Syzygy, and the mop-up /
king-activity eval terms — weeks of work.

Take current `main/search.py` **verbatim** and change one line: `from evaluate import
evaluate` → import the NNUE `evaluate`. Same for `agent.py`. The NNUE contract is
already identical (`chess.Board -> int centipawns, side-to-move relative`), so this is
a drop-in swap.

### 2. Scale the data to 5–10 M positions (a day)

3,875 samples against a 205k-parameter net is pure overfitting. `run_*.json` shows
train_loss (0.004) far below val_loss (0.013) — the tell.

- **Source:** self-play. Play `diag-8` against itself and lightly randomised copies
  (add 1–2 random plies at the start, or pick from a large opening set — the eight in
  `harness/rules.py` are a sample, not the set). Also fold in positions from the team's
  rated games in `GameHistory/` and their continuations — that's the real distribution.
- **Sample ~10–16 positions per game**, and skip: the first ~8 plies, any position in
  check, and any position with a capture or promotion available. The net evaluates
  *quiet* leaves, so train it on quiet positions.
- **Label** each position with a blend:
  `t = λ · sigmoid(shallow_eval / 400) + (1 − λ) · game_result`
  where `game_result ∈ {0, 0.5, 1}` from White's point of view and `shallow_eval` is
  `diag-8` at **depth 6–8** (fast, noisy — that's fine). Start `λ ≈ 0.7`. Stockfish
  ramps λ toward 0 (pure WDL) across training; fixed is fine for a first pass.
- **Do not** label with deep-engine centipawns or with move-match targets. Training the
  net to imitate a strong engine is the mimicry pattern that triggers retroactive DQ
  (`docs/PLAN.md`). Shallow eval + game outcome is the safe recipe.
- **De-duplicate by `_transposition_key` and its horizontal mirror** before the
  train/val split. A random split that drops near-identical positions on both sides
  inflates the validation number (`docs/PLAN.md`).

### 3. Retrain properly (hours of compute)

The current run is 10 flat epochs of MSE at lr 1e-3. Leaving a lot on the table.

- **Loss:** binary cross-entropy on win probability, not MSE:
  `−(t·log p + (1−t)·log(1−p))`.
- **LR schedule:** a few hundred steps of warmup, then step or cosine decay. Not flat.
- **Batch size:** 8k–16k. It's a small net; big batches train fast and stably.
- **Epochs:** many more — 100–400 passes with early stopping on val loss. With enough
  data, train and val loss should track closely.
- Keep logging seed + dataset hash + git commit per run. You already do — good.

**Decision point:** arena `nnue_agent` vs `versions/phase5jit` *and* vs current
classical `main`, 100+ games each. If it only ties phase5jit, it is not worth shipping
over the classical engine that already beats phase5jit.

---

## Strength wins — after the critical path

### Dual-perspective king-relative features

The 768 absolute one-hot input is the "get it working" version. The real strength jump
is **HalfKP / HalfKA_v2**: features indexed relative to a king's square, computed twice
— once from White's king perspective, once from Black's — then concatenated in
side-to-move order (own perspective first) before the dense head. This is what "NNUE"
actually is, and it's a large Elo gain over absolute features because the net stops
having to relearn the same pattern in 64 places.

### Clipped ReLU

Use `clamp(x, 0, 1)` (float) / `clamp(x, 0, 127)` (int8), not plain ReLU. It's the
NNUE standard: trains more stably for this target and quantizes without accuracy loss.

### Architecture shape

Wide feature transformer, small output, tiny head — e.g. `HalfKA (~45k features) → 512`
per perspective, then `1024 → 16 → 32 → 1`. The current 256 hidden is on the small
side; 512 with the accumulator trick below is affordable. Do not go past ~1024 wide —
onnxruntime glued to Python search can't feed a bigger net enough times per move
(`docs/IDEAS.md`: "a net you can evaluate thousands of times per move is worth more
than a better net you can evaluate fifty times").

---

## Speed — the part that makes it usable

Right now `nnue_agent/evaluate.py` calls `session.run` once per node. That pays
onnxruntime's per-call dispatch overhead on every node and is why the node rate
collapses. Three fixes, biggest first:

### Incremental accumulator

The feature-transformer output for a position differs from its parent's by only the
moved piece (plus a captured piece, plus rook-shift on castling, plus the EP pawn). On
`board.push`, subtract the from-square feature column and add the to-square column; on
`board.pop`, undo. Eval goes from an O(features) matmul to O(1) amortised. This is why
real NNUE is fast. Caveat: with king-relative features, a king move invalidates that
side's accumulator and forces a full refresh for that perspective — kings move rarely,
so it's cheap on average.

### Batch the leaf evaluations

Collect a search pass's leaf positions and score them in one `session.run`. Needs a
search restructure: the leaf can't call `evaluate()` and return immediately — it has to
enqueue the position and get resolved after the batch runs. `nnue_eval.py`'s
`NNUEBatchEvaluator.evaluate_batch` already exists; the search side is the missing half.
A cheaper interim step: a bounded eval cache plus batching within one node's move list.

### Quantise to int8

Train in float, then `onnxruntime.quantization` (dynamic, or static with a small
calibration set). 2–4× inference speedup, negligible Elo loss when the net uses clipped
ReLU. NNUE is designed for this.

**Target:** `tools/bench.py` node rate within 2–3× of the classical engine (~22k nps
today). Below ~5k nps the net loses on depth no matter how good the eval is.

**Fallback:** keep the classical eval compiled in. If the net returns NaN or an
out-of-range score, use classical for that leaf.

---

## More ideas / hedges

### Texel-tune the hand-crafted eval — do this in parallel

`docs/PLAN.md` and `evaluate.py` reference a `texel_tune.py` that doesn't exist yet.
The self-play → labelled-positions pipeline you've built is 90% of what Texel tuning
needs. Tuning the PST tables and term weights in `evaluate.py` against the same
WDL-blended labels is **lower risk, ships in the classical framework** (keeps persistent
TT / Syzygy / everything), and typically buys +30–80 Elo for a fraction of the effort.
It also de-risks the NNUE bet — if NNUE stalls, this is the fallback that still improves
the engine. `evaluate.py` already has a `_evaluate_reference` pure-Python path built
specifically to tune against; register any new table in `get_tunable_tables()`.

### Net as a correction term, not a replacement

If NNUE inference stays too slow, use a **tiny** net (128 hidden, single perspective,
int8, incremental) that outputs a *delta* added to the classical eval, not a full
replacement. The classical eval anchors it, the net only has to learn the residual, and
a small net is fast. Lower ceiling than full NNUE, much lower risk.

### Data quality knobs worth trying

- Weight late-game positions higher — the endgame is where the classical eval is
  weakest and where the rated losses cluster (R57/R60/R61/R62/R64 were all endgame
  conversions; Syzygy fixes ≤3 men, the net could help the 4–8 man band).
- Include a slice of positions from *lost* games specifically, so the net learns what
  losing looks like, not just balanced middlegames.
- Filter out positions where `diag-8`'s shallow eval and the game result disagree
  sharply (|sigmoid(eval) − result| > 0.6) — usually a blunder later in the game
  poisoning the label. Or down-weight them.

### Don't

- Don't use the net for move ordering — keep MVV-LVA / killers / history. The net is
  leaf eval only.
- Don't spawn threads. `torch.set_num_threads(1)`, `intra_op_num_threads=1`,
  `inter_op_num_threads=1` (already set in `nnue_eval.py`). More threads lose time on
  one core.
- Don't ship a net anyone else trained, or one fine-tuned / re-exported from a
  published net. Checked after games, not just at upload (`AGENTS.md`).

---

## Rules / deploy checklist

- `torch` 2.13 (CPU) and `onnxruntime` 1.29 are in the base image. Nothing else
  installs. No network at runtime — weights ship in the zip.
- Whole zip < 50 MB unzipped. `model.onnx.data` is 821 KB now; a 512-wide HalfKA
  transformer is still only a few MB. Fine.
- Native binaries are rejected. `.onnx` / `.pt` / `.safetensors` data files are fine.
- Package must include the weights folder. The classical build already does this for
  `syzygy/` via `--include`; the NNUE build needs `--include <weights-dir>` (or name it
  `weights/`, which `harness.package` bundles by default).
- Warm the onnxruntime session at import (one dummy `session.run` with a real-shaped
  batch) so the first move doesn't pay first-inference cost on the clock.
- Keep `make gate` green — ruff, mypy strict, pytest.

---

## Suggested order

1. Rebase `nnue_agent` search onto `main`. — hours
2. Scale data to 5–10 M positions, WDL-blended labels, de-duped and mirror-de-duped. — day
3. Retrain: BCE loss, LR schedule, big batches, many epochs, early stop. — compute
4. Arena vs `phase5jit` **and** vs classical `main`, 100+ games each. **Decision point.**
5. If competitive: incremental accumulator → batched inference → int8. — days
6. If still too slow: correction-term net, or park NNUE and ship the Texel-tuned
   classical eval.

Run Texel tuning (the parallel hedge) alongside from step 2 — it reuses the same data.
