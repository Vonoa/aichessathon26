"""Pairwise-ranking eval tune: learn the positional weights that make the engine
*prefer the moves strong players actually chose*.

Texel fits the eval to game outcomes; this fits it to move choices. For each quiet
(non-capture, non-check) move a strong player made, we want

    eval(position after the played move)  >=  eval(position after a random legal
    alternative)  -  MARGIN

from the mover's point of view. The loss is the mean hinge over sampled
alternatives; coordinate descent minimises it over evaluate.py's ~35 scalar
weights. This is the "don't trade the good bishop / don't sortie the queen for
nothing / consolidate when worse" signal that win/loss data can't give.

Rules: human/GM game databases are unrestricted training data (aichessathon.com
/docs/rules.md). Nothing here is shipped -- it moves integer eval weights.

    uv run python -m tools.policy_tune tuning/ExtraData "tuning/top comeptitiors" \
        --competitor-weight 20 --max-sweeps 60

Reuses the njit eval replica and the tunable-weight layout from tools.texel_tune.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import numpy.typing as npt
import zstandard

import evaluate as ev
from tools.texel_tune import (
    _PARAMS,
    _UNSAFE_FOR_QUIET_WDL,
    _eval_batch,
    print_constant_blocks,
    seed_vector,
)


def _iter_games(paths: list[Path]):
    """Yield (source_path, chess.pgn.Game) from .pgn / .pgn.zst files and dirs."""
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(p.rglob("*.pgn")))
            files.extend(sorted(p.rglob("*.pgn.zst")))
        else:
            files.append(p)
    for f in files:
        if f.suffix == ".zst":
            raw = f.open("rb")
            handle = io.TextIOWrapper(
                zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace"
            )
        else:
            handle = f.open(encoding="utf-8", errors="replace")
        with handle:
            while True:
                try:
                    game = chess.pgn.read_game(handle)
                except (ValueError, RuntimeError):
                    continue
                if game is None:
                    break
                yield f, game


def _enc(board: chess.Board) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], int, int]:
    pieces, occ, _ = ev._encode(board)
    return pieces, occ, board.king(chess.WHITE), board.king(chess.BLACK)


_SEED_W = seed_vector()


def _seed_top_alts(board: chess.Board, played: chess.Move, k: int) -> list[chess.Move] | None:
    """The k QUIET legal moves (other than `played`) the SEED eval most prefers, from
    the mover's point of view -- 'hard negatives'. Quiet-only so the comparison is
    positional-vs-positional (a capture alt vs a quiet GM move is an unfixable material
    gap). Random alts are almost all blunders the eval already ranks right, so no
    gradient; the signal is 'the eval is tempted by quiet move X, a strong player chose
    quiet move Y'.
    """
    legal = [
        m
        for m in board.legal_moves
        if m != played and m.promotion is None and not board.is_capture(m)
    ]
    if len(legal) < k:
        return None
    rows = []
    for m in legal:
        board.push(m)
        rows.append(_enc(board))
        board.pop()
    p = np.stack([r[0] for r in rows])
    o = np.stack([r[1] for r in rows])
    wk = np.asarray([r[2] for r in rows], dtype=np.int64)
    bk = np.asarray([r[3] for r in rows], dtype=np.int64)
    out = np.empty(len(legal), dtype=np.float64)
    _eval_batch(p, o, wk, bk, _SEED_W, out)
    if board.turn == chess.BLACK:
        out = -out  # mover POV
    order = np.argsort(-out)[:k]
    return [legal[i] for i in order]


def extract(
    paths: list[Path],
    skip_plies: int,
    max_plies: int,
    stride: int,
    k_alts: int,
    competitor_weight: int,
    seed: int,
) -> dict[str, npt.NDArray]:
    """Walk every game, sample quiet strong-player moves, and build stacked arrays:
    played_* (M, ...) and alt_* (M, k_alts, ...) plus stm (M,) and gid (M,).
    Positions from an aichessathon competitor PGN are emitted competitor_weight times
    (there are only a few dozen, but they are the exact conditions we compete in).
    """
    _ = seed  # kept for signature symmetry; hard negatives are deterministic
    pp, po, pwk, pbk = [], [], [], []
    ap, ao, awk, abk = [], [], [], []
    stm, gid = [], []
    n_games = n_samp = 0
    started = time.time()

    for src, game in _iter_games(paths):
        if game.headers.get("Variant", "Standard") != "Standard":
            continue
        is_comp = "aichessathon" in src.name.lower()
        reps = competitor_weight if is_comp else 1
        board = game.board()
        n_games += 1
        if n_games % 2000 == 0:
            print(f"  {n_games} games  {n_samp} samples  ({time.time() - started:.0f}s)")
        for ply, move in enumerate(game.mainline_moves()):
            if ply >= max_plies:
                break
            take = (
                ply >= skip_plies
                and ply % stride == 0
                and not board.is_check()
                and not board.is_capture(move)
                and move.promotion is None
            )
            if take:
                board.push(move)
                played_gives_check = board.is_check()
                board.pop()
                alts = None if played_gives_check else _seed_top_alts(board, move, k_alts)
                if alts is not None:
                    board.push(move)
                    p0 = _enc(board)
                    board.pop()
                    arows = []
                    for am in alts:
                        board.push(am)
                        arows.append(_enc(board))
                        board.pop()
                    side = 0 if board.turn == chess.WHITE else 1
                    for _ in range(reps):
                        pp.append(p0[0])
                        po.append(p0[1])
                        pwk.append(p0[2])
                        pbk.append(p0[3])
                        ap.append(np.stack([r[0] for r in arows]))
                        ao.append(np.stack([r[1] for r in arows]))
                        awk.append([r[2] for r in arows])
                        abk.append([r[3] for r in arows])
                        stm.append(side)
                        gid.append(f"{src.stem}:{n_games}")
                    n_samp += 1
            board.push(move)

    print(
        f"extracted {len(pp)} rows ({n_samp} unique) from {n_games} games "
        f"in {time.time() - started:.0f}s"
    )
    return {
        "played_p": np.stack(pp),
        "played_o": np.stack(po),
        "played_wk": np.asarray(pwk, dtype=np.int64),
        "played_bk": np.asarray(pbk, dtype=np.int64),
        "alt_p": np.stack(ap),
        "alt_o": np.stack(ao),
        "alt_wk": np.asarray(awk, dtype=np.int64),
        "alt_bk": np.asarray(abk, dtype=np.int64),
        "stm": np.asarray(stm, dtype=np.int64),
        "gid": np.asarray(gid),
    }


def hinge_loss(
    w: npt.NDArray[np.int16], d: dict[str, npt.NDArray], margin: float, buf: dict[str, npt.NDArray]
) -> float:
    n, k = d["alt_wk"].shape
    _eval_batch(d["played_p"], d["played_o"], d["played_wk"], d["played_bk"], w, buf["sp"])
    _eval_batch(
        d["alt_p"].reshape(n * k, 2, 6),
        d["alt_o"].reshape(n * k, 3),
        d["alt_wk"].reshape(n * k),
        d["alt_bk"].reshape(n * k),
        w,
        buf["sa"],
    )
    sign = np.where(d["stm"] == 0, 1.0, -1.0)  # mover POV = white POV, negated for black
    mp = buf["sp"] * sign
    ma = buf["sa"].reshape(n, k) * sign[:, None]
    return float(np.maximum(0.0, margin - (mp[:, None] - ma)).mean())


def tune(
    tr: dict[str, npt.NDArray], margin: float, reg: float, max_sweeps: int
) -> tuple[npt.NDArray[np.int16], float, float]:
    w = seed_vector()
    seed = seed_vector()
    n, k = tr["alt_wk"].shape
    buf = {"sp": np.empty(n, dtype=np.float64), "sa": np.empty(n * k, dtype=np.float64)}
    steps = {name: step for name, _i, _lo, _hi, step in _PARAMS}

    def objective(ww: npt.NDArray[np.int16]) -> float:
        h = hinge_loss(ww, tr, margin, buf)
        # gentle pull toward the seed: cost grows with (drift / this param's step size)^2,
        # so a param needs real signal to move more than a few steps from PeSTO's value.
        r = sum(
            ((int(ww[idx]) - int(seed[idx])) / max(1, steps[name])) ** 2
            for name, idx, _lo, _hi, _s in _PARAMS
        )
        return h + reg * r

    best = start = objective(w)
    momentum = {name: 1 for name, *_ in _PARAMS}
    print(f"start obj={start:.4f}  ({n} rows, k={k}, margin={margin:.0f}, reg={reg:g})")

    for sweep in range(max_sweeps):
        improved = False
        for name, idx, lo, hi, _s in _PARAMS:
            step = steps[name]
            if step < 1:
                continue
            for delta in (momentum[name] * step, -momentum[name] * step):
                orig = int(w[idx])
                cand = orig + delta
                if cand < lo or cand > hi:
                    continue
                w[idx] = cand
                err = objective(w)
                if err + 1e-9 < best:
                    best, improved = err, True
                    momentum[name] = 1 if delta > 0 else -1
                    break
                w[idx] = orig
        print(f"  sweep {sweep + 1:2d}: obj={best:.4f}  hinge={hinge_loss(w, tr, margin, buf):.4f}")
        if not improved:
            for name in steps:
                steps[name] //= 2
            if all(s < 1 for s in steps.values()):
                break
    return w, start, best



def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path, nargs="+", help=".pgn / .pgn.zst files or dirs")
    p.add_argument("--skip-plies", type=int, default=10)
    p.add_argument(
        "--max-plies", type=int, default=70, help="cap so endgame shuffles do not dominate"
    )
    p.add_argument("--stride", type=int, default=4, help="sample every this many plies")
    p.add_argument("--k-alts", type=int, default=8, help="hard negatives per position")
    p.add_argument("--margin", type=float, default=40.0)
    p.add_argument(
        "--reg",
        type=float,
        default=0.1,
        help="pull toward the PeSTO seed: cost = reg * sum((drift / step)^2). Small "
        "resists runaway/collapse without freezing genuine small gains.",
    )
    p.add_argument("--competitor-weight", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--max-sweeps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tune-values",
        action="store_true",
        help="also tune piece values (default pins them: a pawn-anchored scale stops the "
        "ranking loss from collapsing every weight toward zero, and PeSTO's material is "
        "already right -- move choice should only move the positional terms)",
    )
    p.add_argument(
        "--no-king-safety",
        action="store_true",
        help="also pin the king-safety cluster (shield / open-file / attack-unit / danger)",
    )
    a = p.parse_args()

    global _PARAMS
    drop: set[str] = set()
    if not a.tune_values:
        drop |= {q[0] for q in _PARAMS if q[0].startswith("value_")}
    if a.no_king_safety:
        drop |= _UNSAFE_FOR_QUIET_WDL - {q[0] for q in _PARAMS if q[0].startswith("value_")}
    if drop:
        _PARAMS[:] = [q for q in _PARAMS if q[0] not in drop]
    print(f"tuning {len(_PARAMS)} terms: {', '.join(q[0] for q in _PARAMS)}")

    d = extract(a.data, a.skip_plies, a.max_plies, a.stride, a.k_alts, a.competitor_weight, a.seed)

    rng = np.random.default_rng(a.seed)
    uniq = sorted(set(d["gid"].tolist()))
    n_val = max(1, int(len(uniq) * a.val_frac))
    val_games = {uniq[i] for i in rng.choice(len(uniq), size=n_val, replace=False)}
    is_val = np.array([g in val_games for g in d["gid"]])
    tr = {key: d[key][~is_val] for key in d}
    va = {key: d[key][is_val] for key in d}
    print(f"{len(uniq)} games; train {(~is_val).sum()}  val {is_val.sum()}")

    tuned, start, end = tune(tr, a.margin, a.reg, a.max_sweeps)

    seed = seed_vector()
    vb = {
        "sp": np.empty(len(va["stm"]), dtype=np.float64),
        "sa": np.empty(va["alt_wk"].size, dtype=np.float64),
    }
    va_seed = hinge_loss(seed, va, a.margin, vb)
    va_tuned = hinge_loss(tuned, va, a.margin, vb)
    print(f"\ntrain hinge {start:.4f} -> {end:.4f}")
    print(f"val   hinge {va_seed:.4f} -> {va_tuned:.4f}")
    print("\nweight        seed  ->  tuned   (delta)")
    for name, idx, _lo, _hi, _s in _PARAMS:
        s, t = int(seed[idx]), int(tuned[idx])
        if s != t:
            print(f"  {name:16s} {s:5d}  -> {t:5d}   ({t - s:+d})")

    print_constant_blocks(tuned)

    out_dir = Path("tuning/policy_runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"policy_{int(time.time())}.json"
    dest.write_text(
        json.dumps(
            {
                "data": [str(x) for x in a.data],
                "margin": a.margin,
                "k_alts": a.k_alts,
                "competitor_weight": a.competitor_weight,
                "n_train": int((~is_val).sum()),
                "n_val": int(is_val.sum()),
                "train_hinge_start": start,
                "train_hinge_end": end,
                "val_hinge_seed": va_seed,
                "val_hinge_tuned": va_tuned,
                "seed_weights": [int(x) for x in seed],
                "tuned_weights": [int(x) for x in tuned],
                "param_names": [q[0] for q in _PARAMS],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {dest}")



if __name__ == "__main__":
    main()
