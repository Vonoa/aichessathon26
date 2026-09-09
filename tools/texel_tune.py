"""Texel tuning for evaluate.py's scalar weights.

Reads a dataset of ``FEN<TAB>target`` lines -- ``target`` in [0, 1] from White's
point of view, the game outcome blended with a shallow eval, exactly the format
``versions/phase7/label.py`` emits -- and coordinate-descends the evaluation
weights to minimise

    mean( (target - sigmoid(eval_white(fen) / K))**2 )

over the dataset, solving the sigmoid scale ``K`` alongside.

Scope (v1): the ~35 SCALAR weights -- piece values, mobility (mg/eg), passed-pawn
bonus by rank, doubled / isolated penalties, king-safety shield / open-file /
attacker-unit / danger terms. The 768 piece-square-table entries are left as-is;
they need far more data and mirror-symmetry constraints, and the PeSTO seed
values are already sane.

numba freezes a global array's contents into an ``@njit`` kernel at compile time,
so evaluate.py's own kernels can't have their weights moved from outside. Rather
than thread a weight array through the shipped hot path, this file carries a
faithful *replica* of the three weight-dependent kernels (``_pawn_side``,
``_mob_side``, ``_ks_side``) plus the assembly, all reading a passed-in ``W``
vector. ``_assert_replica_matches_shipped`` checks the replica reproduces
``evaluate.evaluate`` exactly at the seed weights on every loaded position before
tuning starts, so the optimum found is valid for the real eval. Paste the printed
constant blocks back into evaluate.py; the frozen-global kernels pick them up at
the next import and tests/test_evaljit.py's pin still holds.

Run (from the repo root):

    uv run python -m tools.texel_tune tuning/labelled_combined_v3.tsv
    uv run python -m tools.texel_tune tuning/labelled_combined_v3.tsv --limit 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import chess
import numpy as np
import numpy.typing as npt
from numba import njit

import evaluate as ev
import movegen

# --- weight-vector layout -----------------------------------------------------
# One flat int16 vector. Index constants are plain ints, which numba is happy to
# freeze; only the *array contents* problem forces the replica below.
WI_VALUE = 0  # [0:6]   pawn, knight, bishop, rook, queen, king
WI_MOB_MG = 6  # [6:10]  knight, bishop, rook, queen
WI_MOB_EG = 10  # [10:14] knight, bishop, rook, queen
WI_KATK = 14  # [14:20] pawn, knight, bishop, rook, queen, king  (king slot 0)
WI_PASSED = 20  # [20:28] by rank from own side, 0..7
WI_DOUBLED = 28
WI_ISOLATED = 29
WI_SHIELD = 30
WI_OPEN_FILE = 31
WI_SEMI_OPEN = 32
WI_DANGER_SCALE = 33
WI_DANGER_MAX = 34
W_LEN = 35

_PT_ORDER = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)
_MINOR_ORDER = (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)

# name, index, lo, hi, initial coordinate step (centipawns)
_PARAMS: list[tuple[str, int, int, int, int]] = [
    ("value_pawn", WI_VALUE + 0, 60, 160, 6),
    ("value_knight", WI_VALUE + 1, 240, 420, 8),
    ("value_bishop", WI_VALUE + 2, 240, 420, 8),
    ("value_rook", WI_VALUE + 3, 400, 640, 10),
    ("value_queen", WI_VALUE + 4, 760, 1140, 16),
    ("mob_mg_knight", WI_MOB_MG + 0, 0, 16, 1),
    ("mob_mg_bishop", WI_MOB_MG + 1, 0, 16, 1),
    ("mob_mg_rook", WI_MOB_MG + 2, 0, 16, 1),
    ("mob_mg_queen", WI_MOB_MG + 3, 0, 16, 1),
    ("mob_eg_knight", WI_MOB_EG + 0, 0, 16, 1),
    ("mob_eg_bishop", WI_MOB_EG + 1, 0, 16, 1),
    ("mob_eg_rook", WI_MOB_EG + 2, 0, 16, 1),
    ("mob_eg_queen", WI_MOB_EG + 3, 0, 16, 1),
    ("passed_r1", WI_PASSED + 1, 0, 40, 4),
    ("passed_r2", WI_PASSED + 2, 0, 60, 4),
    ("passed_r3", WI_PASSED + 3, 0, 90, 5),
    ("passed_r4", WI_PASSED + 4, 10, 130, 6),
    ("passed_r5", WI_PASSED + 5, 20, 180, 8),
    ("passed_r6", WI_PASSED + 6, 40, 260, 10),
    ("doubled", WI_DOUBLED, -50, 0, 2),
    ("isolated", WI_ISOLATED, -50, 0, 2),
    ("shield", WI_SHIELD, 0, 36, 2),
    ("open_file", WI_OPEN_FILE, -55, 0, 3),
    ("semi_open_file", WI_SEMI_OPEN, -40, 0, 2),
    ("katk_pawn", WI_KATK + 0, 0, 12, 1),
    ("katk_knight", WI_KATK + 1, 0, 16, 1),
    ("katk_bishop", WI_KATK + 2, 0, 16, 1),
    ("katk_rook", WI_KATK + 3, 0, 20, 1),
    ("katk_queen", WI_KATK + 4, 0, 24, 1),
    ("danger_scale", WI_DANGER_SCALE, 10, 200, 4),
    ("danger_max", WI_DANGER_MAX, 120, 900, 20),
]

# Terms Texel on quiet WDL positions cannot tune honestly: piece values (pawn-count
# outweighs majors in quiet positions from decisive games -> ratios drift low) and
# king safety (disasters live in check/capture positions the quiet filter removes).
# --positional-only keeps just the structural terms that DO have quiet-position
# signal. See the run notes in docs/PLAN.md.
_UNSAFE_FOR_QUIET_WDL = {
    "value_pawn",
    "value_knight",
    "value_bishop",
    "value_rook",
    "value_queen",
    "shield",
    "open_file",
    "semi_open_file",
    "katk_pawn",
    "katk_knight",
    "katk_bishop",
    "katk_rook",
    "katk_queen",
    "danger_scale",
    "danger_max",
}


def seed_vector() -> npt.NDArray[np.int16]:
    """The current evaluate.py weights, packed into the flat vector."""
    w = np.zeros(W_LEN, dtype=np.int16)
    for i, pt in enumerate(_PT_ORDER):
        w[WI_VALUE + i] = ev.PIECE_VALUES[pt]
    for i, pt in enumerate(_MINOR_ORDER):
        w[WI_MOB_MG + i] = ev.MOBILITY_MG[pt]
        w[WI_MOB_EG + i] = ev.MOBILITY_EG[pt]
    for i, pt in enumerate((chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)):
        w[WI_KATK + i] = ev.KING_ATTACK_UNIT[pt]
    w[WI_PASSED : WI_PASSED + 8] = ev.PASSED_PAWN_BONUS_BY_RANK
    w[WI_DOUBLED] = ev.DOUBLED_PAWN_PENALTY
    w[WI_ISOLATED] = ev.ISOLATED_PAWN_PENALTY
    w[WI_SHIELD] = ev.SHIELD_PAWN_BONUS
    w[WI_OPEN_FILE] = ev.OPEN_FILE_PENALTY
    w[WI_SEMI_OPEN] = ev.SEMI_OPEN_FILE_PENALTY
    w[WI_DANGER_SCALE] = ev.KING_DANGER_SCALE
    w[WI_DANGER_MAX] = ev.KING_DANGER_MAX
    return w


# --- replica of evaluate.py's three weight-dependent kernels ------------------
# Byte-for-byte the same arithmetic as evaluate._pawn_structure_side /
# _mobility_side / _king_safety_side / _evaluate_jit, with every tunable read from
# W instead of a module global. Kept honest by _assert_replica_matches_shipped.
_U0 = np.uint64(0)
_U1 = np.uint64(1)


@njit(cache=False)
def _mat_pst(
    pieces: npt.NDArray[np.uint64],
    w: npt.NDArray[np.int16],
    pst_mg: npt.NDArray[np.int16],
    pst_eg: npt.NDArray[np.int16],
) -> tuple[int, int]:
    mg = 0
    eg = 0
    for pt in range(6):
        v = int(w[WI_VALUE + pt])
        bb = pieces[0, pt]
        while bb != _U0:
            lsb = bb & (~bb + _U1)
            sq = int(ev._popcount(lsb - _U1))
            mg += v + int(pst_mg[pt, sq])
            eg += v + int(pst_eg[pt, sq])
            bb &= bb - _U1
        bb = pieces[1, pt]
        while bb != _U0:
            lsb = bb & (~bb + _U1)
            idx = int(ev._popcount(lsb - _U1)) ^ 56
            mg -= v + int(pst_mg[pt, idx])
            eg -= v + int(pst_eg[pt, idx])
            bb &= bb - _U1
    return mg, eg


@njit(cache=False)
def _pawn_side(own: np.uint64, enemy: np.uint64, is_white: bool, w: npt.NDArray[np.int16]) -> int:
    score = 0
    counts = np.empty(8, dtype=np.int64)
    for f in range(8):
        counts[f] = ev._popcount(own & ev._FILE_MASK_ARR[f])
    for f in range(8):
        c = int(counts[f])
        if c > 1:
            score += int(w[WI_DOUBLED]) * (c - 1)
        if c > 0:
            left = int(counts[f - 1]) if f > 0 else 0
            right = int(counts[f + 1]) if f < 7 else 0
            if left + right == 0:
                score += int(w[WI_ISOLATED]) * c
    bb = own
    while bb != _U0:
        lsb = bb & (~bb + _U1)
        sq = ev._popcount(lsb - _U1)
        if is_white:
            span = ev._PASSED_MASK_WHITE[sq]
            rank = sq >> 3
        else:
            span = ev._PASSED_MASK_BLACK[sq]
            rank = 7 - (sq >> 3)
        if (enemy & span) == _U0:
            score += int(w[WI_PASSED + rank])
        bb &= bb - _U1
    return score


@njit(cache=False)
def _mob_side(
    pieces: npt.NDArray[np.uint64], occ_all: np.uint64, colour: int, w: npt.NDArray[np.int16]
) -> tuple[int, int]:
    mg = 0
    eg = 0
    for pt in range(1, 5):
        wmg = int(w[WI_MOB_MG + pt - 1])
        weg = int(w[WI_MOB_EG + pt - 1])
        bb = pieces[colour, pt]
        while bb != _U0:
            lsb = bb & (~bb + _U1)
            sq = int(ev._popcount(lsb - _U1))
            n = int(ev._popcount(ev._piece_attacks(occ_all, sq, pt)))
            mg += wmg * n
            eg += weg * n
            bb &= bb - _U1
    return mg, eg


@njit(cache=False)
def _ks_side(
    pieces: npt.NDArray[np.uint64],
    occ_all: np.uint64,
    king_sq: int,
    colour: int,
    w: npt.NDArray[np.int16],
) -> int:
    own_pawns = pieces[colour, 0]
    enemy = 1 - colour
    enemy_pawns = pieces[enemy, 0]
    score = 0

    shield = ev._SHIELD_ARR[colour, king_sq]
    score += int(w[WI_SHIELD]) * int(ev._popcount(own_pawns & shield))

    king_file = king_sq % 8
    for nf in range(king_file - 1, king_file + 2):
        if nf < 0 or nf > 7:
            continue
        file_mask = ev._FILE_MASK_ARR[nf]
        has_own = (own_pawns & file_mask) != _U0
        has_enemy = (enemy_pawns & file_mask) != _U0
        if not has_own and not has_enemy:
            score += int(w[WI_OPEN_FILE])
        elif not has_own and has_enemy:
            score += int(w[WI_SEMI_OPEN])

    zone = ev._KING_ZONE_ARR[king_sq]
    danger = 0
    attackers = 0
    for pt in range(5):
        unit = int(w[WI_KATK + pt])
        bb = pieces[enemy, pt]
        while bb != _U0:
            lsb = bb & (~bb + _U1)
            sq = int(ev._popcount(lsb - _U1))
            att = ev._PAWN_ATTACKS_ARR[enemy, sq] if pt == 0 else ev._piece_attacks(occ_all, sq, pt)
            hits = int(ev._popcount(att & zone))
            if hits > 0:
                attackers += 1
                danger += unit * hits
            bb &= bb - _U1
    if attackers >= 2:
        penalty = danger * danger * int(w[WI_DANGER_SCALE]) // 100
        cap = int(w[WI_DANGER_MAX])
        if penalty > cap:
            penalty = cap
        score -= penalty
    else:
        score -= danger
    return int(score)


@njit(cache=False)
def _eval_white(
    pieces: npt.NDArray[np.uint64],
    occ: npt.NDArray[np.uint64],
    white_king: int,
    black_king: int,
    w: npt.NDArray[np.int16],
) -> int:
    """evaluate._evaluate_jit's body, White-positive (no side-to-move flip)."""
    phase = ev._game_phase_jit(pieces)
    material_mg, material_eg = _mat_pst(pieces, w, ev._PST_MG, ev._PST_EG)
    pawn = _pawn_side(pieces[0, 0], pieces[1, 0], True, w) - _pawn_side(
        pieces[1, 0], pieces[0, 0], False, w
    )
    all_occ = occ[2]
    w_mg, w_eg = _mob_side(pieces, all_occ, 0, w)
    b_mg, b_eg = _mob_side(pieces, all_occ, 1, w)
    mob_mg = w_mg - b_mg
    mob_eg = w_eg - b_eg
    king_safety = _ks_side(pieces, all_occ, white_king, 0, w) - _ks_side(
        pieces, all_occ, black_king, 1, w
    )
    mg_total = material_mg + pawn + mob_mg + king_safety
    eg_total = material_eg + pawn + mob_eg
    blended = mg_total * phase + eg_total * (24 - phase)
    return (
        int(blended / 24)
        + ev._mopup_jit(occ, pieces, white_king, black_king)
        + ev._king_activity_jit(occ, pieces, white_king, black_king)
    )


@njit(cache=False)
def _eval_batch(
    pieces_all: npt.NDArray[np.uint64],
    occ_all: npt.NDArray[np.uint64],
    wk_all: npt.NDArray[np.int64],
    bk_all: npt.NDArray[np.int64],
    w: npt.NDArray[np.int16],
    out: npt.NDArray[np.float64],
) -> None:
    for i in range(pieces_all.shape[0]):
        out[i] = _eval_white(pieces_all[i], occ_all[i], wk_all[i], bk_all[i], w)


# --- data -------------------------------------------------------------------


def load_dataset(
    path: Path, limit: int | None
) -> tuple[list[str], npt.NDArray[np.float64], list[str]]:
    """Accepts either row format, auto-detected by column count. Returns the FEN list,
    targets normalised to White's point of view in [0, 1], and a game-id per row (""
    when the format carries none) so the train/val split can avoid cross-game leakage:

      fen <TAB> target                 -- target already White-POV (label.py output)
      game_id <TAB> fen <TAB> result   -- result is side-to-move-POV game outcome
                                          (tuning/*.tsv); flipped to White here
    """
    fens: list[str] = []
    targets: list[float] = []
    game_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 3:
                game_id, fen, result = parts
                value = float(result)
                if fen.split()[1] != "w":  # result is from the mover's side
                    value = 1.0 - value
            elif len(parts) == 2:
                game_id, fen, raw = "", parts[0], parts[1]
                value = float(raw)
            else:
                continue
            fens.append(fen)
            targets.append(value)
            game_ids.append(game_id)
            if limit is not None and len(fens) >= limit:
                break
    return fens, np.asarray(targets, dtype=np.float64), game_ids


_QBUF = np.empty(256, dtype=np.int32)


def _is_quiet(bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64]) -> bool:
    """No SEE>=0 capture available for the side to move -- the standard Texel filter.
    A position with a hanging piece has a static eval that lies about its true value,
    so tuning on it drifts weights (v1 saw piece values fall this way).
    """
    turn = int(state[0])
    enemy = int(movegen._occ_of(bb, 1 - turn))
    n = movegen._gen_legal(bb, state, _QBUF)
    for i in range(n):
        code = int(_QBUF[i])
        to = (code >> 6) & 0x3F
        flag = (code >> 15) & 7
        if (flag == 2 or ((enemy >> to) & 1)) and int(movegen._see(bb, turn, code)) >= 0:
            return False
    return True


def encode_dataset(
    fens: list[str],
    quiet_filter: bool,
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
]:
    """FEN list -> stacked (pieces, occ, white_king, black_king, shipped_eval) arrays for
    the KEPT rows only, plus the kept-row indices into `fens`. Drops: unparseable, no
    king, in check, stalemate, insufficient material, and -- when quiet_filter -- any
    position with a SEE>=0 capture on the board. shipped_eval (seed weights, White POV)
    feeds the optional WDL blend. Building only the kept rows keeps the peak allocation
    at the filtered size, not the raw size, so the dataset can be large.
    """
    p_list: list[npt.NDArray[np.uint64]] = []
    o_list: list[npt.NDArray[np.uint64]] = []
    wk_list: list[int] = []
    bk_list: list[int] = []
    sev_list: list[int] = []
    kept: list[int] = []
    for i, fen in enumerate(fens):
        if i % 250_000 == 0 and i:
            print(f"  scanned {i}/{len(fens)}  kept {len(kept)} ...")
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        wksq = board.king(chess.WHITE)
        bksq = board.king(chess.BLACK)
        if (
            wksq is None
            or bksq is None
            or board.is_check()
            or board.is_stalemate()
            or board.is_insufficient_material()
        ):
            continue
        bb, state = movegen.encode(board)
        if quiet_filter and not _is_quiet(bb, state):
            continue
        p, o, _ = ev._encode(board)
        p_list.append(p.copy())
        o_list.append(o.copy())
        wk_list.append(wksq)
        bk_list.append(bksq)
        shipped = ev.evaluate(board)
        sev_list.append(shipped if board.turn == chess.WHITE else -shipped)
        kept.append(i)
    if not kept:
        raise SystemExit("no usable positions after filtering")
    pieces = np.stack(p_list)
    occ = np.stack(o_list)
    return (
        pieces,
        occ,
        np.asarray(wk_list, dtype=np.int64),
        np.asarray(bk_list, dtype=np.int64),
        np.asarray(sev_list, dtype=np.int32),
        np.asarray(kept, dtype=np.int64),
    )


def _assert_replica_matches_shipped(
    fens: list[str],
    pieces: npt.NDArray[np.uint64],
    occ: npt.NDArray[np.uint64],
    wk: npt.NDArray[np.int64],
    bk: npt.NDArray[np.int64],
    sample: int = 6000,
) -> None:
    """Check the njit replica reproduces evaluate.evaluate on a random sample (the full
    2M-row loop of chess.Board + evaluate would be minutes) at the seed weights, so the
    optimum found is valid for the shipped eval."""
    seed = seed_vector()
    out = np.zeros(len(fens), dtype=np.float64)
    _eval_batch(pieces, occ, wk, bk, seed, out)
    rng = np.random.default_rng(1)
    idx = rng.choice(len(fens), size=min(sample, len(fens)), replace=False)
    bad = 0
    for i in idx:
        board = chess.Board(fens[i])
        shipped = ev.evaluate(board)
        white_pov = shipped if board.turn == chess.WHITE else -shipped
        if int(out[i]) != white_pov:
            bad += 1
            if bad <= 5:
                print(f"  replica mismatch: {fens[i]}  replica={int(out[i])} shipped={white_pov}")
    if bad:
        raise SystemExit(
            f"replica disagrees with evaluate.evaluate on {bad}/{len(idx)} sampled positions "
            "-- the kernels in evaluate.py changed; update the replica in this file."
        )
    print(f"replica == evaluate.evaluate on {len(idx)} sampled positions (seed weights)")


# --- optimiser ------------------------------------------------------------------


def sigmoid(centipawns: npt.NDArray[np.float64], k: float) -> npt.NDArray[np.float64]:
    return 1.0 / (1.0 + np.exp(-centipawns / k))


def mse(targets: npt.NDArray[np.float64], evals: npt.NDArray[np.float64], k: float) -> float:
    diff = targets - sigmoid(evals, k)
    return float(np.mean(diff * diff))


def solve_k(targets: npt.NDArray[np.float64], evals: npt.NDArray[np.float64]) -> float:
    best_k, best_err = 400.0, float("inf")
    for k in list(np.arange(120.0, 720.0, 20.0)):
        err = mse(targets, evals, k)
        if err < best_err:
            best_err, best_k = err, float(k)
    for k in list(np.arange(best_k - 20.0, best_k + 20.0, 2.0)):
        if k <= 0:
            continue
        err = mse(targets, evals, k)
        if err < best_err:
            best_err, best_k = err, float(k)
    return best_k


def tune(
    targets: npt.NDArray[np.float64],
    pieces: npt.NDArray[np.uint64],
    occ: npt.NDArray[np.uint64],
    wk: npt.NDArray[np.int64],
    bk: npt.NDArray[np.int64],
    max_sweeps: int,
) -> tuple[npt.NDArray[np.int16], float, float, float]:
    w = seed_vector()
    out = np.zeros(len(targets), dtype=np.float64)

    _eval_batch(pieces, occ, wk, bk, w, out)
    k = solve_k(targets, out)
    start_err = mse(targets, out, k)
    best_err = start_err
    steps = {name: step for name, _idx, _lo, _hi, step in _PARAMS}
    momentum = {name: 1 for name, *_ in _PARAMS}  # last successful direction, tried first
    print(f"K={k:.0f}  start MSE={start_err:.6f}  ({len(targets)} positions)")

    for sweep in range(max_sweeps):
        improved = False
        for name, idx, lo, hi, _step0 in _PARAMS:
            step = steps[name]
            if step < 1:
                continue
            first = momentum[name]
            for delta in (first * step, -first * step):
                original = int(w[idx])
                candidate = original + delta
                if candidate < lo or candidate > hi:
                    continue
                w[idx] = candidate
                _eval_batch(pieces, occ, wk, bk, w, out)
                err = mse(targets, out, k)
                if err + 1e-12 < best_err:
                    best_err = err
                    improved = True
                    momentum[name] = 1 if delta > 0 else -1
                    break
                w[idx] = original
        _eval_batch(pieces, occ, wk, bk, w, out)
        k = solve_k(targets, out)
        best_err = mse(targets, out, k)
        print(f"  sweep {sweep + 1:2d}: MSE={best_err:.6f}  K={k:.0f}")
        if not improved:
            for name in steps:
                steps[name] = steps[name] // 2
            if all(step < 1 for step in steps.values()):
                break
    return w, k, start_err, best_err


# --- reporting ----------------------------------------------------------------


def _fmt_list(values: list[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def print_constant_blocks(w: npt.NDArray[np.int16]) -> None:
    v = [int(x) for x in w]
    print("\n--- paste into evaluate.py (only the numbers changed) ---\n")
    print("PIECE_VALUES = {")
    for i, name in enumerate(("PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN")):
        print(f"    chess.{name}: {v[WI_VALUE + i]},")
    print("    chess.KING: 0,\n}")
    mob_mg = [v[WI_MOB_MG + i] for i in range(4)]
    mob_eg = [v[WI_MOB_EG + i] for i in range(4)]
    print(
        "\nMOBILITY_MG = {"
        f"chess.KNIGHT: {mob_mg[0]}, chess.BISHOP: {mob_mg[1]}, "
        f"chess.ROOK: {mob_mg[2]}, chess.QUEEN: {mob_mg[3]}}}"
    )
    print(
        "MOBILITY_EG = {"
        f"chess.KNIGHT: {mob_eg[0]}, chess.BISHOP: {mob_eg[1]}, "
        f"chess.ROOK: {mob_eg[2]}, chess.QUEEN: {mob_eg[3]}}}"
    )
    print(f"\nPASSED_PAWN_BONUS_BY_RANK = {_fmt_list([v[WI_PASSED + r] for r in range(8)])}")
    print(f"DOUBLED_PAWN_PENALTY = {v[WI_DOUBLED]}")
    print(f"ISOLATED_PAWN_PENALTY = {v[WI_ISOLATED]}")
    print(f"\nSHIELD_PAWN_BONUS = {v[WI_SHIELD]}")
    print(f"OPEN_FILE_PENALTY = {v[WI_OPEN_FILE]}")
    print(f"SEMI_OPEN_FILE_PENALTY = {v[WI_SEMI_OPEN]}")
    print(
        "\nKING_ATTACK_UNIT = {"
        f"chess.PAWN: {v[WI_KATK + 0]}, chess.KNIGHT: {v[WI_KATK + 1]}, "
        f"chess.BISHOP: {v[WI_KATK + 2]}, chess.ROOK: {v[WI_KATK + 3]}, "
        f"chess.QUEEN: {v[WI_KATK + 4]}}}"
    )
    print(f"KING_DANGER_SCALE = {v[WI_DANGER_SCALE]}")
    print(f"KING_DANGER_MAX = {v[WI_DANGER_MAX]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="+",
        help="one or more .tsv files, or directories searched recursively for *.tsv",
    )
    parser.add_argument("--limit", type=int, default=None, help="cap positions (smoke runs)")
    parser.add_argument("--val-frac", type=float, default=0.1, help="held-out fraction")
    parser.add_argument("--max-sweeps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0, help="train/val split seed")
    parser.add_argument(
        "--positional-only",
        action="store_true",
        help="tune only terms with honest quiet-position signal (drops piece values "
        "and king safety -- see _UNSAFE_FOR_QUIET_WDL)",
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=1.0,
        metavar="LAMBDA",
        help="target = LAMBDA*game_result + (1-LAMBDA)*sigmoid(shipped_eval/400). "
        "1.0 (default) = pure game outcome; 0.6-0.7 adds per-position signal that "
        "steadies the fit. The eval half is our own, so keep LAMBDA >= ~0.5.",
    )
    parser.add_argument(
        "--no-quiet-filter",
        action="store_true",
        help="keep positions with a SEE>=0 capture on the board (default drops them; "
        "tuning on tactically-unresolved positions drifts weights)",
    )
    args = parser.parse_args()

    global _PARAMS
    if args.positional_only:
        drop = _UNSAFE_FOR_QUIET_WDL | {p[0] for p in _PARAMS if p[0].startswith("mob_")}
        _PARAMS = [p for p in _PARAMS if p[0] not in drop]
        print(f"positional-only: tuning {len(_PARAMS)} terms ({', '.join(p[0] for p in _PARAMS)})")

    files: list[Path] = []
    for entry in args.dataset:
        if entry.is_dir():
            files.extend(sorted(entry.rglob("*.tsv")))
        else:
            files.append(entry)
    if not files:
        raise SystemExit("no .tsv files found")

    hasher = hashlib.sha256()
    fens: list[str] = []
    game_ids: list[str] = []
    targets_list: list[npt.NDArray[np.float64]] = []
    for path in files:
        hasher.update(path.read_bytes())
        part_fens, part_targets, part_ids = load_dataset(path, None)
        # A blank id (2-column format) gets a per-row unique id -> a plain random split
        # for those rows instead of dumping the whole file into one train/val bucket.
        game_ids.extend(
            f"{path.stem}:{g}" if g else f"{path.stem}:r{len(fens) + n}"
            for n, g in enumerate(part_ids)
        )
        fens.extend(part_fens)
        targets_list.append(part_targets)
        print(f"  {path}  rows={len(part_fens)}")
    targets = np.concatenate(targets_list)
    # Shuffle before any --limit so the cap samples every file, not just the first ones.
    order = np.random.default_rng(args.seed).permutation(len(fens))
    fens = [fens[i] for i in order]
    game_ids = [game_ids[i] for i in order]
    targets = targets[order]
    if args.limit is not None and len(fens) > args.limit:
        fens, game_ids, targets = fens[: args.limit], game_ids[: args.limit], targets[: args.limit]
    digest = hasher.hexdigest()[:16]
    dataset_dir = files[0].parent
    print(f"{len(files)} file(s)  sha256[:16]={digest}  rows={len(fens)}")

    t0 = time.time()
    pieces, occ, wk, bk, seval, kept = encode_dataset(fens, not args.no_quiet_filter)
    n_scanned = len(fens)
    fens = [fens[i] for i in kept]
    game_ids = [game_ids[i] for i in kept]
    targets = targets[kept]
    filt = "no quiet filter" if args.no_quiet_filter else "quiet-filtered"
    print(
        f"kept {len(fens)} / {n_scanned} positions in {time.time() - t0:.1f}s "
        f"(dropped check / terminal / no-king / {filt})"
    )

    if args.blend < 1.0:
        prob = 1.0 / (1.0 + np.exp(-np.clip(seval.astype(np.float64), -2000.0, 2000.0) / 400.0))
        targets = args.blend * targets + (1.0 - args.blend) * prob
        print(f"blended targets: {args.blend:.2f}*WDL + {1 - args.blend:.2f}*sigmoid(eval/400)")

    _assert_replica_matches_shipped(fens, pieces, occ, wk, bk)

    # Split so no game straddles train/val (2-column rows have per-row ids -> random).
    rng = np.random.default_rng(args.seed)
    uniq = sorted(set(game_ids))
    n_val_groups = max(1, int(len(uniq) * args.val_frac))
    val_lut = {uniq[i] for i in rng.choice(len(uniq), size=n_val_groups, replace=False)}
    is_val = np.array([g in val_lut for g in game_ids])
    val_idx = np.nonzero(is_val)[0]
    train_idx = np.nonzero(~is_val)[0]
    print(f"{len(uniq)} groups; train {len(train_idx)}  val {len(val_idx)}")

    def split(a: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        return a[train_idx], a[val_idx]

    tr_t, va_t = split(targets)
    tr_p, va_p = split(pieces)
    tr_o, va_o = split(occ)
    tr_wk, va_wk = split(wk)
    tr_bk, va_bk = split(bk)

    tuned, k, start_err, end_err = tune(tr_t, tr_p, tr_o, tr_wk, tr_bk, args.max_sweeps)

    seed = seed_vector()
    va_out = np.zeros(len(va_t), dtype=np.float64)
    _eval_batch(va_p, va_o, va_wk, va_bk, seed, va_out)
    va_seed = mse(va_t, va_out, k)
    _eval_batch(va_p, va_o, va_wk, va_bk, tuned, va_out)
    va_tuned = mse(va_t, va_out, k)

    print(f"\ntrain MSE {start_err:.6f} -> {end_err:.6f}")
    print(f"val   MSE {va_seed:.6f} -> {va_tuned:.6f}  (K={k:.0f})")
    print("\nweight        seed  ->  tuned   (delta)")
    for name, idx, _lo, _hi, _step in _PARAMS:
        s, t = int(seed[idx]), int(tuned[idx])
        if s != t:
            print(f"  {name:16s} {s:5d}  -> {t:5d}   ({t - s:+d})")

    print_constant_blocks(tuned)

    runs_dir = dataset_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    out_path = runs_dir / f"texel_{int(time.time())}.json"
    out_path.write_text(
        json.dumps(
            {
                "dataset": [str(p) for p in files],
                "dataset_sha256_16": digest,
                "blend_lambda": args.blend,
                "quiet_filter": not args.no_quiet_filter,
                "n_positions": len(fens),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "k": k,
                "train_mse_start": start_err,
                "train_mse_end": end_err,
                "val_mse_seed": va_seed,
                "val_mse_tuned": va_tuned,
                "seed_weights": [int(x) for x in seed],
                "tuned_weights": [int(x) for x in tuned],
                "param_names": [p[0] for p in _PARAMS],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
