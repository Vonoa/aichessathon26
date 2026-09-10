"""
Incremental (the "efficiently updatable" in NNUE) evaluation.

SimpleNNUE's first layer (fc1: 769 -> 256) is a huge, mostly-zero one-hot
layer -- at most 33 of 769 inputs are ever 1. Recomputing x @ W1 + b1 from
scratch at every node is ~197,000 multiply-adds thrown away on 736+ zero
inputs. A single move changes at most 4 of those 769 features (2 for a quiet
move: the mover's old square off, new square on; +1 for a capture; +2 more
for castling's rook; the side-to-move bit flips every ply too), so instead of
recomputing, this maintains a running 256-wide accumulator and adds/subtracts
only the changed feature columns of W1 on push, reversing it on pop -- the
same pattern PLAN.md already uses for incremental material/PST, applied to
the net's first layer. Only fc2/fc3 (256->32->1, cheap either way) still run
as a plain forward pass, once per leaf.

The feature indexing here MUST exactly match features.py's fen_to_sparse_indices
-- this reuses its constants rather than redefining them, since any drift
between training-time and runtime feature layout would silently score every
position through the wrong weights.
"""

from __future__ import annotations

import math
from pathlib import Path

import chess
import numpy as np
import numpy.typing as npt

import movegen
from features import _PIECE_TYPE_INDEX, N_COLORS, N_PIECE_FEATURES, N_SQUARES

_WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.npz"
_weights = np.load(_WEIGHTS_PATH)
W1: npt.NDArray[np.float32] = _weights["fc1_weight"]  # (256, 769)
B1: npt.NDArray[np.float32] = _weights["fc1_bias"]  # (256,)
W2: npt.NDArray[np.float32] = _weights["fc2_weight"]  # (32, 256)
B2: npt.NDArray[np.float32] = _weights["fc2_bias"]  # (32,)
W3: npt.NDArray[np.float32] = _weights["fc3_weight"]  # (1, 32)
B3: npt.NDArray[np.float32] = _weights["fc3_bias"]  # (1,)

HIDDEN1 = W1.shape[0]
_STM_FEATURE = N_PIECE_FEATURES  # feature 768, on iff Black to move
_MAX_STACK = 256  # generous vs. _MAX_DEPTH (64) + _QS_MAX_PLY (96)


def _feature_idx(piece_type: int, color: bool, square: int) -> int:
    color_idx = 0 if color == chess.WHITE else 1
    type_idx = _PIECE_TYPE_INDEX[piece_type]
    return (type_idx * N_COLORS + color_idx) * N_SQUARES + square


def move_deltas(board: chess.Board, move: chess.Move) -> list[tuple[int, int]]:
    """Feature-index deltas caused by pushing `move` on `board`, given board is
    STILL in its pre-move state (call this before board.push(move)). Does not
    include the side-to-move bit -- Accumulator.push() handles that
    unconditionally, since every move flips it the same way regardless of
    what else happened.
    """
    mover = board.piece_at(move.from_square)
    assert mover is not None, "move_deltas called with no piece on from_square"
    deltas = [(_feature_idx(mover.piece_type, mover.color, move.from_square), -1)]

    if board.is_en_passant(move):
        captured_sq = move.to_square + (-8 if mover.color == chess.WHITE else 8)
        deltas.append((_feature_idx(chess.PAWN, not mover.color, captured_sq), -1))
    else:
        captured = board.piece_at(move.to_square)
        if captured is not None:
            deltas.append((_feature_idx(captured.piece_type, captured.color, move.to_square), -1))

    new_type = move.promotion if move.promotion else mover.piece_type
    deltas.append((_feature_idx(new_type, mover.color, move.to_square), +1))

    if board.is_castling(move):
        kingside = chess.square_file(move.to_square) == 6
        rank = 0 if mover.color == chess.WHITE else 7
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        deltas.append((_feature_idx(chess.ROOK, mover.color, rook_from), -1))
        deltas.append((_feature_idx(chess.ROOK, mover.color, rook_to), +1))

    return deltas


def move_deltas_bb(
    bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64], code: int
) -> list[tuple[int, int]]:
    """move_deltas(), for the jitted (bb, state, move-code) substrate current
    main/search.py runs on instead of chess.Board/chess.Move. `code` is the
    int32 move encoding from movegen.py (bits 0-5 from, 6-11 to, 12-14 promo
    [0 none, 1 N, 2 B, 3 R, 4 Q], 15-17 flag [0 normal/capture, 1 double pawn
    push, 2 en passant, 3 castle]) -- see movegen.py's own docstring. Reuses
    _feature_idx unchanged: movegen._piece_at already returns piece types in
    python-chess's own 1..6 numbering, and movegen's colour index (0 White, 1
    Black) matches board.turn's True/False the same way. Call with bb/state
    STILL in their pre-move state, same as move_deltas().
    """
    frm = code & 0x3F
    to = (code >> 6) & 0x3F
    promo = (code >> 12) & 7
    flag = (code >> 15) & 7
    mover_idx = int(state[0])
    enemy_idx = 1 - mover_idx
    mover_color = chess.WHITE if mover_idx == 0 else chess.BLACK
    enemy_color = chess.BLACK if mover_idx == 0 else chess.WHITE

    movpt = movegen._piece_at(bb, mover_idx, frm)
    deltas = [(_feature_idx(movpt, mover_color, frm), -1)]

    if flag == 2:  # en passant
        cap_sq = to - 8 if mover_idx == 0 else to + 8
        deltas.append((_feature_idx(chess.PAWN, enemy_color, cap_sq), -1))
    else:
        cappt = movegen._piece_at(bb, enemy_idx, to)
        if cappt != 0:
            deltas.append((_feature_idx(cappt, enemy_color, to), -1))

    new_pt = (promo + 1) if promo != 0 else movpt
    deltas.append((_feature_idx(new_pt, mover_color, to), +1))

    if flag == 3:  # castle
        rook_from = movegen._rook_hop_from(to)
        rook_to = movegen._rook_hop_to(to)
        deltas.append((_feature_idx(chess.ROOK, mover_color, rook_from), -1))
        deltas.append((_feature_idx(chess.ROOK, mover_color, rook_to), +1))

    return deltas


def accumulator_from_scratch(board: chess.Board) -> npt.NDArray[np.float32]:
    """Full recompute -- used once, to seed the accumulator at the search root."""
    acc = B1.copy()
    for square, piece in board.piece_map().items():
        acc += W1[:, _feature_idx(piece.piece_type, piece.color, square)]
    if board.turn == chess.BLACK:
        acc += W1[:, _STM_FEATURE]
    return acc


def forward(acc: npt.NDArray[np.float32]) -> float:
    """The cheap remaining layers, 256 -> 32 -> 1, sigmoid output in [0, 1],
    White's POV -- same convention as label.py's training target, regardless
    of whose turn it is (the side-to-move bit is an input feature, not an
    output-perspective flip)."""
    h1 = np.maximum(acc, 0.0)
    h2 = np.maximum(W2 @ h1 + B2, 0.0)
    out = W3 @ h2 + B3
    return float(1.0 / (1.0 + np.exp(-out[0])))


def score_to_centipawns(score_0_to_1: float, k: float = 400.0) -> int:
    score_0_to_1 = min(max(score_0_to_1, 1e-6), 1 - 1e-6)
    return int(k * math.log(score_0_to_1 / (1 - score_0_to_1)))


class Accumulator:
    """A preallocated push/pop stack of accumulator states, one row per ply,
    so push/pop never allocates -- matches PLAN.md's "no allocation in the
    node loop" rule. Mirrors the search's own board.push()/board.pop() calls
    one-for-one: call push() with the PRE-move board immediately before
    board.push(move), and pop() immediately after board.pop().
    """

    def __init__(self) -> None:
        self._stack = np.zeros((_MAX_STACK, HIDDEN1), dtype=np.float32)
        self._ply = 0

    def set_root(self, board: chess.Board) -> None:
        self._ply = 0
        self._stack[0] = accumulator_from_scratch(board)

    def push(self, board: chess.Board, move: chess.Move) -> None:
        deltas = move_deltas(board, move)
        stm_sign = 1 if board.turn == chess.WHITE else -1  # turn flips after this push
        parent = self._stack[self._ply]
        self._ply += 1
        child = self._stack[self._ply]
        child[:] = parent
        for idx, sign in deltas:
            if sign > 0:
                child += W1[:, idx]
            else:
                child -= W1[:, idx]
        child += stm_sign * W1[:, _STM_FEATURE]

    def pop(self) -> None:
        self._ply -= 1

    def current(self) -> npt.NDArray[np.float32]:
        return self._stack[self._ply]

    def evaluate(self) -> float:
        return forward(self.current())

    def evaluate_cp(self, board: chess.Board) -> int:
        """Centipawns, relative to the side to move -- the same contract
        search.py already expects from evaluate.evaluate(board)."""
        cp = score_to_centipawns(self.evaluate())
        return cp if board.turn == chess.WHITE else -cp

    def push_bb(self, bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64], code: int) -> None:
        """Same contract as push(), for the jitted (bb, state) substrate current
        main/search.py runs on: call with bb/state STILL in their pre-move state,
        immediately before movegen._make(bb, state, code)."""
        deltas = move_deltas_bb(bb, state, code)
        stm_sign = 1 if int(state[0]) == 0 else -1  # turn flips after this push
        parent = self._stack[self._ply]
        self._ply += 1
        child = self._stack[self._ply]
        child[:] = parent
        for idx, sign in deltas:
            if sign > 0:
                child += W1[:, idx]
            else:
                child -= W1[:, idx]
        child += stm_sign * W1[:, _STM_FEATURE]

    def evaluate_cp_bb(self, state: npt.NDArray[np.int64]) -> int:
        """evaluate_cp(), for the jitted (bb, state) substrate: state[0] == 0
        means White to move (movegen's own convention), same as board.turn ==
        chess.WHITE."""
        cp = score_to_centipawns(self.evaluate())
        return cp if int(state[0]) == 0 else -cp
