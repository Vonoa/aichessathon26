"""
Correctness check for nnue_incremental.py: the incremental accumulator must
match a from-scratch recompute after every single push and pop, exactly (up
to float32 rounding). Run standalone: uv run python test_incremental.py

Random games exercise ordinary quiet moves and captures at volume; the
hand-built FENs below force the edge cases random play rarely reaches enough
times to trust -- castling (both sides, both colours), en passant (both
colours), and promotion (plain and capturing) -- since those are exactly the
cases move_deltas() has special-cased code for for, and a bug in any one of
them would silently corrupt every position reached after it in a real game.
"""

from __future__ import annotations

import random

import chess
import numpy as np
from features import fen_to_dense_vector
from nnue_incremental import Accumulator, accumulator_from_scratch, forward

_TOL = 1e-4


def reference_forward(fen: str) -> float:
    """Ground truth: the exact same computation train.py's model performs,
    from features.py's dense vector, independent of the accumulator code."""
    from nnue_incremental import B1, B2, B3, W1, W2, W3

    x = fen_to_dense_vector(fen)
    h1 = np.maximum(W1 @ x + B1, 0.0)
    h2 = np.maximum(W2 @ h1 + B2, 0.0)
    out = W3 @ h2 + B3
    return float(1.0 / (1.0 + np.exp(-out[0])))


def play_and_check(board: chess.Board, moves: list[chess.Move], label: str) -> None:
    acc = Accumulator()
    acc.set_root(board)
    ref_root = reference_forward(board.fen())
    if abs(ref_root - forward(acc.current())) > _TOL:
        raise AssertionError(f"{label}: forward mismatch at root, {board.fen()}")
    for i, move in enumerate(moves):
        assert move in board.legal_moves, (
            f"{label}: move {i} ({move.uci()}) illegal at {board.fen()}"
        )
        acc.push(board, move)
        board.push(move)
        expected = accumulator_from_scratch(board)
        got = acc.current()
        if not np.allclose(expected, got, atol=_TOL):
            raise AssertionError(
                f"{label}: mismatch after move {i} ({move.uci()}) at {board.fen()}"
            )
        ref = reference_forward(board.fen())
        out = forward(got)
        if abs(ref - out) > _TOL:
            raise AssertionError(
                f"{label}: forward mismatch after move {i} ({move.uci()}): {ref} vs {out}"
            )
    # pop back to the root and confirm we land exactly where set_root put us
    for _ in moves:
        acc.pop()
        board.pop()
    if not np.allclose(acc.current(), accumulator_from_scratch(board), atol=_TOL):
        raise AssertionError(f"{label}: accumulator wrong after popping back to root")


def random_games(n_games: int, max_plies: int, seed: int) -> None:
    rng = random.Random(seed)
    for g in range(n_games):
        board = chess.Board()
        moves = []
        for _ in range(max_plies):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            move = rng.choice(legal)
            moves.append(move)
            board.push(move)
        for _ in moves:
            board.pop()
        play_and_check(board, moves, f"random game {g}")


def edge_cases() -> None:
    cases = [
        ("white kingside castle", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
         ["e1g1"]),
        ("white queenside castle", "r3kbnr/pppqpppp/2np4/8/3P4/2N1B3/PPPQPPPP/R3KBNR w KQkq - 6 5",
         ["e1c1"]),
        ("black kingside castle", "rnbqk2r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 4 4",
         ["e8g8"]),
        ("black queenside castle", "r3kbnr/pppqpppp/2np4/8/3P4/2N1B3/PPPQPPPP/R3KBNR b KQkq - 6 5",
         ["e8c8"]),
        ("white en passant", "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
         ["e5d6"]),
        ("black en passant", "rnbqkbnr/ppp1p1pp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
         ["e4d3"]),
        ("promotion plain", "8/P6k/8/8/8/8/7K/8 w - - 0 1", ["a7a8q"]),
        ("promotion capturing", "n6k/1P6/8/8/8/8/7K/8 w - - 0 1", ["b7a8q"]),
    ]
    for label, fen, uci_moves in cases:
        board = chess.Board(fen)
        moves = [chess.Move.from_uci(u) for u in uci_moves]
        play_and_check(board, moves, label)


if __name__ == "__main__":
    edge_cases()
    print("edge cases passed (castling x4, en passant x2, promotion x2)")
    random_games(n_games=100, max_plies=80, seed=0)
    print("100 random games passed, every ply matched from-scratch recompute")
