"""
Correctness check for nnue_incremental.py's push_bb()/move_deltas_bb(), the
path the live search actually uses now that search.py runs on movegen.py's
jitted (bb, state) substrate instead of chess.Board. Mirrors
test_incremental.py's structure and edge cases exactly, but drives the game
through movegen._gen_legal()/_make()/_unmake() -- the real production code
path -- instead of chess.Board.push(), so this tests what search.py actually
does, not an approximation of it. A parallel chess.Board is kept in lockstep
(via movegen.decode_move()) purely to get a ground-truth from-scratch
accumulator and ground-truth FEN for reference_forward(); it never drives the
moves itself.

Run standalone: uv run python test_incremental_bb.py
"""

from __future__ import annotations

import random

import chess
import numpy as np

import movegen
from features import fen_to_dense_vector
from nnue_incremental import Accumulator, accumulator_from_scratch, forward

_TOL = 1e-4


def reference_forward(fen: str) -> float:
    from nnue_incremental import B1, B2, B3, W1, W2, W3

    x = fen_to_dense_vector(fen)
    h1 = np.maximum(W1 @ x + B1, 0.0)
    h2 = np.maximum(W2 @ h1 + B2, 0.0)
    out = W3 @ h2 + B3
    return float(1.0 / (1.0 + np.exp(-out[0])))


def _find_code(bb: np.ndarray, state: np.ndarray, uci: str) -> int:
    """The move codes movegen._gen_legal produces aren't looked up by UCI
    anywhere in movegen.py -- this scans the legal list and decodes each to
    find the one matching the requested move, same as a test harness would
    for any move-code-based engine."""
    buf = np.zeros(256, dtype=np.int32)
    n = movegen._gen_legal(bb, state, buf)
    target = chess.Move.from_uci(uci)
    for i in range(n):
        code = int(buf[i])
        if movegen.decode_move(code) == target:
            return code
    raise AssertionError(f"move {uci} not found in legal moves for state {state}")


def play_and_check(start_fen: str, ucis: list[str], label: str) -> None:
    board = chess.Board(start_fen)
    bb, state = movegen.encode(board)
    acc = Accumulator()
    acc.set_root(board)
    ref_root = reference_forward(board.fen())
    if abs(ref_root - forward(acc.current())) > _TOL:
        raise AssertionError(f"{label}: forward mismatch at root, {board.fen()}")

    codes = []
    for i, uci in enumerate(ucis):
        code = _find_code(bb, state, uci)
        codes.append(code)
        acc.push_bb(bb, state, code)
        movegen._make(bb, state, code)
        board.push(chess.Move.from_uci(uci))

        expected = accumulator_from_scratch(board)
        got = acc.current()
        if not np.allclose(expected, got, atol=_TOL):
            raise AssertionError(f"{label}: mismatch after move {i} ({uci}) at {board.fen()}")
        ref = reference_forward(board.fen())
        out = forward(got)
        if abs(ref - out) > _TOL:
            raise AssertionError(
                f"{label}: forward mismatch after move {i} ({uci}): {ref} vs {out}"
            )
        # bb/state must also exactly match a fresh encode() of the board python-chess
        # arrived at independently -- catches a move_deltas_bb bug that happens to
        # cancel out in the accumulator's float weights but not in board state.
        bb_ref, state_ref = movegen.encode(board)
        if not np.array_equal(bb, bb_ref) or not np.array_equal(state, state_ref):
            raise AssertionError(f"{label}: bb/state diverged from board after move {i} ({uci})")

    # _unmake needs the undo word _make returned, which push_bb's loop above never kept --
    # replay codes on a fresh (bb2, state2) to collect them, then unmake in reverse.
    undos = []
    bb2, state2 = movegen.encode(chess.Board(start_fen))
    for code in codes:
        undos.append(movegen._make(bb2, state2, code))
    for code, undo in zip(reversed(codes), reversed(undos)):
        movegen._unmake(bb2, state2, code, undo)
        acc.pop()
    if not np.array_equal(bb2, movegen.encode(chess.Board(start_fen))[0]):
        raise AssertionError(f"{label}: bb wrong after unmaking back to root")
    if not np.allclose(acc.current(), accumulator_from_scratch(chess.Board(start_fen)), atol=_TOL):
        raise AssertionError(f"{label}: accumulator wrong after popping back to root")


def random_games(n_games: int, max_plies: int, seed: int) -> None:
    rng = random.Random(seed)
    for g in range(n_games):
        board = chess.Board()
        bb, state = movegen.encode(board)
        ucis = []
        for _ in range(max_plies):
            buf = np.zeros(256, dtype=np.int32)
            n = movegen._gen_legal(bb, state, buf)
            if n == 0:
                break
            code = int(buf[rng.randrange(n)])
            ucis.append(movegen.decode_move(code).uci())
            movegen._make(bb, state, code)
        play_and_check(chess.Board().fen(), ucis, f"random game {g}")


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
    for label, fen, ucis in cases:
        play_and_check(fen, ucis, label)


if __name__ == "__main__":
    edge_cases()
    print("bb edge cases passed (castling x4, en passant x2, promotion x2)")
    random_games(n_games=100, max_plies=80, seed=0)
    print("100 bb-driven random games passed, every ply matched from-scratch recompute")
