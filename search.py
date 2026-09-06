"""Negamax with alpha-beta, iterative deepening, and a per-move time budget.

Phase 2 adds root-relative mate scores, repetition awareness, and the edge-case sweep.
Phase 3 replaces the python-chess move loop with a jitted generator. Phase 4 adds the
pruning stack: transposition table, killers/history, null-move, late-move reductions.
"""

import time

import chess

from evaluate import evaluate

MATE = 1_000_000
_SAFETY_MS = 300
_CHECK_INTERVAL = 1023  # test the wall clock once per this many nodes

_PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

_nodes = 0


class _Timeout(Exception):
    """Raised inside the search when the per-move budget is spent."""


def search_move(board: chess.Board, time_left_ms: int) -> str:
    """Search the position and return the best move found, in UCI notation.

    Iterative deepening: each pass keeps the best move from the last *completed* depth,
    so whenever the budget runs out there is always a finished answer to return.
    """
    global _nodes
    _nodes = 0
    deadline = time.monotonic() + _budget_s(board, time_left_ms)
    best = next(iter(board.legal_moves))
    for depth in range(1, 64):
        try:
            best = _search_root(board, depth, deadline)
        except _Timeout:
            break
        if time.monotonic() >= deadline:
            break
    return best.uci()


def _budget_s(board: chess.Board, time_left_ms: int) -> float:
    """Spend a slice of the remaining clock on this move, keeping a watchdog margin."""
    moves_left = max(20, 50 - board.fullmove_number)
    share = time_left_ms / moves_left
    capped = min(share, float(time_left_ms - _SAFETY_MS))
    return max(capped, 10.0) / 1000.0


def _search_root(board: chess.Board, depth: int, deadline: float) -> chess.Move:
    best_move = next(iter(board.legal_moves))
    best_score = -MATE - 1
    alpha = -MATE - 1
    for move in _ordered(board, list(board.legal_moves)):
        board.push(move)
        score = -_negamax(board, depth - 1, -MATE - 1, -alpha, deadline)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
    return best_move


def _negamax(board: chess.Board, depth: int, alpha: int, beta: int, deadline: float) -> int:
    _tick(deadline)
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0
    if depth <= 0:
        return evaluate(board)
    value = -MATE - 1
    for move in _ordered(board, moves):
        board.push(move)
        value = max(value, -_negamax(board, depth - 1, -beta, -alpha, deadline))
        board.pop()
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return value


def _ordered(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Captures first, ordered most-valuable-victim / least-valuable-attacker."""

    def score(move: chess.Move) -> int:
        if not board.is_capture(move):
            return 0
        victim = board.piece_type_at(move.to_square) or chess.PAWN
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        return 10 * _PIECE_VALUE.get(victim, 0) - _PIECE_VALUE.get(attacker, 0)

    return sorted(moves, key=score, reverse=True)


def _tick(deadline: float) -> None:
    global _nodes
    _nodes += 1
    if _nodes % _CHECK_INTERVAL == 0 and time.monotonic() >= deadline:
        raise _Timeout
