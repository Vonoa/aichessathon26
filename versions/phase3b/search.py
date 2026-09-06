"""Negamax with alpha-beta, iterative deepening, a transposition table, and a time budget.

Phase 3b adds the transposition table: a search result is cached by position, so a
position reached by a different move order is not re-searched, and the best move from the
last iteration is tried first on the next. The table is cleared each move for now; Phase 4
makes it persistent and fixed-size and adds the rest of the pruning stack. Phase 3e
replaces the python-chess move loop with a jitted generator.
"""

import os
import time
from collections.abc import Hashable

import chess

from evaluate import evaluate

MATE = 1_000_000
_MATE_THRESHOLD = MATE - 1_000  # a score past this is a forced mate
_SAFETY_MS = 300
_CHECK_INTERVAL = 255  # test the wall clock once per this many nodes
_MAX_DEPTH = 64

_EXACT, _LOWER, _UPPER = 0, 1, 2  # transposition-table bound kinds
_TT_MAX = 1_000_000  # entries; clear rather than grow past this

_DEBUG = os.environ.get("AGENT_DEBUG") == "1"

_nodes = 0
_last_depth = 0  # deepest fully completed pass of the last search; read by tools/bench.py
_seen: frozenset[Hashable] = frozenset()
_tt: dict[Hashable, tuple[int, int, int, chess.Move | None]] = {}


class _Timeout(Exception):
    """Raised inside the search when the per-move budget is spent."""


def search_move(
    board: chess.Board, time_left_ms: int, history: dict[Hashable, int] | None = None
) -> str:
    """Search the position and return the best move found, in UCI notation.

    Iterative deepening: each pass keeps the best move from the last *completed* depth,
    so whenever the budget runs out there is always a finished answer to return, and that
    move is tried first on the next, deeper pass.
    """
    global _nodes, _seen, _last_depth
    _nodes = 0
    _last_depth = 0
    _seen = frozenset(history) if history else frozenset()
    _tt.clear()

    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        return legal[0].uci()

    started = time.monotonic()
    deadline = started + _budget_s(board, time_left_ms)
    best = legal[0]
    for depth in range(1, _MAX_DEPTH + 1):
        try:
            move, score = _search_root(board, depth, deadline, best)
        except _Timeout:
            break
        best = move
        _last_depth = depth
        if _DEBUG:
            elapsed = (time.monotonic() - started) * 1000.0
            print(f"depth {depth:2d}  score {score:+7d}  nodes {_nodes:>9d}  {elapsed:6.0f} ms")
        if abs(score) >= _MATE_THRESHOLD:
            break  # forced mate found; a deeper search cannot improve on it
        if time.monotonic() >= deadline:
            break
    return best.uci()


def _budget_s(board: chess.Board, time_left_ms: int) -> float:
    """Spend a slice of the remaining clock on this move, keeping a watchdog margin."""
    moves_left = max(20, 50 - board.fullmove_number)
    share = time_left_ms / moves_left
    capped = min(share, float(time_left_ms - _SAFETY_MS))
    return max(capped, 10.0) / 1000.0


def _search_root(
    board: chess.Board, depth: int, deadline: float, first: chess.Move
) -> tuple[chess.Move, int]:
    moves = _ordered(board, list(board.legal_moves))
    if first in moves:
        moves.remove(first)
        moves.insert(0, first)

    best_move = moves[0]
    best_score = -MATE - 1
    alpha = -MATE - 1
    for move in moves:
        board.push(move)
        score = -_negamax(board, depth - 1, 1, -MATE - 1, -alpha, deadline)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
    return best_move, best_score


def _negamax(
    board: chess.Board, depth: int, ply: int, alpha: int, beta: int, deadline: float
) -> int:
    _tick(deadline)
    if board.is_fifty_moves():
        return 0
    if chess.popcount(board.occupied) <= 4 and board.is_insufficient_material():
        return 0

    moves = list(board.legal_moves)
    if not moves:
        return -MATE + ply if board.is_check() else 0

    # A repetition or an already-seen position is a draw even when it lands exactly on the
    # horizon, so this must run before the depth<=0 leaf return. The key is only computed
    # once halfmove_clock makes a repetition possible, so quiet leaves still skip it.
    key = board._transposition_key() if board.halfmove_clock >= 4 else None
    if key is not None and (board.is_repetition(2) or key in _seen):
        return 0
    if depth <= 0:
        return evaluate(board)

    if key is None:
        key = board._transposition_key()
    tt_move: chess.Move | None = None
    entry = _tt.get(key)
    if entry is not None:
        e_depth, e_value, e_flag, tt_move = entry
        if e_depth >= depth:
            if e_flag == _EXACT:
                return e_value
            if e_flag == _LOWER and e_value >= beta:
                return e_value
            if e_flag == _UPPER and e_value <= alpha:
                return e_value

    ordered = _ordered(board, moves)
    if tt_move is not None and tt_move in ordered:
        ordered.remove(tt_move)
        ordered.insert(0, tt_move)

    alpha_orig = alpha
    value = -MATE - 1
    best_move: chess.Move | None = None
    for move in ordered:
        board.push(move)
        score = -_negamax(board, depth - 1, ply + 1, -beta, -alpha, deadline)
        board.pop()
        if score > value:
            value = score
            best_move = move
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    # Do not cache mate scores: ours are measured from the root, so they are wrong down a
    # different path. (A 0 from an in-search repetition is path-dependent too, but harmless
    # while the table is cleared every move; revisit when the Phase 4 table persists.)
    if abs(value) < _MATE_THRESHOLD:
        if value <= alpha_orig:
            flag = _UPPER
        elif value >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        if len(_tt) >= _TT_MAX:
            _tt.clear()
        _tt[key] = (depth, value, flag, best_move)
    return value


def _ordered(board: chess.Board, moves: list[chess.Move]) -> list[chess.Move]:
    """Best-first: captures by MVV-LVA on piece-type ordinals, then queen promotions."""

    def score(move: chess.Move) -> int:
        value = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            value = 8 * victim - attacker
        if move.promotion is not None:
            value += 100 if move.promotion == chess.QUEEN else 10
        return value

    return sorted(moves, key=score, reverse=True)


def _tick(deadline: float) -> None:
    global _nodes
    _nodes += 1
    if _nodes % _CHECK_INTERVAL == 0 and time.monotonic() >= deadline:
        raise _Timeout
