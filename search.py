"""Negamax with alpha-beta, iterative deepening, a transposition table, and a time budget.

Phase 3b adds the transposition table: a search result is cached by position, so a
position reached by a different move order is not re-searched, and the best move from the
last iteration is tried first on the next. The table is cleared each move for now; Phase 4
makes it persistent and fixed-size and adds the rest of the pruning stack. Phase 3e
replaces the python-chess move loop with a jitted generator.

Phase 4a adds quiescence search: at the horizon, keep searching captures (and check
evasions) until the position is quiet before calling evaluate(), so the score is never
read in the middle of an exchange.

Phase 4b adds killer moves and a history heuristic: a quiet move that caused a beta
cutoff is tried early in sibling nodes (killer, per ply) and its from/to square pair
accrues a score that ranks the remaining quiet moves. Both reset each move.

The engine is deterministic by construction: no RNG is imported, move ordering is a
stable sort over python-chess's fixed generation order, and ties are broken by first-seen.
The same position and clock always produce the same move.
"""

import os
import time
from collections.abc import Hashable

import chess

from evaluate import evaluate

MATE = 1_000_000
_MATE_THRESHOLD = MATE - 1_000  # a score past this is a forced mate
_RESERVE_MS = 500  # keep at least this on the clock; the watchdog does not forgive
_CHECK_INTERVAL = 255  # test the wall clock once per this many nodes
_MAX_DEPTH = 64
_QS_MAX_PLY = _MAX_DEPTH + 32  # hard cap on quiescence recursion, a safety net

_EXACT, _LOWER, _UPPER = 0, 1, 2  # transposition-table bound kinds
_TT_MAX = 1_000_000  # entries; clear rather than grow past this

# Move-ordering score bands: captures and promotions on top, then the two killer slots
# for this ply, then quiet moves ranked by the history heuristic (well below these).
_CAPTURE_BASE = 10_000_000
_KILLER_0 = 9_000_000
_KILLER_1 = 8_000_000
_KILLER_SLOTS = (_MAX_DEPTH + 1) * 2

_DEBUG = os.environ.get("AGENT_DEBUG") == "1"

_nodes = 0
_last_depth = 0  # deepest fully completed pass of the last search; read by tools/bench.py
_seen: frozenset[Hashable] = frozenset()
_tt: dict[Hashable, tuple[int, int, int, chess.Move | None]] = {}
_killers: list[chess.Move | None] = [None] * _KILLER_SLOTS  # two per ply, flat: ply*2, ply*2+1
_hist: list[int] = [0] * 4096  # quiet-move cutoff counts, indexed from_square*64 + to_square


class _Timeout(Exception):
    """Raised inside the search when the per-move budget is spent."""


def search_move(
    board: chess.Board,
    time_left_ms: int,
    history: dict[Hashable, int] | None = None,
    increment_ms: float = 0.0,
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
    _killers[:] = [None] * _KILLER_SLOTS
    _hist[:] = [0] * 4096

    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        return legal[0].uci()

    started = time.monotonic()
    deadline = started + _budget_s(board, time_left_ms, increment_ms)
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


def _budget_s(board: chess.Board, time_left_ms: int, increment_ms: float) -> float:
    """Time to spend on this move, in seconds.

    A sustainable share of the clock plus half the increment (each move refills the clock
    by the increment, so it is time to spend, not hoard). Assume the game still has a fair
    number of moves left - never divide by fewer than 30 - then never commit more than a
    third of the clock to one move and always leave a reserve. Floored at 10 ms.
    """
    moves_left = max(30, 56 - board.fullmove_number)
    budget_ms = time_left_ms / moves_left + 0.5 * increment_ms
    budget_ms = min(budget_ms, time_left_ms / 3.0, float(time_left_ms - _RESERVE_MS))
    return max(budget_ms, 10.0) / 1000.0


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
        return _qsearch(board, ply, alpha, beta, deadline)

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

    ordered = _ordered(board, moves, ply)
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
            if not board.is_capture(move) and move.promotion is None:
                _record_cutoff(move, ply, depth)
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


def _qsearch(board: chess.Board, ply: int, alpha: int, beta: int, deadline: float) -> int:
    """Search only captures (all evasions when in check) until the position is quiet.

    Without this the evaluation is read mid-exchange - "up a queen" one ply before the
    recapture - and is wrong. The stand-pat score assumes the side to move can hold at
    least the static eval, which fails only in zugzwang and is a standard trade-off.
    """
    _tick(deadline)
    if ply >= _QS_MAX_PLY:
        return evaluate(board)

    if board.is_check():
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply
        best = -MATE - 1
    else:
        best = evaluate(board)
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion is not None]

    for move in _ordered(board, moves):
        board.push(move)
        score = -_qsearch(board, ply + 1, -beta, -alpha, deadline)
        board.pop()
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _ordered(board: chess.Board, moves: list[chess.Move], ply: int = -1) -> list[chess.Move]:
    """Best-first: captures/promotions by MVV-LVA, then this ply's killer moves, then quiet
    moves by history score. The caller places any transposition-table move ahead of all.
    """
    base = ply * 2
    if 0 <= base < _KILLER_SLOTS - 1:
        killer0, killer1 = _killers[base], _killers[base + 1]
    else:
        killer0 = killer1 = None

    def score(move: chess.Move) -> int:
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            attacker = board.piece_type_at(move.from_square) or chess.PAWN
            promo = 0
            if move.promotion == chess.QUEEN:
                promo = 100
            elif move.promotion is not None:
                promo = 10
            return _CAPTURE_BASE + 8 * victim - attacker + promo
        if move.promotion is not None:
            return _CAPTURE_BASE + (100 if move.promotion == chess.QUEEN else 10)
        if move == killer0:
            return _KILLER_0
        if move == killer1:
            return _KILLER_1
        return _hist[move.from_square * 64 + move.to_square]

    return sorted(moves, key=score, reverse=True)


def _record_cutoff(move: chess.Move, ply: int, depth: int) -> None:
    """A quiet move caused a beta cutoff: remember it as a killer for this ply and add to
    its history score, weighted by depth so deep cutoffs count for more."""
    base = ply * 2
    if 0 <= base < _KILLER_SLOTS - 1 and _killers[base] != move:
        _killers[base + 1] = _killers[base]
        _killers[base] = move
    _hist[move.from_square * 64 + move.to_square] += depth * depth


def _tick(deadline: float) -> None:
    global _nodes
    _nodes += 1
    if _nodes % _CHECK_INTERVAL == 0 and time.monotonic() >= deadline:
        raise _Timeout
