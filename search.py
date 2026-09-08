"""Negamax with alpha-beta, iterative deepening, a transposition table, and a time budget.

Phase 3b adds the transposition table: a search result is cached by position, so a
position reached by a different move order is not re-searched, and the best move from the
last iteration is tried first on the next. The table is cleared each move for now; Phase 4
makes it persistent and fixed-size and adds the rest of the pruning stack. Phase 3e
replaces the python-chess move loop with a jitted generator.

Phase 4a adds quiescence search: at the horizon, keep searching captures (and check
evasions) until the position is quiet before calling evaluate(), so the score is never
read in the middle of an exchange. It also follows one ply of quiet checks past the
horizon, so a forcing shot (knight fork with check, back-rank skewer) is not missed.

Phase 4b adds killer moves and a history heuristic: a quiet move that caused a beta
cutoff is tried early in sibling nodes (killer, per ply) and its from/to square pair
accrues a score that ranks the remaining quiet moves. Both reset each move.

Phase 5f adds contempt: every draw path scores _CONTEMPT below equal from the root side's
point of view, so the engine only accepts a draw when it genuinely believes it is worse.

Check extension: a node that is in check is searched one ply deeper, so a forcing line
resolves before it is evaluated. Late-move reductions: quiet moves ordered late are
searched shallower first and only re-searched at full depth if they beat alpha.

The engine is deterministic by construction: no RNG is imported, move ordering is a
stable sort over python-chess's fixed generation order, and ties are broken by first-seen.
The same position and clock always produce the same move.
"""

import os
import time
from collections.abc import Hashable

import chess
import numpy as np
import numpy.typing as npt

from evaluate import evaluate

MATE = 1_000_000
_MATE_THRESHOLD = MATE - 1_000  # a score past this is a forced mate
_CONTEMPT = 25  # a draw scores this many centipawns below equal, so the engine plays to win
_RESERVE_MS = 500  # keep at least this on the clock; the watchdog does not forgive
_CHECK_INTERVAL = 255  # test the wall clock once per this many nodes
_MAX_DEPTH = 64
_QS_MAX_PLY = _MAX_DEPTH + 32  # hard cap on quiescence recursion, a safety net
_QS_CHECK_PLIES = 1  # follow non-capturing checks this many plies past the horizon
_QS_CHECK_CAP = 6  # at most this many quiet checking moves added per quiescence node

_EXACT, _LOWER, _UPPER = 0, 1, 2  # transposition-table bound kinds
_LMR_MIN_DEPTH = 3  # only reduce late moves with this much depth left
_LMR_MIN_MOVE = 3  # first this many moves at each node are searched at full depth

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
_killers: list[chess.Move | None] = [None] * _KILLER_SLOTS  # two per ply, flat: ply*2, ply*2+1
_hist: list[int] = [0] * 4096  # quiet-move cutoff counts, indexed from_square*64 + to_square

# Transposition table. Fixed-size and kept across moves within a game -- a fresh process
# per game resets it for free; tests call _reset_tt(). Two flat uint64 arrays, no
# per-entry Python objects: an unbounded dict here churns GC and eats the 2 GB budget
# (docs/PLAN.md, Phase 4). Open-addressed, one probe at slot = key & mask.
_TT_BITS = 22
_TT_SIZE = 1 << _TT_BITS  # 4,194,304 slots; 64 MB for the pair of arrays
_TT_MASK = _TT_SIZE - 1
_TT_VALUE_MAX = 30_000  # values outside +-this are not stored: they cannot fit the 16-bit
#                         field and a real evaluate() score never comes near it anyway.
#                         This also excludes mate scores (measured from the root, wrong
#                         down another path) without a separate check.
_tt_key: npt.NDArray[np.uint64] = np.zeros(_TT_SIZE, dtype=np.uint64)  # 0 == empty slot
_tt_data: npt.NDArray[np.uint64] = np.zeros(_TT_SIZE, dtype=np.uint64)
_tt_gen = 0  # bumped per search; a slot from an older generation is always replaceable

# _tt_data packs one entry into 64 bits:
#   bits  0-15  value, offset-encoded (value + 0x8000) so negatives round-trip
#   bits 16-23  depth (0..255; mate scores are never stored, so 16 bits of value is plenty)
#   bits 24-25  bound flag (_EXACT / _LOWER / _UPPER)
#   bits 26-41  best-move code: from | to << 6 | promo << 12  (0 == no move)
#   bits 42-57  generation, low 16 bits
_PROMO_CODE = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
_CODE_PROMO: tuple[int | None, ...] = (None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)


def _move_code(move: chess.Move | None) -> int:
    if move is None:
        return 0
    return move.from_square | (move.to_square << 6) | (_PROMO_CODE[move.promotion] << 12)


def _code_move(code: int) -> chess.Move | None:
    if code == 0:
        return None
    return chess.Move(code & 0x3F, (code >> 6) & 0x3F, _CODE_PROMO[(code >> 12) & 0x7])


def _reset_tt() -> None:
    """Wipe the table. The engine never calls this -- a game is one process and the table
    is meant to live the whole game. Tests that play several positions in one process do,
    the same way they reset agent._history / agent._clock.
    """
    global _tt_gen
    _tt_key.fill(0)
    _tt_data.fill(0)
    _tt_gen = 0


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
    global _nodes, _seen, _last_depth, _tt_gen
    _nodes = 0
    _last_depth = 0
    _seen = frozenset(history) if history else frozenset()
    _tt_gen = (_tt_gen + 1) & 0xFFFF  # keep the table; mark this search's entries fresh
    _killers[:] = [None] * _KILLER_SLOTS
    _hist[:] = [0] * 4096

    # Capture the move label now: an interrupted search unwinds through _Timeout without
    # popping, so board.fullmove_number / board.turn are unreliable once the loop ends.
    label = f"{board.fullmove_number}{'w' if board.turn else 'b'}"

    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        _log_move(label, legal[0], 0, 0, time_left_ms, time.monotonic())
        return legal[0].uci()

    started = time.monotonic()
    deadline = started + _budget_s(board, time_left_ms, increment_ms)
    # Ordered so that an interrupted first pass (severe time pressure) still returns the
    # best-looking move rather than whatever python-chess happened to generate first.
    best = _ordered(board, legal)[0]
    score = 0
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
    _log_move(label, best, score, _last_depth, time_left_ms, started)
    return best.uci()


def _log_move(
    label: str, move: chess.Move, score: int, depth: int, clock_ms: int, started: float
) -> None:
    """One compact line per move to stderr, kept in the platform's per-game log so a rated
    game can be diagnosed after the fact (real depth reached, nodes, time actually spent).
    """
    ms = (time.monotonic() - started) * 1000.0
    print(
        f"[{label}] {move.uci()} d{depth} score {score:+d} nodes {_nodes} "
        f"{ms:.0f}ms clock {clock_ms}",
        flush=True,
    )


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
    in_check = board.is_check()
    if in_check and ply < _MAX_DEPTH:
        depth += 1  # check extension: let a forcing line resolve before we evaluate it

    if board.is_fifty_moves():
        return _draw_score(ply)
    if chess.popcount(board.occupied) <= 4 and board.is_insufficient_material():
        return _draw_score(ply)

    moves = list(board.legal_moves)
    if not moves:
        return -MATE + ply if in_check else _draw_score(ply)

    # A repetition or an already-seen position is a draw even when it lands exactly on the
    # horizon, so this must run before the depth<=0 leaf return. The key is only computed
    # once halfmove_clock makes a repetition possible, so quiet leaves still skip it.
    tkey = board._transposition_key() if board.halfmove_clock >= 4 else None
    if tkey is not None and (board.is_repetition(2) or tkey in _seen):
        return _draw_score(ply)
    if depth <= 0:
        return _qsearch(board, ply, alpha, beta, deadline)

    if tkey is None:
        tkey = board._transposition_key()
    key64 = hash(tkey) & 0xFFFFFFFFFFFFFFFF or 1  # 0 is the empty-slot marker
    slot = key64 & _TT_MASK
    tt_move: chess.Move | None = None
    if int(_tt_key[slot]) == key64:
        data = int(_tt_data[slot])
        e_depth = (data >> 16) & 0xFF
        e_flag = (data >> 24) & 0x3
        tt_move = _code_move((data >> 26) & 0xFFFF)
        if e_depth >= depth:
            e_value = (data & 0xFFFF) - 0x8000
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
    for move_index, move in enumerate(ordered):
        quiet = move.promotion is None and not board.is_capture(move)
        board.push(move)

        # Late-move reduction: quiet moves ordered late are probably bad, so search them
        # shallower first; if one beats alpha anyway, re-search it at full depth.
        reduce = (
            quiet
            and not in_check
            and depth >= _LMR_MIN_DEPTH
            and move_index >= _LMR_MIN_MOVE
            and not board.is_check()
        )
        if reduce:
            r = 2 if move_index >= _LMR_MIN_MOVE + 3 else 1
            score = -_negamax(board, depth - 1 - r, ply + 1, -beta, -alpha, deadline)
            if score > alpha:
                score = -_negamax(board, depth - 1, ply + 1, -beta, -alpha, deadline)
        else:
            score = -_negamax(board, depth - 1, ply + 1, -beta, -alpha, deadline)

        board.pop()
        if score > value:
            value = score
            best_move = move
        alpha = max(alpha, value)
        if alpha >= beta:
            if quiet:
                _record_cutoff(move, ply, depth)
            break

    # Mate scores are not stored (see _TT_VALUE_MAX): measured from the root, they are
    # wrong down a different path. A 0 from an in-search repetition is mildly
    # path-dependent too, but the generation stamp below refreshes such entries within a
    # move or two and contempt keeps the error to +-_CONTEMPT -- a standard trade-off for
    # a persistent table.
    if abs(value) < _TT_VALUE_MAX:
        if value <= alpha_orig:
            flag = _UPPER
        elif value >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        prev = int(_tt_data[slot])
        # Replace unless the slot already holds a deeper result for THIS position from
        # THIS search; a stale-generation or different-position slot is always taken.
        if (
            int(_tt_key[slot]) != key64
            or ((prev >> 42) & 0xFFFF) != _tt_gen
            or ((prev >> 16) & 0xFF) <= depth
        ):
            _tt_key[slot] = key64
            _tt_data[slot] = (
                ((value + 0x8000) & 0xFFFF)
                | (min(depth, 255) << 16)
                | (flag << 24)
                | (_move_code(best_move) << 26)
                | ((_tt_gen & 0xFFFF) << 42)
            )
    return value


def _qsearch(
    board: chess.Board, ply: int, alpha: int, beta: int, deadline: float, qply: int = 0
) -> int:
    """Search captures, promotions, and - for the first _QS_CHECK_PLIES plies past the
    horizon - quiet checks, until the position is quiet. All evasions when in check.

    Without this the evaluation is read mid-exchange - "up a queen" one ply before the
    recapture - and is wrong. The stand-pat score assumes the side to move can hold at
    least the static eval, which fails only in zugzwang and is a standard trade-off.

    Following one ply of quiet checks catches the forcing shot - a knight fork with check,
    a back-rank skewer - that a captures-only qsearch walks past. It can only raise the
    score (a bad check just scores low and is ignored), so stand-pat stays sound; qply
    gates the extra generation only, not the recursion depth.
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
        want_checks = qply < _QS_CHECK_PLIES
        captures: list[chess.Move] = []
        checks: list[chess.Move] = []
        for m in board.legal_moves:
            if board.is_capture(m) or m.promotion is not None:
                captures.append(m)
            elif want_checks and len(checks) < _QS_CHECK_CAP and board.gives_check(m):
                checks.append(m)
        moves = captures + checks

    for move in _ordered(board, moves):
        board.push(move)
        score = -_qsearch(board, ply + 1, -beta, -alpha, deadline, qply + 1)
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


def _draw_score(ply: int) -> int:
    """Contempt: a draw is worth _CONTEMPT below equal from the root side's point of view.
    ply is even on the root side's turn, odd on the opponent's; the sign flips so the value
    stays consistent through negamax's per-ply negation (and per-position, since a position
    always recurs at the same ply parity).
    """
    return -_CONTEMPT if ply % 2 == 0 else _CONTEMPT


def _tick(deadline: float) -> None:
    global _nodes
    _nodes += 1
    if _nodes % _CHECK_INTERVAL == 0 and time.monotonic() >= deadline:
        raise _Timeout
