"""Negamax with alpha-beta, iterative deepening, a transposition table, and a time budget.

The search runs on the jitted bitboard board from movegen.py: `(bb, state)` numpy arrays,
`movegen._gen_legal` / `_make` / `_unmake` in place of `board.legal_moves` / `push` / pop,
and `movegen._zobrist` for the transposition and repetition keys. A `chess.Board` is only
touched at the root -- to parse the FEN, probe Syzygy, and format the UCI reply.

Phase 4a: quiescence search -- at the horizon keep searching captures, promotions and one
ply of quiet checks until the position is quiet, so the eval is never read mid-exchange.
Phase 4b: killer moves + a history heuristic order the quiet moves, reset each move.
Phase 5f: contempt -- every draw path scores _CONTEMPT below equal from the root side.
Check extension: a node in check is searched one ply deeper. Late-move reductions: quiet
moves ordered late are scouted shallower first. Principal variation search + aspiration
windows narrow the search around the previous iteration's score.

Deterministic by construction: no RNG in the search, move ordering is a stable sort over
the generator's fixed order, ties broken by first-seen. Same position + clock -> same move.
"""

import math
import os
import time

import chess
import chess.syzygy
import numpy as np
import numpy.typing as npt

import movegen
from evaluate import _evaluate_jit

MATE = 1_000_000
_MATE_THRESHOLD = MATE - 1_000  # a score past this is a forced mate
_CONTEMPT = 40  # diag-21c: harder draw aversion, measured not shipped
_RESERVE_MS = 500  # keep at least this on the clock; the watchdog does not forgive
_CHECK_INTERVAL = 255  # test the wall clock once per this many nodes
_MAX_DEPTH = 64
_QS_MAX_PLY = _MAX_DEPTH + 32  # hard cap on quiescence recursion, a safety net
_QS_CHECK_PLIES = 1  # follow non-capturing checks this many plies past the horizon
_QS_CHECK_CAP = 6  # at most this many quiet checking moves added per quiescence node
_MAX_PLY = _MAX_DEPTH + _QS_MAX_PLY + 16  # size of the per-ply scratch buffers

_EXACT, _LOWER, _UPPER = 0, 1, 2  # transposition-table bound kinds
_LMR_MIN_DEPTH = 3  # only reduce late moves with this much depth left
_LMR_MIN_MOVE = 3  # first this many moves at each node are searched at full depth
_ASPIRATION = 40  # centipawns; the half-width of the first window around the last score
_NMP_MIN_DEPTH = 3  # only try a null move with at least this much depth left
_SEE_QS_MARGIN = 90  # quiescence keeps a capture unless SEE is worse than -this
_RFP_MAX_DEPTH = 6  # reverse-futility pruning only near the frontier
_RFP_MARGIN = 75  # centipawns per ply the static eval must clear beta by
_FUTILITY_MAX_DEPTH = 2  # futility-prune quiet moves only at the frontier
_FUTILITY_MARGIN = 120  # centipawns per ply a quiet move must come within alpha

# Fifty-move-rule damping. Once a line has gone _FIFTY_FADE_START plies with no pawn move
# or capture, the position is drifting toward the draw claim, so _eval_bb fades the score
# ~1% per further quiet ply (floor 0.20x). This makes the search treat aimless shuffling
# of a won position as a loss of value and steer toward a zeroing move (pawn break, trade)
# while the edge is still real. Below the threshold the eval is untouched. (Round 97: a
# +400 rook-up game shuffled straight into a threefold; the eval bled +399 -> -25 over
# ~18 quiet moves with no course correction.)
# diag-21: lowered 30 -> 16. Round 104 bled a +230 into a threefold *through* trades --
# a capture every 2-3 moves kept the halfmove clock under ~15, so the fade never engaged
# until the position was already dead. 16 plies of quiet play is enough to start dimming.
_FIFTY_FADE_START = 16

# Late-move reduction depth by [depth][move_index] (both clamped to 63). The classic
# log formula -- reduce more the deeper the search and the later the move. The call
# site shaves one off on the PV and for killers, floors at 0, and caps it so the
# reduced re-search is always at least one ply (never straight into quiescence).
_LMR_TABLE: list[list[int]] = [[0] * 64 for _ in range(64)]
for _d in range(1, 64):
    for _m in range(1, 64):
        _LMR_TABLE[_d][_m] = int(0.8 + math.log(_d) * math.log(_m) / 2.5)

# Syzygy endgame tablebases. When the board is down to this few men and ./syzygy holds
# the files, search_move picks the move straight from the tables (WDL for the outcome,
# DTZ for the fastest conversion) and skips the search entirely -- perfect endgame play.
# A missing / empty dir just leaves _tablebase None and nothing changes. The package step
# must be told to ship the folder:  uv run python -m harness.package --include syzygy
_TB_MAX_PIECES = 5  # use the tables at this many men or fewer (3-4-5 syzygy; we ship 3-man)
_SYZYGY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "syzygy")
_tablebase: chess.syzygy.Tablebase | None = None
try:
    if os.path.isdir(_SYZYGY_DIR) and any(f.endswith(".rtbw") for f in os.listdir(_SYZYGY_DIR)):
        _tablebase = chess.syzygy.open_tablebase(_SYZYGY_DIR)
        print(f"syzygy: loaded from {_SYZYGY_DIR}", flush=True)
except Exception as _tb_exc:  # a tablebase must never break import
    print(f"syzygy: disabled ({type(_tb_exc).__name__}: {_tb_exc})", flush=True)
    _tablebase = None

# Move-ordering score bands: captures and promotions on top, then the two killer slots
# for this ply, then quiet moves ranked by the history heuristic (well below these). A
# transposition-table move is spliced to the very front after the sort.
_CAPTURE_BASE = 10_000_000
_KILLER_0 = 9_000_000
_KILLER_1 = 8_000_000
_KILLER_SLOTS = (_MAX_DEPTH + 1) * 2

_DEBUG = os.environ.get("AGENT_DEBUG") == "1"

_nodes = 0
_last_depth = 0  # deepest fully completed pass of the last search; read by tools/bench.py
_seen: frozenset[int] = frozenset()  # zobrist keys of positions already seen this game
_killers: list[int] = [0] * _KILLER_SLOTS  # two move codes per ply, flat: ply*2, ply*2+1
_hist: list[int] = [0] * 4096  # quiet-move cutoff counts, indexed from_square*64 + to_square

# Per-ply scratch: one legal-move buffer per recursion level, one occupancy triple, and
# the chain of position keys down the current line (for repetition detection).
_MBUF: npt.NDArray[np.int32] = np.empty((_MAX_PLY, 256), dtype=np.int32)
_OCC3: npt.NDArray[np.uint64] = np.empty(3, dtype=np.uint64)
_PATH: npt.NDArray[np.uint64] = np.zeros(_MAX_PLY, dtype=np.uint64)  # zobrist keys, full 64-bit

# Transposition table. Fixed-size and kept across moves within a game -- a fresh process
# per game resets it for free; tests call _reset_tt(). Two flat uint64 arrays, no
# per-entry Python objects: an unbounded dict here churns GC and eats the 2 GB budget
# (docs/PLAN.md, Phase 4). Open-addressed, one probe at slot = key & mask.
_TT_BITS = 24
_TT_SIZE = 1 << _TT_BITS  # 16,777,216 slots; 256 MB for the pair (of a 2 GB budget)
_TT_MASK = _TT_SIZE - 1
_TT_VALUE_MAX = 30_000  # values outside +-this are not stored: they cannot fit the 16-bit
#                         field and a real eval score never comes near it anyway. This
#                         also excludes mate scores (measured from the root, wrong down
#                         another path) without a separate check.
_tt_key: npt.NDArray[np.uint64] = np.zeros(_TT_SIZE, dtype=np.uint64)  # 0 == empty slot
_tt_data: npt.NDArray[np.uint64] = np.zeros(_TT_SIZE, dtype=np.uint64)
_tt_gen = 0  # bumped per search; a slot from an older generation is always replaceable

# _tt_data packs one entry into 64 bits:
#   bits  0-15  value, offset-encoded (value + 0x8000) so negatives round-trip
#   bits 16-23  depth (0..255; mate scores are never stored, so 16 bits of value is plenty)
#   bits 24-25  bound flag (_EXACT / _LOWER / _UPPER)
#   bits 26-40  best-move code: from | to << 6 | promo << 12  (0 == no move)
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


# --- board-state helpers on (bb, state) --------------------------------------------


def _eval_bb(bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64]) -> int:
    """Static eval from the side-to-move's point of view -- the jitted eval read straight
    off the bitboard board (validated identical to evaluate.evaluate)."""
    movegen._occ3(bb, _OCC3)
    wk = movegen._king_sq(bb, 0)
    bk = movegen._king_sq(bb, 1)
    score = int(_evaluate_jit(bb, _OCC3, wk, bk, int(state[0]) == 0))
    half = int(state[3])
    if half > _FIFTY_FADE_START:
        over = half - _FIFTY_FADE_START  # quiet plies past the threshold
        score = score * max(20, 100 - over) // 100  # ~1% eval per quiet ply, floor 0.20x
    return score


def _in_check(bb: npt.NDArray[np.uint64], state: npt.NDArray[np.int64]) -> bool:
    turn = int(state[0])
    movegen._occ3(bb, _OCC3)
    return movegen._attacked_by(bb, _OCC3[2], movegen._king_sq(bb, turn), 1 - turn)


def _insufficient(bb: npt.NDArray[np.uint64]) -> bool:
    """K vs K, K+minor vs K -- the common insufficient-material draws. Same-colour KBvKB
    and KNNvK are left to the eval / repetition (rare, never a blunder)."""
    if int(bb[0, 0] | bb[1, 0] | bb[0, 3] | bb[1, 3] | bb[0, 4] | bb[1, 4]):
        return False
    minors = int(bb[0, 1] | bb[1, 1] | bb[0, 2] | bb[1, 2])
    return minors == (minors & -minors)  # zero or one bit set


def _repeats(key: int, ply: int) -> bool:
    """Has this exact key already appeared higher in the current line?"""
    return any(int(_PATH[j]) == key for j in range(ply))


def _has_non_pawn_material(bb: npt.NDArray[np.uint64], turn: int) -> bool:
    """True if `turn` has a knight, bishop, rook or queen -- null-move pruning is unsafe
    without one (in a king-and-pawns position, passing can be forced-best: zugzwang)."""
    return bool(int(bb[turn, 1] | bb[turn, 2] | bb[turn, 3] | bb[turn, 4]))


# --- iterative deepening at the root ---------------------------------------------


def search_move(
    board: chess.Board,
    time_left_ms: int,
    history: dict[int, int] | None = None,
    increment_ms: float = 0.0,
) -> str:
    """Search the position and return the best move found, in UCI notation.

    Iterative deepening: each pass keeps the best move from the last *completed* depth,
    so whenever the budget runs out there is always a finished answer to return.
    """
    global _nodes, _seen, _last_depth, _tt_gen
    _nodes = 0
    _last_depth = 0
    _seen = frozenset(int(k) for k in history) if history else frozenset()
    _tt_gen = (_tt_gen + 1) & 0xFFFF  # keep the table; mark this search's entries fresh
    _killers[:] = [0] * _KILLER_SLOTS
    _hist[:] = [0] * 4096

    label = f"{board.fullmove_number}{'w' if board.turn else 'b'}"

    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    if len(legal) == 1:
        _log_move(label, legal[0], 0, 0, time_left_ms, time.monotonic())
        return legal[0].uci()

    tb_move = _tb_root_move(board)
    if tb_move is not None:
        _log_move(label, chess.Move.from_uci(tb_move), 0, 0, time_left_ms, time.monotonic())
        return tb_move

    bb0, state0 = movegen.encode(board)
    turn = int(state0[0])
    enemy_occ = int(movegen._occ_of(bb0, 1 - turn))
    n = movegen._gen_legal(bb0, state0, _MBUF[0])
    started = time.monotonic()
    deadline = started + _budget_s(board, time_left_ms, increment_ms)
    # Ordered so an interrupted first pass still returns the best-looking move.
    best_code = int(_ordered(bb0, turn, enemy_occ, _MBUF[0][:n].tolist(), -1, 0)[0])
    score = 0
    for depth in range(1, _MAX_DEPTH + 1):
        # Aspiration: past the shallow passes, a narrow window around the last score.
        if depth <= 3:
            alpha, beta = -MATE - 1, MATE + 1
        else:
            alpha, beta = score - _ASPIRATION, score + _ASPIRATION
        try:
            move_code, score = _aspiration_search(
                bb0, state0, depth, deadline, best_code, alpha, beta
            )
        except _Timeout:
            break
        best_code = move_code
        _last_depth = depth
        if _DEBUG:
            elapsed = (time.monotonic() - started) * 1000.0
            print(f"depth {depth:2d}  score {score:+7d}  nodes {_nodes:>9d}  {elapsed:6.0f} ms")
        if abs(score) >= _MATE_THRESHOLD:
            break  # forced mate found; a deeper search cannot improve on it
        if time.monotonic() >= deadline:
            break
    best_move = movegen.decode_move(best_code)
    _log_move(label, best_move, score, _last_depth, time_left_ms, started)
    return best_move.uci()


def _aspiration_search(
    bb0: npt.NDArray[np.uint64],
    state0: npt.NDArray[np.int64],
    depth: int,
    deadline: float,
    first_code: int,
    alpha: int,
    beta: int,
) -> tuple[int, int]:
    """Run _search_root on a fresh copy of the root state; if the result falls outside
    (alpha, beta) widen that side to infinity and try once more."""
    while True:
        bb = bb0.copy()
        state = state0.copy()
        move_code, score = _search_root(bb, state, depth, deadline, first_code, alpha, beta)
        if score <= alpha and alpha > -MATE - 1:
            alpha = -MATE - 1
        elif score >= beta and beta < MATE + 1:
            beta = MATE + 1
        else:
            return move_code, score


def _search_root(
    bb: npt.NDArray[np.uint64],
    state: npt.NDArray[np.int64],
    depth: int,
    deadline: float,
    first_code: int,
    alpha: int,
    beta: int,
) -> tuple[int, int]:
    _PATH[0] = int(movegen._zobrist(bb, state))
    turn = int(state[0])
    enemy_occ = int(movegen._occ_of(bb, 1 - turn))
    n = movegen._gen_legal(bb, state, _MBUF[0])
    first_core = (first_code & 0x7FFF) if first_code else 0
    ordered = _ordered(bb, turn, enemy_occ, _MBUF[0][:n].tolist(), -1, first_core)

    best_code = int(ordered[0])
    best_score = -MATE - 1
    for i, raw in enumerate(ordered):
        code = int(raw)
        undo = movegen._make(bb, state, code)
        if i == 0:
            move_score = -_negamax(bb, state, depth - 1, 1, -beta, -alpha, deadline)
        else:
            move_score = -_negamax(bb, state, depth - 1, 1, -alpha - 1, -alpha, deadline)
            if alpha < move_score < beta:
                move_score = -_negamax(bb, state, depth - 1, 1, -beta, -alpha, deadline)
        movegen._unmake(bb, state, code, undo)
        if move_score > best_score:
            best_score = move_score
            best_code = code
        if move_score > alpha:
            alpha = move_score
        if alpha >= beta:
            break  # fail-high at the root; _aspiration_search widens and re-searches
    return best_code, best_score


def _negamax(
    bb: npt.NDArray[np.uint64],
    state: npt.NDArray[np.int64],
    depth: int,
    ply: int,
    alpha: int,
    beta: int,
    deadline: float,
) -> int:
    _tick(deadline)
    if ply >= _MAX_PLY - 1:
        return _eval_bb(bb, state)

    turn = int(state[0])
    movegen._occ3(bb, _OCC3)
    in_check = movegen._attacked_by(bb, _OCC3[2], movegen._king_sq(bb, turn), 1 - turn)
    if in_check and ply < _MAX_DEPTH:
        depth += 1  # check extension: let a forcing line resolve before we evaluate it

    if int(state[3]) >= 100:
        return _draw_score(ply)
    if int(_OCC3[2]).bit_count() <= 4 and _insufficient(bb):
        return _draw_score(ply)

    n = movegen._gen_legal(bb, state, _MBUF[ply])
    if n == 0:
        return -MATE + ply if in_check else _draw_score(ply)

    key = int(movegen._zobrist(bb, state))
    if int(state[3]) >= 4 and (key in _seen or _repeats(key, ply)):
        return _draw_score(ply)
    _PATH[ply] = key
    if depth <= 0:
        return _qsearch(bb, state, ply, alpha, beta, deadline, 0)

    key64 = key or 1  # 0 is the empty-slot marker
    slot = key64 & _TT_MASK
    tt_move = 0
    if int(_tt_key[slot]) == key64:
        data = int(_tt_data[slot])
        e_depth = (data >> 16) & 0xFF
        e_flag = (data >> 24) & 0x3
        tt_move = (data >> 26) & 0x7FFF
        if e_depth >= depth:
            e_value = (data & 0xFFFF) - 0x8000
            if e_flag == _EXACT:
                return e_value
            if e_flag == _LOWER and e_value >= beta:
                return e_value
            if e_flag == _UPPER and e_value <= alpha:
                return e_value

    # Static eval, shared by reverse-futility and null-move pruning below. Meaningless
    # in check (no stand-pat), so we skip it there and neither prune can fire.
    static_eval = 0 if in_check else _eval_bb(bb, state)

    # Reverse futility pruning: in a non-PV node near the frontier, if the static eval
    # clears beta by a generous depth-scaled margin, the opponent is very unlikely to
    # claw it back -- fail high now instead of searching. Held off near mate scores.
    if (
        not in_check
        and beta - alpha == 1
        and depth <= _RFP_MAX_DEPTH
        and abs(beta) < _MATE_THRESHOLD
        and static_eval - _RFP_MARGIN * depth >= beta
    ):
        return static_eval - _RFP_MARGIN * depth

    # Null-move pruning: hand the opponent a free move and search shallower; if we are
    # still >= beta even a tempo down, the real search would only confirm the cutoff.
    # Gated on the static eval already being >= beta (without it the null search is pure
    # overhead in every equal-or-worse position), not in check, not when beta is a mate
    # score, and not when the side to move has only pawns (zugzwang).
    if (
        not in_check
        and depth >= _NMP_MIN_DEPTH
        and abs(beta) < _MATE_THRESHOLD
        and _has_non_pawn_material(bb, turn)
        and static_eval >= beta
    ):
        r = 3 if depth >= 6 else 2
        old_turn, old_ep, old_half = int(state[0]), int(state[2]), int(state[3])
        state[0] = 1 - old_turn
        state[2] = -1
        state[3] = old_half + 1
        null_score = -_negamax(bb, state, depth - 1 - r, ply + 1, -beta, -beta + 1, deadline)
        state[0], state[2], state[3] = old_turn, old_ep, old_half
        if null_score >= beta:
            return beta if null_score >= _MATE_THRESHOLD else null_score

    enemy_occ = int(movegen._occ_of(bb, 1 - turn))
    ordered = _ordered(bb, turn, enemy_occ, _MBUF[ply][:n].tolist(), ply, tt_move)

    alpha_orig = alpha
    value = -MATE - 1
    best_code = 0
    for move_index, raw in enumerate(ordered):
        code = int(raw)
        to = (code >> 6) & 0x3F
        flag = (code >> 15) & 7
        is_cap = flag == 2 or ((enemy_occ >> to) & 1)
        quiet = ((code >> 12) & 7) == 0 and not is_cap
        undo = movegen._make(bb, state, code)

        if move_index == 0:
            # The principal variation: search it in full to establish a real bound.
            score = -_negamax(bb, state, depth - 1, ply + 1, -beta, -alpha, deadline)
        else:
            gives_check = _in_check(bb, state)  # opponent now to move

            # Futility pruning: at the frontier in a non-PV node, a quiet non-checking
            # move whose static eval is a full margin below alpha cannot raise it --
            # skip it. Held off in check and near mate scores.
            if (
                quiet
                and not gives_check
                and not in_check
                and beta - alpha == 1
                and depth <= _FUTILITY_MAX_DEPTH
                and abs(alpha) < _MATE_THRESHOLD
                and static_eval + _FUTILITY_MARGIN * depth <= alpha
            ):
                movegen._unmake(bb, state, code, undo)
                continue

            reduce = (
                quiet
                and not in_check
                and depth >= _LMR_MIN_DEPTH
                and move_index >= _LMR_MIN_MOVE
                and not gives_check
            )
            if reduce:
                # conditional expressions, not min(): a builtin call per reduced move
                # is ~15% of a cheap endgame node (see tools/bench.py rook endgame).
                di = depth if depth < 64 else 63
                mi = move_index if move_index < 64 else 63
                r = _LMR_TABLE[di][mi]
                if beta - alpha > 1:  # on the PV, reduce one less
                    r -= 1
                if r < 1:
                    r = 0
                elif r > depth - 2:
                    r = depth - 2
            else:
                r = 0
            score = -_negamax(bb, state, depth - 1 - r, ply + 1, -alpha - 1, -alpha, deadline)
            if score > alpha and (r > 0 or score < beta):
                score = -_negamax(bb, state, depth - 1, ply + 1, -beta, -alpha, deadline)

        movegen._unmake(bb, state, code, undo)
        if score > value:
            value = score
            best_code = code
        alpha = max(alpha, value)
        if alpha >= beta:
            if quiet:
                _record_cutoff(code, ply, depth)
            break

    if abs(value) < _TT_VALUE_MAX:
        if value <= alpha_orig:
            flag_out = _UPPER
        elif value >= beta:
            flag_out = _LOWER
        else:
            flag_out = _EXACT
        prev = int(_tt_data[slot])
        if (
            int(_tt_key[slot]) != key64
            or ((prev >> 42) & 0xFFFF) != _tt_gen
            or ((prev >> 16) & 0xFF) <= depth
        ):
            _tt_key[slot] = key64
            _tt_data[slot] = (
                ((value + 0x8000) & 0xFFFF)
                | (min(depth, 255) << 16)
                | (flag_out << 24)
                | ((best_code & 0x7FFF) << 26)
                | ((_tt_gen & 0xFFFF) << 42)
            )
    return value


def _qsearch(
    bb: npt.NDArray[np.uint64],
    state: npt.NDArray[np.int64],
    ply: int,
    alpha: int,
    beta: int,
    deadline: float,
    qply: int = 0,
) -> int:
    """Search captures, promotions, and - for the first _QS_CHECK_PLIES plies past the
    horizon - quiet checks, until the position is quiet. All evasions when in check.

    Without this the eval is read mid-exchange and is wrong. The stand-pat score assumes
    the side to move can hold at least the static eval (fails only in zugzwang). Following
    one ply of quiet checks catches a forcing shot a captures-only qsearch walks past --
    it can only raise the score, so stand-pat stays sound.
    """
    _tick(deadline)
    if ply >= _MAX_PLY - 1 or ply >= _QS_MAX_PLY:
        return _eval_bb(bb, state)

    turn = int(state[0])
    in_check = _in_check(bb, state)
    n = movegen._gen_legal(bb, state, _MBUF[ply])

    if in_check:
        if n == 0:
            return -MATE + ply
        best = -MATE - 1
        enemy_occ = int(movegen._occ_of(bb, 1 - turn))
        move_list: list[int] = list(_ordered(bb, turn, enemy_occ, _MBUF[ply][:n].tolist(), -1, 0))
    else:
        best = _eval_bb(bb, state)
        if best >= beta:
            return best
        if best > alpha:
            alpha = best
        want_checks = qply < _QS_CHECK_PLIES
        enemy_occ = int(movegen._occ_of(bb, 1 - turn))
        captures: list[int] = []
        checks: list[int] = []
        for i in range(n):
            code = int(_MBUF[ply][i])
            to = (code >> 6) & 0x3F
            frm = code & 0x3F
            flag = (code >> 15) & 7
            promo = (code >> 12) & 7
            is_cap = flag == 2 or ((enemy_occ >> to) & 1)
            if promo != 0:
                captures.append(code)  # promotions always resolved
            elif is_cap:
                victim = _pt_at(bb, 1 - turn, to) if flag != 2 else chess.PAWN
                attacker = _pt_at(bb, turn, frm)
                # capturing equal-or-up is structurally safe; only SEE the "capturing
                # down" moves, where a losing capture hides. Keep any capture giving check.
                if victim >= attacker or movegen._see(bb, turn, code) >= -_SEE_QS_MARGIN:
                    captures.append(code)
                else:
                    undo = movegen._make(bb, state, code)
                    if _in_check(bb, state):
                        captures.append(code)
                    movegen._unmake(bb, state, code, undo)
            elif want_checks and len(checks) < _QS_CHECK_CAP:
                undo = movegen._make(bb, state, code)
                if _in_check(bb, state):
                    checks.append(code)
                movegen._unmake(bb, state, code, undo)
        move_list = list(_ordered(bb, turn, enemy_occ, captures + checks, -1, 0))

    for raw in move_list:
        code = int(raw)
        undo = movegen._make(bb, state, code)
        score = -_qsearch(bb, state, ply + 1, -beta, -alpha, deadline, qply + 1)
        movegen._unmake(bb, state, code, undo)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best


def _ordered(
    bb: npt.NDArray[np.uint64],
    turn: int,
    enemy_occ: int,
    codes: list[int],
    ply: int,
    tt_move: int,
) -> list[int]:
    """Best-first ordering of move codes: TT move, then captures / promotions by MVV-LVA,
    then this ply's killers, then quiet moves by history score."""
    base = ply * 2
    if 0 <= base < _KILLER_SLOTS - 1:
        killer0, killer1 = _killers[base], _killers[base + 1]
    else:
        killer0 = killer1 = 0

    def score(code: int) -> int:
        frm = code & 0x3F
        to = (code >> 6) & 0x3F
        promo = (code >> 12) & 7
        flag = (code >> 15) & 7
        is_ep = flag == 2
        if is_ep or ((enemy_occ >> to) & 1):
            victim = chess.PAWN if is_ep else _pt_at(bb, 1 - turn, to)
            attacker = _pt_at(bb, turn, frm)
            promo_bonus = 100 if promo == 4 else (10 if promo else 0)
            return _CAPTURE_BASE + 8 * victim - attacker + promo_bonus
        if promo != 0:
            return _CAPTURE_BASE + (100 if promo == 4 else 10)
        core = code & 0x7FFF
        if core == killer0:
            return _KILLER_0
        if core == killer1:
            return _KILLER_1
        return _hist[frm * 64 + to]

    result = sorted(codes, key=score, reverse=True)
    if tt_move:
        for i, c in enumerate(result):
            if (c & 0x7FFF) == tt_move:
                result.insert(0, result.pop(i))
                break
    return result


def _pt_at(bb: npt.NDArray[np.uint64], colour: int, sq: int) -> int:
    """1..6 piece type of `colour` on `sq`, else PAWN (matches the old `or chess.PAWN`)."""
    b = 1 << sq
    for pt in range(6):
        if int(bb[colour, pt]) & b:
            return pt + 1
    return chess.PAWN


def _record_cutoff(code: int, ply: int, depth: int) -> None:
    """A quiet move caused a beta cutoff: remember it as a killer for this ply and add to
    its history score, weighted by depth so deep cutoffs count for more."""
    core = code & 0x7FFF
    base = ply * 2
    if 0 <= base < _KILLER_SLOTS - 1 and _killers[base] != core:
        _killers[base + 1] = _killers[base]
        _killers[base] = core
    _hist[(code & 0x3F) * 64 + ((code >> 6) & 0x3F)] += depth * depth


def _draw_score(ply: int) -> int:
    """Contempt: a draw is worth _CONTEMPT below equal from the root side's point of view.
    ply is even on the root side's turn, odd on the opponent's; the sign flips so the value
    stays consistent through negamax's per-ply negation.
    """
    return -_CONTEMPT if ply % 2 == 0 else _CONTEMPT


def _log_move(
    label: str, move: chess.Move, score: int, depth: int, clock_ms: int, started: float
) -> None:
    """One compact line per move to stderr, kept in the platform's per-game log so a rated
    game can be diagnosed after the fact."""
    ms = (time.monotonic() - started) * 1000.0
    print(
        f"[{label}] {move.uci()} d{depth} score {score:+d} nodes {_nodes} "
        f"{ms:.0f}ms clock {clock_ms}",
        flush=True,
    )


def _budget_s(board: chess.Board, time_left_ms: int, increment_ms: float) -> float:
    """Time to spend on this move, in seconds. A sustainable share of the clock plus half
    the increment, capped at a third of the clock, always leaving a reserve, floored at
    10 ms. Assume at least 30 moves left so we do not drain the clock in a long game."""
    moves_left = max(30, 56 - board.fullmove_number)
    budget_ms = time_left_ms / moves_left + 0.5 * increment_ms
    budget_ms = min(budget_ms, time_left_ms / 3.0, float(time_left_ms - _RESERVE_MS))
    return max(budget_ms, 10.0) / 1000.0


def _tb_root_move(board: chess.Board) -> str | None:
    """Best move straight from the Syzygy tables, in UCI, or None if the position is not
    fully covered. Outcome first (WDL), then Distance-To-Zero for the fastest win / most
    stubborn loss; a move that resets the fifty-move counter is preferred when winning and
    avoided when losing, so a real conversion never stalls on the fifty-move rule.
    """
    if _tablebase is None or chess.popcount(board.occupied) > _TB_MAX_PIECES:
        return None
    scored: list[tuple[chess.Move, int, int, bool, bool]] = []
    try:
        for move in board.legal_moves:
            zeroing = board.is_zeroing(move)
            board.push(move)
            try:
                mate = board.is_checkmate()
                wdl = 2 if mate else -_tablebase.probe_wdl(board)
                dtz = 0 if mate else -_tablebase.probe_dtz(board)
            finally:
                board.pop()
            scored.append((move, wdl, dtz, zeroing, mate))
    except Exception:  # any gap in the tables -> let the search handle it
        return None
    if not scored:
        return None

    best_wdl = max(s[1] for s in scored)
    pool = [s for s in scored if s[1] == best_wdl]
    if best_wdl > 0:  # winning: mate now, else a counter-resetting move, else shortest DTZ
        pool.sort(key=lambda s: (not s[4], not s[3], s[2] if s[2] >= 0 else 1_000_000))
    elif best_wdl < 0:  # losing: drag it out -- keep the counter running, largest |DTZ|
        pool.sort(key=lambda s: (s[3], s[2]))
    else:  # drawn: hold it; stable order is fine
        pool.sort(key=lambda s: s[3])
    return pool[0][0].uci()


def _tick(deadline: float) -> None:
    global _nodes
    _nodes += 1
    if _nodes % _CHECK_INTERVAL == 0 and time.monotonic() >= deadline:
        raise _Timeout


def warm_up() -> None:
    """Run one tiny search so every jitted specialisation the search path needs compiles
    inside the 90 s init budget, not on the clock during move one. agent.py calls this at
    import; tests just compile lazily on first use.
    """
    global _nodes, _last_depth, _seen, _tt_gen
    try:
        search_move(chess.Board(), 200)
    finally:
        _reset_tt()
        _nodes = 0
        _last_depth = 0
        _seen = frozenset()
        _tt_gen = 0
        _killers[:] = [0] * _KILLER_SLOTS
        _hist[:] = [0] * 4096
