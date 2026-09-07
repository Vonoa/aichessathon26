"""Phase 5 evaluation: tapered midgame/endgame PST, pawn structure, mobility, king safety.

This merges two previously separate pieces of work into one file:
  - tapered material + PST + pawn structure + mobility (the bulk of Phase 5's scope)
  - king safety (pawn shield, open files, attacker-zone pressure), SCALED by game phase
    so it applies at near-full weight in the middlegame and fades to ~0 in the endgame,
    where an exposed king is not a liability -- it's supposed to walk to the center.

All weights below are PLACEHOLDERS, seeded from public-style values so the function is
correct and produces sane scores before any tuning. texel_tune.py owns setting the real
numbers -- do not hand-adjust these from watching games. Register any new table you add
here in texel_tune.py's get_tunable_tables() or it silently won't get tuned.

evaluate() runs the numba-jitted path (_evaluate_jit, built from the kernels in the
"Jitted evaluation" section). _evaluate_reference() is the pure-python equivalent, kept
verbatim as the oracle for tests/test_engine.py's golden values and the equivalence test
in tests/test_evaljit.py. Tune against _evaluate_reference (easy to debug); the jitted
kernels read the same PST_MG/PST_EG/PIECE_VALUES tables, so a retune flows through both.

Returned score is from the side-to-move's point of view (negamax convention), matching
search.py's expectations exactly as the current evaluate.py does.
"""

import chess
import numpy as np
import numpy.typing as npt
from numba import njit

# ---------------------------------------------------------------------------
# Material (centipawns).
# ---------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Phase weight per piece type: total phase = 24 at game start, interpolates to 0 as
# non-pawn material comes off. Used both for MG/EG PST blending and to fade king safety.
PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
TOTAL_PHASE = 24

# ---------------------------------------------------------------------------
# Piece-square tables, White's perspective, a1=index 0 ... h8=index 63.
# Seeded from common open-source-style starting values -- not tuned yet.
# ---------------------------------------------------------------------------

PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]
KNIGHT_MG = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]
KNIGHT_EG = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]
BISHOP_MG = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
BISHOP_EG = [
    -14, -21, -11,  -8, -7,  -9, -17, -24,
     -8,  -4,   7, -12, -3, -13,  -4, -14,
      2,  -8,   0,  -1, -2,   6,   0,   4,
     -3,   9,  12,   9, 14,  10,   3,   2,
     -6,   3,  13,  19,  7,  10,  -3,  -9,
    -12,  -3,   8,  10, 13,   3,  -7, -15,
    -14, -18,  -7,  -1,  4,  -9, -15, -27,
    -23,  -9, -23,  -5, -9, -16,  -5, -17,
]
ROOK_MG = [
     32,  42,  32,  51, 63,  9,  31,  43,
     27,  32,  58,  62, 80, 67,  26,  44,
     -5,  19,  26,  36, 17, 45,  61,  16,
    -24, -11,   7,  26, 24, 35,  -8, -20,
    -36, -26, -12,  -1,  9, -7,   6, -23,
    -45, -25, -16, -17,  3,  0,  -5, -33,
    -44, -16, -20,  -9, -1, 11,  -6, -71,
    -19, -13,   1,  17, 16,  7, -37, -26,
]
ROOK_EG = [
    13, 10, 18, 15, 12,  12,   8,   5,
    11, 13, 13, 11, -3,   3,   8,   3,
     7,  7,  7,  5,  4,  -3,  -5,  -3,
     4,  3, 13,  1,  2,   1,  -1,   2,
     3,  5,  8,  4, -5,  -6,  -8, -11,
    -4,  0, -5, -1, -7, -12,  -8, -16,
    -6, -6,  0,  2, -9,  -9, -11,  -3,
    -9,  2,  3, -1, -5, -13,   4, -20,
]
QUEEN_MG = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]
QUEEN_EG = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]
KING_MG = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
KING_EG = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

def _flip_ranks(table: list[int]) -> list[int]:
    """The tables above are written rank 8 first (a8 = index 0), the PeSTO source layout.
    This repo indexes squares a1 = 0, so flip the ranks once here to line them up.
    """
    return [table[(7 - sq // 8) * 8 + sq % 8] for sq in range(64)]


PST_MG = {
    chess.PAWN: _flip_ranks(PAWN_MG), chess.KNIGHT: _flip_ranks(KNIGHT_MG),
    chess.BISHOP: _flip_ranks(BISHOP_MG), chess.ROOK: _flip_ranks(ROOK_MG),
    chess.QUEEN: _flip_ranks(QUEEN_MG), chess.KING: _flip_ranks(KING_MG),
}
PST_EG = {
    chess.PAWN: _flip_ranks(PAWN_EG), chess.KNIGHT: _flip_ranks(KNIGHT_EG),
    chess.BISHOP: _flip_ranks(BISHOP_EG), chess.ROOK: _flip_ranks(ROOK_EG),
    chess.QUEEN: _flip_ranks(QUEEN_EG), chess.KING: _flip_ranks(KING_EG),
}

# Pawn structure (centipawns).
DOUBLED_PAWN_PENALTY = -12
ISOLATED_PAWN_PENALTY = -10
PASSED_PAWN_BONUS_BY_RANK = [0, 5, 10, 20, 35, 60, 100, 0]  # index = rank from own side

# Mobility (centipawns per legal-attack-square, MG/EG).
MOBILITY_MG = {chess.KNIGHT: 4, chess.BISHOP: 4, chess.ROOK: 2, chess.QUEEN: 1}
MOBILITY_EG = {chess.KNIGHT: 4, chess.BISHOP: 5, chess.ROOK: 3, chess.QUEEN: 2}

# KX-vs-K mate driver (centipawns). Only active when one side is a bare king: push the
# lone king off the centre toward a corner, and march the winning king up to support the
# mate. Small vs the material lead -- a gradient to convert by, not a material term.
MOPUP_CENTER_WEIGHT = 10  # per unit of the lone king's centre-manhattan distance (0..6)
MOPUP_KINGS_WEIGHT = 4  # per unit the winning king is closer than 7 (Chebyshev)

# King safety (centipawns, MG-scale -- faded by phase before being added, see evaluate()).
SHIELD_PAWN_BONUS = 12
OPEN_FILE_PENALTY = -22
SEMI_OPEN_FILE_PENALTY = -12
ATTACKER_ZONE_WEIGHT = {
    chess.PAWN: 2,
    chess.KNIGHT: 6,
    chess.BISHOP: 6,
    chess.ROOK: 9,
    chess.QUEEN: 14,
}

_FILE_MASK = [0x0101010101010101 << f for f in range(8)]
_KING_ZONE: list[int] = [chess.BB_SQUARES[sq] | chess.BB_KING_ATTACKS[sq] for sq in range(64)]

_WHITE_SHIELD: list[int] = [0] * 64
_BLACK_SHIELD: list[int] = [0] * 64
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    _w = 0
    _b = 0
    for _nf in (_f - 1, _f, _f + 1):
        if 0 <= _nf <= 7:
            if _r + 1 <= 7:
                _w |= chess.BB_SQUARES[chess.square(_nf, _r + 1)]
            if _r - 1 >= 0:
                _b |= chess.BB_SQUARES[chess.square(_nf, _r - 1)]
    _WHITE_SHIELD[_sq] = _w
    _BLACK_SHIELD[_sq] = _b


def _mirror(square: int) -> int:
    return square ^ 56


def _center_manhattan_distance(square: int) -> int:
    """0 on the four centre squares, 6 in a corner -- how far a square is from the middle.
    Used to push a lone king toward the edge, where it can be mated."""
    f, r = square & 7, square >> 3
    file_d = f - 4 if f > 3 else 3 - f
    rank_d = r - 4 if r > 3 else 3 - r
    return file_d + rank_d


def _mopup(board: chess.Board) -> int:
    """White-relative KX-vs-K driver, 0 unless exactly one side is a bare king. Rewards the
    winning side for cornering the lone king and closing the distance between the kings.
    """
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_bare = white == board.kings & white
    black_bare = black == board.kings & black
    if white_bare == black_bare:  # both bare (KvK) or neither -- no driver
        return 0

    loser = chess.WHITE if white_bare else chess.BLACK
    lone_king = board.king(loser)
    winning_king = board.king(not loser)
    if lone_king is None or winning_king is None:
        return 0

    score = MOPUP_CENTER_WEIGHT * _center_manhattan_distance(lone_king)
    score += MOPUP_KINGS_WEIGHT * (7 - chess.square_distance(winning_king, lone_king))
    return -score if loser == chess.WHITE else score


def _game_phase(board: chess.Board) -> int:
    """Non-pawn material still on the board: 24 at the start (pure midgame), 0 at bare
    kings (pure endgame). evaluate() blends mg_score by phase and eg_score by 24 - phase.
    """
    phase = (
        PHASE_WEIGHTS[chess.KNIGHT] * board.knights.bit_count()
        + PHASE_WEIGHTS[chess.BISHOP] * board.bishops.bit_count()
        + PHASE_WEIGHTS[chess.ROOK] * board.rooks.bit_count()
        + PHASE_WEIGHTS[chess.QUEEN] * board.queens.bit_count()
    )
    return min(TOTAL_PHASE, phase)


def _pawn_structure(board: chess.Board, color: bool) -> int:
    score = 0
    pawns = board.pawns & board.occupied_co[color]
    files = [0] * 8
    for sq in chess.scan_forward(pawns):
        files[chess.square_file(sq)] += 1

    for f in range(8):
        if files[f] > 1:
            score += DOUBLED_PAWN_PENALTY * (files[f] - 1)
        if files[f] > 0:
            neighbours = (files[f - 1] if f > 0 else 0) + (files[f + 1] if f < 7 else 0)
            if neighbours == 0:
                score += ISOLATED_PAWN_PENALTY * files[f]

    enemy_pawns = board.pawns & board.occupied_co[not color]
    enemy_files_by_rank: dict[int, list[int]] = {}
    for sq in chess.scan_forward(enemy_pawns):
        enemy_files_by_rank.setdefault(chess.square_file(sq), []).append(chess.square_rank(sq))

    for sq in chess.scan_forward(pawns):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        blocked = False
        for nf in (f - 1, f, f + 1):
            if nf < 0 or nf > 7:
                continue
            for er in enemy_files_by_rank.get(nf, []):
                if (color == chess.WHITE and er > r) or (color == chess.BLACK and er < r):
                    blocked = True
        if not blocked:
            rank_from_own_side = r if color == chess.WHITE else 7 - r
            score += PASSED_PAWN_BONUS_BY_RANK[rank_from_own_side]

    return score


def _mobility(board: chess.Board, color: bool) -> tuple[int, int]:
    mg = 0
    eg = 0
    for piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        weight_mg = MOBILITY_MG[piece_type]
        weight_eg = MOBILITY_EG[piece_type]
        for sq in chess.scan_forward(board.pieces_mask(piece_type, color)):
            n_moves = board.attacks_mask(sq).bit_count()
            mg += weight_mg * n_moves
            eg += weight_eg * n_moves
    return mg, eg


def _king_safety_mg(board: chess.Board, king_sq: int, color: bool, white: int, black: int) -> int:
    """MG-scale king safety score. Caller fades this by game phase before adding it in,
    since an exposed king is not a liability once material is off the board."""
    own = white if color == chess.WHITE else black
    enemy = black if color == chess.WHITE else white
    own_pawns = board.pawns & own
    enemy_pawns = board.pawns & enemy

    score = 0

    shield_mask = _WHITE_SHIELD[king_sq] if color == chess.WHITE else _BLACK_SHIELD[king_sq]
    score += SHIELD_PAWN_BONUS * (own_pawns & shield_mask).bit_count()

    f = chess.square_file(king_sq)
    for nf in (f - 1, f, f + 1):
        if not (0 <= nf <= 7):
            continue
        file_mask = _FILE_MASK[nf]
        has_own = bool(own_pawns & file_mask)
        has_enemy = bool(enemy_pawns & file_mask)
        if not has_own and not has_enemy:
            score += OPEN_FILE_PENALTY
        elif not has_own and has_enemy:
            score += SEMI_OPEN_FILE_PENALTY

    zone = _KING_ZONE[king_sq]
    enemy_color = not color
    for piece_type, weight in ATTACKER_ZONE_WEIGHT.items():
        for sq in chess.scan_forward(board.pieces_mask(piece_type, enemy_color)):
            if board.attacks_mask(sq) & zone:
                score -= weight

    return score


_PIECE_TYPES = (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)


# ---------------------------------------------------------------------------
# Jitted evaluation. The tapered eval above still runs the search (see evaluate());
# this block is the jitted port, built as one verifiable increment per commit:
#   1  _encode + _ray_attacks + import-time warm-up                        [done]
#   2  jitted material + tapered PST                                       [done]
#   3  jitted pawn structure (doubled / isolated / passed)                 [done]
#   4  jitted mobility + king safety (via _ray_attacks -- the speed win)   [done]
#   5  evaluate() calls the jitted path (_evaluate_jit); the old body is    [done]
#      _evaluate_reference, the oracle for the golden + equivalence tests.
#   6  set-bit iteration in _material_pst; _encode fills reused buffers     [done]
# Each kernel is pinned against the Python eval above by tests/test_evaljit.py.
# ---------------------------------------------------------------------------

# One uint64 per square, so the jitted code indexes a table instead of doing
# width-fragile `1 << sq` shifts on numba's mixed int/uint types.
_BB_SQUARES: npt.NDArray[np.uint64] = np.array(
    [1 << sq for sq in range(64)], dtype=np.uint64
)

# Precomputed leaper attack sets, straight from python-chess so they match exactly.
_KNIGHT_ATTACKS: npt.NDArray[np.uint64] = np.array(chess.BB_KNIGHT_ATTACKS, dtype=np.uint64)
_KING_ATTACKS: npt.NDArray[np.uint64] = np.array(chess.BB_KING_ATTACKS, dtype=np.uint64)

# Slider directions as (file step, rank step) pairs; a queen is the union of both.
_BISHOP_DIRS: npt.NDArray[np.int64] = np.array(
    [(1, 1), (1, -1), (-1, 1), (-1, -1)], dtype=np.int64
)
_ROOK_DIRS: npt.NDArray[np.int64] = np.array(
    [(1, 0), (-1, 0), (0, 1), (0, -1)], dtype=np.int64
)
_QUEEN_DIRS: npt.NDArray[np.int64] = np.concatenate((_BISHOP_DIRS, _ROOK_DIRS))


@njit(cache=False)
def _ray_attacks(occ: np.uint64, sq: int, dirs: npt.NDArray[np.int64]) -> np.uint64:
    """Classical slider attacks: step each direction from `sq` until off-board or a
    blocker, and include that blocker square (a capture target, same as python-chess).
    """
    attacks = np.uint64(0)
    f0 = sq % 8
    r0 = sq // 8
    for i in range(dirs.shape[0]):
        df = dirs[i, 0]
        dr = dirs[i, 1]
        f = f0 + df
        r = r0 + dr
        while 0 <= f <= 7 and 0 <= r <= 7:
            bit = _BB_SQUARES[r * 8 + f]
            attacks |= bit
            if (occ & bit) != np.uint64(0):
                break
            f += df
            r += dr
    return attacks


def _encode(
    board: chess.Board,
    pieces: npt.NDArray[np.uint64] | None = None,
    occ: npt.NDArray[np.uint64] | None = None,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], bool]:
    """Board -> fixed arrays the jitted eval reads.

    Returns:
      pieces: shape (2, 6) uint64 -- pieces[colour][piece_type - 1] bitboard,
              colour 0 = White, 1 = Black; piece_type is chess.PAWN..chess.KING.
      occ:    shape (3,) uint64 -- [white occupancy, black occupancy, all occupancy].
      turn:   True if White is to move (for the final side-to-move sign flip).

    `pieces` and `occ`, if given, are filled in place -- the search passes two module
    buffers so a leaf's eval allocates nothing. Omit them for a fresh pair (tests).
    """
    if pieces is None:
        pieces = np.empty((2, 6), dtype=np.uint64)
    if occ is None:
        occ = np.empty(3, dtype=np.uint64)
    for pt in range(1, 7):
        pieces[0, pt - 1] = board.pieces_mask(pt, chess.WHITE)
        pieces[1, pt - 1] = board.pieces_mask(pt, chess.BLACK)
    occ[0] = board.occupied_co[chess.WHITE]
    occ[1] = board.occupied_co[chess.BLACK]
    occ[2] = board.occupied
    return pieces, occ, board.turn


# --- step 2: material + tapered PST -----------------------------------------
#
# The Python tables above, baked into fixed arrays the jitted kernels read. Rows
# follow _PIECE_TYPES order: 0 = pawn ... 5 = king, matching _encode()'s pieces[c].
# PST_MG / PST_EG are already _flip_ranks'd (a1 = 0), so the jit indexes `sq`
# directly for White and `sq ^ 56` for Black -- no second flip.
_PIECE_VALUE_ARR: npt.NDArray[np.int16] = np.array(
    [PIECE_VALUES[pt] for pt in _PIECE_TYPES], dtype=np.int16
)
_PST_MG: npt.NDArray[np.int16] = np.array(
    [PST_MG[pt] for pt in _PIECE_TYPES], dtype=np.int16
)
_PST_EG: npt.NDArray[np.int16] = np.array(
    [PST_EG[pt] for pt in _PIECE_TYPES], dtype=np.int16
)


@njit(cache=False)
def _popcount(bb: np.uint64) -> np.int64:
    """SWAR population count. All-uint64 internally so numba never sees a mixed-width
    shift; the result is cast to int64 so callers can do plain `x >> 3` / `2 * x`.
    """
    bb = bb - ((bb >> np.uint64(1)) & np.uint64(0x5555555555555555))
    bb = (bb & np.uint64(0x3333333333333333)) + (
        (bb >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    bb = (bb + (bb >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((bb * np.uint64(0x0101010101010101)) >> np.uint64(56))


@njit(cache=False)
def _game_phase_jit(pieces: npt.NDArray[np.uint64]) -> int:
    """Non-pawn material on the board: 24 at the start (midgame), 0 at bare kings.
    Mirrors evaluate._game_phase -- N/B weight 1, R weight 2, Q weight 4, capped at 24.
    """
    knights = _popcount(pieces[0, 1] | pieces[1, 1])
    bishops = _popcount(pieces[0, 2] | pieces[1, 2])
    rooks = _popcount(pieces[0, 3] | pieces[1, 3])
    queens = _popcount(pieces[0, 4] | pieces[1, 4])
    phase = knights + bishops + 2 * rooks + 4 * queens
    return int(24 if phase > 24 else phase)


@njit(cache=False)
def _material_pst(
    pieces: npt.NDArray[np.uint64],
    values: npt.NDArray[np.int16],
    pst_mg: npt.NDArray[np.int16],
    pst_eg: npt.NDArray[np.int16],
) -> tuple[int, int]:
    """(mg, eg) material + piece-square sums, White-positive. Black squares are
    mirrored (sq ^ 56) and subtracted, exactly as evaluate()'s piece loop does.

    Iterates set bits (a board has ~16-32 pieces) rather than scanning all 64 squares.
    """
    mg = 0
    eg = 0
    for pt in range(6):
        v = int(values[pt])

        bb = pieces[0, pt]
        while bb != np.uint64(0):
            lsb = bb & (~bb + np.uint64(1))
            sq = int(_popcount(lsb - np.uint64(1)))
            mg += v + int(pst_mg[pt, sq])
            eg += v + int(pst_eg[pt, sq])
            bb &= bb - np.uint64(1)

        bb = pieces[1, pt]
        while bb != np.uint64(0):
            lsb = bb & (~bb + np.uint64(1))
            idx = int(_popcount(lsb - np.uint64(1))) ^ 56
            mg -= v + int(pst_mg[pt, idx])
            eg -= v + int(pst_eg[pt, idx])
            bb &= bb - np.uint64(1)
    return mg, eg


@njit(cache=False)
def _eval_material_pst_tapered(
    pieces: npt.NDArray[np.uint64],
    values: npt.NDArray[np.int16],
    pst_mg: npt.NDArray[np.int16],
    pst_eg: npt.NDArray[np.int16],
) -> int:
    """Step-2 entry point: material + tapered PST only, White-positive (no side-to-move
    flip yet). Grows into the full jitted eval as steps 3-4 land. Blend matches
    evaluate() -- int(.../24) truncation toward zero keeps the score colour-symmetric.
    """
    phase = _game_phase_jit(pieces)
    mg, eg = _material_pst(pieces, values, pst_mg, pst_eg)
    blended = mg * phase + eg * (24 - phase)
    return int(blended / 24)


# --- step 3: pawn structure ------------------------------------------------------
#
# Doubled / isolated by per-file pawn count (popcount of own & file mask); passed
# by a front-span mask -- a pawn is passed iff no enemy pawn stands on its own or
# an adjacent file, anywhere ahead of it. Masks baked here to match evaluate.
# _pawn_structure()'s "for nf in (f-1, f, f+1): for er ahead of r" check exactly.
_FILE_MASK_ARR: npt.NDArray[np.uint64] = np.array(
    [0x0101010101010101 << f for f in range(8)], dtype=np.uint64
)
_PASSED_BONUS_ARR: npt.NDArray[np.int16] = np.array(
    PASSED_PAWN_BONUS_BY_RANK, dtype=np.int16
)


def _build_passed_masks() -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    white: list[int] = []
    black: list[int] = []
    for sq in range(64):
        file_, rank = sq & 7, sq >> 3
        wm = 0
        bm = 0
        for nf in (file_ - 1, file_, file_ + 1):
            if 0 <= nf <= 7:
                for nr in range(8):
                    target = 1 << (nr * 8 + nf)
                    if nr > rank:
                        wm |= target
                    if nr < rank:
                        bm |= target
        white.append(wm)
        black.append(bm)
    return np.array(white, dtype=np.uint64), np.array(black, dtype=np.uint64)


_PASSED_MASK_WHITE, _PASSED_MASK_BLACK = _build_passed_masks()


@njit(cache=False)
def _pawn_structure_side(own: np.uint64, enemy: np.uint64, is_white: bool) -> int:
    """One side's pawn score (centipawns), mirroring evaluate._pawn_structure():
    -12 per doubled pawn, -10 per isolated pawn, a by-rank bonus for each passed one.
    """
    score = 0

    counts = np.empty(8, dtype=np.int64)
    for f in range(8):
        counts[f] = _popcount(own & _FILE_MASK_ARR[f])

    for f in range(8):
        c = int(counts[f])
        if c > 1:
            score += DOUBLED_PAWN_PENALTY * (c - 1)
        if c > 0:
            left = int(counts[f - 1]) if f > 0 else 0
            right = int(counts[f + 1]) if f < 7 else 0
            if left + right == 0:
                score += ISOLATED_PAWN_PENALTY * c

    bb = own
    while bb != np.uint64(0):
        lsb = bb & (~bb + np.uint64(1))
        sq = _popcount(lsb - np.uint64(1))  # trailing zeros = square index
        if is_white:
            span = _PASSED_MASK_WHITE[sq]
            rank = sq >> 3
        else:
            span = _PASSED_MASK_BLACK[sq]
            rank = 7 - (sq >> 3)
        if (enemy & span) == np.uint64(0):
            score += int(_PASSED_BONUS_ARR[rank])
        bb &= bb - np.uint64(1)

    return score


@njit(cache=False)
def _pawn_structure_jit(pieces: npt.NDArray[np.uint64]) -> int:
    """White pawn score minus Black's -- the sign*pawn_score combination evaluate()
    folds into both mg and eg. Added to the tapered blend unchanged at step 5.
    """
    white_pawns = pieces[0, 0]
    black_pawns = pieces[1, 0]
    return int(
        _pawn_structure_side(white_pawns, black_pawns, True)
        - _pawn_structure_side(black_pawns, white_pawns, False)
    )


# --- step 4: mobility + king safety --------------------------------------------
#
# The speed win: evaluate()'s _mobility and _king_safety_mg call board.attacks_mask()
# ~24x per leaf (python-chess slider rays); here it is a jitted _ray_attacks loop.
# Weights indexed by piece type 0=pawn..5=king (0 where the term does not apply).
_MOBILITY_MG_ARR: npt.NDArray[np.int16] = np.array(
    [0, MOBILITY_MG[chess.KNIGHT], MOBILITY_MG[chess.BISHOP],
     MOBILITY_MG[chess.ROOK], MOBILITY_MG[chess.QUEEN], 0], dtype=np.int16,
)
_MOBILITY_EG_ARR: npt.NDArray[np.int16] = np.array(
    [0, MOBILITY_EG[chess.KNIGHT], MOBILITY_EG[chess.BISHOP],
     MOBILITY_EG[chess.ROOK], MOBILITY_EG[chess.QUEEN], 0], dtype=np.int16,
)
_ATTACKER_WEIGHT_ARR: npt.NDArray[np.int16] = np.array(
    [ATTACKER_ZONE_WEIGHT[chess.PAWN], ATTACKER_ZONE_WEIGHT[chess.KNIGHT],
     ATTACKER_ZONE_WEIGHT[chess.BISHOP], ATTACKER_ZONE_WEIGHT[chess.ROOK],
     ATTACKER_ZONE_WEIGHT[chess.QUEEN], 0], dtype=np.int16,
)
_KING_ZONE_ARR: npt.NDArray[np.uint64] = np.array(_KING_ZONE, dtype=np.uint64)
# Shield / pawn-attack tables in this file's colour order: index 0 = White, 1 = Black.
_SHIELD_ARR: npt.NDArray[np.uint64] = np.array(
    [_WHITE_SHIELD, _BLACK_SHIELD], dtype=np.uint64
)
_PAWN_ATTACKS_ARR: npt.NDArray[np.uint64] = np.array(
    [list(chess.BB_PAWN_ATTACKS[chess.WHITE]), list(chess.BB_PAWN_ATTACKS[chess.BLACK])],
    dtype=np.uint64,
)


@njit(cache=False)
def _piece_attacks(occ_all: np.uint64, sq: int, pt: int) -> np.uint64:
    """attacks_mask for a non-pawn piece type: 1=knight, 2=bishop, 3=rook, 4=queen."""
    if pt == 1:
        return np.uint64(_KNIGHT_ATTACKS[sq])
    if pt == 2:
        return _ray_attacks(occ_all, sq, _BISHOP_DIRS)
    if pt == 3:
        return _ray_attacks(occ_all, sq, _ROOK_DIRS)
    return _ray_attacks(occ_all, sq, _QUEEN_DIRS)


@njit(cache=False)
def _mobility_side(
    pieces: npt.NDArray[np.uint64], occ_all: np.uint64, colour: int
) -> tuple[int, int]:
    """(mg, eg) mobility for one side: attack-square count per N/B/R/Q times its weight,
    mirroring evaluate._mobility (which counts every attacked square, own pieces included).
    """
    mg = 0
    eg = 0
    for pt in range(1, 5):  # KNIGHT, BISHOP, ROOK, QUEEN
        wmg = int(_MOBILITY_MG_ARR[pt])
        weg = int(_MOBILITY_EG_ARR[pt])
        bb = pieces[colour, pt]
        while bb != np.uint64(0):
            lsb = bb & (~bb + np.uint64(1))
            sq = int(_popcount(lsb - np.uint64(1)))
            att = _piece_attacks(occ_all, sq, pt)
            n = int(_popcount(att))
            mg += wmg * n
            eg += weg * n
            bb &= bb - np.uint64(1)
    return mg, eg


@njit(cache=False)
def _mobility_jit(
    pieces: npt.NDArray[np.uint64], occ: npt.NDArray[np.uint64]
) -> tuple[int, int]:
    """(mg, eg) = White mobility minus Black's -- the sign*mob combination evaluate() folds
    into mg_score and eg_score respectively."""
    all_occ = occ[2]
    white_mg, white_eg = _mobility_side(pieces, all_occ, 0)
    black_mg, black_eg = _mobility_side(pieces, all_occ, 1)
    return int(white_mg - black_mg), int(white_eg - black_eg)


@njit(cache=False)
def _king_safety_side(
    pieces: npt.NDArray[np.uint64], occ_all: np.uint64, king_sq: int, colour: int
) -> int:
    """MG-scale king safety for one side, mirroring evaluate._king_safety_mg: pawn shield,
    open / semi-open files beside the king, and enemy attacker-zone pressure. The caller
    fades this by game phase.
    """
    own_pawns = pieces[colour, 0]
    enemy = 1 - colour
    enemy_pawns = pieces[enemy, 0]
    score = 0

    shield = _SHIELD_ARR[colour, king_sq]
    score += SHIELD_PAWN_BONUS * int(_popcount(own_pawns & shield))

    king_file = king_sq % 8
    for nf in range(king_file - 1, king_file + 2):
        if nf < 0 or nf > 7:
            continue
        file_mask = _FILE_MASK_ARR[nf]
        has_own = (own_pawns & file_mask) != np.uint64(0)
        has_enemy = (enemy_pawns & file_mask) != np.uint64(0)
        if not has_own and not has_enemy:
            score += OPEN_FILE_PENALTY
        elif not has_own and has_enemy:
            score += SEMI_OPEN_FILE_PENALTY

    zone = _KING_ZONE_ARR[king_sq]
    for pt in range(5):  # PAWN, KNIGHT, BISHOP, ROOK, QUEEN
        weight = int(_ATTACKER_WEIGHT_ARR[pt])
        bb = pieces[enemy, pt]
        while bb != np.uint64(0):
            lsb = bb & (~bb + np.uint64(1))
            sq = int(_popcount(lsb - np.uint64(1)))
            # both branches are uint64, so the ternary is numba-safe here (unlike a
            # ternary that mixes an int with a promoted-to-float shift result)
            att = _PAWN_ATTACKS_ARR[enemy, sq] if pt == 0 else _piece_attacks(occ_all, sq, pt)
            if (att & zone) != np.uint64(0):
                score -= weight
            bb &= bb - np.uint64(1)

    return int(score)


@njit(cache=False)
def _king_safety_jit(
    pieces: npt.NDArray[np.uint64],
    occ: npt.NDArray[np.uint64],
    white_king: int,
    black_king: int,
) -> int:
    """White king safety minus Black's (MG-scale, unfaded). evaluate() adds this to
    mg_score only, then the phase blend fades it toward 0 in the endgame."""
    all_occ = occ[2]
    return int(
        _king_safety_side(pieces, all_occ, white_king, 0)
        - _king_safety_side(pieces, all_occ, black_king, 1)
    )


@njit(cache=False)
def _center_manhattan_distance_jit(square: int) -> int:
    f = square & 7
    r = square >> 3
    file_d = f - 4 if f > 3 else 3 - f
    rank_d = r - 4 if r > 3 else 3 - r
    return file_d + rank_d


@njit(cache=False)
def _mopup_jit(
    occ: npt.NDArray[np.uint64],
    pieces: npt.NDArray[np.uint64],
    white_king: int,
    black_king: int,
) -> int:
    """White-relative KX-vs-K driver, 0 unless exactly one side is a bare king. Mirrors
    evaluate._mopup: corner the lone king, close the distance between the kings."""
    white_bare = occ[0] == pieces[0, 5]
    black_bare = occ[1] == pieces[1, 5]
    if white_bare == black_bare:
        return 0

    if white_bare:
        lone_king = white_king
        winning_king = black_king
        sign = -1
    else:
        lone_king = black_king
        winning_king = white_king
        sign = 1

    lf = lone_king & 7
    lr = lone_king >> 3
    wf = winning_king & 7
    wr = winning_king >> 3
    kings_dist = max(abs(wf - lf), abs(wr - lr))
    score = MOPUP_CENTER_WEIGHT * _center_manhattan_distance_jit(lone_king)
    score += MOPUP_KINGS_WEIGHT * (7 - kings_dist)
    return sign * score


@njit(cache=False)
def _evaluate_jit(
    pieces: npt.NDArray[np.uint64],
    occ: npt.NDArray[np.uint64],
    white_king: int,
    black_king: int,
    white_to_move: bool,
) -> int:
    """Full tapered eval, side-to-move relative -- the jitted equivalent of
    _evaluate_reference()'s body (minus the stalemate / insufficient-material guard,
    which the Python wrapper still does). Assembly matches the reference exactly:
    king safety enters mg only; pawn structure enters both mg and eg.
    """
    phase = _game_phase_jit(pieces)
    material_mg, material_eg = _material_pst(pieces, _PIECE_VALUE_ARR, _PST_MG, _PST_EG)
    pawn = _pawn_structure_jit(pieces)
    mob_mg, mob_eg = _mobility_jit(pieces, occ)
    king_safety = _king_safety_jit(pieces, occ, white_king, black_king)

    mg_total = material_mg + pawn + mob_mg + king_safety
    eg_total = material_eg + pawn + mob_eg
    blended = mg_total * phase + eg_total * (24 - phase)
    tapered = int(blended / 24) + _mopup_jit(occ, pieces, white_king, black_king)
    if white_to_move:
        return tapered
    return -tapered


def _warm_up() -> None:
    """Compile the jitted kernels at import, with the exact dtypes the search feeds
    them, so numba never compiles on the clock (its `/tmp` cache is wiped per game).
    """
    occ_scalar = np.uint64(0x00FF00000000FF00)
    for dirs in (_BISHOP_DIRS, _ROOK_DIRS, _QUEEN_DIRS):
        _ray_attacks(occ_scalar, 27, dirs)
    pieces, occ, _turn = _encode(chess.Board())
    _popcount(np.uint64(0xFFFF00000000FFFF))
    _game_phase_jit(pieces)
    _eval_material_pst_tapered(pieces, _PIECE_VALUE_ARR, _PST_MG, _PST_EG)
    _pawn_structure_jit(pieces)
    _mobility_jit(pieces, occ)
    _king_safety_jit(pieces, occ, 4, 60)  # e1 / e8 kings of the start position
    _mopup_jit(occ, pieces, 4, 60)
    _evaluate_jit(pieces, occ, 4, 60, True)


_warm_up()


def _evaluate_reference(board: chess.Board) -> int:
    """Pure-python tapered eval -- the oracle evaluate() (the jitted path) is checked
    against. Kept verbatim; tune weights here, and register new tables in texel_tune.py.

    Tapered eval, centipawns, from side-to-move's perspective (negamax convention). The
    search never calls this on a checkmate, so there is no is_checkmate() guard here.
    """
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = _game_phase(board)  # 24 = full material (MG), 0 = bare kings (EG)

    mg_score = 0
    eg_score = 0

    for piece_type in _PIECE_TYPES:
        value = PIECE_VALUES[piece_type]
        table_mg = PST_MG[piece_type]
        table_eg = PST_EG[piece_type]
        for sq in chess.scan_forward(board.pieces_mask(piece_type, chess.WHITE)):
            mg_score += value + table_mg[sq]
            eg_score += value + table_eg[sq]
        for sq in chess.scan_forward(board.pieces_mask(piece_type, chess.BLACK)):
            idx = _mirror(sq)
            mg_score -= value + table_mg[idx]
            eg_score -= value + table_eg[idx]

    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]

    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        pawn_score = _pawn_structure(board, color)
        mob_mg, mob_eg = _mobility(board, color)
        mg_score += sign * (pawn_score + mob_mg)
        eg_score += sign * (pawn_score + mob_eg)

    # King safety is MG-only by nature (an exposed king is a middlegame liability, not an
    # endgame one), so it's added into mg_score alone and faded out by the phase blend below
    # -- at phase=0 (bare kings) it contributes exactly 0, at phase=24 it's at full weight.
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is not None:
        mg_score += _king_safety_mg(board, white_king, chess.WHITE, white, black)
    if black_king is not None:
        mg_score -= _king_safety_mg(board, black_king, chess.BLACK, white, black)

    # int() truncates toward zero so the score stays exactly colour-symmetric; // would
    # round toward negative infinity and make a position and its mirror differ by 1.
    blended = mg_score * phase + eg_score * (TOTAL_PHASE - phase)
    tapered = int(blended / TOTAL_PHASE) + _mopup(board)

    return tapered if board.turn == chess.WHITE else -tapered


# Reused every evaluate() call so a leaf's eval allocates nothing. Safe because the
# search is single-threaded and _evaluate_jit reads them before evaluate() returns.
_EVAL_PIECES: npt.NDArray[np.uint64] = np.empty((2, 6), dtype=np.uint64)
_EVAL_OCC: npt.NDArray[np.uint64] = np.empty(3, dtype=np.uint64)


def evaluate(board: chess.Board) -> int:
    """Tapered eval, centipawns, from the side-to-move's point of view (negamax
    convention). Runs the numba-jitted path; _evaluate_reference() is the pure-python
    equivalent, kept as the golden-test oracle.

    The search never calls this on a checkmate (a node with no legal moves returns a mate
    score), so there is no is_checkmate() guard -- it would be a wasted movegen per leaf.
    """
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        return _evaluate_reference(board)  # not a legal search position; stay safe

    pieces, occ, white_to_move = _encode(board, _EVAL_PIECES, _EVAL_OCC)
    return int(_evaluate_jit(pieces, occ, white_king, black_king, white_to_move))
