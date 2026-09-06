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

NOT YET JITTED. This uses python-chess objects (piece_map(), pieces(), attacks_mask())
for clarity while the weights are still being tuned -- tune this version first, since it's
far easier to debug than numba, then port the tuned tables into Phase 3's jitted bitboard
eval. Don't tune the jitted version directly.

Returned score is from the side-to-move's point of view (negamax convention), matching
search.py's expectations exactly as the current evaluate.py does.
"""

import chess

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

PST_MG = {
    chess.PAWN: PAWN_MG, chess.KNIGHT: KNIGHT_MG, chess.BISHOP: BISHOP_MG,
    chess.ROOK: ROOK_MG, chess.QUEEN: QUEEN_MG, chess.KING: KING_MG,
}
PST_EG = {
    chess.PAWN: PAWN_EG, chess.KNIGHT: KNIGHT_EG, chess.BISHOP: BISHOP_EG,
    chess.ROOK: ROOK_EG, chess.QUEEN: QUEEN_EG, chess.KING: KING_EG,
}

# Pawn structure (centipawns).
DOUBLED_PAWN_PENALTY = -12
ISOLATED_PAWN_PENALTY = -10
PASSED_PAWN_BONUS_BY_RANK = [0, 5, 10, 20, 35, 60, 100, 0]  # index = rank from own side

# Mobility (centipawns per legal-attack-square, MG/EG).
MOBILITY_MG = {chess.KNIGHT: 4, chess.BISHOP: 4, chess.ROOK: 2, chess.QUEEN: 1}
MOBILITY_EG = {chess.KNIGHT: 4, chess.BISHOP: 5, chess.ROOK: 3, chess.QUEEN: 2}

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


def _game_phase(board: chess.Board) -> int:
    phase = TOTAL_PHASE
    for piece_type, weight in PHASE_WEIGHTS.items():
        phase -= weight * (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
        )
    return max(0, min(TOTAL_PHASE, phase))


def _pawn_structure(board: chess.Board, color: bool) -> int:
    score = 0
    pawns = board.pieces(chess.PAWN, color)
    files = [0] * 8
    for sq in pawns:
        files[chess.square_file(sq)] += 1

    for f in range(8):
        if files[f] > 1:
            score += DOUBLED_PAWN_PENALTY * (files[f] - 1)
        if files[f] > 0:
            neighbours = (files[f - 1] if f > 0 else 0) + (files[f + 1] if f < 7 else 0)
            if neighbours == 0:
                score += ISOLATED_PAWN_PENALTY * files[f]

    enemy_pawns = board.pieces(chess.PAWN, not color)
    enemy_files_by_rank: dict[int, list[int]] = {}
    for sq in enemy_pawns:
        enemy_files_by_rank.setdefault(chess.square_file(sq), []).append(chess.square_rank(sq))

    for sq in pawns:
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
        for sq in board.pieces(piece_type, color):
            n_moves = chess.popcount(board.attacks_mask(sq))
            mg += MOBILITY_MG[piece_type] * n_moves
            eg += MOBILITY_EG[piece_type] * n_moves
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
        for sq in board.pieces(piece_type, enemy_color):
            if board.attacks_mask(sq) & zone:
                score -= weight

    return score


def evaluate(board: chess.Board) -> int:
    """Tapered eval, centipawns, from side-to-move's perspective (negamax convention)."""
    if board.is_checkmate():
        return -100_000  # safety net; search's mate-distance logic should pre-empt this
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    phase = _game_phase(board)  # 24 = full material (MG), 0 = bare kings (EG)

    mg_score = 0
    eg_score = 0

    for square, piece in board.piece_map().items():
        value = PIECE_VALUES[piece.piece_type]
        idx = square if piece.color == chess.WHITE else _mirror(square)
        pst_mg = PST_MG[piece.piece_type][idx]
        pst_eg = PST_EG[piece.piece_type][idx]
        sign = 1 if piece.color == chess.WHITE else -1
        mg_score += sign * (value + pst_mg)
        eg_score += sign * (value + pst_eg)

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

    tapered = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) // TOTAL_PHASE

    return tapered if board.turn == chess.WHITE else -tapered
