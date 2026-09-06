"""Static evaluation: bitboard material plus a pawn piece-square table.

Phase 3c: material is counted with int.bit_count() on masked bitboards and the pawn table
is walked by bit iteration, so evaluate() no longer builds a SquareSet per piece type.
Phase 4 tapers this into midgame/endgame tables for every piece; a later phase can swap
the whole function for a network we trained ourselves.
"""

import chess

_PAWN = 100
_KNIGHT = 320
_BISHOP = 330
_ROOK = 500
_QUEEN = 900

# Centipawn bonus for a white pawn on each square, indexed a1..h8. Black mirrors vertically
# (square ^ 56 flips the rank), and its contribution is subtracted.
_PAWN_PST: tuple[int, ...] = (
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, -20, -20, 10, 10, 5,
    5, -5, -10, 0, 0, -10, -5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, 5, 10, 25, 25, 10, 5, 5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
    0, 0, 0, 0, 0, 0, 0, 0,
)


def evaluate(board: chess.Board) -> int:
    """Centipawn score from the point of view of the side to move."""
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]

    score = (
        _PAWN * ((board.pawns & white).bit_count() - (board.pawns & black).bit_count())
        + _KNIGHT * ((board.knights & white).bit_count() - (board.knights & black).bit_count())
        + _BISHOP * ((board.bishops & white).bit_count() - (board.bishops & black).bit_count())
        + _ROOK * ((board.rooks & white).bit_count() - (board.rooks & black).bit_count())
        + _QUEEN * ((board.queens & white).bit_count() - (board.queens & black).bit_count())
    )

    wp = board.pawns & white
    while wp:
        score += _PAWN_PST[(wp & -wp).bit_length() - 1]
        wp &= wp - 1
    bp = board.pawns & black
    while bp:
        score -= _PAWN_PST[((bp & -bp).bit_length() - 1) ^ 56]
        bp &= bp - 1

    return score if board.turn == chess.WHITE else -score
