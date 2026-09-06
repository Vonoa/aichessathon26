"""Static evaluation. Phase 1: material plus a single pawn piece-square table.

Phase 4 replaces this with tapered midgame/endgame tables for every piece, and a later
phase can swap the whole function for a network we trained ourselves.
"""

import chess

_PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

# Centipawn bonus for a white pawn on each square, indexed a1..h8. Black mirrors vertically.
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
    score = 0
    for piece_type, value in _PIECE_VALUE.items():
        score += value * len(board.pieces(piece_type, chess.WHITE))
        score -= value * len(board.pieces(piece_type, chess.BLACK))
    for square in board.pieces(chess.PAWN, chess.WHITE):
        score += _PAWN_PST[square]
    for square in board.pieces(chess.PAWN, chess.BLACK):
        score -= _PAWN_PST[chess.square_mirror(square)]
    return score if board.turn == chess.WHITE else -score
