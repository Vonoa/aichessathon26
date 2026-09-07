"""Jitted-eval step 1: board encoding, classical ray attacks, precomputed leaper tables.

These pin the numba primitives against python-chess's own attack tables before the
tapered eval (evaluate.py) is ported onto them. tests/ is never packaged.
"""

import chess
import numpy as np

import evaluate

# A spread of occupancies: empty-ish, dense middlegame, endgame, castling position.
_OCC_FENS = [
    chess.STARTING_FEN,
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1",
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1",
]


def _slider_oracle(sq: int, occ: int) -> tuple[int, int]:
    """python-chess's own magic-free lookup: (diagonal attacks, rank|file attacks),
    each including the first blocker square -- exactly what board.attacks_mask() uses.
    """
    diag = chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occ]
    line = (
        chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occ]
        | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occ]
    )
    return diag, line


def test_ray_attacks_matches_python_chess() -> None:
    for fen in _OCC_FENS:
        occ = chess.Board(fen).occupied
        occ_u = np.uint64(occ)
        for sq in chess.SQUARES:
            diag, line = _slider_oracle(sq, occ)
            bishop = int(evaluate._ray_attacks(occ_u, sq, evaluate._BISHOP_DIRS))
            rook = int(evaluate._ray_attacks(occ_u, sq, evaluate._ROOK_DIRS))
            queen = int(evaluate._ray_attacks(occ_u, sq, evaluate._QUEEN_DIRS))
            assert bishop == diag, (fen, chess.square_name(sq))
            assert rook == line, (fen, chess.square_name(sq))
            assert queen == (diag | line), (fen, chess.square_name(sq))


def test_leaper_tables_match_python_chess() -> None:
    for sq in chess.SQUARES:
        assert int(evaluate._KNIGHT_ATTACKS[sq]) == chess.BB_KNIGHT_ATTACKS[sq]
        assert int(evaluate._KING_ATTACKS[sq]) == chess.BB_KING_ATTACKS[sq]


def test_bb_squares_table() -> None:
    for sq in chess.SQUARES:
        assert int(evaluate._BB_SQUARES[sq]) == (1 << sq)


def test_encode_matches_board() -> None:
    for fen in _OCC_FENS:
        board = chess.Board(fen)
        pieces, occ, turn = evaluate._encode(board)
        assert pieces.shape == (2, 6)
        assert turn == board.turn
        assert int(occ[0]) == board.occupied_co[chess.WHITE]
        assert int(occ[1]) == board.occupied_co[chess.BLACK]
        assert int(occ[2]) == board.occupied
        for pt in range(1, 7):
            assert int(pieces[0, pt - 1]) == board.pieces_mask(pt, chess.WHITE)
            assert int(pieces[1, pt - 1]) == board.pieces_mask(pt, chess.BLACK)
