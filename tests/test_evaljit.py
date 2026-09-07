"""Jitted-eval increments, pinned against the Python eval in evaluate.py.

step 1: board encoding, classical ray attacks, precomputed leaper tables
step 2: material + tapered PST kernels
step 3: pawn structure (doubled / isolated / passed)

tests/ is never packaged.
"""

import chess
import numpy as np
import pytest

import evaluate

# A spread of occupancies: empty-ish, dense middlegame, endgame, castling position.
_OCC_FENS = [
    chess.STARTING_FEN,
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1",
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1",
]

# The golden FENs from tests/test_engine.py plus a couple of lopsided ones, so the
# jitted material+PST slice is exercised across the phase range and both signs.
_EVAL_FENS = [
    chess.STARTING_FEN,
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1",
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1",
    "8/1p3pk1/p5p1/3P4/2P5/6P1/5K2/8 w - - 0 1",
    "2r3k1/5ppp/p7/1p1Pp3/8/1P3N2/P4PPP/3R2K1 b - - 0 1",
    "4k3/8/8/8/8/8/8/3QK3 w - - 0 1",
    "3rk3/8/8/8/4N3/8/8/3RK3 w - - 0 1",
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


# --- step 2: material + tapered PST ---------------------------------------------


_POPCOUNT_CASES = [
    0,
    1,
    2,
    0xFF,
    0x8000000000000000,
    0xFFFFFFFFFFFFFFFF,
    0x00FF00000000FF00,
    0xDEADBEEFCAFEF00D,
]


@pytest.mark.parametrize("value", _POPCOUNT_CASES)
def test_popcount_matches_bit_count(value: int) -> None:
    assert evaluate._popcount(np.uint64(value)) == value.bit_count()


@pytest.mark.parametrize("fen", _EVAL_FENS)
def test_game_phase_jit_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    pieces, _occ, _turn = evaluate._encode(board)
    assert evaluate._game_phase_jit(pieces) == evaluate._game_phase(board)


def _material_pst_reference(board: chess.Board) -> int:
    """The material + tapered-PST slice of evaluate(), White-positive, no side flip."""
    phase = evaluate._game_phase(board)
    mg = eg = 0
    for pt in evaluate._PIECE_TYPES:
        value = evaluate.PIECE_VALUES[pt]
        tmg, teg = evaluate.PST_MG[pt], evaluate.PST_EG[pt]
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.WHITE)):
            mg += value + tmg[sq]
            eg += value + teg[sq]
        for sq in chess.scan_forward(board.pieces_mask(pt, chess.BLACK)):
            idx = sq ^ 56
            mg -= value + tmg[idx]
            eg -= value + teg[idx]
    blended = mg * phase + eg * (evaluate.TOTAL_PHASE - phase)
    return int(blended / evaluate.TOTAL_PHASE)


@pytest.mark.parametrize("fen", _EVAL_FENS)
def test_jitted_material_pst_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    pieces, _occ, _turn = evaluate._encode(board)
    got = evaluate._eval_material_pst_tapered(
        pieces, evaluate._PIECE_VALUE_ARR, evaluate._PST_MG, evaluate._PST_EG
    )
    assert got == _material_pst_reference(board)


def test_jitted_material_pst_is_colour_symmetric() -> None:
    for fen in _EVAL_FENS:
        board = chess.Board(fen)
        pieces, _o, _t = evaluate._encode(board)
        mpieces, _o2, _t2 = evaluate._encode(board.mirror())
        forward = evaluate._eval_material_pst_tapered(
            pieces, evaluate._PIECE_VALUE_ARR, evaluate._PST_MG, evaluate._PST_EG
        )
        mirrored = evaluate._eval_material_pst_tapered(
            mpieces, evaluate._PIECE_VALUE_ARR, evaluate._PST_MG, evaluate._PST_EG
        )
        assert forward == -mirrored


# --- step 3: pawn structure --------------------------------------------------------

# Positions that exercise doubled, isolated, blocked and clean passed pawns.
_PAWN_FENS = [
    chess.STARTING_FEN,
    "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",          # lone white pawn: isolated + passed
    "4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1",        # black d7 blocks it ahead: not passed
    "4k3/8/8/8/8/3P4/3P4/4K3 w - - 0 1",        # white d2/d3: doubled + isolated
    "4k3/pp3ppp/8/8/8/8/PP3PPP/4K3 w - - 0 1",  # symmetric wings, e/d files empty
    "8/1p3pk1/p5p1/3P4/2P5/6P1/5K2/8 w - - 0 1",
    "2r3k1/5ppp/p7/1p1Pp3/8/1P3N2/P4PPP/3R2K1 b - - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1",
]


@pytest.mark.parametrize("fen", _PAWN_FENS)
def test_pawn_structure_side_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    white_pawns = np.uint64(board.pawns & board.occupied_co[chess.WHITE])
    black_pawns = np.uint64(board.pawns & board.occupied_co[chess.BLACK])
    assert evaluate._pawn_structure_side(white_pawns, black_pawns, True) == (
        evaluate._pawn_structure(board, chess.WHITE)
    )
    assert evaluate._pawn_structure_side(black_pawns, white_pawns, False) == (
        evaluate._pawn_structure(board, chess.BLACK)
    )


@pytest.mark.parametrize("fen", _PAWN_FENS)
def test_pawn_structure_jit_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    pieces, _occ, _turn = evaluate._encode(board)
    combined = evaluate._pawn_structure(board, chess.WHITE) - evaluate._pawn_structure(
        board, chess.BLACK
    )
    assert evaluate._pawn_structure_jit(pieces) == combined


def test_pawn_structure_jit_is_colour_symmetric() -> None:
    for fen in _PAWN_FENS:
        board = chess.Board(fen)
        pieces, _o, _t = evaluate._encode(board)
        mpieces, _o2, _t2 = evaluate._encode(board.mirror())
        assert evaluate._pawn_structure_jit(pieces) == -evaluate._pawn_structure_jit(mpieces)


# --- step 4: mobility + king safety ----------------------------------------------

# Positions with pieces bearing on the king zone and varied mobility.
_MOB_KS_FENS = [
    *_EVAL_FENS,
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 b kq - 5 5",
    "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",           # rook on an open file
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N1P/1PP1QPP1/R4RK1 w - - 0 11",
    "2r3k1/5ppp/p7/1p1Pp3/8/1P3N2/P4PPP/3R2K1 b - - 0 1",
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1",               # bare-ish, exposed kings
]


@pytest.mark.parametrize("fen", _MOB_KS_FENS)
def test_mobility_side_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    pieces, occ, _turn = evaluate._encode(board)
    for colour in (chess.WHITE, chess.BLACK):
        got = evaluate._mobility_side(pieces, occ[2], 0 if colour == chess.WHITE else 1)
        assert tuple(got) == evaluate._mobility(board, colour)


@pytest.mark.parametrize("fen", _MOB_KS_FENS)
def test_mobility_jit_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    pieces, occ, _turn = evaluate._encode(board)
    white_mg, white_eg = evaluate._mobility(board, chess.WHITE)
    black_mg, black_eg = evaluate._mobility(board, chess.BLACK)
    got = evaluate._mobility_jit(pieces, occ)
    assert tuple(got) == (white_mg - black_mg, white_eg - black_eg)


@pytest.mark.parametrize("fen", _MOB_KS_FENS)
def test_king_safety_side_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    pieces, occ, _turn = evaluate._encode(board)
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    assert white_king is not None and black_king is not None
    assert evaluate._king_safety_side(pieces, occ[2], white_king, 0) == (
        evaluate._king_safety_mg(board, white_king, chess.WHITE, white, black)
    )
    assert evaluate._king_safety_side(pieces, occ[2], black_king, 1) == (
        evaluate._king_safety_mg(board, black_king, chess.BLACK, white, black)
    )


@pytest.mark.parametrize("fen", _MOB_KS_FENS)
def test_king_safety_jit_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    pieces, occ, _turn = evaluate._encode(board)
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    assert white_king is not None and black_king is not None
    ref = evaluate._king_safety_mg(
        board, white_king, chess.WHITE, white, black
    ) - evaluate._king_safety_mg(board, black_king, chess.BLACK, white, black)
    assert evaluate._king_safety_jit(pieces, occ, white_king, black_king) == ref


def test_mobility_and_king_safety_jit_are_colour_symmetric() -> None:
    for fen in _MOB_KS_FENS:
        board = chess.Board(fen)
        mirror = board.mirror()
        pieces, occ, _t = evaluate._encode(board)
        mpieces, mocc, _t2 = evaluate._encode(mirror)
        wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
        mwk, mbk = mirror.king(chess.WHITE), mirror.king(chess.BLACK)
        assert wk is not None and bk is not None and mwk is not None and mbk is not None

        mob = evaluate._mobility_jit(pieces, occ)
        mob_mirror = evaluate._mobility_jit(mpieces, mocc)
        assert tuple(mob) == tuple(-v for v in mob_mirror)

        ks = evaluate._king_safety_jit(pieces, occ, wk, bk)
        ks_mirror = evaluate._king_safety_jit(mpieces, mocc, mwk, mbk)
        assert ks == -ks_mirror


# --- step 5: evaluate() runs the jitted path ------------------------------------

# Every FEN this file exercises, plus the test_engine.py golden set, plus terminal
# positions so the stalemate / insufficient-material guard and the kingless fallback
# are covered too.
_EQUIV_FENS = [
    *dict.fromkeys(_OCC_FENS + _EVAL_FENS + _PAWN_FENS + _MOB_KS_FENS),
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "r1b1k2r/ppppqppp/2n2n2/2b5/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 6",
    "7k/8/8/8/8/8/8/5B1K w - - 0 1",       # KB vs K: insufficient material -> 0
    "8/8/8/8/8/5k2/5p2/5K2 w - - 0 1",     # stalemate for White -> 0
    "4k3/8/8/8/8/8/8/4K2R w - - 0 1",      # KR vs K, not terminal
    "8/8/8/4k3/8/2K5/8/8 w - - 0 1",       # bare kings (insufficient) -> 0
]


@pytest.mark.parametrize("fen", _EQUIV_FENS)
def test_evaluate_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    assert evaluate.evaluate(board) == evaluate._evaluate_reference(board)


@pytest.mark.parametrize("fen", _EQUIV_FENS)
def test_evaluate_matches_reference_mirrored(fen: str) -> None:
    board = chess.Board(fen).mirror()
    assert evaluate.evaluate(board) == evaluate._evaluate_reference(board)


def test_evaluate_still_side_to_move_relative() -> None:
    # A position and the same position with the other side to move negate.
    for fen in ("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
                "2r3k1/5ppp/p7/1p1Pp3/8/1P3N2/P4PPP/3R2K1 b - - 0 1"):
        board = chess.Board(fen)
        flipped = board.copy()
        flipped.turn = not board.turn
        assert evaluate.evaluate(board) == -evaluate.evaluate(flipped)
