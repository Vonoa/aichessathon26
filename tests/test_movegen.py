"""The jitted move generator (movegen.py), validated against python-chess.

perft: node counts to a fixed depth from the six standard test positions, checked
against the published numbers. divergence: the legal-move set at every position of a
batch of pseudo-random games, checked move-for-move. A movegen bug is a lost game, so
this must be exact before movegen.py goes anywhere near search.py.
"""

import random

import chess

import movegen

# fen -> {depth: published perft}. Depths kept modest so the gate stays fast; the deeper
# numbers (initial d5, Kiwipete d4, ...) are checked in scratch runs, not here.
_PERFT = {
    chess.STARTING_FEN: {1: 20, 2: 400, 3: 8902, 4: 197281},
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1": {
        1: 48, 2: 2039, 3: 97862,
    },
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1": {1: 14, 2: 191, 3: 2812, 4: 43238},
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1": {1: 6, 2: 264, 3: 9467},
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 0 1": {1: 44, 2: 1486, 3: 62379},
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 1": {
        1: 46, 2: 2079, 3: 89890,
    },
}


def test_perft_matches_the_published_numbers() -> None:
    for fen, depths in _PERFT.items():
        board = chess.Board(fen)
        for depth, want in depths.items():
            assert movegen.perft(board, depth) == want, (fen, depth)


def test_legal_move_set_matches_python_chess() -> None:
    rng = random.Random(20260908)
    checked = 0
    for _ in range(60):
        board = chess.Board()
        for _ in range(rng.randint(0, 50)):
            moves = list(board.legal_moves)
            if not moves:
                break
            assert {m.uci() for m in moves} == movegen.legal_ucis(board), board.fen()
            checked += 1
            board.push(rng.choice(moves))
    assert checked > 800


def test_encode_decode_round_trip() -> None:
    for fen in (
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1",
        "8/P7/8/8/8/8/8/k5K1 w - - 0 1",
    ):
        board = chess.Board(fen)
        bb, state = movegen.encode(board)
        for pt in range(1, 7):
            assert int(bb[0, pt - 1]) == board.pieces_mask(pt, chess.WHITE)
            assert int(bb[1, pt - 1]) == board.pieces_mask(pt, chess.BLACK)
        assert bool(state[0]) == (board.turn == chess.BLACK)
        assert int(state[2]) == (board.ep_square if board.ep_square is not None else -1)
