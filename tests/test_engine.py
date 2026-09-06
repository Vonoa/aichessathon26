"""Fast, deterministic checks on the move contract and the Phase 2 edge cases.

tests/ is never packaged into the submission. `make gate` runs this.
"""

import time

import chess
import pytest

import agent
import evaluate
import search


@pytest.fixture(autouse=True)
def _reset_state():
    agent._history.clear()
    agent._clock.clear()
    yield
    agent._history.clear()
    agent._clock.clear()


def test_finds_mate_in_one() -> None:
    # Ra8 is the only move, and it is mate; the search must see MATE and play it.
    fen = "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"
    assert agent.get_move(fen, 5000) == "a1a8"


def test_short_circuits_single_legal_move() -> None:
    # Black king on a8, cut off by the rook on c7: Kb8 is the one legal move.
    fen = "k7/2R5/8/8/8/8/8/K7 b - - 0 1"
    assert agent.get_move(fen, 5000) == "a8b8"


def test_promotes_to_queen() -> None:
    fen = "8/P7/8/8/8/8/8/k5K1 w - - 0 1"
    assert agent.get_move(fen, 2000) == "a7a8q"


TRICKY_FENS = [
    chess.STARTING_FEN,
    "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1",     # en passant capture available
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",  # castling both sides
    "8/P7/8/8/8/8/8/k5K1 w - - 0 1",         # promotion available
    "7k/8/8/8/8/8/8/5B1K w - - 0 1",         # KB vs K, insufficient material
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1",       # plain rook endgame
]


@pytest.mark.parametrize("fen", TRICKY_FENS)
def test_returns_a_legal_move(fen: str) -> None:
    board = chess.Board(fen)
    move = chess.Move.from_uci(agent.get_move(fen, 1000))
    assert move in board.legal_moves


def test_is_deterministic() -> None:
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    first = agent.get_move(fen, 2000)
    agent._history.clear()
    agent._clock.clear()
    second = agent.get_move(fen, 2000)
    assert first == second


def test_seen_position_is_a_draw_at_the_horizon() -> None:
    # Regression (Phase 3b): a position already seen in the game must score 0 even at
    # depth 0, not be evaluated by material.
    board = chess.Board("8/8/8/4k3/8/8/3RK3/8 w - - 10 40")  # white is up a whole rook
    search._seen = frozenset({board._transposition_key()})
    try:
        at_horizon = search._negamax(board, 0, 1, -search.MATE, search.MATE, time.monotonic() + 5)
    finally:
        search._seen = frozenset()
    assert at_horizon == 0


def test_budget_stays_sane_across_clocks() -> None:
    board = chess.Board()
    for clock in (150, 1_000, 8_000, 60_000, 120_000):
        for inc in (0.0, 100.0, 500.0):
            secs = search._budget_s(board, clock, inc)
            assert 0.01 <= secs <= clock / 1000.0, f"clock {clock}, inc {inc} -> {secs}s"


_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def _reference_evaluate(board: chess.Board) -> int:
    """The pre-3c SquareSet implementation, kept here to pin the bitboard rewrite."""
    pst = evaluate._PAWN_PST
    score = sum(
        v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
        for pt, v in _VALUES.items()
    )
    score += sum(pst[sq] for sq in board.pieces(chess.PAWN, chess.WHITE))
    score -= sum(pst[chess.square_mirror(sq)] for sq in board.pieces(chess.PAWN, chess.BLACK))
    return score if board.turn == chess.WHITE else -score


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
        "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1",
        "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "8/PP3kpp/8/8/8/8/pp3KPP/8 b - - 0 1",
        "8/8/8/8/8/8/8/K6k w - - 0 1",
    ],
)
def test_bitboard_eval_matches_reference(fen: str) -> None:
    board = chess.Board(fen)
    assert evaluate.evaluate(board) == _reference_evaluate(board)


def test_infers_increment_from_clock_deltas() -> None:
    assert agent._infer_increment(120_000) == 0.0  # no prior move yet
    agent._clock["prev"] = 120_000.0
    agent._clock["spent"] = 2_000.0
    # clock 120000 - spent 2000 = 118000, refilled to 118500 -> increment ~= 500
    assert 400.0 <= agent._infer_increment(118_500) <= 600.0


def test_move_one_stays_within_budget_and_searches() -> None:
    # A guard for Phase 3: if numba ever compiles on the clock, move 1 blows this.
    started = time.monotonic()
    move = chess.Move.from_uci(agent.get_move(chess.STARTING_FEN, 20000))
    elapsed = time.monotonic() - started
    assert move in chess.Board(chess.STARTING_FEN).legal_moves
    assert elapsed < 2.0, f"move 1 took {elapsed:.2f}s"
    assert search._nodes > 500, f"only {search._nodes} nodes searched"
