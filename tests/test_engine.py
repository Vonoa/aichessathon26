"""Fast, deterministic checks on the move contract and the Phase 2 edge cases.

tests/ is never packaged into the submission. `make gate` runs this.
"""

import time

import chess
import pytest

import agent
import search


@pytest.fixture(autouse=True)
def _reset_history():
    agent._history.clear()
    yield
    agent._history.clear()


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
    second = agent.get_move(fen, 2000)
    assert first == second


def test_move_one_stays_within_budget_and_searches() -> None:
    # A guard for Phase 3: if numba ever compiles on the clock, move 1 blows this.
    started = time.monotonic()
    move = chess.Move.from_uci(agent.get_move(chess.STARTING_FEN, 20000))
    elapsed = time.monotonic() - started
    assert move in chess.Board(chess.STARTING_FEN).legal_moves
    assert elapsed < 1.5, f"move 1 took {elapsed:.2f}s"
    assert search._nodes > 500, f"only {search._nodes} nodes searched"
