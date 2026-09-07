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
    # Regression (Phase 3b): a position already seen in the game must score as a draw even
    # at depth 0, not be evaluated by material. With contempt (5f) a draw is not exactly 0.
    board = chess.Board("8/8/8/4k3/8/8/3RK3/8 w - - 10 40")  # white is up a whole rook
    search._seen = frozenset({board._transposition_key()})
    try:
        even = search._negamax(board, 0, 0, -search.MATE, search.MATE, time.monotonic() + 5)
        odd = search._negamax(board, 0, 1, -search.MATE, search.MATE, time.monotonic() + 5)
    finally:
        search._seen = frozenset()
    assert even == -search._CONTEMPT
    assert odd == search._CONTEMPT


def test_qsearch_resolves_a_hanging_piece() -> None:
    # White's rook can take Black's undefended rook. Static eval sees material equal;
    # quiescence must see the win.
    board = chess.Board("4k3/8/8/8/8/3r4/3R4/4K3 w - - 0 1")
    q = search._qsearch(board, 0, -search.MATE, search.MATE, time.monotonic() + 5)
    assert q >= 450
    assert q > evaluate.evaluate(board)


def test_qsearch_leaves_a_quiet_position_at_the_static_eval() -> None:
    board = chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1")  # nothing to capture, no check
    q = search._qsearch(board, 0, -search.MATE, search.MATE, time.monotonic() + 5)
    assert q == evaluate.evaluate(board)


def test_cutoff_updates_killers_and_history() -> None:
    move = chess.Move.from_uci("e2e4")
    base = 3 * 2
    search._killers[base] = None
    search._killers[base + 1] = None
    idx = move.from_square * 64 + move.to_square
    before = search._hist[idx]
    try:
        search._record_cutoff(move, ply=3, depth=5)
        assert search._killers[base] == move
        assert search._hist[idx] == before + 25
    finally:
        search._killers[base] = None
        search._killers[base + 1] = None
        search._hist[idx] = before


def test_budget_stays_sane_across_clocks() -> None:
    board = chess.Board()
    for clock in (150, 1_000, 8_000, 60_000, 120_000):
        for inc in (0.0, 100.0, 500.0):
            secs = search._budget_s(board, clock, inc)
            assert 0.01 <= secs <= clock / 1000.0, f"clock {clock}, inc {inc} -> {secs}s"


def test_budget_leaves_a_reserve_and_caps_at_a_third() -> None:
    board = chess.Board()
    for clock in (700, 2_000, 12_000, 90_000):
        secs = search._budget_s(board, clock, 500.0)
        assert secs <= clock / 3000.0 + 1e-9
        assert secs <= max(0.01, (clock - search._RESERVE_MS) / 1000.0)


@pytest.mark.parametrize(
    "fen",
    [
        chess.STARTING_FEN,
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
        "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1",
        "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1",
        "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1",
    ],
)
def test_eval_is_colour_symmetric(fen: str) -> None:
    board = chess.Board(fen)
    assert evaluate.evaluate(board) == evaluate.evaluate(board.mirror())


# Exact outputs of evaluate() on fixed positions. A pure speed change must not move these;
# the KX-vs-K entry moved 540 -> 578 when the mate driver (evaluate._mopup) was added --
# a deliberate eval change, correct only for a bare-king endgame.
_EVAL_GOLDEN = {
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9": 69,
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1": 0,
    "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1": 8,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": 22,
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1": 578,
    "8/1p3pk1/p5p1/3P4/2P5/6P1/5K2/8 w - - 0 1": -86,
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N1P/1PP1QPP1/R4RK1 w - - 0 11": -2,
    "2r3k1/5ppp/p7/1p1Pp3/8/1P3N2/P4PPP/3R2K1 b - - 0 1": -383,
    "r1b1k2r/ppppqppp/2n2n2/2b5/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 6": -262,
}


@pytest.mark.parametrize(("fen", "expected"), list(_EVAL_GOLDEN.items()))
def test_eval_golden_values(fen: str, expected: int) -> None:
    assert evaluate.evaluate(chess.Board(fen)) == expected


def test_eval_start_position_is_balanced() -> None:
    assert evaluate.evaluate(chess.Board()) == 0


def test_eval_rewards_a_material_lead() -> None:
    # White has an extra queen; the score must be decisively positive.
    board = chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")
    assert evaluate.evaluate(board) > 700


def test_eval_prefers_a_centralised_knight() -> None:
    # Rooks on both sides so the position is not an insufficient-material draw.
    centre = chess.Board("3rk3/8/8/8/4N3/8/8/3RK3 w - - 0 1")
    rim = chess.Board("3rk3/8/8/8/N7/8/8/3RK3 w - - 0 1")
    assert evaluate.evaluate(centre) > evaluate.evaluate(rim)


def test_pst_tables_are_oriented_a1_first() -> None:
    # A pawn one step from promotion must score far above one on its home rank.
    assert evaluate.PST_EG[chess.PAWN][chess.A7] > evaluate.PST_EG[chess.PAWN][chess.A2] + 100
    # A midgame king belongs on the back rank, not marching up the board.
    assert evaluate.PST_MG[chess.KING][chess.E1] > evaluate.PST_MG[chess.KING][chess.E5]


def test_eval_wants_pawns_advanced() -> None:
    advanced = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    home = chess.Board("4k3/8/8/8/8/8/P7/4K3 w - - 0 1")
    assert evaluate.evaluate(advanced) > evaluate.evaluate(home) + 80


def test_mopup_drives_the_lone_king_to_the_edge() -> None:
    # KQ vs K, White winning. The mate driver must score a cornered black king with
    # White's king near it well above a centralised black king with White's king far.
    near = chess.Board("7k/8/6K1/8/8/8/8/7Q w - - 0 1")
    far = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert evaluate.evaluate(near) > evaluate.evaluate(far)


def test_mopup_is_silent_with_pieces_on_both_sides() -> None:
    # No bare king -> the driver contributes nothing, which is what keeps every non-KX
    # golden value stable.
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9")
    assert evaluate._mopup(board) == 0


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
