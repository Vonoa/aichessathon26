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
    search._reset_tt()  # the table lives for a game; a test is a fresh "game"
    yield
    agent._history.clear()
    agent._clock.clear()
    search._reset_tt()


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
    search._reset_tt()  # the carried-over table is a hidden input; wipe it to isolate
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


def test_qsearch_follows_a_quiet_check_to_win_material() -> None:
    # White is down a queen for a knight, but Nf4+ forks the king and the queen. There is
    # no capture in the position, so a captures-only quiescence just stands pat here;
    # following one ply of quiet checks finds the fork and swings the score back.
    board = chess.Board("r7/8/4k1q1/8/8/3N4/6PP/1R4K1 w - - 0 1")
    deadline = time.monotonic() + 5
    stand_pat = evaluate.evaluate(board)
    q = search._qsearch(board, 1, -search.MATE - 1, search.MATE + 1, deadline)
    assert stand_pat < -300  # really is down material before the tactic
    assert q > stand_pat + 400  # the quiet check is searched and the fork is found
    # With the check window exhausted (qply past _QS_CHECK_PLIES) it reverts to stand-pat.
    no_checks = search._qsearch(board, 1, -search.MATE - 1, search.MATE + 1, deadline, 9)
    assert no_checks == stand_pat


def test_tt_entry_round_trips() -> None:
    # Pack the same fields the store path packs, unpack them the way the probe does.
    cases = [
        (0, 1, search._EXACT, None),
        (-29_999, 64, search._UPPER, chess.Move.from_uci("e2e4")),
        (29_999, 255, search._LOWER, chess.Move.from_uci("e7e8q")),
        (-1, 5, search._EXACT, chess.Move.from_uci("a7b8n")),
    ]
    for value, depth, flag, move in cases:
        data = (
            ((value + 0x8000) & 0xFFFF)
            | (min(depth, 255) << 16)
            | (flag << 24)
            | (search._move_code(move) << 26)
            | ((123 & 0xFFFF) << 42)
        )
        assert (data & 0xFFFF) - 0x8000 == value
        assert (data >> 16) & 0xFF == depth
        assert (data >> 24) & 0x3 == flag
        assert search._code_move((data >> 26) & 0xFFFF) == move
        assert (data >> 42) & 0xFFFF == 123


def test_tt_persists_across_moves_and_cuts_nodes() -> None:
    # Two searches of the same position at a fixed depth with no deadline, the way two
    # moves of a game hit the same transposition. The second keeps the first's entries
    # (only the generation stamp advances) and must reach the same score for far fewer
    # nodes. The move can differ between equal-scored choices, so don't assert on it.
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    far = time.monotonic() + 120
    first_move = next(iter(board.legal_moves))
    search._killers[:] = [None] * search._KILLER_SLOTS
    search._hist[:] = [0] * 4096

    search._tt_gen = (search._tt_gen + 1) & 0xFFFF
    search._nodes = 0
    _, cold_score = search._search_root(board, 4, far, first_move)
    cold_nodes = search._nodes

    search._tt_gen = (search._tt_gen + 1) & 0xFFFF
    search._nodes = 0
    _, warm_score = search._search_root(board, 4, far, first_move)
    warm_nodes = search._nodes

    assert warm_score == cold_score
    assert warm_nodes < cold_nodes // 2  # carried-over entries spare most of the re-search


def test_tt_stores_no_value_it_cannot_encode() -> None:
    # KR vs K, no mate for many plies: a fixed-depth search fills the table with real
    # scores. Every stored value must sit inside the band -- a mate score (or anything
    # past +-_TT_VALUE_MAX) has to be skipped, never wrapped into the 16-bit field.
    assert search._TT_VALUE_MAX < search._MATE_THRESHOLD  # so mate scores can't be stored
    board = chess.Board("8/6k1/8/8/8/8/1R4K1/8 w - - 0 1")
    search._killers[:] = [None] * search._KILLER_SLOTS
    search._hist[:] = [0] * 4096
    search._tt_gen = (search._tt_gen + 1) & 0xFFFF
    search._nodes = 0
    search._search_root(board, 6, time.monotonic() + 120, next(iter(board.legal_moves)))

    occupied = search._tt_key != 0
    assert int(occupied.sum()) > 0
    values = (search._tt_data & 0xFFFF).astype("int64") - 0x8000
    oversized = occupied & ((values >= search._TT_VALUE_MAX) | (values <= -search._TT_VALUE_MAX))
    assert int(oversized.sum()) == 0


def test_reset_tt_wipes_the_table() -> None:
    agent.get_move("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", 2000)
    assert int((search._tt_key != 0).sum()) > 0  # something got stored
    search._reset_tt()
    assert int((search._tt_key != 0).sum()) == 0
    assert search._tt_gen == 0


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


# Exact outputs of evaluate() on fixed positions. A pure speed change must not move these.
# Deliberate eval changes so far: the KX-vs-K entry 540 -> 578 (mate driver, evaluate.
# _mopup) -> 604 (mop-up weights bumped 10/4 -> 16/8); three middlegame entries shifted
# +-3 when king safety went non-linear. The symmetric and phase-faded endgame entries
# stay put, as they should.
_EVAL_GOLDEN = {
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9": 66,
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1": 0,
    "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1": 9,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": 22,
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1": 604,
    "8/1p3pk1/p5p1/3P4/2P5/6P1/5K2/8 w - - 0 1": -86,
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N1P/1PP1QPP1/R4RK1 w - - 0 11": -2,
    "2r3k1/5ppp/p7/1p1Pp3/8/1P3N2/P4PPP/3R2K1 b - - 0 1": -383,
    "r1b1k2r/ppppqppp/2n2n2/2b5/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 6": -259,
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


def test_king_activity_marches_the_leading_king_in() -> None:
    # White up a rook, deep endgame, Black king + a pawn: leading king near the enemy
    # scores above the same king shoved into a corner.
    near = chess.Board("8/5k2/4p3/8/8/3K4/5R2/8 w - - 0 1")
    far = chess.Board("8/5k2/4p3/8/8/8/5R2/K7 w - - 0 1")
    assert evaluate.evaluate(near) > evaluate.evaluate(far)
    assert evaluate._king_activity(near) > evaluate._king_activity(far) > 0


def test_king_activity_silent_in_the_middlegame_and_when_level() -> None:
    mid = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9")
    level = chess.Board("8/5k2/8/8/8/3K4/3R1r2/8 w - - 0 1")   # R vs R, no lead
    bare = chess.Board("8/5k2/8/8/8/3K4/5R2/8 w - - 0 1")      # KR vs K -> mopup's job
    assert evaluate._king_activity(mid) == 0
    assert evaluate._king_activity(level) == 0
    assert evaluate._king_activity(bare) == 0


def _black_king_safety(fen: str) -> int:
    board = chess.Board(fen)
    return evaluate._king_safety_mg(
        board, chess.G8, chess.BLACK,
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
    )


def test_king_safety_ramps_with_multiple_attackers() -> None:
    calm = "6k1/5ppp/8/8/8/8/8/R2Q2K1 w - - 0 1"      # nothing near Black's king
    swarm = "6k1/5p1p/6p1/6NQ/8/8/8/R5K1 w - - 0 1"   # knight + queen both on the zone
    # Two attackers must cost far more than the pawn shield is worth -- a real threat
    # signal, and much worse than a lone attacker's linear penalty.
    assert _black_king_safety(swarm) < _black_king_safety(calm) - 80
    assert _black_king_safety(swarm) < -50


def test_king_safety_lone_attacker_is_cheap() -> None:
    # One bishop eyeing f8: a small linear touch, not the quadratic ramp -- the shielded
    # king still scores clearly safe.
    assert _black_king_safety("6k1/5ppp/8/8/1B6/8/8/6K1 w - - 0 1") > 20


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
