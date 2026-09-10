"""Fast, deterministic checks on the move contract and the Phase 2 edge cases.

tests/ is never packaged into the submission. `make gate` runs this.
"""

import time

import chess
import pytest

import agent
import evaluate
import movegen
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


def _negamax_fen(fen: str, depth: int, ply: int) -> int:
    board = chess.Board(fen)
    bb, state = movegen.encode(board)
    search._PATH[:4] = 0
    return search._negamax(bb, state, depth, ply, -search.MATE, search.MATE, time.monotonic() + 5)


def _qsearch_fen(fen: str, ply: int, qply: int = 0) -> int:
    bb, state = movegen.encode(chess.Board(fen))
    lo, hi = -search.MATE - 1, search.MATE + 1
    return search._qsearch(bb, state, ply, lo, hi, time.monotonic() + 5, qply)


def _code_of(board: chess.Board, uci: str) -> int:
    import numpy as np

    bb, state = movegen.encode(board)
    out = np.empty(256, dtype=np.int32)
    n = movegen._gen_legal(bb, state, out)
    want = chess.Move.from_uci(uci)
    for i in range(n):
        if movegen.decode_move(int(out[i])) == want:
            return int(out[i])
    raise AssertionError(uci)


def test_see_static_exchange_evaluation() -> None:
    cases = [
        ("4k3/8/8/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", 100),      # win an undefended pawn
        ("4k3/8/2p5/3p4/8/8/8/3RK3 w - - 0 1", "d1d5", -400),   # rook for a defended pawn
        ("4k3/3r4/8/3r4/8/8/8/3RK3 w - - 0 1", "d1d5", 0),      # RxR, RxR -- level
        ("3rk3/8/8/3p4/8/8/3R4/3RK3 w - - 0 1", "d2d5", 100),   # x-ray: doubled rooks
        ("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1", "d5c6", 100),     # en passant
    ]
    for fen, uci, want in cases:
        board = chess.Board(fen)
        turn = 0 if board.turn == chess.WHITE else 1
        assert movegen._see(movegen.encode(board)[0], turn, _code_of(board, uci)) == want, fen


def test_has_non_pawn_material() -> None:
    bb, _ = movegen.encode(chess.Board())
    assert search._has_non_pawn_material(bb, 0) and search._has_non_pawn_material(bb, 1)
    bb, _ = movegen.encode(chess.Board("8/5k2/8/8/8/3K4/4P3/8 w - - 0 1"))  # K+P vs K
    assert not search._has_non_pawn_material(bb, 0)
    assert not search._has_non_pawn_material(bb, 1)


def test_null_move_pruning_cuts_nodes() -> None:
    # White is a clean knight up with a full board: the null search fails high all over
    # the tree, so NMP prunes. (A near-equal position makes NMP's saving too fragile to
    # assert -- an eval tweak can flip it, as the king-safety bump did.)
    board = chess.Board("r2q1rk1/pp2bppp/2n1b3/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 11")
    far = time.monotonic() + 120
    window = (-search.MATE - 1, search.MATE + 1)

    def run(depth: int) -> tuple[int, int]:
        search._reset_tt()
        search._killers[:] = [0] * search._KILLER_SLOTS
        search._hist[:] = [0] * 4096
        search._tt_gen = (search._tt_gen + 1) & 0xFFFF
        search._nodes = 0
        bb, state = movegen.encode(board)
        _, sc = search._search_root(bb, state, depth, far, 0, *window)
        return sc, search._nodes

    # Hold the other two pruners out so this isolates NMP -- with futility + RFP live,
    # they already thin the frontier enough that NMP's marginal saving can go negative.
    saved_rfp, saved_fut = search._RFP_MAX_DEPTH, search._FUTILITY_MAX_DEPTH
    search._RFP_MAX_DEPTH = 0
    search._FUTILITY_MAX_DEPTH = 0
    try:
        with_score, with_nodes = run(6)
        saved = search._NMP_MIN_DEPTH
        search._NMP_MIN_DEPTH = 99
        try:
            without_score, without_nodes = run(6)
        finally:
            search._NMP_MIN_DEPTH = saved
    finally:
        search._RFP_MAX_DEPTH, search._FUTILITY_MAX_DEPTH = saved_rfp, saved_fut
    assert abs(with_score - without_score) <= search._CONTEMPT
    assert with_nodes < without_nodes


def test_reverse_futility_prunes_nodes() -> None:
    # White is up a rook and three pawns; the static eval clears beta by miles at
    # shallow interior nodes, so reverse-futility fails them high without searching.
    board = chess.Board("4k3/7p/8/8/8/8/PPP5/1K1R4 w - - 0 1")
    far = time.monotonic() + 120
    window = (-search.MATE - 1, search.MATE + 1)

    def run() -> tuple[int, int]:
        search._reset_tt()
        search._killers[:] = [0] * search._KILLER_SLOTS
        search._hist[:] = [0] * 4096
        search._tt_gen = (search._tt_gen + 1) & 0xFFFF
        search._nodes = 0
        bb, state = movegen.encode(board)
        _, sc = search._search_root(bb, state, 6, far, 0, *window)
        return sc, search._nodes

    with_score, with_nodes = run()
    saved = search._RFP_MAX_DEPTH
    search._RFP_MAX_DEPTH = 0  # _negamax never reaches the block with depth <= 0
    try:
        without_score, without_nodes = run()
    finally:
        search._RFP_MAX_DEPTH = saved
    assert with_score > 300 and without_score > 300  # the win survives the pruning
    assert with_nodes < without_nodes


def test_futility_prunes_quiet_frontier_moves() -> None:
    # White is a rook and four pawns down; at the frontier its quiet king shuffles are
    # a full margin below alpha and cannot raise it, so futility pruning skips them.
    board = chess.Board("4k3/7p/8/8/8/8/ppp5/1K1r4 w - - 0 1")
    far = time.monotonic() + 120
    window = (-search.MATE - 1, search.MATE + 1)

    def run() -> tuple[int, int]:
        search._reset_tt()
        search._killers[:] = [0] * search._KILLER_SLOTS
        search._hist[:] = [0] * 4096
        search._tt_gen = (search._tt_gen + 1) & 0xFFFF
        search._nodes = 0
        bb, state = movegen.encode(board)
        _, sc = search._search_root(bb, state, 6, far, 0, *window)
        return sc, search._nodes

    with_score, with_nodes = run()
    saved = search._FUTILITY_MAX_DEPTH
    search._FUTILITY_MAX_DEPTH = 0  # _negamax never reaches the block with depth <= 0
    try:
        without_score, without_nodes = run()
    finally:
        search._FUTILITY_MAX_DEPTH = saved
    assert with_score < -300 and without_score < -300  # still losing, pruning didn't lie
    assert with_nodes < without_nodes


def test_late_move_reductions_cut_nodes() -> None:
    # A wide, level middlegame: the log-formula reduction searches the tail of the
    # quiet list shallower, so far fewer nodes for a score the re-search keeps honest.
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7")
    far = time.monotonic() + 120
    window = (-search.MATE - 1, search.MATE + 1)

    def run() -> tuple[int, int]:
        search._reset_tt()
        search._killers[:] = [0] * search._KILLER_SLOTS
        search._hist[:] = [0] * 4096
        search._tt_gen = (search._tt_gen + 1) & 0xFFFF
        search._nodes = 0
        bb, state = movegen.encode(board)
        _, sc = search._search_root(bb, state, 7, far, 0, *window)
        return sc, search._nodes

    with_score, with_nodes = run()
    saved = search._LMR_MIN_DEPTH
    search._LMR_MIN_DEPTH = 99  # disables the `reduce` gate; every move searched full depth
    try:
        without_score, without_nodes = run()
    finally:
        search._LMR_MIN_DEPTH = saved
    assert abs(with_score - without_score) <= 2 * search._CONTEMPT
    assert with_nodes < without_nodes


def test_seen_position_is_a_draw_at_the_horizon() -> None:
    # A position already seen in the game must score as a draw even at the horizon, not
    # be evaluated by material. With contempt a draw is not exactly 0.
    board = chess.Board("8/8/8/4k3/8/8/3RK3/8 w - - 10 40")  # white is up a whole rook
    search._seen = frozenset({movegen.zobrist(board)})
    try:
        even = _negamax_fen(board.fen(), 0, 0)
        odd = _negamax_fen(board.fen(), 0, 1)
    finally:
        search._seen = frozenset()
    assert even == -search._CONTEMPT
    assert odd == search._CONTEMPT


def test_fifty_move_fade_dampens_a_stale_advantage() -> None:
    # White up a rook. Below the fade threshold the eval is untouched; once the halfmove
    # clock climbs the same position is worth meaningfully less, so the search is pushed
    # to convert with a zeroing move before the draw claim (Round 97).
    won = "8/8/8/4k3/8/8/3RK3/8 w - - {} 40"
    fresh = search._eval_bb(*movegen.encode(chess.Board(won.format(0))))
    early = search._eval_bb(*movegen.encode(chess.Board(won.format(search._FIFTY_FADE_START))))
    stale = search._eval_bb(*movegen.encode(chess.Board(won.format(90))))
    assert fresh > 400
    assert early == fresh                 # threshold not yet crossed
    assert stale < fresh * 3 // 5         # ~0.45x near the claim
    assert stale > 100                    # still clearly winning, just discounted


def test_qsearch_resolves_a_hanging_piece() -> None:
    # White's rook can take Black's undefended rook. Static eval sees material equal;
    # quiescence must see the win.
    fen = "4k3/8/8/8/8/3r4/3R4/4K3 w - - 0 1"
    q = _qsearch_fen(fen, 0)
    assert q >= 450
    assert q > evaluate.evaluate(chess.Board(fen))


def test_qsearch_leaves_a_quiet_position_at_the_static_eval() -> None:
    fen = "4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"  # nothing to capture, no check
    assert _qsearch_fen(fen, 0) == evaluate.evaluate(chess.Board(fen))


def test_qsearch_follows_a_quiet_check_to_win_material() -> None:
    # White is down a queen for a knight, but Nf4+ forks the king and the queen. No
    # capture in the position, so a captures-only quiescence just stands pat; following
    # one ply of quiet checks finds the fork and swings the score back. Black keeps a
    # g7 pawn so White's g2/h2 aren't scored as passed (which would muddy stand_pat).
    fen = "r7/6p1/4k1q1/8/8/3N4/6PP/1R4K1 w - - 0 1"
    stand_pat = evaluate.evaluate(chess.Board(fen))
    assert stand_pat < -300
    assert _qsearch_fen(fen, 1) > stand_pat + 400
    # With the check window exhausted (qply past _QS_CHECK_PLIES) it reverts to stand-pat.
    assert _qsearch_fen(fen, 1, 9) == stand_pat


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
    search._killers[:] = [0] * search._KILLER_SLOTS
    search._hist[:] = [0] * 4096
    window = (-search.MATE - 1, search.MATE + 1)

    search._tt_gen = (search._tt_gen + 1) & 0xFFFF
    search._nodes = 0
    bb, state = movegen.encode(board)
    _, cold_score = search._search_root(bb, state, 4, far, 0, *window)
    cold_nodes = search._nodes

    search._tt_gen = (search._tt_gen + 1) & 0xFFFF
    search._nodes = 0
    bb, state = movegen.encode(board)
    _, warm_score = search._search_root(bb, state, 4, far, 0, *window)
    warm_nodes = search._nodes

    assert warm_score == cold_score
    assert warm_nodes < cold_nodes // 2  # carried-over entries spare most of the re-search


def test_tt_stores_no_value_it_cannot_encode() -> None:
    # KR vs K, no mate for many plies: a fixed-depth search fills the table with real
    # scores. Every stored value must sit inside the band -- a mate score (or anything
    # past +-_TT_VALUE_MAX) has to be skipped, never wrapped into the 16-bit field.
    assert search._TT_VALUE_MAX < search._MATE_THRESHOLD  # so mate scores can't be stored
    board = chess.Board("8/6k1/8/8/8/8/1R4K1/8 w - - 0 1")
    search._killers[:] = [0] * search._KILLER_SLOTS
    search._hist[:] = [0] * 4096
    search._tt_gen = (search._tt_gen + 1) & 0xFFFF
    search._nodes = 0
    bb, state = movegen.encode(board)
    search._search_root(bb, state, 6, time.monotonic() + 120, 0, -search.MATE - 1, search.MATE + 1)

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


def _play_out(fen: str, cap: int = 60) -> chess.Board:
    board = chess.Board(fen)
    agent._history.clear()
    agent._clock.clear()
    search._reset_tt()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < cap:
        board.push(chess.Move.from_uci(agent.get_move(board.fen(), 60_000)))
        plies += 1
    return board


def test_syzygy_returns_none_outside_the_tables() -> None:
    # The opening position has 32 men; the tablebase path must decline it.
    assert search._tb_root_move(chess.Board()) is None


@pytest.mark.skipif(search._tablebase is None, reason="no syzygy files in ./syzygy")
def test_syzygy_mates_a_rook_ending() -> None:
    board = _play_out("8/8/8/4k3/8/8/R7/4K3 w - - 0 1")
    assert board.is_checkmate(), board.fen()


@pytest.mark.skipif(search._tablebase is None, reason="no syzygy files in ./syzygy")
def test_syzygy_holds_a_drawn_pawn_ending() -> None:
    # K+P vs K with the defending king on the queening square: a book draw. The engine
    # must not throw it (this is the Rated 61 / 62 failure mode).
    board = _play_out("8/8/8/3k4/8/3K4/3P4/8 w - - 0 1")
    result = board.result(claim_draw=True)
    assert result == "1/2-1/2", f"{result} {board.fen()}"


@pytest.mark.skipif(search._tablebase is None, reason="no syzygy files in ./syzygy")
def test_syzygy_converts_a_won_pawn_ending() -> None:
    # White king two ranks ahead of the pawn: a book win. DTZ play must finish it.
    board = _play_out("8/8/3k4/8/3K4/8/3P4/8 w - - 0 1")
    assert board.result(claim_draw=True) == "1-0", board.fen()


def test_cutoff_updates_killers_and_history() -> None:
    code = chess.E2 | (chess.E4 << 6)  # move code for e2e4 (from | to << 6)
    base = 3 * 2
    search._killers[base] = 0
    search._killers[base + 1] = 0
    idx = chess.E2 * 64 + chess.E4
    before = search._hist[idx]
    try:
        search._record_cutoff(code, ply=3, depth=5)
        assert search._killers[base] == code
        assert search._hist[idx] == before + 25
    finally:
        search._killers[base] = 0
        search._killers[base + 1] = 0
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
# +-3 when king safety went non-linear; the two pawn-heavy endgame entries moved (-86 ->
# -66, -383 -> -439) when passed/doubled/isolated were Texel-tuned for diag-10.
_EVAL_GOLDEN = {
    "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9": 79,
    "r3k2r/pppq1ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPPQ1PPP/R3K2R w KQkq - 0 1": 0,
    "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 b - - 0 1": -5,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2": 26,
    "8/2k5/8/8/8/8/5K2/6R1 w - - 0 1": 657,
    "8/1p3pk1/p5p1/3P4/2P5/6P1/5K2/8 w - - 0 1": -66,
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N1P/1PP1QPP1/R4RK1 w - - 0 11": -2,
    "2r3k1/5ppp/p7/1p1Pp3/8/1P3N2/P4PPP/3R2K1 b - - 0 1": -442,
    "r1b1k2r/ppppqppp/2n2n2/2b5/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 6": -268,
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
