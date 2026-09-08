"""Positions from real rated losses. Each is a move the engine got wrong; the assertion
is what it should do instead. `xfail(strict=False)` marks the ones the current engine
still fails -- flip them to a plain assert when the fix lands, so a regression re-breaks
the gate. tests/ is never packaged.
"""

import pytest

import agent


@pytest.fixture(autouse=True)
def _reset_state():
    agent._history.clear()
    agent._clock.clear()
    yield
    agent._history.clear()
    agent._clock.clear()


@pytest.mark.xfail(
    reason="king-danger term too weak to decline a piece sac near the king", strict=False
)
def test_r56_declines_the_greek_gift() -> None:
    # Round 56 vs QueenC4: 13.Bxh7+ and the engine grabbed the bishop with 13...Kxh7,
    # opening its king to Ng5+/Qh5 and getting mated. Kh8 declines and keeps the king shut.
    fen = "r1bq1rk1/5p1B/p1nbpp2/1p6/3P4/4BN2/PP3PPP/R2Q1RK1 b - - 0 13"
    assert agent.get_move(fen, 5000) == "g8h8"
