"""The submission entrypoint. The platform imports this file and calls get_move."""

import time
from collections.abc import Hashable

import chess

from search import search_move

# Import time runs once per game, inside a 90 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move. Warm every numba-jitted
# function here too, so compilation lands in the init budget rather than on the clock.

# Every position we have been asked to move in this game, counted. The process is fresh
# per game so this resets on its own. The search reads it to spot a line that repeats a
# position the game has already seen and score it as a draw.
_history: dict[Hashable, int] = {}

# Last move's clock and our own measured spend, used to infer the increment (get_move is
# not told it). new_clock == old_clock - our_spend + increment, so increment falls out.
_clock: dict[str, float] = {}


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation.

    fen           the position to move in; your colour is the side to move
    time_left_ms  your clock before this move, in milliseconds
    returns       "e2e4", or "e7e8q" for a promotion

    The process stays alive between your moves, so state kept on a module survives to
    the next call in the same game. It does not survive to the next game.

    print() is safe: stdout is redirected away from the protocol stream.
    """
    board = chess.Board(fen)
    key = board._transposition_key()
    _history[key] = _history.get(key, 0) + 1

    increment_ms = _infer_increment(time_left_ms)
    started = time.monotonic()
    try:
        return search_move(board, time_left_ms, _history, increment_ms)
    except Exception:
        # A bug in the search must never forfeit the game: fall back to any legal move.
        legal = list(board.legal_moves)
        return legal[0].uci() if legal else "0000"
    finally:
        _clock["prev"] = float(time_left_ms)
        _clock["spent"] = (time.monotonic() - started) * 1000.0


def _infer_increment(time_left_ms: int) -> float:
    """Back the increment out of the clock delta since our last move, or 0 if unknown.

    Our own spend measurement runs a hair short of the referee's, which biases the result
    low - the safe direction, since a smaller increment means a smaller time budget.
    """
    prev = _clock.get("prev")
    spent = _clock.get("spent")
    if prev is None or spent is None:
        return 0.0
    return min(max(time_left_ms - (prev - spent), 0.0), 2000.0)
