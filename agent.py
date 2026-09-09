"""The submission entrypoint. The platform imports this file and calls get_move."""

import time
import traceback

import chess

import movegen
import search

# Bump this on every upload. It prints once at import, so the per-game log the platform
# keeps names exactly which build played that game.
BUILD = "diag-10 texel pawn-structure"
print(f"agent build: {BUILD}", flush=True)

# Import time runs once per game, inside a 90 second budget, before your clock starts.
# Warm the jitted search path here so numba compiles it inside that budget, not on the
# clock during the first move.
search.warm_up()

# Every position we have been asked to move in this game, keyed by its Zobrist hash. The
# process is fresh per game so this resets on its own. The search reads it to spot a line
# that repeats a position the game has already seen and score it as a draw.
_history: dict[int, int] = {}

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
    key = movegen.zobrist(board)
    _history[key] = _history.get(key, 0) + 1

    increment_ms = _infer_increment(time_left_ms)
    started = time.monotonic()
    try:
        return search.search_move(board, time_left_ms, _history, increment_ms)
    except Exception as exc:
        # A bug in the search must never forfeit the game: fall back to any legal move.
        # Print the failure so a rated game that hits this path can be diagnosed -- the
        # platform keeps stderr, and a silent fallback would leave no trace.
        print(f"FALLBACK on {fen!r}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
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
