"""The submission entrypoint. The platform imports this file and calls get_move."""

import chess

from search import search_move

# Import time runs once per game, inside a 90 second budget, before your clock starts.
# Load weights and build tables out here, not inside get_move. Warm every numba-jitted
# function here too, so compilation lands in the init budget rather than on the clock.


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
    try:
        return search_move(board, time_left_ms)
    except Exception:
        # A bug in the search must never forfeit the game: fall back to any legal move.
        return next(iter(board.legal_moves)).uci()
