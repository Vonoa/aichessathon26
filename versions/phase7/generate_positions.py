"""
Phase 7, step 1: position generation.

Addresses the "opening distribution shift" item in the leakage checklist --
rated games start from curated positions we never see, so training only from
the standard opening biases the net toward positions it won't actually face.

Strategy: play a random number of "reasonable" opening plies (weighted toward
sensible moves via a shallow eval, not literally uniform-random legal moves,
which produces garbage positions no real game reaches), then continue with
your actual engine (import your search module) for some plies to reach
middlegame/endgame-ish positions, sampling one FEN every few plies.

Output: a flat file of one FEN per line, tagged with a game id so you can
later attach game outcomes to every sampled position from that game.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass

import chess


@dataclass
class GameRecord:
    game_id: int
    fens: list[str]
    result: float | None = None  # filled in later, from game outcome


def weighted_random_move(board: chess.Board, rng: random.Random, greediness: float = 0.6) -> chess.Move:
    """Picks a legal move, weighted toward captures and central development
    early on, so 'random' openings still look like plausible chess rather than
    hanging pieces on move 3. greediness in [0,1]: 0 = uniform random,
    1 = always take the move a simple heuristic ranks best."""
    moves = list(board.legal_moves)
    if rng.random() > greediness:
        return rng.choice(moves)

    def heuristic(m: chess.Move) -> float:
        score = 0.0
        if board.is_capture(m):
            score += 3.0
        to_file = chess.square_file(m.to_square)
        to_rank = chess.square_rank(m.to_square)
        # mild pull toward the center
        score += 1.0 - (abs(3.5 - to_file) + abs(3.5 - to_rank)) / 7.0
        score += rng.random() * 0.5  # tie-break noise
        return score

    return max(moves, key=heuristic)


def play_random_opening(rng: random.Random, min_plies: int = 4, max_plies: int = 16) -> chess.Board:
    board = chess.Board()
    n_plies = rng.randint(min_plies, max_plies)
    for _ in range(n_plies):
        if board.is_game_over():
            break
        move = weighted_random_move(board, rng)
        board.push(move)
    return board


def continue_with_engine(board: chess.Board, get_move_fn, rng: random.Random,
                          max_plies: int = 60, sample_every: int = 4,
                          time_left_ms: int = 250) -> list[str]:
    """get_move_fn(fen, time_left_ms) -> uci string. Pass your real agent's
    get_move so generated positions reflect positions YOUR engine actually
    reaches, not some other distribution. Add a small amount of move noise
    (occasionally play the 2nd-best move) upstream in get_move_fn if you want
    more positional diversity -- not done here to keep this file engine-agnostic."""
    fens = []
    for ply in range(max_plies):
        if board.is_game_over():
            break
        if ply % sample_every == 0:
            fens.append(board.fen())
        uci = get_move_fn(board.fen(), time_left_ms)
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break  # defensive: don't poison the dataset with an illegal-move bug
        board.push(move)
    return fens


def outcome_to_result(board: chess.Board) -> float:
    """1.0 = white won, 0.0 = black won, 0.5 = draw/unfinished."""
    result = board.result(claim_draw=True)
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result, 0.5)


def generate_dataset(
    n_games: int,
    out_path: str,
    get_move_fn,
    seed: int = 0,
    time_left_ms: int = 1000,
    reset_fn=None,
) -> None:
    """reset_fn, if given, is called before each simulated game. Required when
    get_move_fn wraps a module with cross-move state meant to persist for one real
    game (agent.py's `_history` / `_clock` globals) -- without resetting it, that
    state leaks between the independent games generated here and corrupts
    repetition detection and increment inference for every game after the first."""
    rng = random.Random(seed)
    with open(out_path, "w") as f:
        for game_id in range(n_games):
            if reset_fn is not None:
                reset_fn()
            board = play_random_opening(rng)
            fens = [board.fen()]
            fens += continue_with_engine(
                board, get_move_fn, rng, time_left_ms=time_left_ms
            )
            result = outcome_to_result(board)
            for fen in fens:
                f.write(f"{game_id}\t{fen}\t{result}\n")
            if game_id % 50 == 0:
                print(f"generated {game_id}/{n_games} games", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=150)
    parser.add_argument("--out", default="raw_positions_selfplay.tsv")
    parser.add_argument("--time-ms", type=int, default=1000,
                         help="fake clock handed to the real agent for each generated "
                              "move -- small on purpose, this is bulk data generation, "
                              "not a real game")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    import agent  # noqa: E402  (path must be set up first)

    def _reset_agent_state() -> None:
        agent._history.clear()
        agent._clock.clear()

    generate_dataset(
        n_games=args.games,
        out_path=args.out,
        get_move_fn=agent.get_move,
        seed=args.seed,
        time_left_ms=args.time_ms,
        reset_fn=_reset_agent_state,
    )
    print(f"wrote {args.out}")