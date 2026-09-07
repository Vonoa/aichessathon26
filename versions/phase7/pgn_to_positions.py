"""
Phase 7, step 1b: turn real human games (a PGN file) into the same
game_id \\t fen \\t result rows that generate_positions.py produces, so both
sources feed the same dedupe.py -> label.py -> train.py chain unchanged.

Deliberately reads ONLY the board position and the final game result from each
PGN -- never a move-annotation comment. Some PGN exports embed an engine's own
centipawn eval per move ("{ [%eval 0.23] }"); if that ever became a training
target it would be exactly the "target leakage -> mimicry" problem in
PLAN.md's leakage checklist (training on deep-engine centipawns risks a net
that imitates that engine's moves). Walking the game via board.push(move) and
reading board.fen() never touches move comments, so this is safe by
construction, not by a filter that could be forgotten.

Games with no decisive-or-drawn result ("*", ongoing/aborted), and non-standard
variants (Chess960, Crazyhouse, ...), are skipped: features.py's encoding
assumes a standard board, and an unclear result has no WDL label to blend.
"""

from __future__ import annotations

import sys

import chess
import chess.pgn


def outcome_to_result(result_tag: str) -> float | None:
    return {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result_tag)


def game_to_fens(game: chess.pgn.Game, sample_every: int, max_plies: int) -> list[str]:
    board = game.board()
    fens = []
    for ply, move in enumerate(game.mainline_moves()):
        if ply >= max_plies:
            break
        if ply % sample_every == 0:
            fens.append(board.fen())
        board.push(move)
    return fens


def convert(pgn_paths: list[str], out_path: str, sample_every: int = 4, max_plies: int = 120) -> tuple[int, int]:
    """Returns (n_games_read, n_games_used)."""
    n_read = 0
    n_used = 0
    game_id = 0
    with open(out_path, "w") as fout:
        for pgn_path in pgn_paths:
            with open(pgn_path, encoding="utf-8", errors="replace") as fin:
                while True:
                    game = chess.pgn.read_game(fin)
                    if game is None:
                        break
                    n_read += 1

                    variant = game.headers.get("Variant", "Standard")
                    if variant not in ("Standard", "From Position"):
                        continue
                    result = outcome_to_result(game.headers.get("Result", "*"))
                    if result is None:
                        continue

                    fens = game_to_fens(game, sample_every, max_plies)
                    if not fens:
                        continue
                    for fen in fens:
                        fout.write(f"{game_id}\t{fen}\t{result}\n")
                    game_id += 1
                    n_used += 1
                    if n_used % 500 == 0:
                        print(f"converted {n_used} games...", file=sys.stderr)
    return n_read, n_used


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pgn_paths", nargs="+", help="one or more .pgn files")
    parser.add_argument("--out", default="raw_positions_human.tsv")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=120)
    args = parser.parse_args()

    n_read, n_used = convert(args.pgn_paths, args.out, args.sample_every, args.max_plies)
    skipped = n_read - n_used
    print(f"read {n_read} games, used {n_used}, skipped {skipped} "
          f"(non-standard variant or no clear result)")
    print(f"wrote {args.out}")
