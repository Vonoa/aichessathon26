"""
Phase 7, step 1c: turn the Carlsen career CSV export (one row per ply, with a
SAN move column, not a PGN file) into the same game_id \\t fen \\t result rows
pgn_to_positions.py produces, so this source feeds the same
dedupe.py -> label.py -> train.py chain unchanged.

The CSV's own "fen" column is board-placement only (no side-to-move, castling
rights, en passant square, or move clocks), so it cannot be fed to
chess.Board() directly. Instead this replays each game from the standard
start position via board.push_san() on the CSV's "notation" column, which
gives an exact, fully-populated FEN at every ply -- the same approach
pgn_to_positions.py uses on real PGN movetext, applied to a SAN column
instead. Only the SAN notation and the final game result are read, same as
pgn_to_positions.py, for the same reason: an engine-annotated eval column
would be a target-leakage risk per PLAN.md's checklist, and this CSV carries
none anyway.

A game whose SAN fails to replay (a malformed row, a rare non-standard
notation) is dropped from the point of failure onward; the positions already
written for it up to that point are kept, since they were reached by moves
that verified fine.
"""

from __future__ import annotations

import csv
import sys

import chess

from pgn_to_positions import outcome_to_result


def load_results(game_info_path: str) -> dict[str, float]:
    results: dict[str, float] = {}
    with open(game_info_path, encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            result = outcome_to_result(row["result"])
            if result is not None:
                results[row["game_id"]] = result
    return results


def convert(
    moves_csv_path: str,
    game_info_csv_path: str,
    out_path: str,
    sample_every: int = 4,
    max_plies: int = 120,
) -> tuple[int, int]:
    """Returns (n_games_seen, n_games_used)."""
    results = load_results(game_info_csv_path)

    n_games_seen = 0
    n_games_used = 0
    current_game_id: str | None = None
    board = chess.Board()
    ply = 0
    result: float | None = None
    alive = False  # False once this game's SAN replay has failed or ended

    with open(moves_csv_path, encoding="utf-8", errors="replace", newline="") as fin, \
            open(out_path, "w") as fout:
        for row in csv.DictReader(fin):
            gid = row["game_id"]
            if gid != current_game_id:
                current_game_id = gid
                n_games_seen += 1
                board = chess.Board()
                ply = 0
                result = results.get(gid)
                alive = result is not None
                if alive:
                    n_games_used += 1
                if n_games_seen % 500 == 0:
                    print(f"processed {n_games_seen} games...", file=sys.stderr)

            if not alive or ply >= max_plies:
                continue

            if ply % sample_every == 0:
                fout.write(f"{gid}\t{board.fen()}\t{result}\n")

            try:
                board.push_san(row["notation"])
            except (ValueError, chess.IllegalMoveError, chess.AmbiguousMoveError):
                alive = False
                continue
            ply += 1

    return n_games_seen, n_games_used


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("moves_csv", help="e.g. Carlsen_moves.csv (one row per ply)")
    parser.add_argument("game_info_csv", help="e.g. Carlsen_game_info.csv (one row per game)")
    parser.add_argument("--out", default="raw_positions_carlsen.tsv")
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=120)
    args = parser.parse_args()

    n_seen, n_used = convert(
        args.moves_csv, args.game_info_csv, args.out, args.sample_every, args.max_plies
    )
    print(f"read {n_seen} games, used {n_used} (had a decisive/drawn result), "
          f"skipped {n_seen - n_used}")
    print(f"wrote {args.out}")
