"""Stream a Lichess ``.pgn.zst`` monthly dump into ``game_id <TAB> fen <TAB> result``
rows -- the same format tuning/carlsen/*.tsv and tools/texel_tune.py already use.

Only WHITE-to-move positions are sampled and ``result`` is the White-POV game outcome
(1.0 / 0.5 / 0.0), matching the existing files, so no side-to-move flipping is needed
anywhere downstream. Never reads a move comment (no ``[%eval ...]`` leakage): the game
is walked by ``board.push`` and positions come from ``board.fen()``.

Filters, all aimed at "positions where a real mistake could happen": standard variant,
decisive-or-drawn result, both players inside a rating band (default 1600-2600 -- keeps
club-to-expert games with genuine king hunts, drops beginner noise and lifts the draw
rate), and a minimum length so quick resignations / aborts are skipped. Opening plies
are skipped (theory), and in-check positions are dropped (not quiet).

    uv run python -m tools.lichess_positions tuning/lichess_db_standard_rated_2014-09.pgn.zst \
        --out tuning/raw_positions_lichess_2014-09.tsv --max-positions 4000000
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import zstandard

_RESULT = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}


def _elo(headers: chess.pgn.Headers, key: str) -> int | None:
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return None


def extract(
    src: Path,
    out: Path,
    min_elo: int,
    max_elo: int,
    sample_every: int,
    skip_plies: int,
    max_plies: int,
    min_plies: int,
    max_positions: int,
) -> None:
    dctx = zstandard.ZstdDecompressor()
    n_read = n_used = n_pos = 0
    started = time.time()
    with src.open("rb") as raw, out.open("w", encoding="utf-8") as fout:
        text = io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8", errors="replace")
        game_id = 0
        while n_pos < max_positions:
            try:
                game = chess.pgn.read_game(text)
            except (ValueError, RuntimeError):
                continue
            if game is None:
                break
            n_read += 1
            if n_read % 20_000 == 0:
                rate = n_pos / max(1e-9, time.time() - started)
                print(
                    f"  {n_read:>8} games  {n_used:>7} used  {n_pos:>9} positions  ({rate:,.0f}/s)",
                    file=sys.stderr,
                )

            h = game.headers
            if h.get("Variant", "Standard") != "Standard":
                continue
            result = _RESULT.get(h.get("Result", "*"))
            if result is None:
                continue
            we, be = _elo(h, "WhiteElo"), _elo(h, "BlackElo")
            if we is None or be is None or min(we, be) < min_elo or max(we, be) > max_elo:
                continue

            board = game.board()
            rows: list[str] = []
            ply = -1  # stays -1 for a 0-move game, so the length check below drops it
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= max_plies:
                    break
                if (
                    ply >= skip_plies
                    and ply % 2 == 0  # White to move
                    and (ply // 2) % sample_every == 0
                    and not board.is_check()
                ):
                    rows.append(f"{game_id}\t{board.fen()}\t{result}\n")
                board.push(move)
            if ply + 1 < min_plies or not rows:
                continue
            fout.writelines(rows)
            n_pos += len(rows)
            n_used += 1
            game_id += 1

    dt = time.time() - started
    print(
        f"read {n_read} games, used {n_used}, wrote {n_pos} positions in {dt:.0f}s -> {out}",
        file=sys.stderr,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="lichess .pgn.zst dump")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-elo", type=int, default=1600)
    p.add_argument("--max-elo", type=int, default=2600)
    p.add_argument("--sample-every", type=int, default=6, help="one FEN per this many full moves")
    p.add_argument("--skip-plies", type=int, default=12, help="ignore this many opening plies")
    p.add_argument("--max-plies", type=int, default=160)
    p.add_argument("--min-plies", type=int, default=24, help="skip games shorter than this")
    p.add_argument(
        "--max-positions",
        type=int,
        default=20_000_000,
        help="hard cap; a 2014 month is well under this",
    )
    args = p.parse_args()
    extract(
        args.src,
        args.out,
        args.min_elo,
        args.max_elo,
        args.sample_every,
        args.skip_plies,
        args.max_plies,
        args.min_plies,
        args.max_positions,
    )


if __name__ == "__main__":
    main()
