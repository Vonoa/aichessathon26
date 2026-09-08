"""
Quiet-filters a raw positions file (game_id\\tfen\\tresult) and randomly
subsamples it down to a target count, in one pass. Used for the Lichess bulk
data, which pgn_to_positions.py samples without the quiet-only filter
generate_positions.py applies (that filter needs board.legal_moves(), which
is cheap enough to run retroactively over millions of rows but not worth
duplicating into two code paths -- this is that filter, reused here).

Random reservoir-free approach: since the file fits in memory (a few hundred
MB of text for a few million rows), this just filters everything first, then
random.sample()s the target count from whatever survives -- simpler and just
as correct as true reservoir sampling at this size.
"""

from __future__ import annotations

import argparse
import random
import sys

import chess

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
from generate_positions import _is_quiet  # noqa: E402


def filter_and_subsample(in_path: str, out_path: str, target: int, min_fullmove: int, seed: int) -> tuple[int, int, int]:
    kept = []
    n_in = 0
    with open(in_path, encoding="utf-8", errors="replace") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            if n_in % 200_000 == 0:
                print(f"scanned {n_in}, kept {len(kept)} so far...", file=sys.stderr)
            _, fen, _result = line.split("\t")
            board = chess.Board(fen)
            if board.fullmove_number < min_fullmove:
                continue
            if _is_quiet(board):
                kept.append(line)

    rng = random.Random(seed)
    sample = kept if len(kept) <= target else rng.sample(kept, target)
    with open(out_path, "w") as fout:
        for line in sample:
            fout.write(line + "\n")
    return n_in, len(kept), len(sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    parser.add_argument("--target", type=int, default=30_000)
    parser.add_argument("--min-fullmove", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_in, n_quiet, n_sampled = filter_and_subsample(
        args.in_path, args.out_path, args.target, args.min_fullmove, args.seed
    )
    print(f"read {n_in}, quiet {n_quiet} ({n_quiet / n_in:.1%}), sampled {n_sampled} -> {args.out_path}")
