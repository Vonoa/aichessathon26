"""
Phase 7, step 2: deduplication.

Addresses "transposition contamination" in the leakage checklist: the same
position reached via different move orders looks like two different rows,
and if a random train/val split puts one copy in train and its near-twin in
val, your validation score is measuring memorization, not generalization.

Run this BEFORE splitting into train/val -- not after. Splitting first and
deduping each half separately does not fix the problem.

Dedup key: python-chess's transposition_key() (board occupancy, castling
rights, ep square, side to move) PLUS the same key computed on the
horizontally mirrored position, so left/right mirror twins collapse too.
"""

from __future__ import annotations

import chess


def transposition_key(board: chess.Board):
    # python-chess's internal cache key already ignores move counters,
    # which is what we want for this purpose.
    return board._transposition_key()


def mirrored_key(board: chess.Board):
    mirrored = board.mirror()
    return mirrored._transposition_key()


def dedupe_file(in_path: str, out_path: str) -> tuple[int, int]:
    """Reads game_id\\tfen\\tresult lines, writes deduped fen\\tresult lines.
    When a position (or its mirror) has been seen before, keeps the FIRST
    occurrence and drops the rest. Returns (n_in, n_out)."""
    seen: set = set()
    n_in = 0
    n_out = 0

    with open(in_path, "r") as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            _, fen, result = line.split("\t")
            board = chess.Board(fen)

            key = transposition_key(board)
            mkey = mirrored_key(board)

            if key in seen or mkey in seen:
                continue

            seen.add(key)
            fout.write(f"{fen}\t{result}\n")
            n_out += 1

    return n_in, n_out


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python dedupe.py <raw_positions.tsv> <deduped.tsv>")
        sys.exit(1)

    n_in, n_out = dedupe_file(sys.argv[1], sys.argv[2])
    dropped = n_in - n_out
    pct = (dropped / n_in * 100) if n_in else 0.0
    print(f"read {n_in} positions, kept {n_out}, dropped {dropped} ({pct:.1f}%) as duplicates/mirrors")
