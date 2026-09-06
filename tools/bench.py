"""Nodes-per-second bench. Run it before and after any speed change.

    make bench

Not packaged. For each fixed position it runs one real search with a large per-move
budget (``_budget_s`` still takes a slice of it, so each search lasts ~1-2 s), then
prints the depth reached, nodes visited, and nodes per second. Speed work that does not
move "overall nodes/sec" is not speed work.
"""

import time

import chess

import search

POSITIONS = {
    "open middlegame": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P1B2/2PBPN2/PP1N1PPP/R2Q1RK1 w - - 0 9",
    "sharp middlegame": "r2q1rk1/1b1nbppp/p2ppn2/1p6/3NP3/1BN1B3/PPP1QPPP/R4RK1 w - - 0 12",
    "rook endgame": "8/5pk1/6p1/7p/3R3P/6P1/5PK1/3r4 w - - 0 1",
}

BUDGET_MS = 60_000


def main() -> None:
    header = f"{'position':17s} {'depth':>6s} {'nodes':>11s} {'nodes/sec':>12s}"
    print(header)
    print("-" * len(header))
    total_nodes = 0
    total_time = 0.0
    for name, fen in POSITIONS.items():
        search._nodes = 0
        started = time.monotonic()
        search.search_move(chess.Board(fen), BUDGET_MS)
        elapsed = time.monotonic() - started
        total_nodes += search._nodes
        total_time += elapsed
        nps = search._nodes / elapsed if elapsed else 0.0
        print(f"{name:17s} {search._last_depth:6d} {search._nodes:11d} {nps:12,.0f}")
    print("-" * len(header))
    overall = total_nodes / total_time if total_time else 0.0
    print(f"{'overall':17s} {'':>6s} {total_nodes:11d} {overall:12,.0f}")


if __name__ == "__main__":
    main()
