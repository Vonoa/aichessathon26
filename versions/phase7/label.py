"""
Phase 7, step 3: labelling.

Addresses "target leakage -> mimicry" in the leakage checklist: training on
deep-engine centipawn evals and then being scored on how closely your net's
moves match that engine's moves is the exact signature that triggers a
retroactive DQ at finals. The rules permit using an engine to LABEL data --
they don't permit shipping something whose behaviour is "play like Stockfish."

Mitigation used here: every training target is a BLEND of
  (a) the actual game outcome (WDL) from generate_positions.py, and
  (b) a SHALLOW eval (a few plies, not a deep search) of the position,
      run with whatever engine you use as a labeller.

Blending with game outcome and keeping the eval shallow is what keeps this
from becoming a pure engine-imitation target. Do not swap in a deep
(20+ ply) engine eval as the sole target later without re-reading this file's
docstring and the leakage checklist again.

You need to supply `shallow_eval_fn(fen) -> centipawns from White's POV`.
If you have Stockfish available locally for labelling (offline, not shipped),
wire it here via python-chess's UCI engine interface at a LOW depth/time
limit (e.g. depth 6-8, not depth 20+) -- shallow is the point, not a
performance shortcut.
"""

from __future__ import annotations

import math

import chess


def sigmoid_to_wdl(centipawns: float, k: float = 1.0) -> float:
    """Converts a centipawn eval to a win-probability-like scale [0,1],
    same convention as texel_tune.py's sigmoid, so blending is apples-to-apples."""
    return 1.0 / (1.0 + math.exp(-k * centipawns / 400.0))


def label_dataset(
    in_path: str,
    out_path: str,
    shallow_eval_fn,
    blend_weight_outcome: float = 0.5,
) -> None:
    """in_path: fen\\tresult lines (result = game outcome, White's POV, from dedupe.py output)
    out_path: fen\\ttarget lines, target in [0,1], blend of outcome and shallow eval
    blend_weight_outcome: 0.5 = equal blend; raise toward 1.0 to lean on real
        game outcomes more (safer, slower signal); lower to lean on the shallow
        eval more (faster-converging signal, but re-introduces some engine-
        mimicry risk the more you lean on it -- don't push below ~0.3 without
        a specific reason)."""
    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            fen, result_str = line.split("\t")
            outcome = float(result_str)

            cp = shallow_eval_fn(fen)
            eval_component = sigmoid_to_wdl(cp)

            target = blend_weight_outcome * outcome + (1 - blend_weight_outcome) * eval_component
            fout.write(f"{fen}\t{target:.6f}\n")


def make_engine_shallow_eval(budget_s: float = 0.3, clip_cp: float = 3000.0):
    """The real shallow eval this file's docstring calls for: the current
    classical engine (repo root search.py -- diag-19 as of this writing:
    persistent TT, PVS/aspiration, jitted bitboard movegen, NMP/RFP/LMP,
    SEE-filtered quiescence, Syzygy), run through its own iterative-deepening
    loop under a wall-clock budget rather than a fixed depth cap. Whichever
    diag root search.py actually is when this runs IS the teacher -- keep
    this branch merged with main so labelling never falls behind the engine's
    own progress the way it silently did for this project's first several
    labelling runs (root search.py sat on a pre-diag-9 fork the whole time).
    docs/phase7-nnue.md calls for "depth 6-8, fast, noisy" -- but a raw
    fixed-depth call on this engine can run away in sharp, wide-open
    positions (no time bound at all), while a time budget reaches a
    comparable real depth in ordinary quiet middlegames and stays bounded
    everywhere, including the sharp ones. Mate scores are clipped to clip_cp
    before the sigmoid conversion in label_dataset, since an unclipped mate
    score (+-1,000,000) overflows math.exp there.

    The search runs on movegen.py's jitted (bb, state) substrate, not
    chess.Board -- this only touches chess.Board to encode the starting
    position and to check for no-legal-moves (checkmate/stalemate). Reaches
    into search.py's module-level globals (_killers/_hist/_seen/_tt_gen) the
    same way search_move() does, and resets them before every call the same
    way -- labelling calls are on independent positions, so nothing should
    carry over between them the way search state carries across moves in one
    game. The persistent transposition table is deliberately NOT reset -- it
    is a plain cache keyed by position hash, harmless and mildly helpful to
    leave warm across many independent label() calls in one process.
    """
    import sys
    import time
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import movegen as engine_movegen
    import search as engine_search

    def _eval(fen: str) -> float:
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        if not legal:
            return 0.0  # checkmate/stalemate; the outcome label already carries this
        root_is_white = board.turn == chess.WHITE

        bb0, state0 = engine_movegen.encode(board)

        engine_search._nodes = 0
        engine_search._seen = frozenset()
        engine_search._tt_gen = (engine_search._tt_gen + 1) & 0xFFFF
        engine_search._killers[:] = [0] * len(engine_search._killers)
        engine_search._hist[:] = [0] * 4096

        deadline = time.monotonic() + budget_s
        best_code = 0  # 0 = "no move-ordering hint yet", same sentinel search_move() uses
        score = 0
        alpha, beta = -engine_search.MATE - 1, engine_search.MATE + 1
        for depth in range(1, 9):  # 8 plies is already deeper than this budget usually reaches
            try:
                best_code, score = engine_search._aspiration_search(
                    bb0, state0, depth, deadline, best_code, alpha, beta
                )
            except engine_search._Timeout:
                break
            alpha, beta = score - 40, score + 40
            if time.monotonic() >= deadline or abs(score) >= engine_search._MATE_THRESHOLD:
                break

        score = max(-clip_cp, min(clip_cp, float(score)))
        # the search's score is relative to the side to move; sigmoid_to_wdl
        # expects centipawns from White's POV.
        return score if root_is_white else -score

    return _eval


def make_material_shallow_eval():
    """Placeholder shallow eval so this file runs standalone for a smoke test:
    material count only, no search. Replace with a real shallow-depth engine
    call (your own Phase 2/3 engine at low depth, or Stockfish at depth 6-8)
    before labelling your real dataset -- material-only is not a real signal."""
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
              chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}

    def _eval(fen: str) -> float:
        board = chess.Board(fen)
        score = 0
        for piece in board.piece_map().values():
            v = values[piece.piece_type]
            score += v if piece.color == chess.WHITE else -v
        return float(score)

    return _eval


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="deduped.tsv from dedupe.py")
    parser.add_argument("out_path", help="where to write labelled.tsv")
    parser.add_argument(
        "--blend-weight-outcome", type=float, default=0.3,
        help="0.3 default = lambda~0.7 on the shallow eval, per docs/phase7-nnue.md "
             "(and PLAN.md's stated floor of ~0.3 before the mimicry risk needs a "
             "specific reason to go lower)",
    )
    parser.add_argument("--material-only", action="store_true",
                         help="use the material-only placeholder eval instead of a real "
                              "shallow search -- for smoke tests only, not real training")
    parser.add_argument("--budget-s", type=float, default=0.3,
                         help="wall-clock budget per position for the shallow engine "
                              "search used as the label's eval-component signal")
    args = parser.parse_args()

    eval_fn = (
        make_material_shallow_eval()
        if args.material_only
        else make_engine_shallow_eval(budget_s=args.budget_s)
    )
    label_dataset(
        args.in_path, args.out_path, eval_fn, blend_weight_outcome=args.blend_weight_outcome
    )
    print(f"wrote {args.out_path}")
