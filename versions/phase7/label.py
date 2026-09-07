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
    with open(in_path, "r") as fin, open(out_path, "w") as fout:
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
    import sys

    if len(sys.argv) != 3:
        print("usage: python label.py <deduped.tsv> <labelled.tsv>")
        print("NOTE: uses a material-only placeholder eval -- swap in a real")
        print("shallow-depth engine call before using this for real training.")
        sys.exit(1)

    label_dataset(sys.argv[1], sys.argv[2], make_material_shallow_eval())
    print(f"wrote {sys.argv[2]}")
