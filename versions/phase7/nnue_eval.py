"""
Phase 7, step 5d: runtime inference wrapper.

This is what would replace evaluate.py's evaluate() function IF the NNUE net wins
the arena test against Phase 5 -- until then, this is a standalone module, not
wired into agent.py/search.py.

Critical performance point from IDEAS.md and your own PLAN.md: batch leaf
evaluations into one onnxruntime call rather than calling per-node. A single-position
call pays onnxruntime's per-call dispatch overhead on every node, which will be far
slower than evaluate.py's bitboard approach. The NNUEBatchEvaluator below collects
positions and scores them together -- your search.py would need restructuring to
collect a batch of leaf positions before scoring them, rather than calling evaluate()
immediately at each leaf. That restructuring is NOT done here -- this file only
provides the batched scoring primitive; wiring it into search.py's leaf handling is
a separate, nontrivial task since your current quiescence search scores leaves
immediately and recursively, not in a collected batch.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from features import fen_to_dense_vector


class NNUEBatchEvaluator:
    def __init__(self, onnx_path: str) -> None:
        # Single-threaded on purpose -- PLAN.md's performance rules and IDEAS.md's
        # "more threads than cores" warning both apply here. Don't let onnxruntime
        # spawn its own thread pool on a 1-core sandbox.
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def evaluate_batch(self, fens: list[str]) -> list[float]:
        """Returns a [0,1] win-probability-style score per FEN, in the same order.
        Convert to centipawns with score_to_centipawns() if you need evaluate.py's
        convention for comparison."""
        if not fens:
            return []
        batch = np.stack([fen_to_dense_vector(fen) for fen in fens])
        outputs = self.session.run(None, {self.input_name: batch})
        return outputs[0].tolist()

    def evaluate_one(self, fen: str) -> float:
        """Single-position convenience wrapper. Calling this per-node in search.py
        defeats the entire point of batching -- only use this for spot checks /
        tests, never inside the actual search loop."""
        return self.evaluate_batch([fen])[0]


def score_to_centipawns(score_0_to_1: float, k: float = 400.0) -> int:
    """Inverse of texel_tune.py's sigmoid, so a [0,1] NNUE output can be compared
    on the same centipawn scale as evaluate.py's output."""
    import math

    score_0_to_1 = min(max(score_0_to_1, 1e-6), 1 - 1e-6)  # avoid log(0)
    return int(k * math.log(score_0_to_1 / (1 - score_0_to_1)))


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python nnue_eval.py <model.onnx>")
        sys.exit(1)

    evaluator = NNUEBatchEvaluator(sys.argv[1])
    import chess

    fens = [chess.STARTING_FEN]
    scores = evaluator.evaluate_batch(fens)
    for fen, score in zip(fens, scores):
        print(f"{fen[:30]}...  score={score:.4f}  cp={score_to_centipawns(score)}")
