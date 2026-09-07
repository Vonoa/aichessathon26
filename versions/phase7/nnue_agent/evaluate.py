"""Evaluation via the Phase 7 NNUE network, for arena-testing it against the
hand-crafted eval. search.py imports evaluate() from this module exactly like
it does from the real evaluate.py -- same contract (chess.Board -> int
centipawns, relative to the side to move), different implementation.

Single-position ONNX inference, one call per leaf -- unbatched, on purpose,
because this is a first correctness/strength test, not the speed-optimised
integration. nnue_eval.py's own docstring flags batching leaf evals into one
onnxruntime call as separate, nontrivial work; this file does not attempt it.
Expect far fewer nodes/sec than the real evaluate.py's jitted eval, so give
both sides a generous time control when arena-testing this, or the comparison
just measures "who survives ONNX call overhead" instead of eval quality.
"""

from __future__ import annotations

import math
from pathlib import Path

import chess
import numpy as np
import onnxruntime as ort

from features import fen_to_dense_vector

_MODEL_PATH = Path(__file__).resolve().parent / "model.onnx"

_options = ort.SessionOptions()
_options.intra_op_num_threads = 1
_options.inter_op_num_threads = 1
_session = ort.InferenceSession(
    str(_MODEL_PATH), sess_options=_options, providers=["CPUExecutionProvider"]
)
_input_name = _session.get_inputs()[0].name


def _score_to_centipawns(score_0_to_1: float, k: float = 400.0) -> int:
    score_0_to_1 = min(max(score_0_to_1, 1e-6), 1 - 1e-6)
    return int(k * math.log(score_0_to_1 / (1 - score_0_to_1)))


def evaluate(board: chess.Board) -> int:
    vec = fen_to_dense_vector(board.fen())
    batch = np.expand_dims(vec, axis=0)
    output = _session.run(None, {_input_name: batch})[0]
    white_pov_cp = _score_to_centipawns(float(output[0]))
    return white_pov_cp if board.turn == chess.WHITE else -white_pov_cp
