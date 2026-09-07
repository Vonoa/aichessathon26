"""
Phase 7, step 4: feature extraction.

Converts a FEN into the input representation the network trains on and runs on.

Design choice: 768 = 64 squares x 6 piece types x 2 colors, one-hot, from White's
absolute perspective (not side-relative). This is the simplest correct NNUE-style
input and a reasonable starting point -- real NNUE nets use king-relative features
(HalfKP or similar) for much better generalization, but that's a meaningfully bigger
step in complexity. Get this simple version training and beating nothing (i.e. just
working end-to-end) before reaching for king-relative features.

Side-to-move is appended as an extra input bit (769th feature) so the network can
tell whose turn it is -- without this, the same board with different side-to-move
looks identical to the network, which is wrong (see evaluate.py's own convention:
score is always returned relative to the side to move).
"""

from __future__ import annotations

import chess

N_SQUARES = 64
N_PIECE_TYPES = 6  # pawn, knight, bishop, rook, queen, king
N_COLORS = 2
N_PIECE_FEATURES = N_SQUARES * N_PIECE_TYPES * N_COLORS  # 768
N_FEATURES = N_PIECE_FEATURES + 1  # + side-to-move bit = 769

_PIECE_TYPE_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def fen_to_sparse_indices(fen: str) -> list[int]:
    """Returns the list of feature indices that are 'on' (value 1) for this position.
    Everything else is implicitly 0 -- this is what makes the input sparse (at most
    32 piece features + 1 side-to-move feature are ever on, out of 769)."""
    board = chess.Board(fen)
    indices = []

    for square, piece in board.piece_map().items():
        color_idx = 0 if piece.color == chess.WHITE else 1
        type_idx = _PIECE_TYPE_INDEX[piece.piece_type]
        feature_idx = (type_idx * N_COLORS + color_idx) * N_SQUARES + square
        indices.append(feature_idx)

    if board.turn == chess.BLACK:
        indices.append(N_PIECE_FEATURES)  # the side-to-move bit

    return indices


def fen_to_dense_vector(fen: str):
    """Dense float32 vector version, for feeding directly into a small torch model
    during training (sparse embedding lookups are the efficient way to do this at
    scale, but a dense vector is simpler to get correct first -- 769 floats per
    sample is small enough that this isn't a real bottleneck at your dataset sizes)."""
    import numpy as np

    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for idx in fen_to_sparse_indices(fen):
        vec[idx] = 1.0
    return vec


if __name__ == "__main__":
    # Smoke test: starting position should have exactly 32 piece features on, plus
    # 0 side-to-move features (White to move -> the bit is 0, not appended).
    start_fen = chess.STARTING_FEN
    indices = fen_to_sparse_indices(start_fen)
    print(f"starting position: {len(indices)} features on (expect 32)")
    assert len(indices) == 32, "starting position should have exactly 32 pieces"

    # Flip side to move, same position otherwise -- should be 33 (the extra bit).
    board = chess.Board(start_fen)
    board.turn = chess.BLACK
    indices_black_to_move = fen_to_sparse_indices(board.fen())
    print(f"same position, black to move: {len(indices_black_to_move)} features on (expect 33)")
    assert len(indices_black_to_move) == 33

    print("feature extraction smoke test passed")
