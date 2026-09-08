"""
Phase 7, step 5c-alt: export the trained torch model to plain numpy arrays.

An alternative to export_onnx.py's ONNX export, for the incremental-accumulator
runtime path (nnue_agent/nnue_incremental.py) instead of the onnxruntime path
(nnue_eval.py). SimpleNNUE is tiny -- 769->256->32->1 -- so at inference time
there is no real need for onnxruntime's graph executor at all; a plain numpy
(or numba-jitted) forward pass over these arrays is both simpler and faster,
since it avoids onnxruntime's per-call dispatch overhead entirely. torch
itself is still training-only -- this just swaps the runtime target from
"onnx graph" to "raw arrays", not from "no torch at runtime" (that was never
true either way).
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from model import SimpleNNUE


def export(checkpoint_path: str, out_path: str) -> None:
    model = SimpleNNUE()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    np.savez(
        out_path,
        fc1_weight=model.fc1.weight.detach().numpy().astype(np.float32),
        fc1_bias=model.fc1.bias.detach().numpy().astype(np.float32),
        fc2_weight=model.fc2.weight.detach().numpy().astype(np.float32),
        fc2_bias=model.fc2.bias.detach().numpy().astype(np.float32),
        fc3_weight=model.fc3.weight.detach().numpy().astype(np.float32),
        fc3_bias=model.fc3.bias.detach().numpy().astype(np.float32),
    )
    print(f"exported weights to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to model.pt from train.py")
    parser.add_argument("--out", default="weights.npz")
    args = parser.parse_args()
    export(args.checkpoint, args.out)
