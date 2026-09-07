"""
Phase 7, step 5c: export the trained torch model to ONNX.

Run via onnxruntime at inference time (torch itself is training-only per your
own rules -- IDEAS.md explicitly says onnxruntime startup is faster and inference
on one core is competitive; there's no reason to ship torch inference).

UNTESTED against your actual torch==2.13.0 / onnxruntime==1.29.0 versions -- the
export API (torch.onnx.export) has changed between torch versions before; if this
throws an API-mismatch error, check torch's release notes for 2.13 against whatever
version this was written against and adjust the export call accordingly.
"""

from __future__ import annotations

import argparse

import torch

from features import N_FEATURES
from model import SimpleNNUE


def export(checkpoint_path: str, onnx_path: str) -> None:
    model = SimpleNNUE()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.zeros(1, N_FEATURES, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["score"],
        dynamic_axes={"features": {0: "batch_size"}, "score": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"exported to {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to model.pt from train.py")
    parser.add_argument("--out", default="model.onnx")
    args = parser.parse_args()
    export(args.checkpoint, args.out)
