"""
Test B from the teammate's overnight-run report: is 0.556-ish val_loss a net
limitation or a label ceiling? Scores the classical eval's OWN sigmoid output
against the already-blended labels on the same val split train.py used
(same seed, same random_split), via BCE. If the classical eval scores about
the same as the net, the labels themselves are the ceiling -- no architecture
change (HalfKP included) would do better on this label recipe.
"""
from __future__ import annotations

import math
import sys

import torch

from label import make_engine_shallow_eval, sigmoid_to_wdl


def main(labelled_path: str, seed: int, val_fraction: float, budget_s: float) -> None:
    rows = []
    with open(labelled_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            fen, target = line.split("\t")
            rows.append((fen, float(target)))

    n_val = int(len(rows) * val_fraction)
    n_train = len(rows) - n_val
    train_idx, val_idx = torch.utils.data.random_split(
        range(len(rows)), [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    val_rows = [rows[i] for i in val_idx.indices]
    print(f"scoring classical eval on {len(val_rows)} val positions (budget_s={budget_s})...",
          file=sys.stderr)

    eval_fn = make_engine_shallow_eval(budget_s=budget_s)
    bce_sum = 0.0
    eps = 1e-7
    for i, (fen, target) in enumerate(val_rows):
        cp = eval_fn(fen)
        p = sigmoid_to_wdl(cp)
        p = min(max(p, eps), 1 - eps)
        bce_sum += -(target * math.log(p) + (1 - target) * math.log(1 - p))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(val_rows)}...", file=sys.stderr)

    bce = bce_sum / len(val_rows)
    print(f"classical eval BCE on val split: {bce:.6f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("labelled_path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--budget-s", type=float, default=0.3)
    args = parser.parse_args()
    main(args.labelled_path, args.seed, args.val_fraction, args.budget_s)
