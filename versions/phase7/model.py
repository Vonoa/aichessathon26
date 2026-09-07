"""
Phase 7, step 5a: the network itself.

Small by design -- per IDEAS.md's own guidance, "a net you can evaluate thousands of
times per move is worth more than a better net you can evaluate fifty times." This is
a plain feedforward net on the 769-feature dense input from features.py, not a real
king-relative NNUE architecture -- start here, get the whole pipeline (train -> export
-> run -> arena-test) working end to end on something simple, THEN consider a fancier
architecture. A correct simple net beats a half-working complex one every time.

Output: a single scalar, trained to predict a value in [0, 1] (via sigmoid), same
scale as texel_tune.py's win-probability convention -- 0 = black wins, 1 = white wins,
0.5 = draw. Convert to centipawns at inference time if you want a score on the same
scale as evaluate.py (see nnue_eval.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from features import N_FEATURES


class SimpleNNUE(nn.Module):
    """Feedforward: 769 -> hidden -> hidden -> 1, sigmoid output.
    Small hidden sizes deliberately -- this needs to run thousands of times per move
    on one CPU core with no GPU. Widen only if arena testing shows it's worth the
    inference cost; measure that cost with tools/bench.py-style timing before widening."""

    def __init__(self, hidden1: int = 256, hidden2: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(N_FEATURES, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x.squeeze(-1)


def build_model(seed: int = 0) -> SimpleNNUE:
    """Random init, explicitly seeded -- required by your own rules (no loading a
    published network) and required by your own working agreement (reproducibility
    for the finals panel). Log this seed in your run log every time you train."""
    torch.manual_seed(seed)
    return SimpleNNUE()
