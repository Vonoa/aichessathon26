"""
Phase 7, step 5b: training loop.

Reads a labelled.tsv (fen\ttarget, target in [0,1], from label.py) and trains
SimpleNNUE against it. Random init only (see model.py).

Per docs/phase7-nnue.md's retraining guidance: binary cross-entropy on the win
probability (not MSE -- the target IS a probability, and BCE is the matching
loss for a sigmoid output), a linear-warmup-then-cosine-decay LR schedule
instead of a flat rate, and early stopping on validation loss instead of a
fixed epoch count, since with a small dataset a fixed large epoch count just
means memorizing it (see the train/val loss gap in early runs' logs).

PROVENANCE: every run writes a run log (JSON) recording the git commit, dataset
file + its hash, all hyperparameters, and final loss. This is not optional --
your PLAN.md commits to walking a panel through how any shipped network was
trained, and "we forgot to log it" is not an acceptable answer at that point.
If this script is run from outside a git repo, commit will be logged as null --
fix that before training anything you intend to actually ship.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from features import fen_to_dense_vector
from model import build_model
from torch.utils.data import DataLoader, Dataset


class PositionDataset(Dataset):
    def __init__(self, path: str) -> None:
        self.fens: list[str] = []
        self.targets: list[float] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fen, target = line.split("\t")
                self.fens.append(fen)
                self.targets.append(float(target))

    def __len__(self) -> int:
        return len(self.fens)

    def __getitem__(self, idx: int):
        x = fen_to_dense_vector(self.fens[idx])
        y = np.float32(self.targets[idx])
        return x, y


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def train(
    dataset_path: str,
    out_dir: str,
    epochs: int = 300,
    batch_size: int = 2048,
    lr: float = 3e-3,
    seed: int = 0,
    val_fraction: float = 0.1,
    patience: int = 20,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    full_dataset = PositionDataset(dataset_path)
    n_val = int(len(full_dataset) * val_fraction)
    n_train = len(full_dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )
    # NOTE: this is a random split, not a leakage-safe one. dedupe.py already removed
    # exact/mirror duplicates from the whole dataset before this point, which is what
    # actually matters for the "transposition contamination" risk in PLAN.md -- run
    # dedupe.py on the FULL dataset before this script, not after splitting here.

    # A batch bigger than the dataset itself is meaningless -- clamp it down instead of
    # silently training on one giant batch per epoch when the dataset is still small.
    effective_batch = min(batch_size, max(32, n_train // 10))
    if effective_batch != batch_size:
        print(f"batch_size {batch_size} > dataset supports; using {effective_batch} instead")

    train_loader = DataLoader(train_set, batch_size=effective_batch, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=effective_batch, shuffle=False)

    model = build_model(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()  # target is a probability; matches the sigmoid output

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=steps_per_epoch * epochs,
        pct_start=min(0.1, 200 / max(1, steps_per_epoch * epochs)),
    )

    history = []
    started = time.time()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0
    stopped_epoch = epochs - 1

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss_total += loss.item() * x.size(0)
        train_loss = train_loss_total / n_train

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss_total += loss_fn(pred, y).item() * x.size(0)
        val_loss = val_loss_total / max(n_val, 1)

        print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                stopped_epoch = epoch
                print(f"early stop at epoch {epoch}: no val improvement in {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = out / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    run_log = {
        "timestamp": time.time(),
        "duration_s": time.time() - started,
        "git_commit": _git_commit(),
        "dataset_path": dataset_path,
        "dataset_hash": _file_hash(dataset_path),
        "dataset_size": len(full_dataset),
        "n_train": n_train,
        "n_val": n_val,
        "hyperparameters": {
            "epochs_requested": epochs,
            "epochs_run": stopped_epoch + 1,
            "batch_size": effective_batch,
            "lr": lr,
            "seed": seed,
            "val_fraction": val_fraction,
            "patience": patience,
            "loss": "bce",
            "lr_schedule": "one_cycle",
        },
        "best_val_loss": best_val_loss,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
    run_log_path = out / f"run_{int(time.time())}.json"
    with open(run_log_path, "w") as f:
        json.dump(run_log, f, indent=2)

    print(f"saved checkpoint (best val_loss={best_val_loss:.6f}): {checkpoint_path}")
    print(f"saved run log: {run_log_path}")
    if run_log["git_commit"] is None:
        print("WARNING: not run inside a git repo -- commit is null in the run log. "
              "Fix this before training anything you intend to ship; the panel will ask.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to labelled.tsv from label.py")
    parser.add_argument("--out", default="runs/", help="output directory")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    train(args.dataset, args.out, epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, seed=args.seed, patience=args.patience)
