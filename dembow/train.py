"""Train the Dembow music Transformer on a folder of reggaeton MIDI."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_windows, encode_files, find_midi_files, split_files
from .model import ModelConfig, MusicTransformer
from .tokenizer import PAD, VOCAB


@torch.no_grad()
def _eval_loss(model: MusicTransformer, data: torch.Tensor, batch_size: int) -> float:
    """Average next-token loss over a held-out set."""
    model.eval()
    losses = []
    for start in range(0, data.shape[0], batch_size):
        batch = data[start : start + batch_size]
        logits = model(batch[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=PAD
        )
        losses.append(float(loss))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(
    data_dir: str = "reggaeton_samples",
    checkpoint: str = "dembow.pt",
    seq_len: int = 384,
    d_model: int = 256,
    n_layers: int = 4,
    n_heads: int = 4,
    num_epochs: int = 80,
    batch_size: int = 16,
    lr: float = 3e-4,
    augment: int = 3,
    val_frac: float = 0.1,
    patience: int = 8,
    seed: int = 0,
    device: str | None = None,
) -> MusicTransformer:
    """Train the Transformer and save the best checkpoint (by validation loss)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    transpositions = list(range(-augment, augment + 1)) if augment else [0]
    files = find_midi_files(data_dir)
    train_files, val_files = split_files(files, val_frac, seed=seed)
    print(f"Loading songs from '{data_dir}' (pitch augmentation: {transpositions})")
    print(f"  {len(train_files)} train songs, {len(val_files)} val songs")

    # Train data is augmented; validation is held-out songs at original pitch only.
    train_windows = build_windows(encode_files(train_files, transpositions), seq_len)
    val_windows = build_windows(encode_files(val_files, [0]), seq_len)
    if train_windows.shape[0] == 0:
        raise SystemExit("No usable training windows -- nothing to train on.")
    print(f"  {train_windows.shape[0]} train windows, {val_windows.shape[0]} val windows (vocab {len(VOCAB)})")

    config = ModelConfig(
        vocab_size=len(VOCAB), d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=seq_len,
    )
    model = MusicTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_data = torch.from_numpy(train_windows).to(device)
    val_data = torch.from_numpy(val_windows).to(device) if val_windows.shape[0] else None
    n = train_data.shape[0]

    print(f"Training for up to {num_epochs} epochs (early-stop patience {patience}) ...")
    best_val = float("inf")
    best_epoch = 0
    since_improved = 0
    model.train()
    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch_size):
            batch = train_data[perm[start : start + batch_size]]
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=PAD
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        if val_data is not None:
            val_loss = _eval_loss(model, val_data, batch_size)
            improved = val_loss < best_val - 1e-4
            if improved:
                best_val, best_epoch, since_improved = val_loss, epoch, 0
                model.save(checkpoint)  # keep the best model, not the last
            else:
                since_improved += 1
            print(f"  epoch {epoch:4d}/{num_epochs}  train {train_loss:.4f}  val {val_loss:.4f}"
                  f"{'  *best (saved)' if improved else ''}")
            if since_improved >= patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                break
        else:
            print(f"  epoch {epoch:4d}/{num_epochs}  train {train_loss:.4f}")
            model.save(checkpoint)

    if val_data is not None:
        print(f"Best model: epoch {best_epoch}, val loss {best_val:.4f} -> '{checkpoint}'")
    else:
        print(f"Saved trained model to '{checkpoint}'")
    return model
