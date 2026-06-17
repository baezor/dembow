"""Train the Dembow music Transformer on a folder of reggaeton MIDI."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .data import build_windows, encode_corpus
from .model import ModelConfig, MusicTransformer
from .tokenizer import PAD, VOCAB


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
    seed: int = 0,
    device: str | None = None,
) -> MusicTransformer:
    """Train the Transformer and save it to ``checkpoint``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    transpositions = list(range(-augment, augment + 1)) if augment else [0]
    print(f"Loading songs from '{data_dir}' (pitch augmentation: {transpositions}) ...")
    songs = encode_corpus(data_dir, transpositions=transpositions)
    print(f"{len(songs)} song variations encoded")
    windows = build_windows(songs, seq_len)
    if windows.shape[0] == 0:
        raise SystemExit("No usable training windows -- nothing to train on.")
    print(f"{windows.shape[0]} training windows of length {seq_len}  (vocab {len(VOCAB)})")

    config = ModelConfig(
        vocab_size=len(VOCAB), d_model=d_model, n_layers=n_layers,
        n_heads=n_heads, max_len=seq_len,
    )
    model = MusicTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.2f}M parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    data = torch.from_numpy(windows).to(device)
    n = data.shape[0]

    print(f"Training for {num_epochs} epochs ...")
    model.train()
    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch_size):
            batch = data[perm[start : start + batch_size]]
            inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=PAD
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        if epoch % max(1, num_epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{num_epochs}  loss {np.mean(losses):.4f}")

    model.save(checkpoint)
    print(f"Saved trained model to '{checkpoint}'")
    return model
