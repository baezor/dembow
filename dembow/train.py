"""Train the Dembow RBM on a folder of reggaeton MIDI files."""

from __future__ import annotations

import numpy as np
import torch

from .dataset import build_examples, load_songs
from .midi_io import SPAN
from .rbm import RBM, RBMConfig


def train(
    data_dir: str = "reggaeton_samples",
    checkpoint: str = "dembow.pt",
    num_timesteps: int = 15,
    n_hidden: int = 50,
    num_epochs: int = 200,
    batch_size: int = 100,
    lr: float = 0.005,
    k: int = 1,
    seed: int = 0,
    device: str | None = None,
) -> RBM:
    """Train an RBM and save it to ``checkpoint``. Returns the trained model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed)

    print(f"Loading songs from '{data_dir}' ...")
    songs = load_songs(data_dir, steps_per_quarter=4)
    print(f"{len(songs)} songs processed")
    if not songs:
        raise SystemExit("No usable songs found -- nothing to train on.")

    examples = build_examples(songs, num_timesteps)
    print(f"{examples.shape[0]} training windows of {examples.shape[1]} units each")

    n_visible = num_timesteps * 2 * SPAN
    config = RBMConfig(
        n_visible=n_visible,
        n_hidden=n_hidden,
        num_timesteps=num_timesteps,
        span=SPAN,
        steps_per_quarter=4,
    )
    model = RBM(config, device=device, seed=seed)

    data = torch.from_numpy(examples).to(device)
    n = data.shape[0]
    print(f"Training on {device} for {num_epochs} epochs ...")
    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(n, device=device)
        errors = []
        for start in range(0, n, batch_size):
            batch = data[perm[start : start + batch_size]]
            errors.append(model.contrastive_divergence(batch, lr=lr, k=k))
        if epoch % max(1, num_epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{num_epochs}  recon-error {np.mean(errors):.5f}")

    model.save(checkpoint)
    print(f"Saved trained model to '{checkpoint}'")
    return model
