"""Train the Dembow RBM on a folder of reggaeton MIDI files."""

from __future__ import annotations

import numpy as np
import torch

from .dataset import build_examples, build_sequences, load_feature_songs, load_songs
from .lstm import DembowLSTM, LSTMConfig
from .midi_io import SPAN
from .rbm import RBM, RBMConfig
from .representation import DRUM_SLICE, N_DRUMS, N_FEATURES


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


def train_lstm(
    data_dir: str = "reggaeton_samples",
    checkpoint: str = "dembow.pt",
    seq_len: int = 64,
    hidden: int = 256,
    layers: int = 2,
    num_epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    pos_weight: float = 4.0,
    normalize_key: bool = True,
    seed: int = 0,
    device: str | None = None,
) -> DembowLSTM:
    """Train the sequence model on the corpus and save it to ``checkpoint``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Loading songs from '{data_dir}' (drums separated, key-normalized={normalize_key}) ...")
    songs = load_feature_songs(data_dir, normalize_key=normalize_key)
    print(f"{len(songs)} songs processed")
    sequences = build_sequences(songs, seq_len)
    if sequences.shape[0] == 0:
        raise SystemExit("No usable sequences found -- nothing to train on.")
    print(f"{sequences.shape[0]} training sequences of length {seq_len}")

    model = DembowLSTM(LSTMConfig(n_features=N_FEATURES, hidden=hidden, layers=layers)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Positive note events are rare; weight them so the model doesn't just
    # predict silence everywhere.
    weight = torch.full((N_FEATURES,), pos_weight, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)

    data = torch.from_numpy(sequences).to(device)
    n = data.shape[0]
    print(f"Training LSTM on {device} for {num_epochs} epochs ...")
    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, batch_size):
            batch = data[perm[start : start + batch_size]]
            inputs, targets = batch[:, :-1, :], batch[:, 1:, :]
            logits, _ = model(inputs)
            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss))
        if epoch % max(1, num_epochs // 20) == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{num_epochs}  loss {np.mean(losses):.4f}")

    model.save(checkpoint)
    print(f"Saved trained LSTM to '{checkpoint}'")
    return model
