"""Generate new reggaeton from a trained Dembow model (LSTM or classic RBM)."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from .dataset import find_midi_files
from .lstm import DembowLSTM
from .midi_io import SPAN, midi_to_note_state_matrix, note_state_matrix_to_midi
from .rbm import RBM
from .representation import N_FEATURES, features_to_midi, song_to_features


def _checkpoint_type(checkpoint: str) -> str:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return ckpt.get("model_type", "rbm")


def generate(
    checkpoint: str = "dembow.pt",
    output_dir: str = "generated",
    num_samples: int = 5,
    num_steps: int = 64,
    prime_steps: int = 16,
    max_pitched: int = 5,
    temperature: float = 1.0,
    seed_dir: Optional[str] = "reggaeton_samples",
    tempo_bpm: float = 95.0,
    random_seed: int = 0,
    device: str | None = None,
    # RBM-only knob, kept for the classic mode.
    k: int = 1,
) -> List[str]:
    """Sample patterns from the model and write them out as MIDI files."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    if _checkpoint_type(checkpoint) == "lstm":
        return _generate_lstm(
            checkpoint, output_dir, num_samples, num_steps, prime_steps,
            max_pitched, temperature, seed_dir, tempo_bpm, device,
        )
    return _generate_rbm(
        checkpoint, output_dir, num_samples, k, seed_dir, tempo_bpm, device,
    )


def _generate_lstm(checkpoint, output_dir, num_samples, num_steps, prime_steps,
                   max_pitched, temperature, seed_dir, tempo_bpm, device) -> List[str]:
    model = DembowLSTM.load(checkpoint, device=device)

    # Prime each generation with the opening bars of a real reggaeton song so the
    # model starts in the pocket -- including the dembow drum groove.
    primes = []
    if seed_dir and os.path.isdir(seed_dir):
        for path in find_midi_files(seed_dir):
            song = song_to_features(path)
            if song.shape[0] > prime_steps:
                primes.append(song[:prime_steps])
            if len(primes) >= num_samples:
                break
    if not primes:
        primes = [np.zeros((prime_steps, N_FEATURES), dtype=np.float32)]

    from .representation import DRUM_SLICE

    written = []
    for i in range(num_samples):
        prime = primes[i % len(primes)]
        # An LSTM can drift into silence (an empty step begets empty steps). If
        # the beat dies, re-roll with a more permissive drum threshold so every
        # track the user gets actually grooves.
        body = None
        for drum_threshold in (0.4, 0.3, 0.22):
            body = model.generate(
                prime, num_steps=num_steps, max_pitched=max_pitched,
                drum_threshold=drum_threshold, temperature=temperature, device=device,
            )
            if body[:, DRUM_SLICE].sum() >= num_steps * 0.3:
                break
        full = np.concatenate([prime, body], axis=0)
        out_path = os.path.join(output_dir, f"dembow_{i}.mid")
        features_to_midi(full, out_path, tempo_bpm=tempo_bpm)
        written.append(out_path)

    print(f"Wrote {len(written)} MIDI file(s) to '{output_dir}/' (LSTM)")
    return written


def _seed_from_song(path: str, num_timesteps: int) -> Optional[torch.Tensor]:
    matrix = midi_to_note_state_matrix(path, steps_per_quarter=4)
    width = 2 * SPAN
    if matrix.shape[0] < num_timesteps:
        return None
    window = matrix[:num_timesteps].reshape(1, num_timesteps * width)
    return torch.from_numpy(window.astype(np.float32))


def _generate_rbm(checkpoint, output_dir, num_samples, k, seed_dir, tempo_bpm, device) -> List[str]:
    model = RBM.load(checkpoint, device=device)
    cfg = model.config

    init = None
    if seed_dir and os.path.isdir(seed_dir):
        seeds = []
        for path in find_midi_files(seed_dir):
            vec = _seed_from_song(path, cfg.num_timesteps)
            if vec is not None:
                seeds.append(vec)
            if len(seeds) >= num_samples:
                break
        if seeds:
            init = torch.cat(seeds, dim=0)
            if init.shape[0] < num_samples:
                reps = (num_samples + init.shape[0] - 1) // init.shape[0]
                init = init.repeat(reps, 1)[:num_samples]

    samples = model.generate(num_samples, k=k, init=init).cpu().numpy()
    width = 2 * SPAN
    written = []
    for i, sample in enumerate(samples):
        if not sample.any():
            continue
        matrix = sample.reshape(cfg.num_timesteps, width)
        out_path = os.path.join(output_dir, f"dembow_{i}.mid")
        note_state_matrix_to_midi(matrix, out_path, steps_per_quarter=cfg.steps_per_quarter, tempo_bpm=tempo_bpm)
        written.append(out_path)

    print(f"Wrote {len(written)} MIDI file(s) to '{output_dir}/' (RBM classic)")
    return written
