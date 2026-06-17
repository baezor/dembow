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
    groove: str = "auto",
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
            max_pitched, temperature, groove, seed_dir, tempo_bpm, device,
        )
    return _generate_rbm(
        checkpoint, output_dir, num_samples, k, seed_dir, tempo_bpm, device,
    )


def _generate_lstm(checkpoint, output_dir, num_samples, num_steps, prime_steps,
                   max_pitched, temperature, groove, seed_dir, tempo_bpm, device) -> List[str]:
    from .groove import canonical_groove, fallback_groove, tile_groove
    from .representation import DRUM_SLICE

    model = DembowLSTM.load(checkpoint, device=device)

    # Load the corpus once: used both to prime generation and to derive the groove.
    corpus = []
    if seed_dir and os.path.isdir(seed_dir):
        for path in find_midi_files(seed_dir):
            corpus.append(song_to_features(path))

    # Prime each generation with the opening bars of a real reggaeton song so the
    # model starts in the pocket.
    primes = [s[:prime_steps] for s in corpus if s.shape[0] > prime_steps]
    if not primes:
        primes = [np.zeros((prime_steps, N_FEATURES), dtype=np.float32)]

    # Build the dembow drum backbone the model will groove over.
    drum_track = None
    if groove == "auto":
        pattern = canonical_groove(corpus) if corpus else fallback_groove()
        drum_track = tile_groove(pattern, num_steps)
    elif groove == "dembow":
        drum_track = tile_groove(fallback_groove(), num_steps)

    written = []
    for i in range(num_samples):
        prime = primes[i % len(primes)]
        body = _roll_out(
            model, prime, num_steps, max_pitched, temperature, drum_track, device,
        )
        full = np.concatenate([prime, body], axis=0)
        out_path = os.path.join(output_dir, f"dembow_{i}.mid")
        features_to_midi(full, out_path, tempo_bpm=tempo_bpm)
        written.append(out_path)

    print(f"Wrote {len(written)} MIDI file(s) to '{output_dir}/' (LSTM)")
    return written


def _roll_out(model, prime, num_steps, max_pitched, temperature, drum_track, device):
    """Generate one body, re-rolling if drums or melody drift into silence.

    An LSTM can collapse into emptiness (an empty step begets empty steps). We
    try progressively more permissive settings until both the beat (when not
    locked to a groove) and the melody clear a minimum density.
    """
    from .representation import PLAY_SLICE

    attempts = [
        dict(drum_threshold=0.40, pitch_threshold=0.50, temperature=temperature),
        dict(drum_threshold=0.30, pitch_threshold=0.40, temperature=max(temperature, 1.1)),
        dict(drum_threshold=0.22, pitch_threshold=0.32, temperature=max(temperature, 1.25)),
    ]
    body = None
    for params in attempts:
        body = model.generate(
            prime, num_steps=num_steps, max_pitched=max_pitched,
            drum_track=drum_track, device=device, **params,
        )
        drums_ok = drum_track is not None or body[:, DRUM_SLICE].sum() >= num_steps * 0.3
        melody_ok = body[:, PLAY_SLICE].sum() >= num_steps * 0.5
        if drums_ok and melody_ok:
            break
    return body


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
