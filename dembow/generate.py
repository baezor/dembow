"""Generate new reggaeton from a trained Dembow RBM."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from .dataset import find_midi_files
from .midi_io import SPAN, midi_to_note_state_matrix, note_state_matrix_to_midi
from .rbm import RBM


def _seed_from_song(path: str, num_timesteps: int, n_visible: int) -> Optional[torch.Tensor]:
    """Build a single visible vector from the start of a real reggaeton song.

    Seeding the Gibbs chain with a genuine groove -- rather than silence -- keeps
    the model in the dembow pocket, which is the whole point of the project.
    """
    song = midi_to_note_state_matrix(path, steps_per_quarter=4)
    width = 2 * SPAN
    if song.shape[0] < num_timesteps:
        return None
    window = song[:num_timesteps].reshape(1, num_timesteps * width)
    return torch.from_numpy(window.astype(np.float32))


def generate(
    checkpoint: str = "dembow.pt",
    output_dir: str = "generated",
    num_samples: int = 10,
    k: int = 1,
    seed_dir: Optional[str] = "reggaeton_samples",
    tempo_bpm: float = 95.0,
    random_seed: int = 0,
    device: str | None = None,
) -> List[str]:
    """Sample patterns from the model and write them out as MIDI files."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = RBM.load(checkpoint, device=device)
    cfg = model.config
    os.makedirs(output_dir, exist_ok=True)

    # Try to seed each chain from a real song so the output grooves.
    init = None
    if seed_dir and os.path.isdir(seed_dir):
        seeds = []
        for path in find_midi_files(seed_dir):
            vec = _seed_from_song(path, cfg.num_timesteps, cfg.n_visible)
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

    print(f"Wrote {len(written)} MIDI file(s) to '{output_dir}/'")
    return written
