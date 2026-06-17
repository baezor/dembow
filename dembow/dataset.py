"""Load the reggaeton MIDI corpus and turn it into RBM training data."""

from __future__ import annotations

import glob
import os
from typing import List

import numpy as np

from .midi_io import SPAN, midi_to_note_state_matrix
from .representation import N_FEATURES, song_to_features


def find_midi_files(directory: str) -> List[str]:
    """Return every MIDI file in ``directory`` (case-insensitive extension).

    The original used ``glob('*.mid*')``, which silently skipped the uppercase
    ``.MID`` files in the corpus. We match regardless of case.
    """
    files = glob.glob(os.path.join(directory, "*"))
    return sorted(f for f in files if f.lower().endswith((".mid", ".midi")))


def load_songs(directory: str, min_steps: int = 50, steps_per_quarter: int = 4) -> List[np.ndarray]:
    """Parse every usable song in ``directory`` into piano-roll matrices."""
    songs = []
    for path in find_midi_files(directory):
        try:
            song = midi_to_note_state_matrix(path, steps_per_quarter=steps_per_quarter)
        except Exception as exc:  # a few corpus files have quirky encodings
            print(f"  skipping {os.path.basename(path)}: {type(exc).__name__}: {exc}")
            continue
        if song.shape[0] > min_steps:
            songs.append(song)
    return songs


def build_examples(songs: List[np.ndarray], num_timesteps: int) -> np.ndarray:
    """Slice songs into fixed-length windows flattened for the RBM.

    Each training example is ``num_timesteps`` consecutive grid steps flattened
    into a single ``num_timesteps * 2 * SPAN`` vector -- exactly the layout the
    original code fed its visible layer.
    """
    width = 2 * SPAN
    examples = []
    for song in songs:
        usable = (song.shape[0] // num_timesteps) * num_timesteps
        if usable == 0:
            continue
        trimmed = song[:usable]
        examples.append(trimmed.reshape(usable // num_timesteps, num_timesteps * width))
    if not examples:
        return np.zeros((0, num_timesteps * width), dtype=np.float32)
    return np.concatenate(examples, axis=0).astype(np.float32)


def load_feature_songs(
    directory: str, min_steps: int = 32, steps_per_quarter: int = 4, normalize_key: bool = True
) -> List[np.ndarray]:
    """Parse songs into the drum+pitched feature representation used by the LSTM."""
    songs = []
    for path in find_midi_files(directory):
        try:
            song = song_to_features(path, steps_per_quarter=steps_per_quarter, normalize_key=normalize_key)
        except Exception as exc:
            print(f"  skipping {os.path.basename(path)}: {type(exc).__name__}: {exc}")
            continue
        if song.shape[0] > min_steps:
            songs.append(song)
    return songs


def build_sequences(songs: List[np.ndarray], seq_len: int, hop: int = 8) -> np.ndarray:
    """Slice songs into overlapping ``seq_len + 1`` windows for next-step training."""
    windows = []
    for song in songs:
        for start in range(0, song.shape[0] - seq_len - 1, hop):
            windows.append(song[start : start + seq_len + 1])
    if not windows:
        return np.zeros((0, seq_len + 1, N_FEATURES), dtype=np.float32)
    return np.stack(windows).astype(np.float32)
