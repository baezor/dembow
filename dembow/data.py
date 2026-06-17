"""Load the reggaeton corpus and turn it into token windows for training."""

from __future__ import annotations

import glob
import os
from typing import List

import numpy as np

from .tokenizer import BOS, EOS, PAD, encode


def find_midi_files(directory: str) -> List[str]:
    """Every MIDI file in ``directory`` (case-insensitive extension)."""
    files = glob.glob(os.path.join(directory, "*"))
    return sorted(f for f in files if f.lower().endswith((".mid", ".midi")))


def split_files(paths: List[str], val_frac: float, seed: int = 0) -> tuple[List[str], List[str]]:
    """Split a file list into (train, val) at the *song* level.

    Splitting by song -- not by window -- matters because pitch augmentation
    creates transposed copies of each song; a window-level split would leak those
    copies across the train/val boundary and make the validation loss a lie.
    """
    rng = np.random.RandomState(seed)
    order = list(paths)
    rng.shuffle(order)
    n_val = int(len(order) * val_frac)
    return order[n_val:], order[:n_val]


def encode_files(paths: List[str], transpositions: List[int] | None = None) -> List[List[int]]:
    """Encode a specific list of MIDI files (optionally pitch-augmented)."""
    if transpositions is None:
        transpositions = [0]
    songs = []
    for path in paths:
        for shift in transpositions:
            try:
                ids = encode(path, transpose=shift)
            except Exception as exc:
                print(f"  skipping {os.path.basename(path)} (shift {shift}): {type(exc).__name__}: {exc}")
                continue
            if len(ids) > 8:
                songs.append(ids)
    return songs


def encode_corpus(directory: str, transpositions: List[int] | None = None) -> List[List[int]]:
    """Encode every song, optionally augmenting with transposed copies.

    Pitch augmentation multiplies our tiny ~76-song corpus: each transpose is a
    musically valid variation, which helps the model generalize instead of just
    memorizing a handful of files.
    """
    return encode_files(find_midi_files(directory), transpositions)


def build_windows(songs: List[List[int]], seq_len: int) -> np.ndarray:
    """Slice token streams into fixed-length training windows.

    Each window is one training example; the model learns to predict every token
    from the ones before it. Short songs are padded; long songs yield several
    overlapping windows.
    """
    examples = []
    stride = seq_len // 2
    for ids in songs:
        if len(ids) <= seq_len:
            examples.append(ids + [PAD] * (seq_len - len(ids)))
        else:
            for start in range(0, len(ids) - seq_len + 1, stride):
                examples.append(ids[start : start + seq_len])
    if not examples:
        return np.zeros((0, seq_len), dtype=np.int64)
    return np.array(examples, dtype=np.int64)
