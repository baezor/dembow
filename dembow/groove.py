"""The dembow drum backbone.

An LSTM trained on ~76 songs grooves most of the time, but it can drop the snare
or drift. The signature of reggaeton is its drum pattern, so this module pins
that down: it derives the canonical one-bar dembow groove straight from the
corpus (averaging drum onsets across every bar of every song, then keeping the
positions that fire often), and lays it under the model's bass and melody.

Run against the bundled corpus, the pattern that emerges is the textbook dembow:

    kick        X...X...X...X...   downbeats
    snare       ...X..X....X..X.   the iconic "boom-ch-boom-chick" (steps 3,6,11,14)
    closed_hat  X..XX.X.X..XX.X.   steady eighths

When no corpus is available we fall back to that exact pattern, hardcoded.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .representation import DRUM_SLICE, N_DRUMS

BAR_STEPS = 16  # one 4/4 bar on a 16th-note grid

# Hardcoded fallback, matching the groove the corpus reveals. Indices follow
# representation.DRUM_CLASSES: kick=0, snare=1, clap=2, closed_hat=3, open_hat=4.
_FALLBACK = {
    0: [0, 4, 8, 12],        # kick
    1: [3, 6, 11, 14],       # snare
    3: [0, 3, 4, 6, 8, 11, 12, 14],  # closed hat
    2: [0, 4, 8, 12],        # clap reinforcing the kick
}


def fallback_groove() -> np.ndarray:
    """The canonical dembow bar, as a ``[BAR_STEPS, N_DRUMS]`` matrix."""
    pattern = np.zeros((BAR_STEPS, N_DRUMS), dtype=np.float32)
    for cls, steps in _FALLBACK.items():
        for s in steps:
            pattern[s, cls] = 1.0
    return pattern


def canonical_groove(songs: List[np.ndarray], bar_steps: int = BAR_STEPS, threshold: float = 0.25) -> np.ndarray:
    """Derive a representative one-bar drum pattern from feature matrices.

    For each within-bar position and drum class we measure how often an onset
    occurs across the whole corpus, then keep the ones that fire in at least
    ``threshold`` of bars. Falls back to the hardcoded dembow if there's no data.
    """
    acc = np.zeros((bar_steps, N_DRUMS), dtype=np.float64)
    num_bars = 0
    for song in songs:
        drums = song[:, DRUM_SLICE]
        bars = drums.shape[0] // bar_steps
        if bars == 0:
            continue
        acc += drums[: bars * bar_steps].reshape(bars, bar_steps, N_DRUMS).sum(axis=0)
        num_bars += bars
    if num_bars == 0:
        return fallback_groove()
    prob = acc / num_bars
    pattern = (prob > threshold).astype(np.float32)
    if pattern.sum() == 0:
        return fallback_groove()
    return pattern


def tile_groove(pattern: np.ndarray, num_steps: int, offset: int = 0) -> np.ndarray:
    """Repeat a one-bar pattern to cover ``num_steps`` steps."""
    bar_steps = pattern.shape[0]
    idx = (np.arange(num_steps) + offset) % bar_steps
    return pattern[idx]


def groove_track(num_steps: int, songs: Optional[List[np.ndarray]] = None, offset: int = 0) -> np.ndarray:
    """Convenience: a full ``[num_steps, N_DRUMS]`` dembow drum track."""
    pattern = canonical_groove(songs) if songs else fallback_groove()
    return tile_groove(pattern, num_steps, offset=offset)
