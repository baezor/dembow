"""A fast end-to-end smoke test: MIDI round-trip, a tiny train, and generation.

Run with:  python -m pytest tests/  (or just  python tests/test_smoke.py)
"""

import os
import tempfile

import numpy as np

from dembow.dataset import build_examples, find_midi_files, load_songs
from dembow.midi_io import (
    SPAN,
    midi_to_note_state_matrix,
    note_state_matrix_to_midi,
)
from dembow.rbm import RBM, RBMConfig

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "reggaeton_samples")


def test_finds_uppercase_midi():
    files = find_midi_files(SAMPLES)
    assert files, "no MIDI files found"
    # The corpus contains uppercase .MID files the old glob missed.
    assert any(f.endswith(".MID") for f in files)


def test_midi_roundtrip():
    files = find_midi_files(SAMPLES)
    matrix = midi_to_note_state_matrix(files[0])
    assert matrix.ndim == 2 and matrix.shape[1] == 2 * SPAN
    with tempfile.TemporaryDirectory() as d:
        out = note_state_matrix_to_midi(matrix, os.path.join(d, "rt.mid"))
        assert os.path.exists(out)
        reparsed = midi_to_note_state_matrix(out)
        # The "play" content should survive the round trip.
        assert reparsed[:, :SPAN].sum() > 0


def test_train_and_generate_tiny():
    num_timesteps = 8
    songs = load_songs(SAMPLES)
    examples = build_examples(songs, num_timesteps)
    assert examples.shape[0] > 0

    n_visible = num_timesteps * 2 * SPAN
    cfg = RBMConfig(n_visible=n_visible, n_hidden=16, num_timesteps=num_timesteps, span=SPAN)
    rbm = RBM(cfg, seed=0)

    import torch

    data = torch.from_numpy(examples[:64])
    err0 = rbm.contrastive_divergence(data, lr=0.01)
    for _ in range(50):
        err = rbm.contrastive_divergence(data, lr=0.01)
    assert err <= err0 + 1e-6  # learning should not blow up

    samples = rbm.generate(4).numpy()
    assert samples.shape == (4, n_visible)
    assert set(np.unique(samples)).issubset({0.0, 1.0})

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt")
        rbm.save(path)
        loaded = RBM.load(path)
        assert loaded.W.shape == rbm.W.shape


if __name__ == "__main__":
    test_finds_uppercase_midi()
    test_midi_roundtrip()
    test_train_and_generate_tiny()
    print("All smoke tests passed.")
