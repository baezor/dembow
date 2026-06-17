"""A fast end-to-end smoke test: MIDI round-trip, a tiny train, and generation.

Run with:  python -m pytest tests/  (or just  python tests/test_smoke.py)
"""

import os
import tempfile

import numpy as np

from dembow.dataset import (
    build_examples,
    build_sequences,
    find_midi_files,
    load_feature_songs,
    load_songs,
)
from dembow.lstm import DembowLSTM, LSTMConfig
from dembow.midi_io import (
    SPAN,
    midi_to_note_state_matrix,
    note_state_matrix_to_midi,
)
from dembow.rbm import RBM, RBMConfig
from dembow.representation import (
    DRUM_SLICE,
    N_FEATURES,
    PLAY_SLICE,
    features_to_midi,
    song_to_features,
)

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


def test_representation_separates_drums():
    files = find_midi_files(SAMPLES)
    feats = song_to_features(files[0])
    assert feats.shape[1] == N_FEATURES
    # A reggaeton file should have both drum and pitched content.
    assert feats[:, DRUM_SLICE].sum() > 0
    assert feats[:, PLAY_SLICE].sum() > 0
    with tempfile.TemporaryDirectory() as d:
        out = features_to_midi(feats[:64], os.path.join(d, "feat.mid"))
        assert os.path.exists(out)


def test_lstm_train_and_generate_tiny():
    songs = load_feature_songs(SAMPLES)
    seqs = build_sequences(songs, seq_len=16, hop=16)
    assert seqs.shape[0] > 0 and seqs.shape[2] == N_FEATURES

    import torch

    model = DembowLSTM(LSTMConfig(hidden=32, layers=1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    data = torch.from_numpy(seqs[:32])
    for _ in range(5):
        logits, _ = model(data[:, :-1, :])
        loss = loss_fn(logits, data[:, 1:, :])
        opt.zero_grad()
        loss.backward()
        opt.step()

    out = model.generate(songs[0][:16], num_steps=24, max_pitched=4)
    assert out.shape == (24, N_FEATURES)
    # Sparse and binary -- music, not a wall of noise.
    assert set(np.unique(out)).issubset({0.0, 1.0})
    assert out[:, PLAY_SLICE].sum(axis=1).max() <= 4

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "lstm.pt")
        model.save(path)
        loaded = DembowLSTM.load(path)
        assert loaded.config.hidden == 32


def test_groove_backbone():
    from dembow.groove import canonical_groove, fallback_groove, tile_groove
    from dembow.representation import N_DRUMS

    songs = load_feature_songs(SAMPLES)
    pattern = canonical_groove(songs)
    assert pattern.shape == (16, N_DRUMS)
    # The corpus groove must include the signature: kick on beat 1, a snare.
    assert pattern[0, 0] == 1.0          # kick on the downbeat
    assert pattern[:, 1].sum() > 0       # snare present

    fb = fallback_groove()
    assert fb[3, 1] == 1.0 and fb[6, 1] == 1.0  # the iconic dembow snare hits

    track = tile_groove(pattern, 40)
    assert track.shape == (40, N_DRUMS)


def test_lstm_generate_with_groove_locks_drums():
    import torch

    from dembow.groove import tile_groove, fallback_groove
    from dembow.representation import DRUM_SLICE

    songs = load_feature_songs(SAMPLES)
    model = DembowLSTM(LSTMConfig(hidden=32, layers=1))
    drum_track = tile_groove(fallback_groove(), 32)
    out = model.generate(songs[0][:16], num_steps=32, drum_track=drum_track)
    # Drums in the output must match the locked groove exactly.
    assert np.array_equal(out[:, DRUM_SLICE], drum_track)


if __name__ == "__main__":
    test_finds_uppercase_midi()
    test_midi_roundtrip()
    test_train_and_generate_tiny()
    test_representation_separates_drums()
    test_lstm_train_and_generate_tiny()
    test_groove_backbone()
    test_lstm_generate_with_groove_locks_drums()
    print("All smoke tests passed.")
