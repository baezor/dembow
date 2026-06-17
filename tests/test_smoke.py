"""Fast end-to-end smoke test: tokenizer round-trip, a tiny train, generation.

Run with:  pytest tests/   (or  python tests/test_smoke.py)
"""

import os
import tempfile

import numpy as np

from dembow.data import build_windows, encode_corpus, find_midi_files
from dembow.model import ModelConfig, MusicTransformer
from dembow.tokenizer import BAR, BOS, EOS, VOCAB, decode, encode

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "reggaeton_samples")


def test_finds_uppercase_midi():
    files = find_midi_files(SAMPLES)
    assert files
    # The corpus contains uppercase .MID files the old glob missed.
    assert any(f.endswith(".MID") for f in files)


def test_tokenizer_roundtrip():
    files = find_midi_files(SAMPLES)
    ids = encode(files[0])
    assert ids[0] == BOS and ids[-1] == EOS
    assert all(0 <= t < len(VOCAB) for t in ids)
    assert BAR in ids

    midi_file = decode(ids)
    assert len(midi_file.tracks) >= 2  # meta + at least one instrument
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rt.mid")
        midi_file.save(path)
        # The decoded file must itself be re-encodable.
        assert len(encode(path)) > 4


def test_transpose_augmentation_shifts_pitch():
    files = find_midi_files(SAMPLES)
    base = encode(files[0], transpose=0)
    up = encode(files[0], transpose=2)
    # Augmentation should change the token stream (different pitches).
    assert base != up


def test_build_windows():
    songs = encode_corpus(SAMPLES, transpositions=[0])
    windows = build_windows(songs, seq_len=128)
    assert windows.ndim == 2 and windows.shape[1] == 128
    assert windows.shape[0] > 0


def test_train_and_generate_tiny():
    import torch

    songs = encode_corpus(SAMPLES, transpositions=[0])
    windows = build_windows(songs, seq_len=128)
    config = ModelConfig(vocab_size=len(VOCAB), d_model=64, n_layers=2, n_heads=2, max_len=128)
    model = MusicTransformer(config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data = torch.from_numpy(windows[:16])
    first = last = None
    for _ in range(8):
        logits = model(data[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), data[:, 1:].reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
        last = float(loss)
    assert last < first  # the model is learning

    ids = model.generate(torch.tensor([BOS]), max_new_tokens=120, eos_id=EOS)
    assert len(ids) > 1 and all(0 <= t < len(VOCAB) for t in ids)
    midi_file = decode(ids)  # generated tokens must decode without error
    assert len(midi_file.tracks) >= 1

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.pt")
        model.save(path)
        loaded = MusicTransformer.load(path)
        assert loaded.config.d_model == 64


if __name__ == "__main__":
    test_finds_uppercase_midi()
    test_tokenizer_roundtrip()
    test_transpose_augmentation_shifts_pitch()
    test_build_windows()
    test_train_and_generate_tiny()
    print("All smoke tests passed.")
