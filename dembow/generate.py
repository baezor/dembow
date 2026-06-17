"""Generate new reggaeton from a trained Dembow Transformer."""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch

from .data import find_midi_files
from .model import MusicTransformer
from .tokenizer import BAR, BOS, EOS, decode, encode


def _prime_tokens(seed_dir: Optional[str], prime_bars: int) -> List[int]:
    """Build a priming prompt from the opening bars of a real reggaeton song."""
    if not seed_dir or not os.path.isdir(seed_dir):
        return [BOS]
    for path in find_midi_files(seed_dir):
        ids = encode(path)
        # Cut after `prime_bars` BAR tokens so we hand the model a few real bars.
        bars = 0
        for i, t in enumerate(ids):
            if t == BAR:
                bars += 1
                if bars > prime_bars:
                    return ids[:i]
        if len(ids) > 4:
            return ids
    return [BOS]


def generate(
    checkpoint: str = "dembow.pt",
    output_dir: str = "generated",
    num_samples: int = 5,
    max_new_tokens: int = 800,
    temperature: float = 1.0,
    top_p: float = 0.92,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 0,
    prime_bars: int = 2,
    seed_dir: Optional[str] = "reggaeton_samples",
    tempo_bpm: float = 95.0,
    random_seed: int = 0,
    device: str | None = None,
) -> List[str]:
    """Sample songs from the model and write them out as MIDI files."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)

    model = MusicTransformer.load(checkpoint, device=device)
    prompt_ids = _prime_tokens(seed_dir, prime_bars)

    written = []
    for i in range(num_samples):
        prompt = torch.tensor(prompt_ids, dtype=torch.long)
        ids = model.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=temperature,
            top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size, eos_id=EOS, device=device,
        )
        midi_file = decode(ids, tempo_bpm=tempo_bpm)
        # A valid song needs at least a couple of note tracks (beyond the meta track).
        if len(midi_file.tracks) < 2:
            continue
        out_path = os.path.join(output_dir, f"dembow_{i}.mid")
        midi_file.save(out_path)
        written.append(out_path)

    print(f"Wrote {len(written)} MIDI file(s) to '{output_dir}/'")
    return written
