"""Dembow -- a Transformer that generates reggaeton.

The first A.I. that generates reggaeton hits, rebuilt around a modern,
event-based music language model.

The 2016 original trained a Restricted Boltzmann Machine on a binary piano roll.
This version replaces that entirely: songs are tokenized into a REMI-style stream
of musical events (bar, position, instrument, pitch, duration, velocity) and a
decoder-only Transformer learns to generate them one token at a time -- the same
recipe used by modern symbolic-music models.
"""

from .tokenizer import VOCAB, Vocab, encode, decode
from .model import MusicTransformer, ModelConfig

__version__ = "2.2.0"

__all__ = [
    "VOCAB",
    "Vocab",
    "encode",
    "decode",
    "MusicTransformer",
    "ModelConfig",
]
