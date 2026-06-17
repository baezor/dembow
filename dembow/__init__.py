"""Dembow -- a small neural network that generates reggaeton.

The first A.I. that generates reggaeton hits, brought back to life.

The original 2016 project trained a Restricted Boltzmann Machine (RBM) on a
corpus of reggaeton MIDI files and Gibbs-sampled new patterns out of it. This
revival keeps that exact idea -- an RBM learning the dembow groove -- but runs
on a modern stack: Python 3, PyTorch, and ``mido`` instead of TensorFlow 1.x,
Python 2, and the long-dead ``python-midi`` library.
"""

from .rbm import RBM
from .lstm import DembowLSTM, LSTMConfig
from .midi_io import (
    LOWER_BOUND,
    UPPER_BOUND,
    SPAN,
    midi_to_note_state_matrix,
    note_state_matrix_to_midi,
)
from .representation import (
    N_FEATURES,
    DRUM_CLASSES,
    song_to_features,
    features_to_midi,
)
from .groove import canonical_groove, fallback_groove, groove_track

__version__ = "1.2.0"

__all__ = [
    "RBM",
    "DembowLSTM",
    "LSTMConfig",
    "LOWER_BOUND",
    "UPPER_BOUND",
    "SPAN",
    "N_FEATURES",
    "DRUM_CLASSES",
    "midi_to_note_state_matrix",
    "note_state_matrix_to_midi",
    "song_to_features",
    "features_to_midi",
    "canonical_groove",
    "fallback_groove",
    "groove_track",
]
