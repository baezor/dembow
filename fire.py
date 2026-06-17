#!/usr/bin/env python3
"""Light the fire. 🔥

The original entry point of Dembow trained the RBM and immediately sampled a few
patterns. This keeps that one-shot spirit while delegating to the modern package
underneath. For more control use the CLI:

    dembow train
    dembow generate

or, without installing:

    python -m dembow.cli train
    python -m dembow.cli generate

Originally based on Daniel Johnson's biaxial-rnn-music-composition and Dan
Shiebler's RBM-in-TensorFlow tutorial.
"""

from dembow.train import train_lstm
from dembow.generate import generate


def main():
    train_lstm(data_dir="reggaeton_samples", checkpoint="dembow.pt")
    generate(checkpoint="dembow.pt", output_dir="generated")


if __name__ == "__main__":
    main()
