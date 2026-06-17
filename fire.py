#!/usr/bin/env python3
"""Light the fire. 🔥

Trains the Dembow Transformer on the reggaeton corpus and immediately generates
a few songs. For more control use the CLI:

    dembow train
    dembow generate

or, without installing:

    python -m dembow.cli train
    python -m dembow.cli generate
"""

from dembow.train import train
from dembow.generate import generate


def main():
    train(data_dir="reggaeton_samples", checkpoint="dembow.pt")
    generate(checkpoint="dembow.pt", output_dir="generated")


if __name__ == "__main__":
    main()
